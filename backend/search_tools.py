from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol

from vector_store import SearchResults, VectorStore


class Tool(ABC):
    """Abstract base class for all tools"""

    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in the course content",
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')",
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)",
                    },
                },
                "required": ["query"],
            },
        }

    def execute(
        self,
        query: str,
        course_name: Optional[str] = None,
        lesson_number: Optional[int] = None,
    ) -> str:
        """
        Execute the search tool with given parameters.

        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter

        Returns:
            Formatted search results or error message
        """

        # Use the vector store's unified search interface
        results = self.store.search(
            query=query, course_name=course_name, lesson_number=lesson_number
        )

        # Handle errors
        if results.error:
            return results.error

        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."

        # Format and return results
        return self._format_results(results)

    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []  # Track sources for the UI with lesson links

        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get("course_title", "unknown")
            lesson_num = meta.get("lesson_number")

            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"

            # Build source with link information
            source_display = course_title
            if lesson_num is not None:
                source_display += f" - Lesson {lesson_num}"

            # Try to get lesson link if lesson number exists
            lesson_link = None
            if lesson_num is not None and course_title != "unknown":
                lesson_link = self.store.get_lesson_link(course_title, lesson_num)

            # Create structured source data
            source_data = {"display_text": source_display, "lesson_link": lesson_link}
            sources.append(source_data)

            formatted.append(f"{header}\n{doc}")

        # Store structured sources for retrieval
        self.last_sources = sources

        return "\n\n".join(formatted)


class ToolManager:
    """Manages available tools for the AI"""

    def __init__(self):
        self.tools = {}
        self.session_sources = []  # Accumulate sources across rounds

    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"

        result = self.tools[tool_name].execute(**kwargs)

        # Accumulate sources from this tool call
        if hasattr(self.tools[tool_name], "last_sources"):
            self.session_sources.extend(self.tools[tool_name].last_sources)

        return result

    def get_last_sources(self) -> list:
        """Get sources from the last search operation"""
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, "last_sources") and tool.last_sources:
                return tool.last_sources
        return []

    def get_all_sources_from_session(self) -> list:
        """Get all sources accumulated during the current session"""
        return self.session_sources.copy()

    def reset_sources(self):
        """Reset all source tracking"""
        self.session_sources = []
        for tool in self.tools.values():
            if hasattr(tool, "last_sources"):
                tool.last_sources = []


class CourseOutlineTool(Tool):
    """Tool for getting course outlines with lesson structure"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "get_course_outline",
            "description": "Get course outline including title, link, and complete lesson list",
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')",
                    }
                },
                "required": ["course_name"],
            },
        }

    def execute(self, course_name: str) -> str:
        """
        Execute the outline tool to get course structure.

        Args:
            course_name: Course name to get outline for

        Returns:
            Formatted course outline or error message
        """
        # Resolve course name using vector store's semantic matching
        course_title = self.store._resolve_course_name(course_name)
        if not course_title:
            return f"No course found matching '{course_name}'"

        # Get course metadata from catalog
        try:
            results = self.store.course_catalog.get(ids=[course_title])
            if not results or not results["metadatas"]:
                return f"Course metadata not found for '{course_title}'"

            metadata = results["metadatas"][0]
            return self._format_outline(metadata)

        except Exception as e:
            return f"Error retrieving course outline: {str(e)}"

    def _format_outline(self, metadata: Dict[str, Any]) -> str:
        """Format course outline from metadata"""
        import json

        title = metadata.get("title", "Unknown Course")
        course_link = metadata.get("course_link", "")
        instructor = metadata.get("instructor", "")
        lessons_json = metadata.get("lessons_json", "[]")

        # Parse lessons
        try:
            lessons = json.loads(lessons_json)
        except json.JSONDecodeError:
            lessons = []

        # Format output
        outline = [f"Course: {title}"]

        if instructor:
            outline.append(f"Instructor: {instructor}")

        if course_link:
            outline.append(f"Course Link: {course_link}")

        if lessons:
            outline.append("\nLessons:")
            for lesson in lessons:
                lesson_num = lesson.get("lesson_number", "")
                lesson_title = lesson.get("lesson_title", "")
                outline.append(f"  Lesson {lesson_num}: {lesson_title}")
        else:
            outline.append("\nNo lessons found")

        # Track source for UI
        source_data = {
            "display_text": f"{title} - Course Outline",
            "lesson_link": course_link,
        }
        self.last_sources = [source_data]

        return "\n".join(outline)
