import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add backend to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import SearchResults, VectorStore


class TestCourseSearchTool:
    """Test suite for CourseSearchTool functionality"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock VectorStore for testing"""
        mock_store = Mock(spec=VectorStore)
        return mock_store

    @pytest.fixture
    def search_tool(self, mock_vector_store):
        """Create CourseSearchTool instance for testing"""
        return CourseSearchTool(mock_vector_store)

    @pytest.fixture
    def sample_search_results(self):
        """Create sample search results"""
        return SearchResults(
            documents=[
                "This is lesson 0 content about introduction",
                "This is lesson 1 content about advanced topics",
            ],
            metadata=[
                {"course_title": "Test Course", "lesson_number": 0},
                {"course_title": "Test Course", "lesson_number": 1},
            ],
            distances=[0.1, 0.2],
        )

    def test_tool_definition(self, search_tool):
        """Test CourseSearchTool tool definition structure"""
        print("\\n=== Testing CourseSearchTool Definition ===")
        try:
            definition = search_tool.get_tool_definition()

            # Check required fields
            assert "name" in definition
            assert definition["name"] == "search_course_content"
            assert "description" in definition
            assert "input_schema" in definition
            print("✅ Tool definition has required fields")

            # Check input schema structure
            schema = definition["input_schema"]
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "required" in schema
            assert "query" in schema["required"]
            print("✅ Input schema structure valid")

            # Check properties
            props = schema["properties"]
            assert "query" in props
            assert "course_name" in props
            assert "lesson_number" in props
            print("✅ Required properties present")

        except Exception as e:
            print(f"❌ Tool definition test failed: {e}")
            raise

    def test_execute_basic_search(
        self, search_tool, mock_vector_store, sample_search_results
    ):
        """Test basic search execution"""
        print("\\n=== Testing Basic Search Execution ===")
        try:
            # Setup mock to return successful results
            mock_vector_store.search.return_value = sample_search_results
            mock_vector_store.get_lesson_link.return_value = (
                "https://example.com/lesson0"
            )

            # Execute search
            result = search_tool.execute("introduction")

            # Verify vector store was called correctly
            mock_vector_store.search.assert_called_once_with(
                query="introduction", course_name=None, lesson_number=None
            )

            # Verify result format
            assert isinstance(result, str)
            assert len(result) > 0
            assert "Test Course" in result
            print("✅ Basic search execution successful")

        except Exception as e:
            print(f"❌ Basic search test failed: {e}")
            raise

    def test_execute_with_course_filter(
        self, search_tool, mock_vector_store, sample_search_results
    ):
        """Test search execution with course filter"""
        print("\\n=== Testing Search with Course Filter ===")
        try:
            mock_vector_store.search.return_value = sample_search_results
            mock_vector_store.get_lesson_link.return_value = None

            result = search_tool.execute("advanced", course_name="Test Course")

            # Verify parameters passed correctly
            mock_vector_store.search.assert_called_once_with(
                query="advanced", course_name="Test Course", lesson_number=None
            )

            assert "Test Course" in result
            print("✅ Course-filtered search successful")

        except Exception as e:
            print(f"❌ Course-filtered search test failed: {e}")
            raise

    def test_execute_with_lesson_filter(
        self, search_tool, mock_vector_store, sample_search_results
    ):
        """Test search execution with lesson filter"""
        print("\\n=== Testing Search with Lesson Filter ===")
        try:
            mock_vector_store.search.return_value = sample_search_results
            mock_vector_store.get_lesson_link.return_value = (
                "https://example.com/lesson1"
            )

            result = search_tool.execute("topics", lesson_number=1)

            # Verify parameters passed correctly
            mock_vector_store.search.assert_called_once_with(
                query="topics", course_name=None, lesson_number=1
            )

            assert "Lesson 1" in result
            print("✅ Lesson-filtered search successful")

        except Exception as e:
            print(f"❌ Lesson-filtered search test failed: {e}")
            raise

    def test_execute_with_both_filters(
        self, search_tool, mock_vector_store, sample_search_results
    ):
        """Test search execution with both course and lesson filters"""
        print("\\n=== Testing Search with Both Filters ===")
        try:
            mock_vector_store.search.return_value = sample_search_results
            mock_vector_store.get_lesson_link.return_value = (
                "https://example.com/lesson1"
            )

            result = search_tool.execute(
                "topics", course_name="Test Course", lesson_number=1
            )

            # Verify parameters passed correctly
            mock_vector_store.search.assert_called_once_with(
                query="topics", course_name="Test Course", lesson_number=1
            )

            assert "Test Course" in result
            assert "Lesson 1" in result
            print("✅ Combined filters search successful")

        except Exception as e:
            print(f"❌ Combined filters search test failed: {e}")
            raise

    def test_execute_error_handling(self, search_tool, mock_vector_store):
        """Test error handling in execute method"""
        print("\\n=== Testing Error Handling ===")
        try:
            # Test vector store error
            error_result = SearchResults.empty("Database connection error")
            mock_vector_store.search.return_value = error_result

            result = search_tool.execute("test query")

            assert result == "Database connection error"
            print("✅ Vector store error handling successful")

        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
            raise

    def test_execute_empty_results(self, search_tool, mock_vector_store):
        """Test handling of empty search results"""
        print("\\n=== Testing Empty Results Handling ===")
        try:
            # Test empty results
            empty_result = SearchResults([], [], [])
            mock_vector_store.search.return_value = empty_result

            result = search_tool.execute("nonexistent topic")

            expected = "No relevant content found."
            assert result == expected
            print("✅ Empty results handling successful")

            # Test empty results with filters
            result = search_tool.execute(
                "topic", course_name="Test Course", lesson_number=1
            )
            expected = "No relevant content found in course 'Test Course' in lesson 1."
            assert result == expected
            print("✅ Empty results with filters handling successful")

        except Exception as e:
            print(f"❌ Empty results test failed: {e}")
            raise

    def test_result_formatting(self, search_tool, mock_vector_store):
        """Test result formatting functionality"""
        print("\\n=== Testing Result Formatting ===")
        try:
            # Create test results with various metadata combinations
            test_results = SearchResults(
                documents=[
                    "Content from lesson 0",
                    "Content from lesson 1",
                    "Content with unknown lesson",
                ],
                metadata=[
                    {"course_title": "Course A", "lesson_number": 0},
                    {"course_title": "Course B", "lesson_number": 1},
                    {"course_title": "Course C"},  # No lesson number
                ],
                distances=[0.1, 0.2, 0.3],
            )

            mock_vector_store.search.return_value = test_results
            mock_vector_store.get_lesson_link.side_effect = [
                "https://example.com/lesson0",
                "https://example.com/lesson1",
                None,
            ]

            result = search_tool.execute("content")

            # Verify formatting
            assert "[Course A - Lesson 0]" in result
            assert "[Course B - Lesson 1]" in result
            assert "[Course C]" in result  # No lesson number case
            assert "Content from lesson 0" in result
            assert "Content from lesson 1" in result
            assert "Content with unknown lesson" in result

            print("✅ Result formatting successful")

        except Exception as e:
            print(f"❌ Result formatting test failed: {e}")
            raise

    def test_source_tracking(
        self, search_tool, mock_vector_store, sample_search_results
    ):
        """Test source tracking functionality"""
        print("\\n=== Testing Source Tracking ===")
        try:
            mock_vector_store.search.return_value = sample_search_results
            mock_vector_store.get_lesson_link.side_effect = [
                "https://example.com/lesson0",
                "https://example.com/lesson1",
            ]

            # Execute search
            result = search_tool.execute("test")

            # Check that sources were tracked
            assert hasattr(search_tool, "last_sources")
            sources = search_tool.last_sources
            assert len(sources) == 2

            # Verify source structure
            for source in sources:
                assert "display_text" in source
                assert "lesson_link" in source

            # Verify specific source content
            assert sources[0]["display_text"] == "Test Course - Lesson 0"
            assert sources[0]["lesson_link"] == "https://example.com/lesson0"
            assert sources[1]["display_text"] == "Test Course - Lesson 1"
            assert sources[1]["lesson_link"] == "https://example.com/lesson1"

            print("✅ Source tracking successful")

        except Exception as e:
            print(f"❌ Source tracking test failed: {e}")
            raise


class TestCourseOutlineTool:
    """Test suite for CourseOutlineTool functionality"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock VectorStore for testing"""
        mock_store = Mock(spec=VectorStore)
        return mock_store

    @pytest.fixture
    def outline_tool(self, mock_vector_store):
        """Create CourseOutlineTool instance for testing"""
        return CourseOutlineTool(mock_vector_store)

    def test_outline_tool_definition(self, outline_tool):
        """Test CourseOutlineTool tool definition"""
        print("\\n=== Testing CourseOutlineTool Definition ===")
        try:
            definition = outline_tool.get_tool_definition()

            assert definition["name"] == "get_course_outline"
            assert "course_name" in definition["input_schema"]["required"]
            print("✅ Outline tool definition successful")

        except Exception as e:
            print(f"❌ Outline tool definition test failed: {e}")
            raise

    def test_outline_execute(self, outline_tool, mock_vector_store):
        """Test outline tool execution"""
        print("\\n=== Testing Outline Tool Execution ===")
        try:
            # Mock course resolution and catalog data
            mock_vector_store._resolve_course_name.return_value = "Test Course"
            mock_vector_store.course_catalog.get.return_value = {
                "metadatas": [
                    {
                        "title": "Test Course",
                        "instructor": "Test Instructor",
                        "course_link": "https://example.com/course",
                        "lessons_json": '[{"lesson_number": 0, "lesson_title": "Introduction"}, {"lesson_number": 1, "lesson_title": "Advanced"}]',
                    }
                ]
            }

            result = outline_tool.execute("Test")

            assert "Course: Test Course" in result
            assert "Instructor: Test Instructor" in result
            assert "Lesson 0: Introduction" in result
            assert "Lesson 1: Advanced" in result
            print("✅ Outline tool execution successful")

        except Exception as e:
            print(f"❌ Outline tool execution test failed: {e}")
            raise

    def test_outline_no_course_found(self, outline_tool, mock_vector_store):
        """Test outline tool with no course found"""
        print("\\n=== Testing Outline Tool No Course Found ===")
        try:
            mock_vector_store._resolve_course_name.return_value = None

            result = outline_tool.execute("NonexistentCourse")
            assert "No course found matching" in result
            print("✅ Outline tool no course found handling successful")

        except Exception as e:
            print(f"❌ Outline tool no course found test failed: {e}")
            raise


class TestToolManager:
    """Test suite for ToolManager functionality"""

    @pytest.fixture
    def tool_manager(self):
        """Create ToolManager instance for testing"""
        return ToolManager()

    @pytest.fixture
    def mock_search_tool(self, mock_vector_store):
        """Create mock search tool for testing"""
        return CourseSearchTool(mock_vector_store)

    def test_tool_registration(self, tool_manager, mock_search_tool):
        """Test tool registration functionality"""
        print("\\n=== Testing Tool Registration ===")
        try:
            # Register tool
            tool_manager.register_tool(mock_search_tool)

            # Verify tool was registered
            assert "search_course_content" in tool_manager.tools
            assert tool_manager.tools["search_course_content"] == mock_search_tool
            print("✅ Tool registration successful")

        except Exception as e:
            print(f"❌ Tool registration test failed: {e}")
            raise

    def test_get_tool_definitions(self, tool_manager, mock_search_tool):
        """Test getting tool definitions"""
        print("\\n=== Testing Get Tool Definitions ===")
        try:
            tool_manager.register_tool(mock_search_tool)

            definitions = tool_manager.get_tool_definitions()

            assert len(definitions) == 1
            assert definitions[0]["name"] == "search_course_content"
            print("✅ Get tool definitions successful")

        except Exception as e:
            print(f"❌ Get tool definitions test failed: {e}")
            raise

    def test_execute_tool(self, tool_manager, mock_search_tool, mock_vector_store):
        """Test tool execution through manager"""
        print("\\n=== Testing Tool Execution ===")
        try:
            # Setup mocks
            sample_results = SearchResults(
                documents=["Test content"],
                metadata=[{"course_title": "Test Course", "lesson_number": 0}],
                distances=[0.1],
            )
            mock_vector_store.search.return_value = sample_results
            mock_vector_store.get_lesson_link.return_value = None

            tool_manager.register_tool(mock_search_tool)

            # Execute tool
            result = tool_manager.execute_tool("search_course_content", query="test")

            assert isinstance(result, str)
            assert len(result) > 0
            print("✅ Tool execution successful")

            # Test nonexistent tool
            result = tool_manager.execute_tool("nonexistent_tool")
            assert "Tool 'nonexistent_tool' not found" in result
            print("✅ Nonexistent tool handling successful")

        except Exception as e:
            print(f"❌ Tool execution test failed: {e}")
            raise

    def test_source_management(self, tool_manager, mock_search_tool, mock_vector_store):
        """Test source tracking and management"""
        print("\\n=== Testing Source Management ===")
        try:
            # Setup mock with sources
            sample_results = SearchResults(
                documents=["Test content"],
                metadata=[{"course_title": "Test Course", "lesson_number": 0}],
                distances=[0.1],
            )
            mock_vector_store.search.return_value = sample_results
            mock_vector_store.get_lesson_link.return_value = (
                "https://example.com/lesson0"
            )

            tool_manager.register_tool(mock_search_tool)

            # Execute tool and check sources
            tool_manager.execute_tool("search_course_content", query="test")

            sources = tool_manager.get_last_sources()
            assert len(sources) == 1
            assert sources[0]["display_text"] == "Test Course - Lesson 0"
            print("✅ Source retrieval successful")

            # Reset sources
            tool_manager.reset_sources()
            sources = tool_manager.get_last_sources()
            assert len(sources) == 0
            print("✅ Source reset successful")

        except Exception as e:
            print(f"❌ Source management test failed: {e}")
            raise


if __name__ == "__main__":
    print("Running CourseSearchTool tests...")
    pytest.main([__file__, "-v", "-s"])
