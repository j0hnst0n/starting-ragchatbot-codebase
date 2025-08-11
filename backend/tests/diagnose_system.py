#!/usr/bin/env python3
"""
System Diagnostic Script for RAG Chatbot

This script performs comprehensive diagnostics to identify why the RAG chatbot
returns "query failed" for content-related questions.
"""

import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

# Add backend to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all the system components
try:
    from ai_generator import AIGenerator
    from config import config
    from document_processor import DocumentProcessor
    from models import Course, CourseChunk, Lesson
    from rag_system import RAGSystem
    from search_tools import CourseSearchTool, ToolManager
    from session_manager import SessionManager
    from vector_store import VectorStore
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class SystemDiagnostics:
    """Comprehensive system diagnostics"""

    def __init__(self):
        self.issues = []
        self.warnings = []
        self.info = []

    def log_issue(self, message: str):
        """Log a critical issue"""
        self.issues.append(message)
        print(f"❌ ISSUE: {message}")

    def log_warning(self, message: str):
        """Log a warning"""
        self.warnings.append(message)
        print(f"⚠️  WARNING: {message}")

    def log_info(self, message: str):
        """Log informational message"""
        self.info.append(message)
        print(f"ℹ️  INFO: {message}")

    def log_success(self, message: str):
        """Log success message"""
        print(f"✅ SUCCESS: {message}")

    def check_environment(self) -> Dict[str, Any]:
        """Check environment and configuration"""
        print("\\n🔍 CHECKING ENVIRONMENT AND CONFIGURATION")
        print("=" * 50)

        env_status = {}

        # Check API key
        api_key = config.ANTHROPIC_API_KEY
        if not api_key:
            self.log_issue("ANTHROPIC_API_KEY not set in environment")
            env_status["api_key"] = "missing"
        elif api_key == "":
            self.log_issue("ANTHROPIC_API_KEY is empty string")
            env_status["api_key"] = "empty"
        elif len(api_key) < 20:
            self.log_warning("ANTHROPIC_API_KEY appears to be too short")
            env_status["api_key"] = "suspicious"
        else:
            self.log_success("ANTHROPIC_API_KEY is present")
            env_status["api_key"] = "present"

        # Check configuration values
        config_values = {
            "ANTHROPIC_MODEL": config.ANTHROPIC_MODEL,
            "EMBEDDING_MODEL": config.EMBEDDING_MODEL,
            "CHUNK_SIZE": config.CHUNK_SIZE,
            "CHUNK_OVERLAP": config.CHUNK_OVERLAP,
            "MAX_RESULTS": config.MAX_RESULTS,
            "MAX_HISTORY": config.MAX_HISTORY,
            "CHROMA_PATH": config.CHROMA_PATH,
        }

        for key, value in config_values.items():
            self.log_info(f"{key}: {value}")
            env_status[key.lower()] = value

        # Check ChromaDB path
        if os.path.exists(config.CHROMA_PATH):
            self.log_success(f"ChromaDB path exists: {config.CHROMA_PATH}")
            env_status["chroma_path_exists"] = True
        else:
            self.log_warning(f"ChromaDB path does not exist: {config.CHROMA_PATH}")
            env_status["chroma_path_exists"] = False

        return env_status

    def check_vector_store(self) -> Dict[str, Any]:
        """Check vector store functionality"""
        print("\\n🔍 CHECKING VECTOR STORE")
        print("=" * 50)

        store_status = {}

        try:
            # Initialize vector store
            vector_store = VectorStore(
                chroma_path=config.CHROMA_PATH,
                embedding_model=config.EMBEDDING_MODEL,
                max_results=config.MAX_RESULTS,
            )
            self.log_success("VectorStore initialized successfully")
            store_status["initialization"] = "success"

            # Check collections
            try:
                course_count = vector_store.get_course_count()
                self.log_info(f"Course count in vector store: {course_count}")
                store_status["course_count"] = course_count

                if course_count == 0:
                    self.log_warning(
                        "Vector store contains no courses - this could explain query failures"
                    )

            except Exception as e:
                self.log_issue(f"Error getting course count: {e}")
                store_status["course_count"] = "error"

            # Check course titles
            try:
                course_titles = vector_store.get_existing_course_titles()
                self.log_info(f"Course titles: {course_titles}")
                store_status["course_titles"] = course_titles

            except Exception as e:
                self.log_issue(f"Error getting course titles: {e}")
                store_status["course_titles"] = "error"

            # Test basic search functionality
            try:
                test_results = vector_store.search("test query")
                if test_results.error:
                    self.log_warning(
                        f"Test search returned error: {test_results.error}"
                    )
                    store_status["search_test"] = f"error: {test_results.error}"
                else:
                    self.log_success(
                        f"Test search successful, returned {len(test_results.documents)} results"
                    )
                    store_status["search_test"] = (
                        f"success: {len(test_results.documents)} results"
                    )

            except Exception as e:
                self.log_issue(f"Error during test search: {e}")
                store_status["search_test"] = f"exception: {str(e)}"

            # Check ChromaDB collections directly
            try:
                collections = vector_store.client.list_collections()
                collection_info = {}
                for collection in collections:
                    try:
                        count = collection.count()
                        collection_info[collection.name] = count
                        self.log_info(
                            f"Collection '{collection.name}' has {count} items"
                        )
                    except Exception as e:
                        collection_info[collection.name] = f"error: {str(e)}"
                        self.log_warning(
                            f"Error counting collection '{collection.name}': {e}"
                        )

                store_status["collections"] = collection_info

            except Exception as e:
                self.log_issue(f"Error accessing ChromaDB collections: {e}")
                store_status["collections"] = f"error: {str(e)}"

        except Exception as e:
            self.log_issue(f"Failed to initialize VectorStore: {e}")
            store_status["initialization"] = f"error: {str(e)}"
            return store_status

        return store_status

    def check_document_processing(self) -> Dict[str, Any]:
        """Check document processing functionality"""
        print("\\n🔍 CHECKING DOCUMENT PROCESSING")
        print("=" * 50)

        doc_status = {}

        try:
            # Check docs folder
            docs_path = "../docs"
            if os.path.exists(docs_path):
                self.log_success(f"Docs folder exists: {docs_path}")

                # List files in docs folder
                doc_files = [
                    f
                    for f in os.listdir(docs_path)
                    if f.lower().endswith((".txt", ".pdf", ".docx"))
                ]

                self.log_info(f"Found {len(doc_files)} document files")
                doc_status["docs_folder_exists"] = True
                doc_status["doc_files"] = doc_files

                if len(doc_files) == 0:
                    self.log_warning("No document files found in docs folder")

                # Test document processor on first file
                if doc_files:
                    try:
                        processor = DocumentProcessor(
                            config.CHUNK_SIZE, config.CHUNK_OVERLAP
                        )
                        first_file = os.path.join(docs_path, doc_files[0])

                        self.log_info(f"Testing document processing on: {doc_files[0]}")
                        course, chunks = processor.process_course_document(first_file)

                        self.log_success(f"Document processed: {course.title}")
                        self.log_info(f"Generated {len(chunks)} chunks")

                        doc_status["processing_test"] = {
                            "success": True,
                            "file_tested": doc_files[0],
                            "course_title": course.title,
                            "chunk_count": len(chunks),
                        }

                    except Exception as e:
                        self.log_issue(f"Error processing document: {e}")
                        doc_status["processing_test"] = f"error: {str(e)}"

            else:
                self.log_warning(f"Docs folder does not exist: {docs_path}")
                doc_status["docs_folder_exists"] = False

        except Exception as e:
            self.log_issue(f"Error checking document processing: {e}")
            doc_status["error"] = str(e)

        return doc_status

    def check_search_tools(self) -> Dict[str, Any]:
        """Check search tools functionality"""
        print("\\n🔍 CHECKING SEARCH TOOLS")
        print("=" * 50)

        tools_status = {}

        try:
            # Initialize components
            vector_store = VectorStore(
                config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
            )

            # Test CourseSearchTool
            search_tool = CourseSearchTool(vector_store)
            tool_def = search_tool.get_tool_definition()

            self.log_success("CourseSearchTool initialized successfully")
            self.log_info(f"Tool name: {tool_def['name']}")

            tools_status["search_tool_init"] = "success"
            tools_status["tool_name"] = tool_def["name"]

            # Test tool execution
            try:
                result = search_tool.execute("test query")
                self.log_success("CourseSearchTool executed successfully")
                self.log_info(
                    f"Result type: {type(result)}, length: {len(str(result))}"
                )
                tools_status["tool_execution"] = "success"
                tools_status["result_sample"] = (
                    str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
                )

            except Exception as e:
                self.log_issue(f"Error executing search tool: {e}")
                tools_status["tool_execution"] = f"error: {str(e)}"

            # Test ToolManager
            try:
                tool_manager = ToolManager()
                tool_manager.register_tool(search_tool)

                definitions = tool_manager.get_tool_definitions()
                self.log_success(
                    f"ToolManager initialized with {len(definitions)} tools"
                )
                tools_status["tool_manager"] = "success"
                tools_status["registered_tools"] = len(definitions)

                # Test tool execution through manager
                try:
                    manager_result = tool_manager.execute_tool(
                        "search_course_content", query="test"
                    )
                    self.log_success("Tool execution through manager successful")
                    tools_status["manager_execution"] = "success"

                except Exception as e:
                    self.log_issue(f"Error executing tool through manager: {e}")
                    tools_status["manager_execution"] = f"error: {str(e)}"

            except Exception as e:
                self.log_issue(f"Error with ToolManager: {e}")
                tools_status["tool_manager"] = f"error: {str(e)}"

        except Exception as e:
            self.log_issue(f"Error initializing search tools: {e}")
            tools_status["initialization"] = f"error: {str(e)}"

        return tools_status

    def check_ai_generator(self) -> Dict[str, Any]:
        """Check AI generator functionality"""
        print("\\n🔍 CHECKING AI GENERATOR")
        print("=" * 50)

        ai_status = {}

        try:
            # Initialize AI generator
            ai_generator = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)
            self.log_success("AIGenerator initialized successfully")
            ai_status["initialization"] = "success"

            # Check system prompt
            if hasattr(ai_generator, "SYSTEM_PROMPT") and ai_generator.SYSTEM_PROMPT:
                self.log_success("System prompt is present")
                prompt_length = len(ai_generator.SYSTEM_PROMPT)
                self.log_info(f"System prompt length: {prompt_length} characters")
                ai_status["system_prompt"] = "present"
                ai_status["prompt_length"] = prompt_length
            else:
                self.log_warning("System prompt is missing or empty")
                ai_status["system_prompt"] = "missing"

            # Note: We can't test actual API calls without potentially using credits
            # Instead, check that the configuration looks correct

            self.log_info(f"Model: {ai_generator.model}")
            self.log_info(f"Base parameters: {ai_generator.base_params}")
            ai_status["model"] = ai_generator.model
            ai_status["base_params"] = ai_generator.base_params

        except Exception as e:
            self.log_issue(f"Error initializing AI generator: {e}")
            ai_status["initialization"] = f"error: {str(e)}"

        return ai_status

    def check_full_system_integration(self) -> Dict[str, Any]:
        """Test full system integration"""
        print("\\n🔍 CHECKING FULL SYSTEM INTEGRATION")
        print("=" * 50)

        integration_status = {}

        try:
            # Initialize RAG system
            rag_system = RAGSystem(config)
            self.log_success("RAGSystem initialized successfully")
            integration_status["rag_init"] = "success"

            # Check analytics
            try:
                analytics = rag_system.get_course_analytics()
                self.log_info(f"Course analytics: {analytics}")
                integration_status["analytics"] = analytics

                if analytics["total_courses"] == 0:
                    self.log_warning("No courses in system - queries will fail")

            except Exception as e:
                self.log_issue(f"Error getting course analytics: {e}")
                integration_status["analytics"] = f"error: {str(e)}"

            # Test session creation
            try:
                session_id = rag_system.session_manager.create_session()
                self.log_success(f"Session created: {session_id}")
                integration_status["session_creation"] = "success"

            except Exception as e:
                self.log_issue(f"Error creating session: {e}")
                integration_status["session_creation"] = f"error: {str(e)}"

            # Note: We can't test actual query without API key, but we can check setup
            self.log_info("Full integration components appear to be properly connected")

        except Exception as e:
            self.log_issue(f"Error in full system integration: {e}")
            integration_status["error"] = str(e)

        return integration_status

    def run_sample_query_test(self) -> Dict[str, Any]:
        """Attempt to run a sample query (if API key available)"""
        print("\\n🔍 RUNNING SAMPLE QUERY TEST")
        print("=" * 50)

        query_status = {}

        # Only run if API key is present and valid-looking
        if not config.ANTHROPIC_API_KEY or len(config.ANTHROPIC_API_KEY) < 20:
            self.log_warning("Skipping query test - API key not available or invalid")
            query_status["skipped"] = "no_api_key"
            return query_status

        try:
            rag_system = RAGSystem(config)

            # Try a simple query
            test_query = "What courses are available?"
            self.log_info(f"Testing query: {test_query}")

            try:
                response, sources = rag_system.query(test_query)

                self.log_success("Query executed successfully")
                self.log_info(f"Response length: {len(response)}")
                self.log_info(f"Number of sources: {len(sources)}")
                self.log_info(f"Response preview: {response[:200]}...")

                query_status["success"] = True
                query_status["response_length"] = len(response)
                query_status["sources_count"] = len(sources)
                query_status["response_preview"] = response[:200]

            except Exception as e:
                self.log_issue(f"Query failed: {e}")
                self.log_info(f"Full error: {traceback.format_exc()}")
                query_status["success"] = False
                query_status["error"] = str(e)
                query_status["traceback"] = traceback.format_exc()

        except Exception as e:
            self.log_issue(f"Error setting up query test: {e}")
            query_status["setup_error"] = str(e)

        return query_status

    def generate_report(self, all_status: Dict[str, Any]) -> str:
        """Generate comprehensive diagnostic report"""
        report = []
        report.append("\\n" + "=" * 60)
        report.append("RAG SYSTEM DIAGNOSTIC REPORT")
        report.append("=" * 60)

        # Summary
        report.append("\\n📊 SUMMARY")
        report.append("-" * 20)
        report.append(f"Issues found: {len(self.issues)}")
        report.append(f"Warnings: {len(self.warnings)}")
        report.append(f"Info messages: {len(self.info)}")

        if self.issues:
            report.append("\\n❌ CRITICAL ISSUES:")
            for issue in self.issues:
                report.append(f"  • {issue}")

        if self.warnings:
            report.append("\\n⚠️  WARNINGS:")
            for warning in self.warnings:
                report.append(f"  • {warning}")

        # Detailed status
        report.append("\\n📋 DETAILED STATUS")
        report.append("-" * 20)

        for component, status in all_status.items():
            report.append(f"\\n{component.upper().replace('_', ' ')}:")
            if isinstance(status, dict):
                for key, value in status.items():
                    report.append(f"  {key}: {value}")
            else:
                report.append(f"  {status}")

        # Recommendations
        report.append("\\n💡 RECOMMENDATIONS")
        report.append("-" * 20)

        if not config.ANTHROPIC_API_KEY:
            report.append("• Set ANTHROPIC_API_KEY environment variable")

        if (
            "vector_store" in all_status
            and all_status["vector_store"].get("course_count") == 0
        ):
            report.append("• Load course documents into the vector store")
            report.append("• Check that documents are in the correct format")
            report.append("• Verify the docs folder path is correct")

        if self.issues:
            report.append(
                "• Address all critical issues before expecting queries to work"
            )

        report.append("\\n" + "=" * 60)

        return "\\n".join(report)


def main():
    """Run comprehensive system diagnostics"""
    print("🚀 RAG SYSTEM DIAGNOSTICS")
    print("=" * 60)

    diagnostics = SystemDiagnostics()
    all_status = {}

    # Run all diagnostic checks
    all_status["environment"] = diagnostics.check_environment()
    all_status["vector_store"] = diagnostics.check_vector_store()
    all_status["document_processing"] = diagnostics.check_document_processing()
    all_status["search_tools"] = diagnostics.check_search_tools()
    all_status["ai_generator"] = diagnostics.check_ai_generator()
    all_status["system_integration"] = diagnostics.check_full_system_integration()
    all_status["query_test"] = diagnostics.run_sample_query_test()

    # Generate and display report
    report = diagnostics.generate_report(all_status)
    print(report)

    # Save report to file
    report_path = "diagnostic_report.txt"
    try:
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\\n📄 Report saved to: {report_path}")
    except Exception as e:
        print(f"\\n❌ Error saving report: {e}")

    # Return status for programmatic use
    return {
        "issues": diagnostics.issues,
        "warnings": diagnostics.warnings,
        "status": all_status,
    }


if __name__ == "__main__":
    result = main()

    # Exit with error code if issues found
    if result["issues"]:
        sys.exit(1)
    else:
        sys.exit(0)
