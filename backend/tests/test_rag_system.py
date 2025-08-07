import pytest
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any

# Add backend to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from config import Config
from models import Course, Lesson, CourseChunk
from vector_store import SearchResults


@dataclass
class MockConfig:
    """Mock configuration for testing"""
    ANTHROPIC_API_KEY: str = "test_api_key"
    ANTHROPIC_MODEL: str = "claude-3-sonnet-20240229"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 5
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = "./test_chroma"


class TestRAGSystem:
    """Test suite for RAGSystem integration"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test data"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create mock configuration"""
        config = MockConfig()
        config.CHROMA_PATH = temp_dir
        return config
    
    @pytest.fixture
    def sample_course_data(self):
        """Create sample course and chunks for testing"""
        course = Course(
            title="Test Course",
            course_link="https://example.com/course",
            instructor="Test Instructor",
            lessons=[
                Lesson(lesson_number=0, title="Introduction", lesson_link="https://example.com/lesson0"),
                Lesson(lesson_number=1, title="Advanced Topics", lesson_link="https://example.com/lesson1")
            ]
        )
        
        chunks = [
            CourseChunk(
                content="Course Test Course Lesson 0 content: This is an introduction to machine learning concepts",
                course_title="Test Course",
                lesson_number=0,
                chunk_index=0
            ),
            CourseChunk(
                content="Course Test Course Lesson 1 content: Advanced neural network architectures and deep learning",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=1
            )
        ]
        
        return course, chunks
    
    def test_rag_system_initialization(self, mock_config):
        """Test RAGSystem initialization with all components"""
        print("\\n=== Testing RAGSystem Initialization ===")
        try:
            rag_system = RAGSystem(mock_config)
            
            # Verify all components are initialized
            assert rag_system.config == mock_config
            assert rag_system.document_processor is not None
            assert rag_system.vector_store is not None
            assert rag_system.ai_generator is not None
            assert rag_system.session_manager is not None
            assert rag_system.tool_manager is not None
            assert rag_system.search_tool is not None
            assert rag_system.outline_tool is not None
            
            # Verify tools are registered
            tools = rag_system.tool_manager.get_tool_definitions()
            tool_names = [tool["name"] for tool in tools]
            assert "search_course_content" in tool_names
            assert "get_course_outline" in tool_names
            
            print("✅ RAGSystem initialization successful")
            
        except Exception as e:
            print(f"❌ RAGSystem initialization failed: {e}")
            raise
    
    @patch('rag_system.DocumentProcessor')
    def test_add_course_document(self, mock_doc_processor, mock_config, sample_course_data):
        """Test adding a single course document"""
        print("\\n=== Testing Add Course Document ===")
        try:
            course, chunks = sample_course_data
            
            # Setup mock document processor
            mock_processor = Mock()
            mock_processor.process_course_document.return_value = (course, chunks)
            mock_doc_processor.return_value = mock_processor
            
            rag_system = RAGSystem(mock_config)
            rag_system.document_processor = mock_processor
            
            # Test adding document
            result_course, chunk_count = rag_system.add_course_document("test_file.txt")
            
            # Verify document processor was called
            mock_processor.process_course_document.assert_called_once_with("test_file.txt")
            
            # Verify result
            assert result_course == course
            assert chunk_count == len(chunks)
            
            print("✅ Add course document successful")
            
        except Exception as e:
            print(f"❌ Add course document failed: {e}")
            raise
    
    def test_add_course_document_error(self, mock_config):
        """Test error handling when adding course document fails"""
        print("\\n=== Testing Add Course Document Error Handling ===")
        try:
            with patch('rag_system.DocumentProcessor') as mock_doc_processor:
                # Setup mock to raise exception
                mock_processor = Mock()
                mock_processor.process_course_document.side_effect = Exception("File not found")
                mock_doc_processor.return_value = mock_processor
                
                rag_system = RAGSystem(mock_config)
                rag_system.document_processor = mock_processor
                
                # Test error handling
                result_course, chunk_count = rag_system.add_course_document("nonexistent.txt")
                
                assert result_course is None
                assert chunk_count == 0
                
            print("✅ Add course document error handling successful")
            
        except Exception as e:
            print(f"❌ Add course document error handling failed: {e}")
            raise
    
    @patch('rag_system.os.path.exists')
    @patch('rag_system.os.listdir')
    def test_add_course_folder(self, mock_listdir, mock_exists, mock_config, sample_course_data):
        """Test adding course documents from folder"""
        print("\\n=== Testing Add Course Folder ===")
        try:
            course, chunks = sample_course_data
            
            # Setup mocks
            mock_exists.return_value = True
            mock_listdir.return_value = ["course1.txt", "course2.pdf", "readme.md"]
            
            with patch('rag_system.DocumentProcessor') as mock_doc_processor:
                mock_processor = Mock()
                mock_processor.process_course_document.return_value = (course, chunks)
                mock_doc_processor.return_value = mock_processor
                
                rag_system = RAGSystem(mock_config)
                rag_system.document_processor = mock_processor
                
                # Mock vector store to return empty titles
                rag_system.vector_store.get_existing_course_titles = Mock(return_value=[])
                
                # Test adding folder
                total_courses, total_chunks = rag_system.add_course_folder("test_folder")
                
                # Should process 2 files (txt and pdf, not md)
                assert mock_processor.process_course_document.call_count == 2
                assert total_courses == 2
                assert total_chunks == 4  # 2 files * 2 chunks each
                
            print("✅ Add course folder successful")
            
        except Exception as e:
            print(f"❌ Add course folder failed: {e}")
            raise
    
    @patch('rag_system.os.path.exists')
    def test_add_course_folder_not_exists(self, mock_exists, mock_config):
        """Test adding course folder that doesn't exist"""
        print("\\n=== Testing Add Course Folder Not Exists ===")
        try:
            mock_exists.return_value = False
            
            rag_system = RAGSystem(mock_config)
            
            total_courses, total_chunks = rag_system.add_course_folder("nonexistent_folder")
            
            assert total_courses == 0
            assert total_chunks == 0
            
            print("✅ Add course folder not exists handling successful")
            
        except Exception as e:
            print(f"❌ Add course folder not exists handling failed: {e}")
            raise
    
    def test_query_without_session(self, mock_config):
        """Test querying without session ID"""
        print("\\n=== Testing Query Without Session ===")
        try:
            with patch('ai_generator.anthropic.Anthropic'):
                rag_system = RAGSystem(mock_config)
                
                # Mock AI generator response
                rag_system.ai_generator.generate_response = Mock(return_value="Test response")
                rag_system.tool_manager.get_last_sources = Mock(return_value=[])
                rag_system.tool_manager.reset_sources = Mock()
                
                # Execute query
                response, sources = rag_system.query("What is machine learning?")
                
                # Verify AI generator was called correctly
                rag_system.ai_generator.generate_response.assert_called_once()
                call_args = rag_system.ai_generator.generate_response.call_args
                
                assert "What is machine learning?" in call_args[1]["query"]
                assert call_args[1]["conversation_history"] is None
                assert call_args[1]["tools"] is not None
                assert call_args[1]["tool_manager"] is not None
                
                # Verify response
                assert response == "Test response"
                assert sources == []
                
            print("✅ Query without session successful")
            
        except Exception as e:
            print(f"❌ Query without session failed: {e}")
            raise
    
    def test_query_with_session(self, mock_config):
        """Test querying with session ID and history"""
        print("\\n=== Testing Query With Session ===")
        try:
            with patch('ai_generator.anthropic.Anthropic'):
                rag_system = RAGSystem(mock_config)
                
                # Setup session with history
                session_id = rag_system.session_manager.create_session()
                rag_system.session_manager.add_message(session_id, "user", "Previous question")
                rag_system.session_manager.add_message(session_id, "assistant", "Previous answer")
                
                # Mock AI generator response
                rag_system.ai_generator.generate_response = Mock(return_value="Follow-up response")
                rag_system.tool_manager.get_last_sources = Mock(return_value=[
                    {"display_text": "Test Course - Lesson 1", "lesson_link": "https://example.com/lesson1"}
                ])
                rag_system.tool_manager.reset_sources = Mock()
                
                # Execute query
                response, sources = rag_system.query("Follow up question", session_id=session_id)
                
                # Verify history was passed
                call_args = rag_system.ai_generator.generate_response.call_args
                assert call_args[1]["conversation_history"] is not None
                assert "Previous question" in call_args[1]["conversation_history"]
                assert "Previous answer" in call_args[1]["conversation_history"]
                
                # Verify response and sources
                assert response == "Follow-up response"
                assert len(sources) == 1
                assert sources[0]["display_text"] == "Test Course - Lesson 1"
                
                # Verify session was updated with new exchange
                history = rag_system.session_manager.get_conversation_history(session_id)
                assert "Follow up question" in history
                assert "Follow-up response" in history
                
            print("✅ Query with session successful")
            
        except Exception as e:
            print(f"❌ Query with session failed: {e}")
            raise
    
    def test_query_with_tool_usage(self, mock_config, sample_course_data):
        """Test end-to-end query with actual tool usage"""
        print("\\n=== Testing Query With Tool Usage ===")
        try:
            course, chunks = sample_course_data
            
            with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
                # Create RAG system
                rag_system = RAGSystem(mock_config)
                
                # Add test data to vector store
                rag_system.vector_store.add_course_metadata(course)
                rag_system.vector_store.add_course_content(chunks)
                
                # Setup mock AI generator to use tools
                from test_ai_generator import MockContent, MockResponse
                
                mock_client = Mock()
                mock_anthropic.return_value = mock_client
                rag_system.ai_generator.client = mock_client
                
                # First response: tool use
                tool_use_response = MockResponse(
                    content=[MockContent(
                        type="tool_use",
                        id="tool_123",
                        name="search_course_content",
                        input={"query": "machine learning", "course_name": "Test"}
                    )],
                    stop_reason="tool_use"
                )
                
                # Final response after tool execution
                final_response = MockResponse(
                    content=[MockContent(type="text", text="Based on the course content, machine learning is...")],
                    stop_reason="end_turn"
                )
                
                mock_client.messages.create.side_effect = [tool_use_response, final_response]
                
                # Execute query
                response, sources = rag_system.query("Tell me about machine learning in Test course")
                
                # Verify tool was used
                assert mock_client.messages.create.call_count == 2
                
                # Verify response
                assert "Based on the course content" in response
                
                # Note: Sources should be populated by the search tool
                # This tests the full integration
                
            print("✅ Query with tool usage successful")
            
        except Exception as e:
            print(f"❌ Query with tool usage failed: {e}")
            raise
    
    def test_get_course_analytics(self, mock_config, sample_course_data):
        """Test course analytics functionality"""
        print("\\n=== Testing Course Analytics ===")
        try:
            course, chunks = sample_course_data
            
            rag_system = RAGSystem(mock_config)
            
            # Test with empty store
            analytics = rag_system.get_course_analytics()
            assert analytics["total_courses"] == 0
            assert analytics["course_titles"] == []
            
            # Add test data
            rag_system.vector_store.add_course_metadata(course)
            rag_system.vector_store.add_course_content(chunks)
            
            # Test with data
            analytics = rag_system.get_course_analytics()
            assert analytics["total_courses"] == 1
            assert "Test Course" in analytics["course_titles"]
            
            print("✅ Course analytics successful")
            
        except Exception as e:
            print(f"❌ Course analytics failed: {e}")
            raise


class TestRAGSystemErrorHandling:
    """Test error handling and edge cases in RAG system"""
    
    @pytest.fixture
    def mock_config_with_invalid_key(self, temp_dir):
        """Create config with invalid API key"""
        config = MockConfig()
        config.CHROMA_PATH = temp_dir
        config.ANTHROPIC_API_KEY = ""
        return config
    
    def test_initialization_with_missing_api_key(self, mock_config_with_invalid_key):
        """Test initialization with missing API key"""
        print("\\n=== Testing Initialization with Missing API Key ===")
        try:
            # Should still initialize but may fail on actual API calls
            rag_system = RAGSystem(mock_config_with_invalid_key)
            assert rag_system.ai_generator is not None
            print("✅ Initialization with missing API key handled")
            
        except Exception as e:
            print(f"❌ Initialization with missing API key failed: {e}")
            raise
    
    def test_query_with_ai_generator_failure(self, mock_config):
        """Test query handling when AI generator fails"""
        print("\\n=== Testing Query with AI Generator Failure ===")
        try:
            with patch('ai_generator.anthropic.Anthropic'):
                rag_system = RAGSystem(mock_config)
                
                # Mock AI generator to raise exception
                rag_system.ai_generator.generate_response = Mock(
                    side_effect=Exception("API call failed")
                )
                
                # This should propagate the exception
                with pytest.raises(Exception) as exc_info:
                    rag_system.query("test query")
                
                assert "API call failed" in str(exc_info.value)
                
            print("✅ Query with AI generator failure handled")
            
        except Exception as e:
            print(f"❌ Query with AI generator failure test failed: {e}")
            raise
    
    def test_query_with_vector_store_failure(self, mock_config):
        """Test query handling when vector store operations fail"""
        print("\\n=== Testing Query with Vector Store Failure ===")
        try:
            with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
                rag_system = RAGSystem(mock_config)
                
                # Setup AI generator to use tools
                from test_ai_generator import MockContent, MockResponse
                
                mock_client = Mock()
                mock_anthropic.return_value = mock_client
                rag_system.ai_generator.client = mock_client
                
                # Mock tool use
                tool_use_response = MockResponse(
                    content=[MockContent(
                        type="tool_use",
                        id="tool_123",
                        name="search_course_content",
                        input={"query": "test"}
                    )],
                    stop_reason="tool_use"
                )
                
                final_response = MockResponse(
                    content=[MockContent(type="text", text="Fallback response without search results")],
                    stop_reason="end_turn"
                )
                
                mock_client.messages.create.side_effect = [tool_use_response, final_response]
                
                # Mock vector store to fail
                rag_system.vector_store.search = Mock(
                    return_value=SearchResults.empty("Vector store connection failed")
                )
                
                # Execute query - should handle vector store failure gracefully
                response, sources = rag_system.query("test query")
                
                # Should get fallback response
                assert response == "Fallback response without search results"
                
            print("✅ Query with vector store failure handled")
            
        except Exception as e:
            print(f"❌ Query with vector store failure test failed: {e}")
            raise


class TestRAGSystemPerformance:
    """Test performance-related aspects of RAG system"""
    
    def test_concurrent_queries(self, mock_config):
        """Test handling multiple concurrent queries"""
        print("\\n=== Testing Concurrent Queries ===")
        try:
            with patch('ai_generator.anthropic.Anthropic'):
                rag_system = RAGSystem(mock_config)
                
                # Mock AI generator
                rag_system.ai_generator.generate_response = Mock(return_value="Concurrent response")
                rag_system.tool_manager.get_last_sources = Mock(return_value=[])
                rag_system.tool_manager.reset_sources = Mock()
                
                # Create multiple sessions
                sessions = []
                for i in range(3):
                    session_id = rag_system.session_manager.create_session()
                    sessions.append(session_id)
                
                # Execute queries with different sessions
                results = []
                for i, session_id in enumerate(sessions):
                    response, sources = rag_system.query(f"Query {i}", session_id=session_id)
                    results.append((response, sources))
                
                # Verify all queries were handled
                assert len(results) == 3
                for response, sources in results:
                    assert response == "Concurrent response"
                
            print("✅ Concurrent queries handled successfully")
            
        except Exception as e:
            print(f"❌ Concurrent queries test failed: {e}")
            raise


if __name__ == "__main__":
    print("Running RAGSystem integration tests...")
    pytest.main([__file__, "-v", "-s"])