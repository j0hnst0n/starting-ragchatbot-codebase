"""
Test configuration and shared fixtures for the RAG system test suite.

This module provides common fixtures and test utilities used across
all test files in the backend test suite.
"""

import pytest
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any
from fastapi.testclient import TestClient
from httpx import AsyncClient
import asyncio

# Add backend to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults


@dataclass
class MockConfig:
    """Mock configuration for testing"""
    ANTHROPIC_API_KEY: str = "test_api_key"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 5
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = "./test_chroma"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_config(temp_dir):
    """Create mock configuration"""
    config = MockConfig()
    config.CHROMA_PATH = temp_dir
    return config


@pytest.fixture
def sample_course_data():
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


@pytest.fixture
def sample_query_request():
    """Sample query request data for API testing"""
    return {
        "query": "What is machine learning?",
        "session_id": None
    }


@pytest.fixture
def sample_query_response():
    """Sample query response data for API testing"""
    return {
        "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
        "sources": [
            {
                "display_text": "Test Course - Lesson 0: Introduction",
                "lesson_link": "https://example.com/lesson0"
            }
        ],
        "session_id": "test_session_123"
    }


@pytest.fixture
def sample_course_stats():
    """Sample course statistics data for API testing"""
    return {
        "total_courses": 2,
        "course_titles": ["Test Course", "Advanced ML Course"]
    }


@pytest.fixture
def mock_rag_system():
    """Create a mock RAG system for API testing"""
    mock_rag = Mock()
    mock_rag.session_manager = Mock()
    mock_rag.session_manager.create_session.return_value = "test_session_123"
    
    mock_rag.query.return_value = (
        "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
        [
            {
                "display_text": "Test Course - Lesson 0: Introduction",
                "lesson_link": "https://example.com/lesson0"
            }
        ]
    )
    
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Test Course", "Advanced ML Course"]
    }
    
    return mock_rag


@pytest.fixture
def test_app(mock_rag_system, temp_dir):
    """Create a test FastAPI application without static file mounting"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    
    # Create test app
    app = FastAPI(title="Course Materials RAG System - Test", root_path="")
    
    # Add middleware
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Pydantic models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None
    
    class SourceData(BaseModel):
        display_text: str
        lesson_link: Optional[str] = None
    
    class QueryResponse(BaseModel):
        answer: str
        sources: List[SourceData]
        session_id: str
    
    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # API endpoints - inline definitions to avoid import issues
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            
            answer, sources = mock_rag_system.query(request.query, session_id)
            
            source_data_list = []
            for source in sources:
                if isinstance(source, dict):
                    source_data_list.append(SourceData(
                        display_text=source.get('display_text', ''),
                        lesson_link=source.get('lesson_link')
                    ))
                else:
                    source_data_list.append(SourceData(
                        display_text=str(source),
                        lesson_link=None
                    ))
            
            return QueryResponse(
                answer=answer,
                sources=source_data_list,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def root():
        return {"message": "RAG System API is running"}
    
    return app


@pytest.fixture
def test_client(test_app):
    """Create a test client for the FastAPI application"""
    return TestClient(test_app)


# Async test client removed for now due to pytest-asyncio configuration issues


class MockContent:
    """Mock content for AI responses"""
    def __init__(self, type: str, text: str = None, id: str = None, name: str = None, input: dict = None):
        self.type = type
        self.text = text
        self.id = id
        self.name = name
        self.input = input


class MockResponse:
    """Mock response for AI API calls"""
    def __init__(self, content: List[MockContent], stop_reason: str = "end_turn"):
        self.content = content
        self.stop_reason = stop_reason


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for AI generator testing"""
    mock_client = Mock()
    mock_client.messages = Mock()
    mock_client.messages.create = Mock()
    return mock_client


@pytest.fixture
def mock_empty_search_results():
    """Create empty search results for testing error cases"""
    return SearchResults.empty("No results found")


@pytest.fixture
def mock_search_results(sample_course_data):
    """Create mock search results with sample data"""
    course, chunks = sample_course_data
    return SearchResults(
        chunks=chunks,
        distances=[0.1, 0.2],
        error=None
    )


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: Mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: Mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "api: Mark test as an API test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names"""
    for item in items:
        # Add unit marker to all tests in test files starting with test_
        if "test_" in item.nodeid and "api" not in item.nodeid.lower():
            item.add_marker(pytest.mark.unit)
        
        # Add api marker to API tests
        if "api" in item.nodeid.lower() or "test_api" in item.nodeid:
            item.add_marker(pytest.mark.api)
        
        # Add integration marker to integration tests
        if "integration" in item.nodeid.lower() or "test_rag_system" in item.nodeid:
            item.add_marker(pytest.mark.integration)