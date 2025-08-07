import pytest
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

# Add backend to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestVectorStore:
    """Test suite for VectorStore functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test ChromaDB"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def vector_store(self, temp_dir):
        """Create VectorStore instance for testing"""
        return VectorStore(
            chroma_path=temp_dir,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )
    
    @pytest.fixture
    def sample_course(self):
        """Create sample course data"""
        return Course(
            title="Test Course",
            course_link="https://example.com/course",
            instructor="Test Instructor",
            lessons=[
                Lesson(lesson_number=0, title="Introduction", lesson_link="https://example.com/lesson0"),
                Lesson(lesson_number=1, title="Advanced Topics", lesson_link="https://example.com/lesson1")
            ]
        )
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample course chunks"""
        return [
            CourseChunk(
                content="This is lesson 0 content about introduction to the topic",
                course_title="Test Course",
                lesson_number=0,
                chunk_index=0
            ),
            CourseChunk(
                content="This is lesson 1 content about advanced concepts",
                course_title="Test Course", 
                lesson_number=1,
                chunk_index=1
            ),
            CourseChunk(
                content="More advanced content in lesson 1",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=2
            )
        ]
    
    def test_vector_store_initialization(self, temp_dir):
        """Test VectorStore initialization and collection creation"""
        print("\\n=== Testing VectorStore Initialization ===")
        try:
            vector_store = VectorStore(temp_dir, "all-MiniLM-L6-v2", 5)
            assert vector_store.max_results == 5
            assert vector_store.course_catalog is not None
            assert vector_store.course_content is not None
            print("✅ VectorStore initialization successful")
            
            # Test collections exist
            collections = vector_store.client.list_collections()
            collection_names = [c.name for c in collections]
            assert "course_catalog" in collection_names
            assert "course_content" in collection_names
            print("✅ Collections created successfully")
            
        except Exception as e:
            print(f"❌ VectorStore initialization failed: {e}")
            raise
    
    def test_add_course_metadata(self, vector_store, sample_course):
        """Test adding course metadata to catalog"""
        print("\\n=== Testing Add Course Metadata ===")
        try:
            # Add course metadata
            vector_store.add_course_metadata(sample_course)
            print("✅ Course metadata added successfully")
            
            # Verify data was added
            results = vector_store.course_catalog.get(ids=["Test Course"])
            assert results is not None
            assert len(results['ids']) == 1
            assert results['ids'][0] == "Test Course"
            
            metadata = results['metadatas'][0]
            assert metadata['title'] == "Test Course"
            assert metadata['instructor'] == "Test Instructor"
            assert metadata['course_link'] == "https://example.com/course"
            assert metadata['lesson_count'] == 2
            print("✅ Course metadata verification successful")
            
        except Exception as e:
            print(f"❌ Add course metadata failed: {e}")
            raise
    
    def test_add_course_content(self, vector_store, sample_chunks):
        """Test adding course content chunks"""
        print("\\n=== Testing Add Course Content ===")
        try:
            # Add course content
            vector_store.add_course_content(sample_chunks)
            print("✅ Course content added successfully")
            
            # Verify data was added
            all_data = vector_store.course_content.get()
            assert len(all_data['ids']) == 3
            assert len(all_data['documents']) == 3
            assert len(all_data['metadatas']) == 3
            
            # Check metadata structure
            for i, metadata in enumerate(all_data['metadatas']):
                assert metadata['course_title'] == "Test Course"
                assert 'lesson_number' in metadata
                assert 'chunk_index' in metadata
            
            print("✅ Course content verification successful")
            
        except Exception as e:
            print(f"❌ Add course content failed: {e}")
            raise
    
    def test_course_name_resolution(self, vector_store, sample_course):
        """Test semantic course name resolution"""
        print("\\n=== Testing Course Name Resolution ===")
        try:
            # Add course first
            vector_store.add_course_metadata(sample_course)
            
            # Test exact match
            result = vector_store._resolve_course_name("Test Course")
            assert result == "Test Course"
            print("✅ Exact course name match successful")
            
            # Test partial match
            result = vector_store._resolve_course_name("Test")
            assert result == "Test Course"
            print("✅ Partial course name match successful")
            
            # Test no match
            result = vector_store._resolve_course_name("NonexistentCourse")
            assert result is None
            print("✅ No match handling successful")
            
        except Exception as e:
            print(f"❌ Course name resolution failed: {e}")
            raise
    
    def test_filter_building(self, vector_store):
        """Test ChromaDB filter building"""
        print("\\n=== Testing Filter Building ===")
        try:
            # Test no filters
            filter_dict = vector_store._build_filter(None, None)
            assert filter_dict is None
            print("✅ No filters case successful")
            
            # Test course only
            filter_dict = vector_store._build_filter("Test Course", None)
            assert filter_dict == {"course_title": "Test Course"}
            print("✅ Course filter only successful")
            
            # Test lesson only  
            filter_dict = vector_store._build_filter(None, 1)
            assert filter_dict == {"lesson_number": 1}
            print("✅ Lesson filter only successful")
            
            # Test both filters
            filter_dict = vector_store._build_filter("Test Course", 1)
            expected = {"$and": [
                {"course_title": "Test Course"},
                {"lesson_number": 1}
            ]}
            assert filter_dict == expected
            print("✅ Combined filters successful")
            
        except Exception as e:
            print(f"❌ Filter building failed: {e}")
            raise
    
    def test_search_functionality(self, vector_store, sample_course, sample_chunks):
        """Test main search functionality"""
        print("\\n=== Testing Search Functionality ===")
        try:
            # Add test data
            vector_store.add_course_metadata(sample_course)
            vector_store.add_course_content(sample_chunks)
            
            # Test basic search
            results = vector_store.search("introduction")
            assert not results.error
            assert len(results.documents) > 0
            print("✅ Basic search successful")
            
            # Test search with course filter
            results = vector_store.search("advanced", course_name="Test Course")
            assert not results.error
            assert len(results.documents) > 0
            # Verify all results are from the specified course
            for metadata in results.metadata:
                assert metadata['course_title'] == "Test Course"
            print("✅ Course-filtered search successful")
            
            # Test search with lesson filter
            results = vector_store.search("content", lesson_number=1)
            assert not results.error
            assert len(results.documents) > 0
            # Verify all results are from the specified lesson
            for metadata in results.metadata:
                assert metadata['lesson_number'] == 1
            print("✅ Lesson-filtered search successful")
            
            # Test search with both filters
            results = vector_store.search("advanced", course_name="Test", lesson_number=1)
            assert not results.error
            print("✅ Combined filter search successful")
            
            # Test search with no matches
            results = vector_store.search("nonexistent topic")
            assert not results.error
            # May or may not have results depending on similarity threshold
            print("✅ No matches search handled")
            
            # Test invalid course name
            results = vector_store.search("content", course_name="NonexistentCourse") 
            assert results.error is not None
            assert "No course found matching" in results.error
            print("✅ Invalid course name handled")
            
        except Exception as e:
            print(f"❌ Search functionality failed: {e}")
            raise
    
    def test_utility_methods(self, vector_store, sample_course, sample_chunks):
        """Test utility methods"""
        print("\\n=== Testing Utility Methods ===")
        try:
            # Test with empty store
            count = vector_store.get_course_count()
            assert count == 0
            print("✅ Empty store course count successful")
            
            titles = vector_store.get_existing_course_titles()
            assert len(titles) == 0
            print("✅ Empty store course titles successful")
            
            # Add data
            vector_store.add_course_metadata(sample_course)
            vector_store.add_course_content(sample_chunks)
            
            # Test with data
            count = vector_store.get_course_count()
            assert count == 1
            print("✅ Course count with data successful")
            
            titles = vector_store.get_existing_course_titles()
            assert len(titles) == 1
            assert "Test Course" in titles
            print("✅ Course titles with data successful")
            
            # Test course link retrieval
            link = vector_store.get_course_link("Test Course")
            assert link == "https://example.com/course"
            print("✅ Course link retrieval successful")
            
            # Test lesson link retrieval
            lesson_link = vector_store.get_lesson_link("Test Course", 0)
            assert lesson_link == "https://example.com/lesson0"
            print("✅ Lesson link retrieval successful")
            
        except Exception as e:
            print(f"❌ Utility methods failed: {e}")
            raise
    
    def test_error_handling(self, vector_store):
        """Test error handling in various scenarios"""
        print("\\n=== Testing Error Handling ===")
        try:
            # Test search with mock that raises exception
            with patch.object(vector_store.course_content, 'query', side_effect=Exception("ChromaDB Error")):
                results = vector_store.search("test query")
                assert results.error is not None
                assert "Search error" in results.error
            print("✅ Search error handling successful")
            
            # Test course name resolution with mock exception
            with patch.object(vector_store.course_catalog, 'query', side_effect=Exception("Resolution Error")):
                result = vector_store._resolve_course_name("Test Course")
                assert result is None
            print("✅ Course resolution error handling successful")
            
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
            raise


def test_search_results_class():
    """Test SearchResults helper class"""
    print("\\n=== Testing SearchResults Class ===")
    try:
        # Test from_chroma creation
        chroma_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [['meta1', 'meta2']], 
            'distances': [[0.1, 0.2]]
        }
        results = SearchResults.from_chroma(chroma_results)
        assert len(results.documents) == 2
        assert len(results.metadata) == 2
        assert len(results.distances) == 2
        print("✅ SearchResults from_chroma successful")
        
        # Test empty results
        empty_results = SearchResults.empty("Test error")
        assert empty_results.error == "Test error"
        assert empty_results.is_empty()
        print("✅ SearchResults empty creation successful")
        
        # Test is_empty method
        non_empty = SearchResults(['doc'], [{}], [0.1])
        assert not non_empty.is_empty()
        print("✅ SearchResults is_empty method successful")
        
    except Exception as e:
        print(f"❌ SearchResults test failed: {e}")
        raise


if __name__ == "__main__":
    print("Running VectorStore tests...")
    pytest.main([__file__, "-v", "-s"])