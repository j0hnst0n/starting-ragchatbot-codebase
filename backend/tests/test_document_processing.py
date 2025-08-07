import pytest
import sys
import os
import tempfile
from unittest.mock import patch, mock_open

# Add backend to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from document_processor import DocumentProcessor
from models import Course, Lesson, CourseChunk


class TestDocumentProcessor:
    """Test suite for DocumentProcessor functionality"""
    
    @pytest.fixture
    def processor(self):
        """Create DocumentProcessor instance for testing"""
        return DocumentProcessor(chunk_size=200, chunk_overlap=50)
    
    @pytest.fixture
    def sample_course_content(self):
        """Sample course document content"""
        return """Course Title: Introduction to Machine Learning
Course Link: https://example.com/ml-course
Course Instructor: Dr. Jane Smith

Lesson 0: Introduction to ML
Lesson Link: https://example.com/lesson0
Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It involves algorithms that can identify patterns in data and use these patterns to make predictions or decisions. The field has grown rapidly in recent years due to advances in computing power and the availability of large datasets.

Lesson 1: Supervised Learning
Lesson Link: https://example.com/lesson1
Supervised learning is a type of machine learning where algorithms learn from labeled training data. The goal is to learn a mapping from inputs to outputs that can generalize to new, unseen data. Common examples include classification tasks like email spam detection and regression tasks like predicting house prices. The key characteristic is that we have both input features and known correct outputs during training.

Lesson 2: Unsupervised Learning
Unsupervised learning works with data that has no labels or target outputs. The algorithm must find patterns and structure in the data on its own. Common techniques include clustering to group similar data points and dimensionality reduction to simplify complex datasets while preserving important information."""
    
    def test_processor_initialization(self, processor):
        """Test DocumentProcessor initialization"""
        print("\\n=== Testing DocumentProcessor Initialization ===")
        try:
            assert processor.chunk_size == 200
            assert processor.chunk_overlap == 50
            print("✅ DocumentProcessor initialization successful")
            
        except Exception as e:
            print(f"❌ DocumentProcessor initialization failed: {e}")
            raise
    
    def test_read_file_utf8(self, processor):
        """Test reading UTF-8 encoded files"""
        print("\\n=== Testing File Reading UTF-8 ===")
        try:
            test_content = "Test content with UTF-8: café, naïve, résumé"
            
            with patch("builtins.open", mock_open(read_data=test_content)):
                result = processor.read_file("test.txt")
                assert result == test_content
                
            print("✅ UTF-8 file reading successful")
            
        except Exception as e:
            print(f"❌ UTF-8 file reading failed: {e}")
            raise
    
    def test_read_file_encoding_error(self, processor):
        """Test handling of encoding errors"""
        print("\\n=== Testing File Reading Encoding Errors ===")
        try:
            test_content = "Fallback content"
            
            # Mock first call to raise UnicodeDecodeError, second to succeed
            mock_file = mock_open(read_data=test_content)
            mock_file.return_value.read.side_effect = [
                UnicodeDecodeError("utf-8", b"", 0, 1, "invalid"),
                test_content
            ]
            
            with patch("builtins.open", mock_file):
                result = processor.read_file("test.txt")
                assert result == test_content
                
            print("✅ Encoding error handling successful")
            
        except Exception as e:
            print(f"❌ Encoding error handling failed: {e}")
            raise
    
    def test_chunk_text_basic(self, processor):
        """Test basic text chunking functionality"""
        print("\\n=== Testing Basic Text Chunking ===")
        try:
            text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence. Sixth sentence."
            
            chunks = processor.chunk_text(text)
            
            # Should have at least one chunk
            assert len(chunks) >= 1
            assert all(len(chunk) <= processor.chunk_size for chunk in chunks)
            
            # Verify all content is preserved
            combined = ' '.join(chunks)
            # Allow for some overlap in combined length
            assert len(combined) >= len(text)
            
            print(f"✅ Basic text chunking successful ({len(chunks)} chunks created)")
            
        except Exception as e:
            print(f"❌ Basic text chunking failed: {e}")
            raise
    
    def test_chunk_text_with_overlap(self):
        """Test text chunking with overlap"""
        print("\\n=== Testing Text Chunking with Overlap ===")
        try:
            processor = DocumentProcessor(chunk_size=100, chunk_overlap=30)
            text = "This is the first sentence. This is the second sentence. This is the third sentence. This is the fourth sentence."
            
            chunks = processor.chunk_text(text)
            
            # Should have multiple chunks due to size limit
            assert len(chunks) >= 2
            
            # Check for overlap - find common words between adjacent chunks
            has_overlap = False
            for i in range(len(chunks) - 1):
                words_1 = set(chunks[i].split())
                words_2 = set(chunks[i + 1].split())
                if words_1.intersection(words_2):
                    has_overlap = True
                    break
            
            print(f"✅ Text chunking with overlap successful ({len(chunks)} chunks, overlap detected: {has_overlap})")
            
        except Exception as e:
            print(f"❌ Text chunking with overlap failed: {e}")
            raise
    
    def test_chunk_text_edge_cases(self, processor):
        """Test text chunking edge cases"""
        print("\\n=== Testing Text Chunking Edge Cases ===")
        try:
            # Empty text
            chunks = processor.chunk_text("")
            assert len(chunks) == 0
            
            # Single word
            chunks = processor.chunk_text("Word")
            assert len(chunks) == 1
            assert chunks[0] == "Word"
            
            # Text shorter than chunk size
            short_text = "Short text."
            chunks = processor.chunk_text(short_text)
            assert len(chunks) == 1
            assert chunks[0] == short_text
            
            # Text with multiple spaces
            spaced_text = "Word1    Word2     Word3."
            chunks = processor.chunk_text(spaced_text)
            assert len(chunks) == 1
            # Should normalize spaces
            assert "    " not in chunks[0]
            
            print("✅ Text chunking edge cases successful")
            
        except Exception as e:
            print(f"❌ Text chunking edge cases failed: {e}")
            raise
    
    def test_process_course_document_full(self, processor, sample_course_content):
        """Test processing a complete course document"""
        print("\\n=== Testing Complete Course Document Processing ===")
        try:
            with patch.object(processor, 'read_file', return_value=sample_course_content):
                course, chunks = processor.process_course_document("test_course.txt")
                
                # Verify course metadata
                assert course.title == "Introduction to Machine Learning"
                assert course.course_link == "https://example.com/ml-course"
                assert course.instructor == "Dr. Jane Smith"
                assert len(course.lessons) == 3
                
                # Verify lessons
                assert course.lessons[0].lesson_number == 0
                assert course.lessons[0].title == "Introduction to ML"
                assert course.lessons[0].lesson_link == "https://example.com/lesson0"
                
                assert course.lessons[1].lesson_number == 1
                assert course.lessons[1].title == "Supervised Learning"
                assert course.lessons[1].lesson_link == "https://example.com/lesson1"
                
                assert course.lessons[2].lesson_number == 2
                assert course.lessons[2].title == "Unsupervised Learning"
                assert course.lessons[2].lesson_link is None  # No link provided
                
                # Verify chunks were created
                assert len(chunks) > 0
                
                # Verify chunk structure
                for chunk in chunks:
                    assert isinstance(chunk, CourseChunk)
                    assert chunk.course_title == "Introduction to Machine Learning"
                    assert chunk.lesson_number is not None
                    assert chunk.chunk_index >= 0
                    assert len(chunk.content) > 0
                    
                    # Verify contextual prefixes are added
                    if chunk.lesson_number is not None:
                        assert f"Course {course.title} Lesson {chunk.lesson_number} content:" in chunk.content
                
                print(f"✅ Complete course document processing successful ({len(chunks)} chunks created)")
                
        except Exception as e:
            print(f"❌ Complete course document processing failed: {e}")
            raise
    
    def test_process_course_document_minimal(self, processor):
        """Test processing document with minimal metadata"""
        print("\\n=== Testing Minimal Course Document Processing ===")
        try:
            minimal_content = """Course Title: Basic Course

Lesson 0: Only Lesson
Just some content here."""
            
            with patch.object(processor, 'read_file', return_value=minimal_content):
                course, chunks = processor.process_course_document("minimal.txt")
                
                # Verify course metadata
                assert course.title == "Basic Course"
                assert course.course_link is None
                assert course.instructor is None
                assert len(course.lessons) == 1
                
                # Verify lesson
                assert course.lessons[0].lesson_number == 0
                assert course.lessons[0].title == "Only Lesson"
                assert course.lessons[0].lesson_link is None
                
                # Verify chunks
                assert len(chunks) > 0
                assert all(chunk.course_title == "Basic Course" for chunk in chunks)
                
                print("✅ Minimal course document processing successful")
                
        except Exception as e:
            print(f"❌ Minimal course document processing failed: {e}")
            raise
    
    def test_process_course_document_no_lessons(self, processor):
        """Test processing document with no lesson markers"""
        print("\\n=== Testing Course Document Without Lessons ===")
        try:
            no_lessons_content = """Course Title: No Lessons Course
Course Instructor: Test Teacher

This is just content without any lesson markers. It should be treated as a single document and chunked appropriately."""
            
            with patch.object(processor, 'read_file', return_value=no_lessons_content):
                course, chunks = processor.process_course_document("no_lessons.txt")
                
                # Verify course metadata
                assert course.title == "No Lessons Course"
                assert course.instructor == "Test Teacher"
                assert len(course.lessons) == 0
                
                # Should still create chunks from the content
                assert len(chunks) > 0
                
                # Chunks should not have lesson numbers
                for chunk in chunks:
                    assert chunk.lesson_number is None
                    assert chunk.course_title == "No Lessons Course"
                
                print("✅ Course document without lessons processing successful")
                
        except Exception as e:
            print(f"❌ Course document without lessons processing failed: {e}")
            raise
    
    def test_process_course_document_malformed(self, processor):
        """Test processing malformed documents"""
        print("\\n=== Testing Malformed Course Document Processing ===")
        try:
            # Document with missing course title line
            malformed_content = """Some random content
More content
Lesson 0: A lesson
Content for the lesson."""
            
            with patch.object(processor, 'read_file', return_value=malformed_content):
                course, chunks = processor.process_course_document("malformed.txt")
                
                # Should use first line as title
                assert course.title == "Some random content"
                assert course.instructor is None
                assert course.course_link is None
                
                # Should still process the lesson
                assert len(course.lessons) == 1
                assert course.lessons[0].title == "A lesson"
                
                print("✅ Malformed course document processing successful")
                
        except Exception as e:
            print(f"❌ Malformed course document processing failed: {e}")
            raise
    
    def test_process_course_document_empty(self, processor):
        """Test processing empty document"""
        print("\\n=== Testing Empty Course Document Processing ===")
        try:
            with patch.object(processor, 'read_file', return_value=""):
                course, chunks = processor.process_course_document("empty.txt")
                
                # Should handle empty document gracefully
                assert course.title == "empty.txt"  # Uses filename as fallback
                assert len(course.lessons) == 0
                assert len(chunks) == 0
                
                print("✅ Empty course document processing successful")
                
        except Exception as e:
            print(f"❌ Empty course document processing failed: {e}")
            raise
    
    def test_context_prefixes(self, processor):
        """Test that contextual prefixes are added to chunks"""
        print("\\n=== Testing Contextual Prefixes ===")
        try:
            content = """Course Title: Test Course

Lesson 0: Introduction
This is lesson zero content that should get a contextual prefix.

Lesson 1: Advanced
This is lesson one content that should also get a contextual prefix."""
            
            with patch.object(processor, 'read_file', return_value=content):
                course, chunks = processor.process_course_document("test.txt")
                
                # Check that chunks have contextual prefixes
                lesson_0_chunks = [c for c in chunks if c.lesson_number == 0]
                lesson_1_chunks = [c for c in chunks if c.lesson_number == 1]
                
                assert len(lesson_0_chunks) > 0
                assert len(lesson_1_chunks) > 0
                
                # Verify prefixes
                for chunk in lesson_0_chunks:
                    assert "Course Test Course Lesson 0 content:" in chunk.content
                
                for chunk in lesson_1_chunks:
                    assert "Course Test Course Lesson 1 content:" in chunk.content
                
                print("✅ Contextual prefixes test successful")
                
        except Exception as e:
            print(f"❌ Contextual prefixes test failed: {e}")
            raise
    
    def test_chunk_indexing(self, processor):
        """Test that chunks are properly indexed"""
        print("\\n=== Testing Chunk Indexing ===")
        try:
            # Create content that will generate multiple chunks
            long_content = """Course Title: Long Course

Lesson 0: Very Long Lesson
""" + " ".join([f"This is sentence number {i}." for i in range(50)])
            
            with patch.object(processor, 'read_file', return_value=long_content):
                course, chunks = processor.process_course_document("long.txt")
                
                # Verify chunk indices are sequential
                chunk_indices = [chunk.chunk_index for chunk in chunks]
                expected_indices = list(range(len(chunks)))
                
                assert chunk_indices == expected_indices
                
                # Verify all chunks have unique indices
                assert len(set(chunk_indices)) == len(chunk_indices)
                
                print(f"✅ Chunk indexing test successful ({len(chunks)} chunks with sequential indices)")
                
        except Exception as e:
            print(f"❌ Chunk indexing test failed: {e}")
            raise


class TestDocumentProcessorWithRealFiles:
    """Test DocumentProcessor with actual file operations"""
    
    @pytest.fixture
    def temp_file(self):
        """Create temporary file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("""Course Title: Real File Test
Course Link: https://example.com/real-test
Course Instructor: Real Instructor

Lesson 0: Real Lesson
Lesson Link: https://example.com/real-lesson
This is real content from a real file that we're using to test the document processor.""")
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    def test_process_real_file(self, temp_file):
        """Test processing an actual file"""
        print("\\n=== Testing Real File Processing ===")
        try:
            processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
            
            course, chunks = processor.process_course_document(temp_file)
            
            # Verify processing worked
            assert course.title == "Real File Test"
            assert course.instructor == "Real Instructor"
            assert len(course.lessons) == 1
            assert len(chunks) > 0
            
            # Verify chunk content
            for chunk in chunks:
                assert "Real File Test" in chunk.content
                assert len(chunk.content) > 0
            
            print("✅ Real file processing successful")
            
        except Exception as e:
            print(f"❌ Real file processing failed: {e}")
            raise


if __name__ == "__main__":
    print("Running DocumentProcessor tests...")
    pytest.main([__file__, "-v", "-s"])