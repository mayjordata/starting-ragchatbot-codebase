"""
Integration tests for the RAG system using real components.

These tests identify issues in the actual integration between components.
"""

import os
import sys

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestVectorStoreIntegration:
    """Test VectorStore with real ChromaDB"""

    def test_vector_store_initialization(self):
        """Test that VectorStore can be initialized"""
        import tempfile

        from vector_store import VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(
                chroma_path=tmpdir, embedding_model="all-MiniLM-L6-v2", max_results=5
            )

            assert store.course_catalog is not None
            assert store.course_content is not None

    def test_vector_store_search_on_empty_db(self):
        """Test search behavior on empty database"""
        import tempfile

        from vector_store import VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(
                chroma_path=tmpdir, embedding_model="all-MiniLM-L6-v2", max_results=5
            )

            # Search on empty database
            results = store.search(query="test query")

            # Should return empty results, NOT an error
            assert results.error is None, f"Unexpected error: {results.error}"
            assert results.is_empty()

    def test_vector_store_add_and_search(self):
        """Test adding content and searching it"""
        import tempfile

        from models import Course, CourseChunk, Lesson
        from vector_store import VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(
                chroma_path=tmpdir, embedding_model="all-MiniLM-L6-v2", max_results=5
            )

            # Add a test course
            course = Course(
                title="Test Course",
                course_link="http://example.com",
                instructor="Test Instructor",
                lessons=[
                    Lesson(
                        lesson_number=1,
                        title="Intro",
                        lesson_link="http://example.com/1",
                    )
                ],
            )
            store.add_course_metadata(course)

            # Add content chunks
            chunks = [
                CourseChunk(
                    content="This is test content about machine learning and AI.",
                    course_title="Test Course",
                    lesson_number=1,
                    chunk_index=0,
                )
            ]
            store.add_course_content(chunks)

            # Search for the content
            results = store.search(query="machine learning")

            # Should find the content
            assert results.error is None, f"Search error: {results.error}"
            assert not results.is_empty(), "Expected to find results"
            assert (
                "machine learning" in results.documents[0].lower()
                or "test content" in results.documents[0].lower()
            )


class TestCourseSearchToolIntegration:
    """Test CourseSearchTool with real VectorStore"""

    def test_search_tool_with_real_vector_store(self):
        """Test search tool against real vector store"""
        import tempfile

        from models import Course, CourseChunk, Lesson
        from search_tools import CourseSearchTool
        from vector_store import VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(
                chroma_path=tmpdir, embedding_model="all-MiniLM-L6-v2", max_results=5
            )

            # Add test data
            course = Course(
                title="AI Fundamentals",
                course_link="http://example.com/ai",
                instructor="Dr. Test",
                lessons=[
                    Lesson(
                        lesson_number=1,
                        title="Introduction to AI",
                        lesson_link="http://example.com/ai/1",
                    )
                ],
            )
            store.add_course_metadata(course)

            chunks = [
                CourseChunk(
                    content="Artificial intelligence is the simulation of human intelligence by machines.",
                    course_title="AI Fundamentals",
                    lesson_number=1,
                    chunk_index=0,
                )
            ]
            store.add_course_content(chunks)

            # Create search tool
            search_tool = CourseSearchTool(store)

            # Execute search
            result = search_tool.execute(query="artificial intelligence")

            # Verify result
            assert "AI Fundamentals" in result
            assert "Lesson 1" in result
            assert (
                "artificial intelligence" in result.lower()
                or "simulation" in result.lower()
            )

    def test_search_tool_on_empty_store(self):
        """Test search tool behavior on empty store"""
        import tempfile

        from search_tools import CourseSearchTool
        from vector_store import VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(
                chroma_path=tmpdir, embedding_model="all-MiniLM-L6-v2", max_results=5
            )

            search_tool = CourseSearchTool(store)
            result = search_tool.execute(query="anything")

            # Should return "no results" message, not an error
            assert "No relevant content found" in result


class TestExistingDatabaseIntegration:
    """Test against the existing ChromaDB database"""

    def test_existing_database_has_courses(self):
        """Test that the existing database has courses loaded"""
        import os

        from vector_store import VectorStore

        chroma_path = "./chroma_db"

        # Skip if database doesn't exist
        if not os.path.exists(chroma_path):
            pytest.skip("No existing database found at ./chroma_db")

        store = VectorStore(
            chroma_path=chroma_path, embedding_model="all-MiniLM-L6-v2", max_results=5
        )

        course_count = store.get_course_count()
        print(f"Found {course_count} courses in database")

        assert course_count > 0, "Database should have courses loaded"

    def test_existing_database_search_works(self):
        """Test that search works on existing database"""
        import os

        from vector_store import VectorStore

        chroma_path = "./chroma_db"

        if not os.path.exists(chroma_path):
            pytest.skip("No existing database found at ./chroma_db")

        store = VectorStore(
            chroma_path=chroma_path, embedding_model="all-MiniLM-L6-v2", max_results=5
        )

        # Search for common term
        results = store.search(query="tool use")

        print(f"Search results: {len(results.documents)} documents")
        print(f"Error: {results.error}")
        if results.documents:
            print(f"First result preview: {results.documents[0][:100]}...")

        assert results.error is None, f"Search failed with error: {results.error}"

    def test_existing_database_course_search_tool(self):
        """Test CourseSearchTool on existing database"""
        import os

        from search_tools import CourseSearchTool
        from vector_store import VectorStore

        chroma_path = "./chroma_db"

        if not os.path.exists(chroma_path):
            pytest.skip("No existing database found at ./chroma_db")

        store = VectorStore(
            chroma_path=chroma_path, embedding_model="all-MiniLM-L6-v2", max_results=5
        )

        search_tool = CourseSearchTool(store)
        result = search_tool.execute(query="What is tool use?")

        print(f"Search tool result: {result[:200]}...")

        # Should not be an error message
        assert "Search error" not in result, f"Search tool returned error: {result}"
        # Should either find content or say no results found
        assert len(result) > 0
