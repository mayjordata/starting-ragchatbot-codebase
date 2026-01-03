"""
Shared pytest fixtures for the RAG system tests.

This module provides common fixtures for:
- MockConfig for consistent test configuration
- Mocked RAG system components (VectorStore, AIGenerator, etc.)
- Test data fixtures for search results and responses
"""
import pytest
import sys
import os
from dataclasses import dataclass
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import SearchResults


@dataclass
class MockConfig:
    """Mock configuration for testing"""
    ANTHROPIC_API_KEY: str = "test-api-key"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 5
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = "./test_chroma_db"


@pytest.fixture
def mock_config():
    """Provide a MockConfig instance"""
    return MockConfig()


@pytest.fixture
def mock_vector_store():
    """Provide a mocked VectorStore"""
    store = Mock()
    store.search.return_value = SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error=None
    )
    store.get_lesson_link.return_value = None
    store.get_course_count.return_value = 0
    store.get_existing_course_titles.return_value = []
    return store


@pytest.fixture
def mock_ai_generator():
    """Provide a mocked AIGenerator"""
    generator = Mock()
    generator.generate_response.return_value = "Test response"
    return generator


@pytest.fixture
def mock_session_manager():
    """Provide a mocked SessionManager"""
    manager = Mock()
    manager.create_session.return_value = "test-session-id"
    manager.get_conversation_history.return_value = None
    return manager


@pytest.fixture
def mock_document_processor():
    """Provide a mocked DocumentProcessor"""
    processor = Mock()
    processor.process_file.return_value = ([], [])
    return processor


@pytest.fixture
def sample_search_results():
    """Provide sample search results for testing"""
    return SearchResults(
        documents=[
            "This is content about tool use in Claude.",
            "More content about MCP protocol."
        ],
        metadata=[
            {"course_title": "Tool Use Course", "lesson_number": 1},
            {"course_title": "MCP Course", "lesson_number": 2}
        ],
        distances=[0.3, 0.5],
        error=None
    )


@pytest.fixture
def empty_search_results():
    """Provide empty search results for testing"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error=None
    )


@pytest.fixture
def error_search_results():
    """Provide search results with an error"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error="Database connection failed"
    )


@pytest.fixture
def mock_rag_system():
    """Provide a fully mocked RAGSystem for API testing"""
    rag = Mock()
    rag.query.return_value = (
        "This is a test response about the course.",
        [{"text": "Test Course - Lesson 1", "url": "https://example.com/lesson1"}]
    )
    rag.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["Course A", "Course B", "Course C"]
    }
    rag.session_manager = Mock()
    rag.session_manager.create_session.return_value = "new-session-id"
    return rag


@pytest.fixture
def rag_system_patches():
    """Context manager fixture that patches all RAG system dependencies"""
    with patch('rag_system.VectorStore') as mock_vs, \
         patch('rag_system.AIGenerator') as mock_ai, \
         patch('rag_system.DocumentProcessor') as mock_dp, \
         patch('rag_system.SessionManager') as mock_sm:

        mock_sm.return_value.get_conversation_history.return_value = None
        mock_sm.return_value.create_session.return_value = "test-session"
        mock_ai.return_value.generate_response.return_value = "Test response"

        yield {
            'vector_store': mock_vs,
            'ai_generator': mock_ai,
            'document_processor': mock_dp,
            'session_manager': mock_sm
        }
