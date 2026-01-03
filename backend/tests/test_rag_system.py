"""
Tests for RAGSystem in rag_system.py

These tests verify:
1. The RAG system correctly integrates all components
2. Queries are processed through the full pipeline
3. Tools are properly registered and available
4. Sources are correctly retrieved and returned
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from unittest.mock import Mock, patch


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


class TestRAGSystemInitialization:
    """Test RAGSystem initialization and component setup"""

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_rag_system_registers_search_tool(
        self, mock_session, mock_doc_proc, mock_ai_gen, mock_vector_store
    ):
        """Test that CourseSearchTool is registered on initialization"""
        from rag_system import RAGSystem

        config = MockConfig()
        rag = RAGSystem(config)

        # Check that search tool is registered
        assert "search_course_content" in rag.tool_manager.tools
        assert rag.search_tool is not None

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_rag_system_registers_outline_tool(
        self, mock_session, mock_doc_proc, mock_ai_gen, mock_vector_store
    ):
        """Test that CourseOutlineTool is registered on initialization"""
        from rag_system import RAGSystem

        config = MockConfig()
        rag = RAGSystem(config)

        # Check that outline tool is registered
        assert "get_course_outline" in rag.tool_manager.tools
        assert rag.outline_tool is not None

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_tool_definitions_are_correct(
        self, mock_session, mock_doc_proc, mock_ai_gen, mock_vector_store
    ):
        """Test that tool definitions have correct structure"""
        from rag_system import RAGSystem

        config = MockConfig()
        rag = RAGSystem(config)

        definitions = rag.tool_manager.get_tool_definitions()

        assert len(definitions) == 2

        # Verify search tool definition
        search_def = next(
            d for d in definitions if d["name"] == "search_course_content"
        )
        assert "input_schema" in search_def
        assert "query" in search_def["input_schema"]["properties"]

        # Verify outline tool definition
        outline_def = next(d for d in definitions if d["name"] == "get_course_outline")
        assert "input_schema" in outline_def
        assert "course_name" in outline_def["input_schema"]["properties"]


class TestRAGSystemQuery:
    """Test RAGSystem.query() method"""

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_query_calls_ai_generator_with_tools(
        self, mock_session_class, mock_doc_proc, mock_ai_gen_class, mock_vector_store
    ):
        """Test that query passes tool definitions to AI generator"""
        from rag_system import RAGSystem

        # Setup mocks
        mock_ai_generator = Mock()
        mock_ai_gen_class.return_value = mock_ai_generator
        mock_ai_generator.generate_response.return_value = "Test response"

        mock_session_manager = Mock()
        mock_session_class.return_value = mock_session_manager
        mock_session_manager.get_conversation_history.return_value = None

        config = MockConfig()
        rag = RAGSystem(config)

        # Act
        response, sources = rag.query("What is tool use?", session_id="test-session")

        # Assert
        mock_ai_generator.generate_response.assert_called_once()
        call_kwargs = mock_ai_generator.generate_response.call_args.kwargs

        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 2
        assert "tool_manager" in call_kwargs
        assert call_kwargs["tool_manager"] == rag.tool_manager

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_query_returns_response_and_sources(
        self, mock_session_class, mock_doc_proc, mock_ai_gen_class, mock_vector_store
    ):
        """Test that query returns both response and sources"""
        from rag_system import RAGSystem

        mock_ai_generator = Mock()
        mock_ai_gen_class.return_value = mock_ai_generator
        mock_ai_generator.generate_response.return_value = "Response text"

        mock_session_manager = Mock()
        mock_session_class.return_value = mock_session_manager
        mock_session_manager.get_conversation_history.return_value = None

        config = MockConfig()
        rag = RAGSystem(config)

        # Simulate sources being set by tool execution
        rag.search_tool.last_sources = [
            {"text": "Course A", "url": "http://example.com"}
        ]

        # Act
        response, sources = rag.query("Test query")

        # Assert
        assert response == "Response text"
        assert len(sources) == 1
        assert sources[0]["text"] == "Course A"

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_query_resets_sources_after_retrieval(
        self, mock_session_class, mock_doc_proc, mock_ai_gen_class, mock_vector_store
    ):
        """Test that sources are reset after being retrieved"""
        from rag_system import RAGSystem

        mock_ai_generator = Mock()
        mock_ai_gen_class.return_value = mock_ai_generator
        mock_ai_generator.generate_response.return_value = "Response"

        mock_session_manager = Mock()
        mock_session_class.return_value = mock_session_manager
        mock_session_manager.get_conversation_history.return_value = None

        config = MockConfig()
        rag = RAGSystem(config)

        rag.search_tool.last_sources = [{"text": "Test", "url": None}]

        # Act
        rag.query("Test query")

        # Assert - sources should be empty after query
        assert rag.search_tool.last_sources == []

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_query_updates_session_history(
        self, mock_session_class, mock_doc_proc, mock_ai_gen_class, mock_vector_store
    ):
        """Test that query updates conversation history"""
        from rag_system import RAGSystem

        mock_ai_generator = Mock()
        mock_ai_gen_class.return_value = mock_ai_generator
        mock_ai_generator.generate_response.return_value = "AI Response"

        mock_session_manager = Mock()
        mock_session_class.return_value = mock_session_manager
        mock_session_manager.get_conversation_history.return_value = None

        config = MockConfig()
        rag = RAGSystem(config)

        # Act
        rag.query("User question", session_id="session-123")

        # Assert
        mock_session_manager.add_exchange.assert_called_once_with(
            "session-123", "User question", "AI Response"
        )

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_query_includes_conversation_history(
        self, mock_session_class, mock_doc_proc, mock_ai_gen_class, mock_vector_store
    ):
        """Test that previous conversation history is passed to AI generator"""
        from rag_system import RAGSystem

        mock_ai_generator = Mock()
        mock_ai_gen_class.return_value = mock_ai_generator
        mock_ai_generator.generate_response.return_value = "Response"

        mock_session_manager = Mock()
        mock_session_class.return_value = mock_session_manager
        mock_session_manager.get_conversation_history.return_value = "Previous Q&A"

        config = MockConfig()
        rag = RAGSystem(config)

        # Act
        rag.query("Follow up", session_id="session-456")

        # Assert
        call_kwargs = mock_ai_generator.generate_response.call_args.kwargs
        assert call_kwargs["conversation_history"] == "Previous Q&A"


class TestRAGSystemIntegration:
    """Integration tests for the full RAG pipeline"""

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_full_query_flow_with_tool_execution(
        self,
        mock_session_class,
        mock_doc_proc,
        mock_ai_gen_class,
        mock_vector_store_class,
    ):
        """Test the complete query flow including tool execution"""
        from rag_system import RAGSystem
        from vector_store import SearchResults

        # Setup vector store mock
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        mock_vector_store.search.return_value = SearchResults(
            documents=["Content about the topic"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.5],
            error=None,
        )
        mock_vector_store.get_lesson_link.return_value = "http://example.com/lesson1"

        # Setup AI generator that simulates tool use
        mock_ai_generator = Mock()
        mock_ai_gen_class.return_value = mock_ai_generator

        def mock_generate(query, conversation_history, tools, tool_manager):
            # Simulate Claude calling the search tool
            result = tool_manager.execute_tool("search_course_content", query="topic")
            return f"Based on the search: {result[:50]}..."

        mock_ai_generator.generate_response.side_effect = mock_generate

        mock_session_manager = Mock()
        mock_session_class.return_value = mock_session_manager
        mock_session_manager.get_conversation_history.return_value = None

        config = MockConfig()
        rag = RAGSystem(config)

        # Act
        response, sources = rag.query("Tell me about the topic")

        # Assert
        assert "Based on the search" in response
        mock_vector_store.search.assert_called_once()


class TestRAGSystemCourseAnalytics:
    """Test course analytics functionality"""

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_get_course_analytics_returns_stats(
        self, mock_session, mock_doc_proc, mock_ai_gen, mock_vector_store_class
    ):
        """Test that get_course_analytics returns correct data"""
        from rag_system import RAGSystem

        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        mock_vector_store.get_course_count.return_value = 5
        mock_vector_store.get_existing_course_titles.return_value = [
            "Course A",
            "Course B",
            "Course C",
            "Course D",
            "Course E",
        ]

        config = MockConfig()
        rag = RAGSystem(config)

        # Act
        analytics = rag.get_course_analytics()

        # Assert
        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5
        assert "Course A" in analytics["course_titles"]
