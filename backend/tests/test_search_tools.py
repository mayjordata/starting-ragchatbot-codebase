"""
Tests for CourseSearchTool.execute() method in search_tools.py

These tests verify:
1. The execute method correctly processes search results
2. Error handling works properly
3. Filtering by course_name and lesson_number works
4. Results are formatted correctly
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import Mock

from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Test suite for CourseSearchTool.execute() method"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mock_vector_store = Mock()
        self.search_tool = CourseSearchTool(self.mock_vector_store)

    def test_execute_with_valid_query_returns_formatted_results(self):
        """Test that valid search results are formatted correctly"""
        # Arrange
        mock_results = SearchResults(
            documents=["This is lesson content about tool use."],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.5],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = (
            "https://example.com/lesson1"
        )

        # Act
        result = self.search_tool.execute(query="tool use")

        # Assert
        assert "Test Course" in result
        assert "Lesson 1" in result
        assert "This is lesson content about tool use." in result
        self.mock_vector_store.search.assert_called_once_with(
            query="tool use", course_name=None, lesson_number=None
        )

    def test_execute_with_no_results_returns_message(self):
        """Test that empty results return appropriate message"""
        # Arrange
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.search_tool.execute(query="nonexistent topic")

        # Assert
        assert "No relevant content found" in result

    def test_execute_with_search_error_returns_error_message(self):
        """Test that search errors are properly returned"""
        # Arrange
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Search error: Database connection failed",
        )
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.search_tool.execute(query="any query")

        # Assert
        assert "Search error" in result
        assert "Database connection failed" in result

    def test_execute_with_course_filter(self):
        """Test that course_name filter is passed correctly"""
        # Arrange
        mock_results = SearchResults(
            documents=["Filtered content"],
            metadata=[{"course_title": "MCP Course", "lesson_number": 2}],
            distances=[0.3],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = None

        # Act
        result = self.search_tool.execute(query="MCP", course_name="MCP")

        # Assert
        self.mock_vector_store.search.assert_called_once_with(
            query="MCP", course_name="MCP", lesson_number=None
        )
        assert "MCP Course" in result

    def test_execute_with_lesson_filter(self):
        """Test that lesson_number filter is passed correctly"""
        # Arrange
        mock_results = SearchResults(
            documents=["Lesson 3 content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 3}],
            distances=[0.2],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = None

        # Act
        result = self.search_tool.execute(query="content", lesson_number=3)

        # Assert
        self.mock_vector_store.search.assert_called_once_with(
            query="content", course_name=None, lesson_number=3
        )
        assert "Lesson 3" in result

    def test_execute_with_both_filters(self):
        """Test that both course_name and lesson_number filters work together"""
        # Arrange
        mock_results = SearchResults(
            documents=["Specific content"],
            metadata=[{"course_title": "Computer Use", "lesson_number": 5}],
            distances=[0.1],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = (
            "https://example.com/lesson5"
        )

        # Act
        result = self.search_tool.execute(
            query="computer use", course_name="Computer Use", lesson_number=5
        )

        # Assert
        self.mock_vector_store.search.assert_called_once_with(
            query="computer use", course_name="Computer Use", lesson_number=5
        )
        assert "Computer Use" in result
        assert "Lesson 5" in result

    def test_format_results_tracks_sources(self):
        """Test that sources are properly tracked after formatting"""
        # Arrange
        mock_results = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course A", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = (
            "https://example.com/lesson"
        )

        # Act
        self.search_tool.execute(query="test")

        # Assert
        assert len(self.search_tool.last_sources) == 2
        assert self.search_tool.last_sources[0]["text"] == "Course A - Lesson 1"

    def test_no_results_with_course_filter_shows_filter_info(self):
        """Test that no results message includes filter information"""
        # Arrange
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.search_tool.execute(query="test", course_name="Specific Course")

        # Assert
        assert "No relevant content found" in result
        assert "Specific Course" in result


class TestCourseSearchToolDefinition:
    """Test the tool definition is correct"""

    def test_get_tool_definition_has_required_fields(self):
        """Test that tool definition has all required Anthropic fields"""
        mock_store = Mock()
        tool = CourseSearchTool(mock_store)
        definition = tool.get_tool_definition()

        assert "name" in definition
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert "query" in definition["input_schema"]["required"]


class TestToolManager:
    """Test ToolManager functionality"""

    def test_register_tool_stores_tool(self):
        """Test that tools are properly registered"""
        manager = ToolManager()
        mock_store = Mock()
        tool = CourseSearchTool(mock_store)

        manager.register_tool(tool)

        assert "search_course_content" in manager.tools
        assert manager.tools["search_course_content"] == tool

    def test_get_tool_definitions_returns_all_tools(self):
        """Test that all tool definitions are returned"""
        manager = ToolManager()
        mock_store = Mock()
        search_tool = CourseSearchTool(mock_store)
        outline_tool = CourseOutlineTool(mock_store)

        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 2
        names = [d["name"] for d in definitions]
        assert "search_course_content" in names
        assert "get_course_outline" in names

    def test_execute_tool_calls_correct_tool(self):
        """Test that execute_tool dispatches to the right tool"""
        manager = ToolManager()
        mock_store = Mock()
        mock_results = SearchResults(
            documents=["test content"],
            metadata=[{"course_title": "Test", "lesson_number": 1}],
            distances=[0.5],
            error=None,
        )
        mock_store.search.return_value = mock_results
        mock_store.get_lesson_link.return_value = None

        tool = CourseSearchTool(mock_store)
        manager.register_tool(tool)

        result = manager.execute_tool("search_course_content", query="test")

        assert "Test" in result
        mock_store.search.assert_called_once()

    def test_execute_tool_unknown_tool_returns_error(self):
        """Test that unknown tool names return error message"""
        manager = ToolManager()

        result = manager.execute_tool("unknown_tool", query="test")

        assert "not found" in result

    def test_get_last_sources_returns_sources(self):
        """Test that sources are retrieved from tools"""
        manager = ToolManager()
        mock_store = Mock()
        mock_results = SearchResults(
            documents=["content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.5],
            error=None,
        )
        mock_store.search.return_value = mock_results
        mock_store.get_lesson_link.return_value = "https://example.com"

        tool = CourseSearchTool(mock_store)
        manager.register_tool(tool)

        # Execute to generate sources
        manager.execute_tool("search_course_content", query="test")

        sources = manager.get_last_sources()
        assert len(sources) == 1
        assert sources[0]["text"] == "Test Course - Lesson 1"

    def test_reset_sources_clears_all_sources(self):
        """Test that reset_sources clears sources from all tools"""
        manager = ToolManager()
        mock_store = Mock()
        mock_results = SearchResults(
            documents=["content"],
            metadata=[{"course_title": "Test", "lesson_number": 1}],
            distances=[0.5],
            error=None,
        )
        mock_store.search.return_value = mock_results
        mock_store.get_lesson_link.return_value = None

        tool = CourseSearchTool(mock_store)
        manager.register_tool(tool)

        # Execute to generate sources
        manager.execute_tool("search_course_content", query="test")
        assert len(manager.get_last_sources()) > 0

        # Reset and verify
        manager.reset_sources()
        assert len(manager.get_last_sources()) == 0
