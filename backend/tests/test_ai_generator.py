"""
Tests for AIGenerator in ai_generator.py

These tests verify:
1. The AIGenerator correctly calls the Anthropic API
2. Single-round tool execution works properly
3. Multi-round sequential tool calling works (up to MAX_TOOL_ROUNDS)
4. Tool results are correctly passed back to Claude
5. Error handling and termination conditions
"""

import os
import sys

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import Mock, patch

from ai_generator import AIGenerator


class MockTextBlock:
    """Mock for Anthropic TextBlock response"""

    def __init__(self, text):
        self.type = "text"
        self.text = text


class MockToolUseBlock:
    """Mock for Anthropic ToolUseBlock response"""

    def __init__(self, tool_id, name, input_data):
        self.type = "tool_use"
        self.id = tool_id
        self.name = name
        self.input = input_data


class MockResponse:
    """Mock for Anthropic API response"""

    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.stop_reason = stop_reason


class TestAIGeneratorWithoutTools:
    """Test AIGenerator when not using tools"""

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_returns_text(self, mock_anthropic_class):
        """Test that a simple query returns text response"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_response = MockResponse(
            content=[MockTextBlock("This is a response about course content.")],
            stop_reason="end_turn",
        )
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        # Act
        result = generator.generate_response(query="What is machine learning?")

        # Assert
        assert result == "This is a response about course content."
        mock_client.messages.create.assert_called_once()

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_includes_conversation_history(
        self, mock_anthropic_class
    ):
        """Test that conversation history is included in system prompt"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_response = MockResponse(
            content=[MockTextBlock("Response with context.")], stop_reason="end_turn"
        )
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        # Act
        result = generator.generate_response(
            query="Follow up question",
            conversation_history="User: Previous question\nAssistant: Previous answer",
        )

        # Assert
        call_args = mock_client.messages.create.call_args
        system_content = call_args.kwargs.get("system", "")
        assert "Previous conversation" in system_content
        assert "Previous question" in system_content


class TestSingleRoundToolUse:
    """Test AIGenerator with single round of tool use (regression tests)"""

    @patch("ai_generator.anthropic.Anthropic")
    def test_single_tool_call_returns_answer(self, mock_anthropic_class):
        """Test that single tool use works and returns answer"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        tool_use_response = MockResponse(
            content=[
                MockToolUseBlock(
                    "tool_123", "search_course_content", {"query": "tool use"}
                )
            ],
            stop_reason="tool_use",
        )

        final_response = MockResponse(
            content=[
                MockTextBlock("Tool use allows Claude to call external functions.")
            ],
            stop_reason="end_turn",
        )

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "[Course] Content about tools"

        generator = AIGenerator(api_key="test-key", model="test-model")

        # Act
        result = generator.generate_response(
            query="What is tool use?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Assert
        assert mock_client.messages.create.call_count == 2
        assert mock_tool_manager.execute_tool.call_count == 1
        assert "Tool use allows Claude" in result

    @patch("ai_generator.anthropic.Anthropic")
    def test_no_tool_use_returns_direct_answer(self, mock_anthropic_class):
        """Test that queries not needing tools return directly"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MockResponse(
            content=[MockTextBlock("Direct answer without tools.")],
            stop_reason="end_turn",
        )
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test-key", model="test-model")

        # Act
        result = generator.generate_response(
            query="What is 2+2?",
            tools=[{"name": "search_course_content"}],
            tool_manager=Mock(),
        )

        # Assert
        assert mock_client.messages.create.call_count == 1
        assert result == "Direct answer without tools."

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_result_passed_to_claude(self, mock_anthropic_class):
        """Test that tool results are correctly formatted and sent back"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        tool_use_response = MockResponse(
            content=[
                MockToolUseBlock("tool_456", "search_course_content", {"query": "MCP"})
            ],
            stop_reason="tool_use",
        )

        final_response = MockResponse(
            content=[MockTextBlock("MCP is a protocol...")], stop_reason="end_turn"
        )

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search result about MCP"

        generator = AIGenerator(api_key="test-key", model="test-model")

        # Act
        generator.generate_response(
            query="What is MCP?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Assert - verify the second API call includes tool_result
        second_call_args = mock_client.messages.create.call_args_list[1]
        messages = second_call_args.kwargs.get("messages", [])

        assert len(messages) == 3  # user, assistant, user(result)
        assert messages[2]["role"] == "user"

        tool_result_content = messages[2]["content"]
        assert tool_result_content[0]["type"] == "tool_result"
        assert tool_result_content[0]["tool_use_id"] == "tool_456"
        assert tool_result_content[0]["content"] == "Search result about MCP"


class TestMultiRoundToolUse:
    """Test sequential tool calling (up to 2 rounds)"""

    @patch("ai_generator.anthropic.Anthropic")
    def test_two_round_tool_use_completes(self, mock_anthropic_class):
        """Test that Claude can use tools in 2 sequential rounds"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Round 1: Claude searches for content
        round1_response = MockResponse(
            content=[
                MockToolUseBlock(
                    "tool_1", "search_course_content", {"query": "MCP basics"}
                )
            ],
            stop_reason="tool_use",
        )

        # Round 2: Claude gets course outline for more context
        round2_response = MockResponse(
            content=[
                MockToolUseBlock("tool_2", "get_course_outline", {"course_name": "MCP"})
            ],
            stop_reason="tool_use",
        )

        # Final: Claude provides answer
        final_response = MockResponse(
            content=[MockTextBlock("MCP is covered in lessons 1-3 of the MCP course.")],
            stop_reason="end_turn",
        )

        mock_client.messages.create.side_effect = [
            round1_response,
            round2_response,
            final_response,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Content about MCP basics...",
            "Course: MCP\nLesson 1: Intro\nLesson 2: Architecture",
        ]

        generator = AIGenerator(api_key="test-key", model="test-model")

        # Act
        result = generator.generate_response(
            query="What is MCP and where is it covered?",
            tools=[{"name": "search_course_content"}, {"name": "get_course_outline"}],
            tool_manager=mock_tool_manager,
        )

        # Assert
        assert mock_client.messages.create.call_count == 3
        assert mock_tool_manager.execute_tool.call_count == 2
        assert "MCP is covered" in result

    @patch("ai_generator.anthropic.Anthropic")
    def test_max_rounds_enforced(self, mock_anthropic_class):
        """Test that third tool_use is not honored (max 2 rounds)"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Round 1: tool use
        round1_response = MockResponse(
            content=[
                MockToolUseBlock("tool_1", "search_course_content", {"query": "first"})
            ],
            stop_reason="tool_use",
        )

        # Round 2: tool use
        round2_response = MockResponse(
            content=[
                MockToolUseBlock("tool_2", "search_course_content", {"query": "second"})
            ],
            stop_reason="tool_use",
        )

        # Round 3: Would want tool use, but should be forced to give text
        # (tools not included in API call, so Claude gives text)
        round3_response = MockResponse(
            content=[MockTextBlock("Final answer after max rounds.")],
            stop_reason="end_turn",
        )

        mock_client.messages.create.side_effect = [
            round1_response,
            round2_response,
            round3_response,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]

        generator = AIGenerator(api_key="test-key", model="test-model")

        # Act
        result = generator.generate_response(
            query="Complex multi-step query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Assert
        assert mock_client.messages.create.call_count == 3
        assert mock_tool_manager.execute_tool.call_count == 2  # Only 2 tool executions
        assert "Final answer" in result

        # Verify third API call does NOT include tools
        third_call_args = mock_client.messages.create.call_args_list[2]
        assert "tools" not in third_call_args.kwargs

    @patch("ai_generator.anthropic.Anthropic")
    def test_early_termination_on_end_turn(self, mock_anthropic_class):
        """Test that loop stops when Claude returns end_turn after first round"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Round 1: tool use
        round1_response = MockResponse(
            content=[
                MockToolUseBlock("tool_1", "search_course_content", {"query": "test"})
            ],
            stop_reason="tool_use",
        )

        # Claude satisfied after first tool, returns text
        final_response = MockResponse(
            content=[MockTextBlock("Got enough info from first search.")],
            stop_reason="end_turn",
        )

        mock_client.messages.create.side_effect = [round1_response, final_response]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Sufficient result"

        generator = AIGenerator(api_key="test-key", model="test-model")

        # Act
        result = generator.generate_response(
            query="Simple query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Assert
        assert mock_client.messages.create.call_count == 2  # Not 3
        assert mock_tool_manager.execute_tool.call_count == 1
        assert "Got enough info" in result

    @patch("ai_generator.anthropic.Anthropic")
    def test_tools_included_in_both_rounds(self, mock_anthropic_class):
        """Test that tools are available in both round 1 and round 2"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        round1_response = MockResponse(
            content=[MockToolUseBlock("t1", "search_course_content", {"query": "a"})],
            stop_reason="tool_use",
        )
        round2_response = MockResponse(
            content=[
                MockToolUseBlock("t2", "get_course_outline", {"course_name": "X"})
            ],
            stop_reason="tool_use",
        )
        final_response = MockResponse(
            content=[MockTextBlock("Done")], stop_reason="end_turn"
        )

        mock_client.messages.create.side_effect = [
            round1_response,
            round2_response,
            final_response,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["R1", "R2"]

        test_tools = [{"name": "search_course_content"}, {"name": "get_course_outline"}]
        generator = AIGenerator(api_key="test-key", model="test-model")

        # Act
        generator.generate_response(
            query="Test", tools=test_tools, tool_manager=mock_tool_manager
        )

        # Assert - tools in first two calls
        first_call = mock_client.messages.create.call_args_list[0]
        second_call = mock_client.messages.create.call_args_list[1]

        assert first_call.kwargs.get("tools") == test_tools
        assert second_call.kwargs.get("tools") == test_tools


class TestToolErrorHandling:
    """Test error handling during tool execution"""

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_error_prevents_further_rounds(self, mock_anthropic_class):
        """Test that tool execution error stops further tool use"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        tool_use_response = MockResponse(
            content=[
                MockToolUseBlock("tool_1", "search_course_content", {"query": "test"})
            ],
            stop_reason="tool_use",
        )

        final_response = MockResponse(
            content=[MockTextBlock("Handled the error gracefully.")],
            stop_reason="end_turn",
        )

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool failed!")

        generator = AIGenerator(api_key="test-key", model="test-model")

        # Act
        result = generator.generate_response(
            query="Test",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Assert
        assert "Handled the error" in result

        # Verify second call has no tools (error prevents more rounds)
        second_call = mock_client.messages.create.call_args_list[1]
        assert "tools" not in second_call.kwargs

        # Verify error was sent to Claude
        messages = second_call.kwargs["messages"]
        tool_result = messages[2]["content"][0]
        assert tool_result["is_error"] is True
        assert "Error executing tool" in tool_result["content"]

    @patch("ai_generator.anthropic.Anthropic")
    def test_api_error_propagates(self, mock_anthropic_class):
        """Test that API errors are not caught"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API Error: Rate limited")

        generator = AIGenerator(api_key="test-key", model="test-model")

        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            generator.generate_response(query="Test query")

        assert "API Error" in str(exc_info.value)

    @patch("ai_generator.anthropic.Anthropic")
    def test_without_tool_manager_returns_text(self, mock_anthropic_class):
        """Test that tool_use without tool_manager returns available text"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        response = MockResponse(
            content=[
                MockTextBlock("I would search but can't."),
                MockToolUseBlock("tool_1", "search", {"query": "x"}),
            ],
            stop_reason="tool_use",
        )

        mock_client.messages.create.return_value = response

        generator = AIGenerator(api_key="test-key", model="test-model")

        # Act - no tool_manager
        result = generator.generate_response(
            query="Search", tools=[{"name": "search"}], tool_manager=None
        )

        # Assert
        assert result == "I would search but can't."


class TestMessageChainConstruction:
    """Test that message history is correctly built across rounds"""

    @patch("ai_generator.anthropic.Anthropic")
    def test_messages_accumulate_correctly(self, mock_anthropic_class):
        """Test that all messages are preserved across rounds"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        round1_response = MockResponse(
            content=[MockToolUseBlock("t1", "search", {"query": "a"})],
            stop_reason="tool_use",
        )
        round2_response = MockResponse(
            content=[MockToolUseBlock("t2", "outline", {"course": "b"})],
            stop_reason="tool_use",
        )
        final_response = MockResponse(
            content=[MockTextBlock("Final")], stop_reason="end_turn"
        )

        mock_client.messages.create.side_effect = [
            round1_response,
            round2_response,
            final_response,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]

        generator = AIGenerator(api_key="test-key", model="test-model")

        # Act
        generator.generate_response(
            query="Complex query",
            tools=[{"name": "search"}, {"name": "outline"}],
            tool_manager=mock_tool_manager,
        )

        # Assert - check third call has full history
        third_call = mock_client.messages.create.call_args_list[2]
        messages = third_call.kwargs["messages"]

        # Should be: user, assistant(tool1), user(result1), assistant(tool2), user(result2)
        assert len(messages) == 5
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "assistant"
        assert messages[4]["role"] == "user"
