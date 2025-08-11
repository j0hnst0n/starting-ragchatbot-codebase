import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add backend to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator
from search_tools import CourseSearchTool, ToolManager


@dataclass
class MockContent:
    """Mock content block for Anthropic responses"""

    type: str
    text: str = None
    id: str = None
    name: str = None
    input: Dict[str, Any] = None


@dataclass
class MockResponse:
    """Mock Anthropic API response"""

    content: List[MockContent]
    stop_reason: str


class TestAIGenerator:
    """Test suite for AIGenerator functionality"""

    @pytest.fixture
    def api_key(self):
        """Mock API key for testing"""
        return "test_api_key"

    @pytest.fixture
    def model(self):
        """Model name for testing"""
        return "claude-3-sonnet-20240229"

    @pytest.fixture
    def ai_generator(self, api_key, model):
        """Create AIGenerator instance for testing"""
        return AIGenerator(api_key, model)

    @pytest.fixture
    def mock_tool_manager(self):
        """Create mock tool manager"""
        manager = Mock(spec=ToolManager)
        manager.get_tool_definitions.return_value = [
            {
                "name": "search_course_content",
                "description": "Search course content",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "course_name": {"type": "string"},
                    },
                    "required": ["query"],
                },
            }
        ]
        return manager

    def test_initialization(self, ai_generator, api_key, model):
        """Test AIGenerator initialization"""
        print("\\n=== Testing AIGenerator Initialization ===")
        try:
            assert ai_generator.model == model
            assert ai_generator.base_params["model"] == model
            assert ai_generator.base_params["temperature"] == 0
            assert ai_generator.base_params["max_tokens"] == 800
            print("✅ AIGenerator initialization successful")

        except Exception as e:
            print(f"❌ AIGenerator initialization failed: {e}")
            raise

    def test_system_prompt(self, ai_generator):
        """Test system prompt content and structure"""
        print("\\n=== Testing System Prompt ===")
        try:
            assert hasattr(ai_generator, "SYSTEM_PROMPT")
            prompt = ai_generator.SYSTEM_PROMPT

            # Check for key elements in system prompt
            assert "course materials" in prompt.lower()
            assert "search tool" in prompt.lower() or "content search" in prompt.lower()
            assert "educational" in prompt.lower()
            print("✅ System prompt structure validated")

        except Exception as e:
            print(f"❌ System prompt test failed: {e}")
            raise

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_without_tools(self, mock_anthropic, ai_generator):
        """Test generating response without tool usage"""
        print("\\n=== Testing Response Generation Without Tools ===")
        try:
            # Setup mock client
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            # Setup mock response
            mock_response = MockResponse(
                content=[MockContent(type="text", text="This is a direct response.")],
                stop_reason="end_turn",
            )
            mock_client.messages.create.return_value = mock_response

            # Replace client in ai_generator
            ai_generator.client = mock_client

            # Generate response
            result = ai_generator.generate_response("What is machine learning?")

            # Verify API call
            mock_client.messages.create.assert_called_once()
            call_args = mock_client.messages.create.call_args[1]

            assert call_args["model"] == ai_generator.model
            assert call_args["temperature"] == 0
            assert call_args["max_tokens"] == 800
            assert len(call_args["messages"]) == 1
            assert call_args["messages"][0]["role"] == "user"
            assert call_args["messages"][0]["content"] == "What is machine learning?"

            # Verify response
            assert result == "This is a direct response."
            print("✅ Response generation without tools successful")

        except Exception as e:
            print(f"❌ Response generation without tools failed: {e}")
            raise

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_with_conversation_history(
        self, mock_anthropic, ai_generator
    ):
        """Test generating response with conversation history"""
        print("\\n=== Testing Response with Conversation History ===")
        try:
            # Setup mock client
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            ai_generator.client = mock_client

            mock_response = MockResponse(
                content=[MockContent(type="text", text="Response with history.")],
                stop_reason="end_turn",
            )
            mock_client.messages.create.return_value = mock_response

            # Generate response with history
            history = "User: Previous question\\nAssistant: Previous answer"
            result = ai_generator.generate_response(
                "Follow up question", conversation_history=history
            )

            # Verify system prompt includes history
            call_args = mock_client.messages.create.call_args[1]
            system_content = call_args["system"]
            assert "Previous conversation:" in system_content
            assert history in system_content

            print("✅ Response with conversation history successful")

        except Exception as e:
            print(f"❌ Response with conversation history failed: {e}")
            raise

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_with_tools_no_use(
        self, mock_anthropic, ai_generator, mock_tool_manager
    ):
        """Test generating response with tools available but not used"""
        print("\\n=== Testing Response with Tools Available (Not Used) ===")
        try:
            # Setup mock client
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            ai_generator.client = mock_client

            mock_response = MockResponse(
                content=[MockContent(type="text", text="Direct answer without tools.")],
                stop_reason="end_turn",
            )
            mock_client.messages.create.return_value = mock_response

            # Generate response with tools
            tools = mock_tool_manager.get_tool_definitions()
            result = ai_generator.generate_response(
                "What is 2 + 2?", tools=tools, tool_manager=mock_tool_manager
            )

            # Verify tools were passed to API
            call_args = mock_client.messages.create.call_args[1]
            assert "tools" in call_args
            assert call_args["tool_choice"] == {"type": "auto"}
            assert len(call_args["tools"]) == 1

            # Verify response
            assert result == "Direct answer without tools."
            print("✅ Response with tools available (not used) successful")

        except Exception as e:
            print(f"❌ Response with tools available failed: {e}")
            raise

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_with_single_tool_use(
        self, mock_anthropic, ai_generator, mock_tool_manager
    ):
        """Test generating response with single tool usage (backward compatibility)"""
        print("\\n=== Testing Response with Single Tool Usage ===")
        try:
            # Setup mock client
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            ai_generator.client = mock_client

            # First response: tool use
            tool_use_response = MockResponse(
                content=[
                    MockContent(
                        type="tool_use",
                        id="tool_123",
                        name="search_course_content",
                        input={"query": "machine learning", "course_name": "AI Course"},
                    )
                ],
                stop_reason="tool_use",
            )

            # Final response after tool execution
            final_response = MockResponse(
                content=[
                    MockContent(
                        type="text",
                        text="Based on the search results, here's what I found...",
                    )
                ],
                stop_reason="end_turn",
            )

            # Setup mock client to return both responses
            mock_client.messages.create.side_effect = [
                tool_use_response,
                final_response,
            ]

            # Setup tool manager to return mock result
            mock_tool_manager.execute_tool.return_value = (
                "Search results: Machine learning is..."
            )

            # Generate response with tool usage
            tools = mock_tool_manager.get_tool_definitions()
            result = ai_generator.generate_response(
                "Tell me about machine learning in AI Course",
                tools=tools,
                tool_manager=mock_tool_manager,
            )

            # Verify tool was executed
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content",
                query="machine learning",
                course_name="AI Course",
            )

            # Verify two API calls were made
            assert mock_client.messages.create.call_count == 2

            # Check second API call structure
            second_call_args = mock_client.messages.create.call_args_list[1][1]
            messages = second_call_args["messages"]

            # Should have original user message + assistant tool use + user tool result
            assert len(messages) == 3
            assert messages[0]["role"] == "user"  # Original query
            assert messages[1]["role"] == "assistant"  # Tool use
            assert messages[2]["role"] == "user"  # Tool result

            # Verify final result
            assert result == "Based on the search results, here's what I found..."
            print("✅ Response with single tool usage successful")

        except Exception as e:
            print(f"❌ Response with single tool usage failed: {e}")
            raise

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_execution_handling(
        self, mock_anthropic, ai_generator, mock_tool_manager
    ):
        """Test detailed tool execution handling"""
        print("\\n=== Testing Tool Execution Handling ===")
        try:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            ai_generator.client = mock_client

            # Create tool use response with multiple tools
            tool_use_response = MockResponse(
                content=[
                    MockContent(
                        type="tool_use",
                        id="tool_1",
                        name="search_course_content",
                        input={"query": "introduction"},
                    ),
                    MockContent(type="text", text="I'm searching for information..."),
                ],
                stop_reason="tool_use",
            )

            final_response = MockResponse(
                content=[MockContent(type="text", text="Here are the results...")],
                stop_reason="end_turn",
            )

            mock_client.messages.create.side_effect = [
                tool_use_response,
                final_response,
            ]
            mock_tool_manager.execute_tool.return_value = "Tool execution result"

            # Test _handle_tool_execution directly
            base_params = {
                "messages": [{"role": "user", "content": "test query"}],
                "system": "test system prompt",
            }

            result = ai_generator._handle_tool_execution(
                tool_use_response, base_params, mock_tool_manager
            )

            # Verify tool execution
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content", query="introduction"
            )

            # Verify final API call structure
            final_call_args = mock_client.messages.create.call_args[1]
            messages = final_call_args["messages"]

            assert len(messages) == 3
            # Check tool result message structure
            tool_result_msg = messages[2]
            assert tool_result_msg["role"] == "user"
            assert isinstance(tool_result_msg["content"], list)
            assert tool_result_msg["content"][0]["type"] == "tool_result"
            assert tool_result_msg["content"][0]["tool_use_id"] == "tool_1"
            assert tool_result_msg["content"][0]["content"] == "Tool execution result"

            print("✅ Tool execution handling successful")

        except Exception as e:
            print(f"❌ Tool execution handling failed: {e}")
            raise

    @patch("ai_generator.anthropic.Anthropic")
    def test_api_error_handling(self, mock_anthropic, ai_generator):
        """Test API error handling"""
        print("\\n=== Testing API Error Handling ===")
        try:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            ai_generator.client = mock_client

            # Simulate API error
            mock_client.messages.create.side_effect = Exception(
                "API Error: Rate limit exceeded"
            )

            # This should raise the exception
            with pytest.raises(Exception) as exc_info:
                ai_generator.generate_response("test query")

            assert "API Error" in str(exc_info.value)
            print("✅ API error handling successful")

        except Exception as e:
            print(f"❌ API error handling test failed: {e}")
            raise

    def test_parameter_validation(self, ai_generator, mock_tool_manager):
        """Test parameter validation and edge cases"""
        print("\\n=== Testing Parameter Validation ===")
        try:
            # Test empty query
            with patch.object(ai_generator.client, "messages") as mock_messages:
                mock_response = MockResponse(
                    content=[MockContent(type="text", text="Empty query response")],
                    stop_reason="end_turn",
                )
                mock_messages.create.return_value = mock_response

                result = ai_generator.generate_response("")
                assert isinstance(result, str)
                print("✅ Empty query handling successful")

            # Test None parameters
            with patch.object(ai_generator.client, "messages") as mock_messages:
                mock_response = MockResponse(
                    content=[MockContent(type="text", text="None params response")],
                    stop_reason="end_turn",
                )
                mock_messages.create.return_value = mock_response

                result = ai_generator.generate_response("query", None, None, None)
                assert isinstance(result, str)
                print("✅ None parameters handling successful")

        except Exception as e:
            print(f"❌ Parameter validation test failed: {e}")
            raise

    @patch("ai_generator.anthropic.Anthropic")
    def test_sequential_tool_calls(
        self, mock_anthropic, ai_generator, mock_tool_manager
    ):
        """Test sequential tool calling with 2 rounds"""
        print("\\n=== Testing Sequential Tool Calls ===")
        try:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            ai_generator.client = mock_client

            # Round 1: First tool call
            first_tool_response = MockResponse(
                content=[
                    MockContent(
                        type="tool_use",
                        id="tool_1",
                        name="get_course_outline",
                        input={"course_name": "MCP Course"},
                    )
                ],
                stop_reason="tool_use",
            )

            # Round 2: Second tool call after seeing first results
            second_tool_response = MockResponse(
                content=[
                    MockContent(
                        type="tool_use",
                        id="tool_2",
                        name="search_course_content",
                        input={
                            "query": "lesson 3 content",
                            "course_name": "MCP Course",
                        },
                    )
                ],
                stop_reason="tool_use",
            )

            # Final response after all tool calls
            final_response = MockResponse(
                content=[
                    MockContent(
                        type="text",
                        text="Here's the comprehensive information about lesson 3...",
                    )
                ],
                stop_reason="end_turn",
            )

            mock_client.messages.create.side_effect = [
                first_tool_response,
                second_tool_response,
                final_response,
            ]

            # Setup tool manager responses
            mock_tool_manager.execute_tool.side_effect = [
                "Course outline: Lesson 1, Lesson 2, Lesson 3...",
                "Lesson 3 detailed content...",
            ]

            # Execute sequential tool calling
            tools = mock_tool_manager.get_tool_definitions()
            result = ai_generator.generate_response(
                "Tell me about lesson 3 in MCP Course with detailed content",
                tools=tools,
                tool_manager=mock_tool_manager,
                max_rounds=2,
            )

            # Verify both tools were executed
            assert mock_tool_manager.execute_tool.call_count == 2

            # Check first tool call
            first_call = mock_tool_manager.execute_tool.call_args_list[0]
            assert first_call[0][0] == "get_course_outline"
            assert first_call[1]["course_name"] == "MCP Course"

            # Check second tool call
            second_call = mock_tool_manager.execute_tool.call_args_list[1]
            assert second_call[0][0] == "search_course_content"

            # Verify 3 API calls were made (2 tool rounds + 1 final synthesis)
            assert mock_client.messages.create.call_count == 3

            # Verify final result
            assert result == "Here's the comprehensive information about lesson 3..."
            print("✅ Sequential tool calls successful")

        except Exception as e:
            print(f"❌ Sequential tool calls failed: {e}")
            raise

    @patch("ai_generator.anthropic.Anthropic")
    def test_max_rounds_enforcement(
        self, mock_anthropic, ai_generator, mock_tool_manager
    ):
        """Test that system respects max_rounds limit"""
        print("\\n=== Testing Max Rounds Enforcement ===")
        try:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            ai_generator.client = mock_client

            # Always return tool use responses (would go infinite without max_rounds)
            tool_response = MockResponse(
                content=[
                    MockContent(
                        type="tool_use",
                        id="tool_x",
                        name="search_course_content",
                        input={"query": "test"},
                    )
                ],
                stop_reason="tool_use",
            )

            final_response = MockResponse(
                content=[
                    MockContent(type="text", text="Final answer after max rounds")
                ],
                stop_reason="end_turn",
            )

            # Return tool responses for first 2 calls, then final response
            mock_client.messages.create.side_effect = [
                tool_response,
                tool_response,
                final_response,
            ]
            mock_tool_manager.execute_tool.return_value = "Tool result"

            # Test with max_rounds=2
            tools = mock_tool_manager.get_tool_definitions()
            result = ai_generator.generate_response(
                "Test max rounds",
                tools=tools,
                tool_manager=mock_tool_manager,
                max_rounds=2,
            )

            # Should have made exactly 3 API calls (2 rounds + 1 final)
            assert mock_client.messages.create.call_count == 3
            assert mock_tool_manager.execute_tool.call_count == 2
            assert result == "Final answer after max rounds"
            print("✅ Max rounds enforcement successful")

        except Exception as e:
            print(f"❌ Max rounds enforcement failed: {e}")
            raise

    @patch("ai_generator.anthropic.Anthropic")
    def test_early_termination_no_tools(
        self, mock_anthropic, ai_generator, mock_tool_manager
    ):
        """Test early termination when Claude doesn't use tools"""
        print("\\n=== Testing Early Termination (No Tools) ===")
        try:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            ai_generator.client = mock_client

            # Response without tool use
            direct_response = MockResponse(
                content=[MockContent(type="text", text="Direct answer without tools")],
                stop_reason="end_turn",
            )

            mock_client.messages.create.return_value = direct_response

            tools = mock_tool_manager.get_tool_definitions()
            result = ai_generator.generate_response(
                "What is 2+2?",
                tools=tools,
                tool_manager=mock_tool_manager,
                max_rounds=2,
            )

            # Should have made only 1 API call
            assert mock_client.messages.create.call_count == 1
            assert mock_tool_manager.execute_tool.call_count == 0
            assert result == "Direct answer without tools"
            print("✅ Early termination successful")

        except Exception as e:
            print(f"❌ Early termination failed: {e}")
            raise

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_execution_error_handling(
        self, mock_anthropic, ai_generator, mock_tool_manager
    ):
        """Test handling of tool execution errors"""
        print("\\n=== Testing Tool Execution Error Handling ===")
        try:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            ai_generator.client = mock_client

            tool_use_response = MockResponse(
                content=[
                    MockContent(
                        type="tool_use",
                        id="tool_error",
                        name="search_course_content",
                        input={"query": "test"},
                    )
                ],
                stop_reason="tool_use",
            )

            final_response = MockResponse(
                content=[MockContent(type="text", text="Handled error gracefully")],
                stop_reason="end_turn",
            )

            mock_client.messages.create.side_effect = [
                tool_use_response,
                final_response,
            ]

            # Simulate tool execution error
            mock_tool_manager.execute_tool.side_effect = Exception("Tool failed")

            tools = mock_tool_manager.get_tool_definitions()
            result = ai_generator.generate_response(
                "Test error handling", tools=tools, tool_manager=mock_tool_manager
            )

            # Should still return a response
            assert result == "Handled error gracefully"
            print("✅ Tool execution error handling successful")

        except Exception as e:
            print(f"❌ Tool execution error handling failed: {e}")
            raise


class TestAIGeneratorIntegration:
    """Integration tests with real tool manager"""

    @pytest.fixture
    def real_tool_manager(self):
        """Create real tool manager with mock vector store"""
        from search_tools import CourseOutlineTool, CourseSearchTool
        from vector_store import VectorStore

        # Create mock vector store
        mock_store = Mock(spec=VectorStore)

        # Create real tool manager with both tools
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_store)
        outline_tool = CourseOutlineTool(mock_store)
        tool_manager.register_tool(search_tool)
        tool_manager.register_tool(outline_tool)

        return tool_manager, mock_store

    @patch("ai_generator.anthropic.Anthropic")
    def test_integration_with_real_tools(self, mock_anthropic, real_tool_manager):
        """Test integration with real tool manager"""
        print("\\n=== Testing Integration with Real Tools ===")
        try:
            tool_manager, mock_store = real_tool_manager

            # Create AI generator
            ai_gen = AIGenerator("test_key", "test_model")

            # Setup mocks
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            ai_gen.client = mock_client

            # Mock vector store to return results
            from vector_store import SearchResults

            mock_store.search.return_value = SearchResults(
                documents=["Test course content"],
                metadata=[{"course_title": "Test Course", "lesson_number": 0}],
                distances=[0.1],
            )
            mock_store.get_lesson_link.return_value = "https://example.com/lesson0"

            # Setup tool use response
            tool_response = MockResponse(
                content=[
                    MockContent(
                        type="tool_use",
                        id="tool_1",
                        name="search_course_content",
                        input={"query": "test content"},
                    )
                ],
                stop_reason="tool_use",
            )

            final_response = MockResponse(
                content=[MockContent(type="text", text="Integration test successful")],
                stop_reason="end_turn",
            )

            mock_client.messages.create.side_effect = [tool_response, final_response]

            # Execute
            result = ai_gen.generate_response(
                "Tell me about test content",
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager,
            )

            # Verify vector store was called
            mock_store.search.assert_called_once_with(
                query="test content", course_name=None, lesson_number=None
            )

            assert result == "Integration test successful"
            print("✅ Integration with real tools successful")

        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            raise

    @patch("ai_generator.anthropic.Anthropic")
    def test_sequential_integration_with_real_tools(
        self, mock_anthropic, real_tool_manager
    ):
        """Test sequential tool calling with real tool manager"""
        print("\\n=== Testing Sequential Integration with Real Tools ===")
        try:
            tool_manager, mock_store = real_tool_manager

            # Create AI generator
            ai_gen = AIGenerator("test_key", "test_model")

            # Setup mocks
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            ai_gen.client = mock_client

            # Mock vector store to return different results for different calls
            from vector_store import SearchResults

            mock_store.search.side_effect = [
                SearchResults(
                    documents=["Course outline content"],
                    metadata=[{"course_title": "Test Course", "lesson_number": 0}],
                    distances=[0.1],
                ),
                SearchResults(
                    documents=["Specific lesson content"],
                    metadata=[{"course_title": "Test Course", "lesson_number": 3}],
                    distances=[0.1],
                ),
            ]
            mock_store.get_lesson_link.return_value = "https://example.com/lesson"
            mock_store._resolve_course_name.return_value = "Test Course"

            # Setup sequential tool responses
            first_tool_response = MockResponse(
                content=[
                    MockContent(
                        type="tool_use",
                        id="tool_1",
                        name="get_course_outline",
                        input={"course_name": "test"},
                    )
                ],
                stop_reason="tool_use",
            )

            second_tool_response = MockResponse(
                content=[
                    MockContent(
                        type="tool_use",
                        id="tool_2",
                        name="search_course_content",
                        input={"query": "lesson 3", "lesson_number": 3},
                    )
                ],
                stop_reason="tool_use",
            )

            final_response = MockResponse(
                content=[
                    MockContent(type="text", text="Sequential integration successful")
                ],
                stop_reason="end_turn",
            )

            mock_client.messages.create.side_effect = [
                first_tool_response,
                second_tool_response,
                final_response,
            ]

            # Execute with sequential tool calling
            result = ai_gen.generate_response(
                "Get course outline then search lesson 3",
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager,
                max_rounds=2,
            )

            # Verify both search operations were called
            assert (
                mock_store.search.call_count == 1
            )  # Only content search, outline uses different method

            assert result == "Sequential integration successful"
            print("✅ Sequential integration with real tools successful")

        except Exception as e:
            print(f"❌ Sequential integration test failed: {e}")
            raise


if __name__ == "__main__":
    print("Running AIGenerator tests...")
    pytest.main([__file__, "-v", "-s"])
