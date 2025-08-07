import anthropic
import time
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search tools for course information.

Tool Usage Guidelines:
- **Content search tool**: Use for questions about specific course content, materials, or detailed educational information within courses
- **Course outline tool**: Use for questions about course structure, lesson lists, course titles, instructors, or course overviews
- **Sequential tool usage**: You can make up to 2 tool calls in separate rounds to gather comprehensive information
- **Multi-step reasoning**: After seeing tool results, you can make additional tool calls if needed to fully answer the question
- **Tool call strategy**: Use results from first tool call to inform parameters for second tool call
- Synthesize all tool results into accurate, fact-based responses
- If any tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course content questions**: Use content search tool first, then additional searches if needed
- **Course structure/outline questions**: Use course outline tool first, then content search if needed
- **Complex queries**: Break down into multiple targeted tool calls across rounds
- **No meta-commentary**: Provide direct answers only — no reasoning process, tool explanations, or question-type analysis

For outline queries, always include:
- Course title
- Course link (if available)
- Complete lesson list with lesson numbers and titles

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None,
                         max_rounds: int = 2) -> str:
        """
        Generate AI response with sequential tool calling support.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool calling rounds (default: 2)
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Initialize conversation with user query
        messages = [{"role": "user", "content": query}]
        
        # Execute sequential tool calling rounds
        for round_num in range(max_rounds):
            # Prepare API call parameters
            api_params = {
                **self.base_params,
                "messages": messages.copy(),
                "system": system_content
            }
            
            # Add tools if available (keep tools available for all rounds)
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}
            
            # Get response from Claude with retry logic
            response = self._api_call_with_retry(api_params)
            
            # Add Claude's response to conversation
            messages.append({"role": "assistant", "content": response.content})
            
            # Check if Claude wants to use tools
            if response.stop_reason == "tool_use" and tool_manager:
                # Execute tools and add results to conversation
                tool_results = self._execute_tools_for_round(response, tool_manager)
                if tool_results:
                    messages.append({"role": "user", "content": tool_results})
                    continue  # Go to next round
                else:
                    # Tool execution failed, break and return current response
                    break
            else:
                # No tool use, we have our final response
                break
        
        # If we exhausted all rounds or no tools were used, get final response
        if messages[-1]["role"] == "user":  # Last message was tool results
            # Need one final call to synthesize
            final_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content
                # Note: Intentionally NOT including tools for final synthesis
            }
            final_response = self._api_call_with_retry(final_params)
            return final_response.content[0].text
        else:
            # Last message was from assistant without tool use
            return response.content[0].text
    
    def _api_call_with_retry(self, api_params: Dict[str, Any], max_retries: int = 2):
        """
        Make API call with retry logic for transient failures.
        
        Args:
            api_params: API call parameters
            max_retries: Maximum retry attempts
            
        Returns:
            API response
        """
        for attempt in range(max_retries + 1):
            try:
                return self.client.messages.create(**api_params)
            except Exception as e:
                if attempt == max_retries:
                    raise e
                
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "network" in error_msg or "connection" in error_msg:
                    # Add exponential backoff for retries
                    time.sleep(2 ** attempt)
                else:
                    # For other errors, don't retry
                    raise e
    
    def _execute_tools_for_round(self, response, tool_manager) -> Optional[List[Dict]]:
        """
        Execute all tool calls from a response and return formatted results.
        
        Args:
            response: Claude's response containing tool use requests
            tool_manager: Manager to execute tools
            
        Returns:
            List of tool results or None if execution failed
        """
        tool_results = []
        
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, 
                        **content_block.input
                    )
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })
                    
                except Exception as e:
                    # Log error but continue with other tools
                    error_msg = str(e)
                    if "rate limit" in error_msg.lower() or "network" in error_msg.lower():
                        # For transient errors, return None to stop rounds
                        return None
                    else:
                        # For other errors, continue with error message
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": f"Tool execution failed: {error_msg}"
                        })
        
        return tool_results if tool_results else None
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text