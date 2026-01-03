import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Maximum number of sequential tool calling rounds
    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Multi-Step Tool Usage:
- You may use tools up to 2 times per query when complex questions require multiple lookups
- After receiving tool results, you can choose to use another tool if the initial results are insufficient
- Use sequential tools when you need to: search broadly then narrow down, get course outline then search specific content, or compare information across courses

Search Tool Usage:
- Use the search tool for questions about specific course content or detailed educational materials
- If initial search results are insufficient, you may refine your search with different terms
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Course Outline Tool Usage:
- Use the course outline tool for questions about course structure, lesson lists, or what topics a course covers
- Returns: course title, course link, and complete list of lessons with their numbers, titles, and links
- Use this instead of the search tool when the user asks about course structure, outline, or lesson listings
- Always include the course title, course link, and all lesson information (number, title, link) in your response

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer (use additional searches if needed)
- **Course structure/outline questions**: Use the outline tool, then present the full course information
- **Complex questions**: You may search, analyze results, then search again or get outlines as needed
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results" or reference tool usage

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
                         tool_manager=None) -> str:
        """
        Generate AI response with optional multi-round tool usage.

        Supports up to MAX_TOOL_ROUNDS sequential tool calls, where each round
        allows Claude to reason about previous tool results and potentially
        call additional tools.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """
        # Build system content
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initialize messages with user query
        messages = [{"role": "user", "content": query}]

        # Prepare API call parameters
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get initial response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution loop if needed
        if response.stop_reason == "tool_use" and tool_manager and tools:
            return self._continue_tool_loop(
                response=response,
                messages=messages,
                system=system_content,
                tools=tools,
                tool_manager=tool_manager,
                depth=1
            )

        # Return direct response (no tool use)
        return self._extract_text_response(response)

    def _continue_tool_loop(self,
                            response,
                            messages: List[Dict[str, Any]],
                            system: str,
                            tools: List,
                            tool_manager,
                            depth: int) -> str:
        """
        Recursively handle tool execution rounds.

        Args:
            response: The API response containing tool_use blocks
            messages: Current conversation messages (will be copied, not mutated)
            system: System prompt
            tools: Tool definitions
            tool_manager: Manager to execute tools
            depth: Current round number (1-based)

        Returns:
            Final response text
        """
        # Copy messages (immutable approach)
        new_messages = messages.copy()

        # Add Claude's response containing tool_use blocks
        new_messages.append({"role": "assistant", "content": response.content})

        # Execute all tool calls and collect results
        tool_results = []
        has_error = False

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
                    # Mark error but continue with other tools
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": f"Error executing tool: {str(e)}",
                        "is_error": True
                    })
                    has_error = True

        # Add tool results to messages
        if tool_results:
            new_messages.append({"role": "user", "content": tool_results})

        # Determine if we should allow more tool calls
        allow_more_tools = depth < self.MAX_TOOL_ROUNDS and not has_error

        # Prepare next API call
        next_params = {
            **self.base_params,
            "messages": new_messages,
            "system": system
        }

        # Only include tools if we're allowing more rounds
        if allow_more_tools:
            next_params["tools"] = tools
            next_params["tool_choice"] = {"type": "auto"}

        # Make the next API call
        next_response = self.client.messages.create(**next_params)

        # Check if Claude wants more tools AND we allow it
        if next_response.stop_reason == "tool_use" and allow_more_tools:
            # Recursive call for next round
            return self._continue_tool_loop(
                response=next_response,
                messages=new_messages,
                system=system,
                tools=tools,
                tool_manager=tool_manager,
                depth=depth + 1
            )

        # Termination: extract text response
        return self._extract_text_response(next_response)

    def _extract_text_response(self, response) -> str:
        """
        Extract text content from API response.

        Args:
            response: The API response object

        Returns:
            Text content from the response, or empty string if none found
        """
        for block in response.content:
            if hasattr(block, 'text'):
                return block.text
        return ""