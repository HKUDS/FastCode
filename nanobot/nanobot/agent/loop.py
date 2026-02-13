"""Agent loop: the core processing engine."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.context import ContextBuilder
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.subagent import SubagentManager
from nanobot.session.manager import SessionManager


class AgentLoop:
    """
    The agent loop is the core processing engine.
    
    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """
    
    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        cron_service: "CronService | None" = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig
        from nanobot.cron.service import CronService
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        
        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )
        
        self._running = False
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools (restrict to workspace if configured)
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))
        
        # Shell tool
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))
        
        # Web tools
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        
        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)
        
        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)
        
        # Cron tool (for scheduling)
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

        # FastCode tools (conditionally loaded when FASTCODE_API_URL is set)
        fastcode_url = os.environ.get("FASTCODE_API_URL")
        if fastcode_url:
            from nanobot.agent.tools.fastcode import create_all_tools
            for tool in create_all_tools(api_url=fastcode_url):
                self.tools.register(tool)
            logger.info(f"FastCode tools registered (API: {fastcode_url})")
    
    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        logger.info("Agent loop started")
        
        while self._running:
            try:
                # Wait for next message
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
                
                # Process it
                try:
                    response = await self._process_message(msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    # Send error response
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))
            except asyncio.TimeoutError:
                continue
    
    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")
    
    async def _execute_tool_with_feedback(
        self,
        msg: InboundMessage,
        tool_call: Any,
        initial_delay: float = 20.0,
        interval: float = 40.0,
        max_updates: int = 8,  # More updates for better feedback on large repos
        max_timeout: float = 1200.0,  # 20 minutes maximum for very large repositories
    ) -> Any:
        """
        Execute a potentially long-running tool while periodically sending
        lightweight status updates back to the user.

        Args:
            msg: The inbound message context
            tool_call: The tool call to execute
            initial_delay: Initial wait time before first update (seconds)
            interval: Interval between status updates (seconds)
            max_updates: Maximum number of status updates to send
            max_timeout: Maximum total time to wait for tool completion (seconds)
        """
        async def _run_tool() -> Any:
            return await self.tools.execute(tool_call.name, tool_call.arguments)

        task = asyncio.create_task(_run_tool())

        # First, wait for a short period – if the tool finishes quickly, we don't spam updates.
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=initial_delay)
        except asyncio.TimeoutError:
            pass

        updates_sent = 0

        # While still running, send periodic "still working" messages.
        while not task.done() and updates_sent < max_updates:
            try:
                if tool_call.name == "fastcode_load_repo":
                    content = (
                        "索引仍在进行中，仓库可能比较大，我会持续等待直到完成。\n\n"
                        "Repository indexing is still running. Large repositories can take a while."
                    )
                elif tool_call.name == "fastcode_query":
                    content = (
                        "我还在检索和分析相关代码片段，请再稍等几秒钟…\n\n"
                        "I am still retrieving and analyzing the relevant code; this can take a few seconds."
                    )
                else:
                    content = (
                        f"正在执行工具 `{tool_call.name}`，时间有点久，我还在等待结果。\n\n"
                        f"The tool `{tool_call.name}` is still running; I'm waiting for it to finish."
                    )

                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=content,
                ))
                updates_sent += 1
            except Exception as notify_err:
                logger.warning(f"Failed to send periodic tool status message: {notify_err}")

            try:
                return await asyncio.wait_for(asyncio.shield(task), timeout=interval)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                logger.warning(f"Tool {tool_call.name} was cancelled during periodic wait")
                try:
                    cancel_msg = (
                        f"⚠️ 工具 `{tool_call.name}` 执行被取消。\n\n"
                        f"⚠️ Tool `{tool_call.name}` execution was cancelled."
                    )
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=cancel_msg,
                    ))
                except Exception:
                    pass
                return "Error: Tool execution was cancelled"

        # Either completed, or we hit the max_updates limit – await the final result with timeout.
        # Check task status before awaiting
        if task.done():
            if task.cancelled():
                logger.warning(f"Tool {tool_call.name} task was cancelled")
                return "Error: Tool execution was cancelled"
            # Task is done, get result (may raise exception if task failed)
            try:
                result = task.result()
            except Exception as e:
                logger.error(f"Tool {tool_call.name} raised exception: {e}")
                return f"Error: Tool execution failed: {e}"
        else:
            # Task still running, await it with maximum timeout
            try:
                # Calculate remaining timeout
                elapsed = initial_delay + (updates_sent * interval)
                remaining_timeout = max(max_timeout - elapsed, 30.0)  # At least 30 seconds

                logger.info(f"Waiting for {tool_call.name} to complete (timeout: {remaining_timeout:.0f}s)")
                result = await asyncio.wait_for(asyncio.shield(task), timeout=remaining_timeout)
            except asyncio.TimeoutError:
                # Tool execution timed out
                logger.error(f"Tool {tool_call.name} timed out after {max_timeout}s")

                # Cancel the task
                task.cancel()

                # Send timeout notification to user
                try:
                    timeout_msg = (
                        f"⚠️ 工具 `{tool_call.name}` 执行超时（{max_timeout:.0f}秒）。\n"
                        f"对于大型仓库，索引可能需要更长时间。\n\n"
                        f"⚠️ Tool `{tool_call.name}` timed out after {max_timeout:.0f} seconds.\n"
                        f"For large repositories, indexing may require more time."
                    )
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=timeout_msg,
                    ))
                except Exception as notify_err:
                    logger.warning(f"Failed to send timeout notification: {notify_err}")

                # Return error message instead of raising exception
                return f"Error: Tool execution timed out after {max_timeout:.0f} seconds"
            except asyncio.CancelledError:
                logger.warning(f"Tool {tool_call.name} was cancelled during execution")

                # Send cancellation notification to user
                try:
                    cancel_msg = (
                        f"⚠️ 工具 `{tool_call.name}` 执行被取消。\n\n"
                        f"⚠️ Tool `{tool_call.name}` execution was cancelled."
                    )
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=cancel_msg,
                    ))
                except Exception as notify_err:
                    logger.warning(f"Failed to send cancellation notification: {notify_err}")

                # Return error message instead of raising exception
                return "Error: Tool execution was cancelled"
        
        # For fastcode_load_repo, send immediate completion notification to Feishu
        if tool_call.name == "fastcode_load_repo":
            try:
                # The tool result already contains formatted summary with Files, Code elements, etc.
                # Send it directly to the user
                if result and isinstance(result, str):
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=result,
                    ))
                    logger.info(f"Sent completion notification for {tool_call.name} to {msg.channel}:{msg.chat_id}")
                else:
                    # Fallback if result is not a string
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content="✓ 仓库索引已完成！\n\n✓ Repository indexing completed!",
                    ))
            except Exception as notify_err:
                logger.warning(f"Failed to send completion notification: {notify_err}")
        
        return result
    
    async def _process_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a single inbound message.
        
        Args:
            msg: The inbound message to process.
        
        Returns:
            The response message, or None if no response needed.
        """
        # Handle system messages (subagent announces)
        # The chat_id contains the original "channel:chat_id" to route back to
        if msg.channel == "system":
            return await self._process_system_message(msg)
        
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}: {preview}")
        
        # Get or create session
        session = self.sessions.get_or_create(msg.session_key)
        
        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(msg.channel, msg.chat_id)
        
        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(msg.channel, msg.chat_id)
        
        cron_tool = self.tools.get("cron")
        if isinstance(cron_tool, CronTool):
            cron_tool.set_context(msg.channel, msg.chat_id)
        
        # Build initial messages (use get_history for LLM-formatted messages)
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )
        
        # Agent loop
        iteration = 0
        final_content = None
        
        while iteration < self.max_iterations:
            iteration += 1
            
            # Call LLM
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model
            )
            
            # Handle tool calls
            if response.has_tool_calls:
                # Add assistant message with tool calls
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)  # Must be JSON string
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )
                
                # Execute tools
                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")

                    # For long-running FastCode tools, execute with periodic user feedback.
                    try:
                        if tool_call.name in {"fastcode_load_repo", "fastcode_query"}:
                            result = await self._execute_tool_with_feedback(msg, tool_call)
                        else:
                            result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    except asyncio.CancelledError:
                        logger.warning(f"Tool {tool_call.name} was cancelled unexpectedly")
                        result = f"Error: Tool {tool_call.name} execution was cancelled"
                    except Exception as e:
                        logger.error(f"Tool {tool_call.name} failed with exception: {e}")
                        result = f"Error: Tool {tool_call.name} failed: {e}"

                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                # No tool calls, we're done
                final_content = response.content
                break
        
        if final_content is None:
            final_content = "I've completed processing but have no response to give."
        
        # Log response preview
        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info(f"Response to {msg.channel}:{msg.sender_id}: {preview}")
        
        # Save to session
        session.add_message("user", msg.content)
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},  # Pass through for channel-specific needs (e.g. Slack thread_ts)
        )
    
    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).
        
        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")
        
        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id
        
        # Use the origin session for context
        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)
        
        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(origin_channel, origin_chat_id)
        
        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(origin_channel, origin_chat_id)
        
        cron_tool = self.tools.get("cron")
        if isinstance(cron_tool, CronTool):
            cron_tool.set_context(origin_channel, origin_chat_id)
        
        # Build messages with the announce content
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            channel=origin_channel,
            chat_id=origin_chat_id,
        )
        
        # Agent loop (limited for announce handling)
        iteration = 0
        final_content = None
        
        while iteration < self.max_iterations:
            iteration += 1
            
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model
            )
            
            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )
                
                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                final_content = response.content
                break
        
        if final_content is None:
            final_content = "Background task completed."
        
        # Save to session (mark as system message in history)
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content
        )
    
    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
    ) -> str:
        """
        Process a message directly (for CLI or cron usage).
        
        Args:
            content: The message content.
            session_key: Session identifier.
            channel: Source channel (for context).
            chat_id: Source chat ID (for context).
        
        Returns:
            The agent's response.
        """
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content
        )
        
        response = await self._process_message(msg)
        return response.content if response else ""
