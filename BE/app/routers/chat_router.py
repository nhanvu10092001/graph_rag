"""FastAPI router for WebSocket-based chat streaming with server-side cancellation."""

import asyncio
import json
import logging
from contextlib import suppress
from typing import Any, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.agent.graph import get_compiled_graph
from app.schemas.chat import Message

logger = logging.getLogger("BE.routers.chat_router")

router = APIRouter()

_INTERNAL_KEYS = frozenset(("state", "config", "store", "callbacks", "run_manager", "runtime", "messages"))

_MAX_TOOL_OUTPUT_LEN = 10000


def _default_serializer(o: Any) -> Any:
    if hasattr(o, "content"):
        return getattr(o, "content")
    if hasattr(o, "dict") and callable(getattr(o, "dict")):
        return o.dict()
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, default=_default_serializer)


def sanitize_tool_input(tool_input: Any) -> Any:
    if isinstance(tool_input, dict):
        return {k: sanitize_tool_input(v) if isinstance(v, dict) else v for k, v in tool_input.items() if k not in _INTERNAL_KEYS}
    return tool_input


async def _stream_to_ws(websocket: WebSocket, messages: List[Message]) -> None:
    from langchain_core.messages import HumanMessage, AIMessage

    formatted_messages = []
    for msg in messages:
        if msg.role == "user":
            formatted_messages.append(HumanMessage(content=msg.content))
        else:
            formatted_messages.append(AIMessage(content=msg.content))

    logger.info(f"Invoking compiled agent graph with {len(formatted_messages)} messages")

    try:
        async for event in get_compiled_graph().astream_events(
            {"messages": formatted_messages},
            version="v2"
        ):
            kind = event.get("event")

            if kind == "on_chat_model_stream":
                node = event.get("metadata", {}).get("langgraph_node")
                if node == "agent":
                    chunk = event["data"].get("chunk")
                    if chunk:
                        tool_chunks = getattr(chunk, "tool_call_chunks", None)
                        if tool_chunks:
                            for tc in tool_chunks:
                                tc_index = tc.get("index", 0) if isinstance(tc, dict) else getattr(tc, "index", 0)
                                tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                                tc_args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", "")
                                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)

                                payload: dict[str, Any] = {
                                    "type": "tool_call_delta",
                                    "index": tc_index,
                                }
                                if tc_name:
                                    payload["name"] = tc_name
                                if tc_id:
                                    payload["id"] = tc_id
                                if tc_args:
                                    payload["args"] = tc_args

                                await websocket.send_json(payload)

                        raw_content = getattr(chunk, "content", None)
                        extracted_text = ""
                        extracted_thinking = ""
                        if isinstance(raw_content, str):
                            extracted_text = raw_content
                        elif isinstance(raw_content, list):
                            text_parts = []
                            thinking_parts = []
                            for part in raw_content:
                                if isinstance(part, str):
                                    text_parts.append(part)
                                elif isinstance(part, dict):
                                    if part.get("type") == "text" and part.get("text"):
                                        text_parts.append(part["text"])
                                    elif part.get("type") == "thinking" and part.get("thinking"):
                                        thinking_parts.append(part["thinking"])
                                    elif "text" in part and isinstance(part["text"], str):
                                        text_parts.append(part["text"])
                                elif hasattr(part, "text") and getattr(part, "text"):
                                    text_parts.append(str(getattr(part, "text")))
                            extracted_text = "".join(text_parts)
                            extracted_thinking = "".join(thinking_parts)

                        additional_kwargs = getattr(chunk, "additional_kwargs", {})
                        reasoning_content = additional_kwargs.get("reasoning_content")
                        if reasoning_content and isinstance(reasoning_content, str):
                            extracted_thinking += reasoning_content

                        if extracted_thinking:
                            await websocket.send_json({"type": "thinking_delta", "content": extracted_thinking})

                        if extracted_text:
                            await websocket.send_json({"type": "text_delta", "text": extracted_text, "content": extracted_text})

            elif kind == "on_tool_start":
                tool_name = event.get("name")
                tool_run_id = event.get("run_id")
                tool_input = event.get("data", {}).get("input")

                payload = {
                    "type": "tool_call_executing",
                    "name": tool_name,
                    "id": tool_run_id,
                    "input": sanitize_tool_input(tool_input),
                }
                await websocket.send_json(payload)

            elif kind == "on_tool_end":
                tool_name = event.get("name")
                tool_run_id = event.get("run_id")
                raw_output = event.get("data", {}).get("output")

                output_str = ""
                if raw_output is not None:
                    content = getattr(raw_output, "content", None)
                    if content is not None:
                        output_str = str(content)
                    elif isinstance(raw_output, str):
                        output_str = raw_output
                    else:
                        output_str = safe_json_dumps(raw_output)

                if len(output_str) > _MAX_TOOL_OUTPUT_LEN:
                    output_str = output_str[:_MAX_TOOL_OUTPUT_LEN] + "... (truncated)"

                payload = {
                    "type": "tool_call_end",
                    "name": tool_name,
                    "id": tool_run_id,
                    "status": "completed",
                    "output": output_str,
                }
                await websocket.send_json(payload)

            elif kind == "on_chain_end" and event.get("name") == "respond_unsafe":
                output = event.get("data", {}).get("output", {})
                msgs = output.get("messages", [])
                for msg in msgs:
                    content = getattr(msg, "content", None) or (msg if isinstance(msg, str) else "")
                    if content:
                        await websocket.send_json({"type": "text_delta", "text": content, "content": content})

        await websocket.send_json({"type": "done"})
        logger.info("Agent response stream completed successfully.")

    except asyncio.CancelledError:
        try:
            await websocket.send_json({"type": "cancelled"})
        except Exception:
            pass
        raise

    except Exception as e:
        logger.error(f"Error during agent execution stream: {e}")
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass


@router.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    await websocket.accept()
    streaming_task: asyncio.Task | None = None

    try:
        while True:
            raw = await websocket.receive_json()
            msg_type = raw.get("type")

            if msg_type == "chat":
                if streaming_task and not streaming_task.done():
                    streaming_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await streaming_task

                raw_messages = raw.get("messages", [])
                messages = [Message(**m) if isinstance(m, dict) else m for m in raw_messages]
                streaming_task = asyncio.create_task(_stream_to_ws(websocket, messages))

            elif msg_type == "cancel":
                if streaming_task and not streaming_task.done():
                    streaming_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await streaming_task
                    await websocket.send_json({"type": "cancelled"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
        if streaming_task and not streaming_task.done():
            streaming_task.cancel()
            with suppress(asyncio.CancelledError):
                await streaming_task
