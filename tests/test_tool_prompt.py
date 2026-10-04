"""Regression guard: tool-enabled prompts keep agent loops working in auto mode.

Covers the prompt actually sent to ChatGPT: the caller's system prompt survives
next to the injected tool block, tool results name their tool and do not forbid
follow-up calls, and the final protocol reminder is sent in auto mode on every
turn (forced modes before a tool result, then an answer-now instruction).
"""

from __future__ import annotations

import unittest

from src.api.openai_routes import (
    _append_tool_protocol_suffix,
    _build_prompt,
    _build_tool_system_prompt,
    _parse_tool_calls,
    _responses_input_to_messages,
    _unwrap_answer,
)
from src.api.openai_schemas import ChatMessage, ToolDefinition

TOOLS = [ToolDefinition(type="function", function={"name": "read_file", "parameters": {"type": "object"}})]
REMINDER = "FINAL RESPONSE FORMAT (call a function or answer)"


def with_tools(messages):
    return [ChatMessage(role="system", content=_build_tool_system_prompt(TOOLS))] + messages


class ToolPromptTest(unittest.TestCase):
    def test_first_turn_keeps_caller_system_prompt(self) -> None:
        messages = with_tools([
            ChatMessage(role="system", content="You are the OMP coding agent."),
            ChatMessage(role="user", content="Open README.md"),
        ])
        prompt = _append_tool_protocol_suffix(_build_prompt(messages), messages, "auto")
        self.assertIn("read_file", prompt)
        self.assertIn("You are the OMP coding agent.", prompt)
        # Auto mode ends with the two-shape schema, not "tool-calling mode" language.
        self.assertIn(REMINDER, prompt)
        self.assertIn('{"answer":', prompt)
        self.assertIn("Do not use built-in web search", prompt)
        self.assertIn("MUST use shape 1", prompt)  # file/path requests default to calling a function

    def test_agent_loop_turn_allows_follow_up_calls(self) -> None:
        messages = with_tools([
            ChatMessage(role="user", content="Summarize README.md"),
            ChatMessage(role="assistant", content="Reading it first.", tool_calls=[
                {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "README.md"}'}}]),
            ChatMessage(role="tool", tool_call_id="call_1", content="# Title"),
        ])
        prompt = _build_prompt(messages)
        self.assertIn("Assistant: Reading it first.", prompt)
        self.assertIn('read_file({"path": "README.md"}) [call_1]', prompt)
        self.assertIn("[Tool result for read_file (call_1)]: # Title", prompt)
        self.assertNotIn("Do NOT call tools again", prompt)
        self.assertIn(REMINDER, _append_tool_protocol_suffix(prompt, messages, "auto"))
        self.assertIn(REMINDER, _append_tool_protocol_suffix(prompt, messages, None))
        forced = _append_tool_protocol_suffix(prompt, messages, "required")
        self.assertNotIn(REMINDER, forced)
        self.assertTrue(forced.endswith("Do NOT call tools again."))

    def test_responses_history_keeps_assistant_turns(self) -> None:
        messages = _responses_input_to_messages([
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hello there."}]},
            {"type": "function_call", "call_id": "call_9", "name": "read_file", "arguments": '{"path": "a"}'},
            {"type": "function_call_output", "call_id": "call_9", "output": "file body"},
        ])
        prompt = _build_prompt(messages)
        self.assertIn("Assistant: Hello there.", prompt)
        self.assertIn("[Tool result for read_file (call_9)]: file body", prompt)

    def test_answer_shape_is_unwrapped_and_other_text_is_left_alone(self) -> None:
        self.assertEqual(_unwrap_answer('{"answer": "It is 42."}'), "It is 42.")
        self.assertEqual(_unwrap_answer('```json\n{"answer": "Line one\\nLine \\"two\\""}\n```'), 'Line one\nLine "two"')
        self.assertIsNone(_unwrap_answer("Just prose that mentions the answer."))
        self.assertIsNone(_unwrap_answer('{"answer": 42}'))
        self.assertIsNone(_unwrap_answer(None))

    def test_reply_parses_into_tool_call(self) -> None:
        reply = 'Thought for 3s\n\n```json\n{"tool_calls": [{"name": "read_file", "arguments": {"path": "a.py"}}]}\n```'
        calls = _parse_tool_calls(reply, TOOLS)
        self.assertEqual([(c.function.name, c.function.arguments) for c in calls], [("read_file", '{"path": "a.py"}')])


if __name__ == "__main__":
    unittest.main()
