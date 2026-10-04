"""The OpenRouter bridge: a mid-loop turn is answered by another model, not ChatGPT.

No network: bridge._post is replaced. Covers routing (and what must NOT route), the
payload sent, mapping the answer back, failure handling, and the full request path,
including that ChatGPT is never touched for a bridged turn.
"""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from src.api import bridge, openai_routes
from src.api.openai_schemas import ChatCompletionRequest, ChatMessage, ToolDefinition

TOOLS = [ToolDefinition(type="function", function={"name": "read_file", "parameters": {"type": "object"}})]
CALL = {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "NOTES.md"}'}}


def env(**kw):
    base = {"OPENROUTER_API_KEY": "sk-test", "CATGPT_BRIDGE_MODEL": "openrouter/free", "CATGPT_BRIDGE_ROUTE": "continuation"}
    return patch.dict(os.environ, {**base, **kw})


def mid_loop(tool_choice=None) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="gpt-5.5-pro", tools=TOOLS, tool_choice=tool_choice,
        messages=[
            ChatMessage(role="user", content="Read NOTES.md"),
            ChatMessage(role="assistant", tool_calls=[CALL]),
            ChatMessage(role="tool", tool_call_id="call_1", content="Purple otters ship on Fridays."),
        ],
    )


def opening_turn() -> ChatCompletionRequest:
    return ChatCompletionRequest(model="gpt-5.5-pro", tools=TOOLS, messages=[ChatMessage(role="user", content="Read NOTES.md")])


def reply(message: dict, usage=None) -> dict:
    return {"choices": [{"message": message, "finish_reason": "stop"}], "usage": usage or {}}


class RoutingTest(unittest.TestCase):
    def test_off_without_a_model_or_a_key(self) -> None:
        for kw in ({"CATGPT_BRIDGE_MODEL": ""}, {"OPENROUTER_API_KEY": ""}):
            with env(**kw):
                self.assertFalse(bridge.enabled())
                self.assertFalse(bridge.routes_to_bridge(mid_loop()))

    def test_only_a_turn_after_a_tool_result_routes(self) -> None:
        with env():
            self.assertTrue(bridge.routes_to_bridge(mid_loop()))
            self.assertFalse(bridge.routes_to_bridge(opening_turn()), "the opening turn stays on ChatGPT")
            self.assertFalse(bridge.routes_to_bridge(mid_loop(tool_choice="none")))

    def test_fallback_mode_never_routes_up_front(self) -> None:
        with env(CATGPT_BRIDGE_ROUTE="fallback"):
            self.assertFalse(bridge.routes_to_bridge(mid_loop()))
            self.assertTrue(bridge.enabled())  # still available as the fallback

    def test_unknown_route_value_means_continuation(self) -> None:
        with env(CATGPT_BRIDGE_ROUTE="nonsense"):
            self.assertEqual(bridge.route(), "continuation")

    def test_model_setting_accepts_a_list_and_ignores_blanks(self) -> None:
        with env(CATGPT_BRIDGE_MODEL=" stealth/space-bunny-alpha , ,deepseek/deepseek-v4.1-flash,"):
            self.assertEqual(bridge.models(), ["stealth/space-bunny-alpha", "deepseek/deepseek-v4.1-flash"])
            self.assertEqual(bridge.model(), "stealth/space-bunny-alpha")
        with env(CATGPT_BRIDGE_MODEL=" , ,"):
            self.assertFalse(bridge.enabled())


class DenialWordingTest(unittest.TestCase):
    """ChatGPT sometimes looks in its own sandbox instead of calling the function."""

    def test_the_replies_actually_seen_are_recognised(self) -> None:
        for text in (
            "I couldn't read /repo/NOTES.md because that path does not exist in the available filesystem.",
            "The tool result says /repo does not exist on the accessible filesystem.",
            "I couldn’t list /repo because that directory does not exist.",  # curly apostrophe
            "I can't access your files from here.",
        ):
            with self.subTest(text=text[:40]):
                self.assertTrue(openai_routes._looks_like_tool_refusal(text))

    def test_ordinary_answers_are_not(self) -> None:
        for text in ("42", "Paris is sunny.", "The file lists three modules.", ""):
            self.assertFalse(openai_routes._looks_like_tool_refusal(text))


class CompleteTest(unittest.TestCase):
    def run_complete(self, request, answer, **kw):
        sent = []
        self.targets_used = []

        def fake(target, body):
            sent.append(body)
            self.targets_used.append(target.spec)
            if isinstance(answer, Exception):
                raise answer
            return answer

        with env(**kw), patch.object(bridge, "_post", fake):
            return asyncio.run(bridge.complete(request)), sent

    def test_payload_carries_the_conversation_tools_and_requires_tool_support(self) -> None:
        _, sent = self.run_complete(mid_loop(), reply({"content": "done"}))
        body = sent[0]
        self.assertEqual(body["model"], "openrouter/free")
        self.assertEqual(body["tool_choice"], "auto")
        self.assertEqual(body["provider"], {"require_parameters": True})
        self.assertEqual(body["tools"][0]["function"]["name"], "read_file")
        self.assertEqual([m["role"] for m in body["messages"]], ["user", "assistant", "tool"])
        self.assertIsNone(body["messages"][1]["content"])  # assistant turn that only called tools
        self.assertEqual(body["messages"][2]["tool_call_id"], "call_1")
        self.assertFalse(body["stream"])

    def test_models_are_tried_in_order_and_the_first_that_works_wins(self) -> None:
        seen = []

        def fake(target, body):
            seen.append(target.spec)
            if target.spec == "stealth/space-bunny-alpha":
                raise OSError("HTTP Error 402: Payment Required")
            return reply({"content": "From the second model."})

        with env(CATGPT_BRIDGE_MODEL="stealth/space-bunny-alpha,deepseek/deepseek-v4.1-flash"), \
             patch.object(bridge, "_post", fake):
            response = asyncio.run(bridge.complete(mid_loop()))
        self.assertEqual(seen, ["stealth/space-bunny-alpha", "deepseek/deepseek-v4.1-flash"])
        self.assertEqual(response.choices[0].message.content, "From the second model.")

    def test_a_working_first_model_means_the_rest_are_never_called(self) -> None:
        self.run_complete(mid_loop(), reply({"content": "x"}), CATGPT_BRIDGE_MODEL="a/first,b/second")
        self.assertEqual(self.targets_used, ["a/first"])

    def test_all_models_failing_returns_none_so_chatgpt_is_used(self) -> None:
        response, _ = self.run_complete(mid_loop(), OSError("down"), CATGPT_BRIDGE_MODEL="a/first,b/second")
        self.assertIsNone(response)
        self.assertEqual(self.targets_used, ["a/first", "b/second"])

    def test_huggingface_prefix_uses_the_hf_router_token_and_bare_model_id(self) -> None:
        _, sent = self.run_complete(mid_loop(), reply({"content": "x"}), HF_TOKEN="hf-test",
                                    CATGPT_BRIDGE_MODEL="huggingface/deepseek-ai/DeepSeek-V4.1-Flash")
        self.assertEqual(sent[0]["model"], "deepseek-ai/DeepSeek-V4.1-Flash")
        self.assertNotIn("provider", sent[0], "the OpenRouter-only routing field is not sent to Hugging Face")
        self.assertEqual(sent[0]["tools"][0]["function"]["name"], "read_file")
        target = bridge.targets()[0] if False else None  # (resolved below, inside env)
        with env(HF_TOKEN="hf-test", CATGPT_BRIDGE_MODEL="huggingface/deepseek-ai/DeepSeek-V4.1-Flash"):
            target = bridge.targets()[0]
        self.assertEqual((target.url, target.key_env), ("https://router.huggingface.co/v1/chat/completions", "HF_TOKEN"))

    def test_an_entry_without_its_backend_key_is_skipped(self) -> None:
        with env(HF_TOKEN="", CATGPT_BRIDGE_MODEL="huggingface/deepseek-ai/DeepSeek-V4.1-Flash,openrouter/free"):
            self.assertEqual([t.spec for t in bridge.targets()], ["openrouter/free"])
        with env(OPENROUTER_API_KEY="", HF_TOKEN="hf-test", CATGPT_BRIDGE_MODEL="openrouter/free,huggingface/x/y"):
            self.assertEqual([t.spec for t in bridge.targets()], ["huggingface/x/y"])
        with env(OPENROUTER_API_KEY="", HF_TOKEN="", CATGPT_BRIDGE_MODEL="openrouter/free,huggingface/x/y"):
            self.assertFalse(bridge.enabled())

    def test_openrouter_then_huggingface_fallback_across_backends(self) -> None:
        seen = []

        def fake(target, body):
            seen.append((target.key_env, body["model"]))
            if target.openrouter:
                raise OSError("HTTP Error 402: Payment Required")
            return reply({"content": "From Hugging Face."})

        with env(HF_TOKEN="hf-test", CATGPT_BRIDGE_MODEL="deepseek/deepseek-v4.1-flash,huggingface/deepseek-ai/DeepSeek-V4.1-Flash"), \
             patch.object(bridge, "_post", fake):
            response = asyncio.run(bridge.complete(mid_loop()))
        self.assertEqual(seen, [("OPENROUTER_API_KEY", "deepseek/deepseek-v4.1-flash"), ("HF_TOKEN", "deepseek-ai/DeepSeek-V4.1-Flash")])
        self.assertEqual(response.choices[0].message.content, "From Hugging Face.")

    def test_the_log_names_the_model_that_actually_answered(self) -> None:
        answer = {**reply({"content": "x"}), "model": "deepseek/deepseek-v4.1-flash"}
        with env(CATGPT_BRIDGE_MODEL="stealth/space-bunny-alpha,deepseek/deepseek-v4.1-flash"), \
             patch.object(bridge, "_post", return_value=answer), self.assertLogs("bridge", "INFO") as logs:
            asyncio.run(bridge.complete(mid_loop()))
        self.assertTrue(any("deepseek/deepseek-v4.1-flash" in line for line in logs.output))

    def test_a_specific_model_is_used_as_given(self) -> None:
        _, sent = self.run_complete(mid_loop(), reply({"content": "x"}), CATGPT_BRIDGE_MODEL="anthropic/claude-sonnet-4.5")
        self.assertEqual(sent[0]["model"], "anthropic/claude-sonnet-4.5")

    def test_text_answer_maps_to_a_stop_response_under_the_clients_model_name(self) -> None:
        response, _ = self.run_complete(mid_loop(), reply({"content": " It is Purple otters. "}, {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}))
        choice = response.choices[0]
        self.assertEqual((choice.finish_reason, choice.message.content, choice.message.tool_calls), ("stop", "It is Purple otters.", None))
        self.assertEqual((response.model, response.usage.total_tokens), ("gpt-5.5-pro", 10))

    def test_tool_call_maps_and_repairs_sloppy_arguments(self) -> None:
        sloppy = {"tool_calls": [{"id": "c9", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py",}'}}]}
        response, _ = self.run_complete(mid_loop(), reply(sloppy))
        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, "tool_calls")
        self.assertIsNone(choice.message.content)
        self.assertEqual(json.loads(choice.message.tool_calls[0].function.arguments), {"path": "a.py"})

    def test_a_tool_the_request_did_not_offer_is_dropped(self) -> None:
        evil = {"content": "", "tool_calls": [{"id": "x", "type": "function", "function": {"name": "rm_rf", "arguments": "{}"}}]}
        response, _ = self.run_complete(mid_loop(), reply(evil))
        self.assertIsNone(response, "nothing usable left, so the caller falls back to ChatGPT")

    def test_failure_or_empty_answer_returns_none(self) -> None:
        for answer in (OSError("429"), reply({"content": ""}), {"unexpected": True}):
            with self.subTest(answer=str(answer)[:30]):
                response, _ = self.run_complete(mid_loop(), answer)
                self.assertIsNone(response)

    def test_disabled_never_calls_out(self) -> None:
        with env(OPENROUTER_API_KEY=""), patch.object(bridge, "_post", side_effect=AssertionError("network used")):
            self.assertIsNone(asyncio.run(bridge.complete(mid_loop())))


class FakeClient:
    """Stands in for the ChatGPT browser: `used` shows whether it was touched and `sent`
    records every message. `replies` (if given) are returned one per message."""

    def __init__(self, reply_text: str = "", replies: list[str] | None = None) -> None:
        self.reply_text = reply_text
        self.replies = list(replies) if replies else None
        self.used = False
        self.sent: list[str] = []
        self.page = SimpleNamespace(url="https://chatgpt.com/?temporary-chat=true")

    async def new_chat(self) -> None:
        self.used = True

    async def send_message(self, text, image_paths=None, file_paths=None):
        self.used = True
        self.sent.append(text)
        return SimpleNamespace(message=self.replies.pop(0) if self.replies else self.reply_text)


class RequestPathTest(unittest.TestCase):
    def setUp(self) -> None:
        saved = (openai_routes._client, openai_routes._pool,
                 openai_routes._last_response_time, openai_routes._lock)
        openai_routes._pool, openai_routes._last_response_time, openai_routes._lock = None, 0.0, None
        self.addCleanup(lambda: (setattr(openai_routes, "_client", saved[0]),
                                 setattr(openai_routes, "_pool", saved[1]),
                                 setattr(openai_routes, "_last_response_time", saved[2]),
                                 setattr(openai_routes, "_lock", saved[3])))

    def complete(self, request, client):
        openai_routes.set_openai_client(client)
        return asyncio.run(openai_routes._run_completion(request))

    def test_a_mid_loop_turn_never_touches_chatgpt(self) -> None:
        client = FakeClient("ChatGPT must not be asked")
        with env(), patch.object(bridge, "_post", return_value=reply({"content": "Purple otters ship on Fridays."})):
            response = self.complete(mid_loop(), client)
        self.assertEqual(response.choices[0].message.content, "Purple otters ship on Fridays.")
        self.assertFalse(client.used)

    def test_the_opening_turn_still_goes_to_chatgpt(self) -> None:
        client = FakeClient('{"tool_calls":[{"name":"read_file","arguments":{"path":"NOTES.md"}}]}')
        with env(), patch.object(bridge, "_post", side_effect=AssertionError("opening turn was bridged")):
            response = self.complete(opening_turn(), client)
        self.assertTrue(client.used)
        self.assertEqual(response.choices[0].finish_reason, "tool_calls")

    def test_bridge_failure_falls_back_to_chatgpt(self) -> None:
        client = FakeClient("From ChatGPT.")
        with env(), patch.object(bridge, "_post", side_effect=OSError("down")):
            response = self.complete(mid_loop(), client)
        self.assertTrue(client.used)
        self.assertEqual(response.choices[0].message.content, "From ChatGPT.")

    def test_an_empty_chatgpt_reply_is_answered_by_the_bridge_instead_of_a_502(self) -> None:
        client = FakeClient("")  # what ChatGPT's rate limiting looks like
        with env(), patch.object(bridge, "_post", return_value=reply({"content": "Bridge answer."})):
            response = self.complete(opening_turn(), client)
        self.assertTrue(client.used)
        self.assertEqual(response.choices[0].message.content, "Bridge answer.")

    DENIAL = '{"answer":"I couldn\'t read /repo/NOTES.md because that path does not exist in the available filesystem."}'

    def test_a_sandbox_denial_on_the_opening_turn_is_taken_over_by_the_bridge(self) -> None:
        client = FakeClient(self.DENIAL)
        call = {"tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "/repo/NOTES.md"}'}}]}
        with env(), patch.object(bridge, "_post", return_value=reply(call)):
            response = self.complete(opening_turn(), client)
        self.assertEqual(response.choices[0].finish_reason, "tool_calls")
        self.assertEqual(len(client.sent), 1, "no correction message was spent on ChatGPT")

    def test_without_a_bridge_a_denial_gets_the_usual_correction_turn(self) -> None:
        good = '{"tool_calls":[{"name":"read_file","arguments":{"path":"/repo/NOTES.md"}}]}'
        client = FakeClient(replies=[self.DENIAL, good])
        with env(CATGPT_BRIDGE_MODEL=""):
            response = self.complete(opening_turn(), client)
        self.assertEqual(len(client.sent), 2)
        self.assertEqual(response.choices[0].finish_reason, "tool_calls")

    def test_if_the_bridge_is_down_a_denial_still_falls_back_to_the_correction_turn(self) -> None:
        good = '{"tool_calls":[{"name":"read_file","arguments":{"path":"/repo/NOTES.md"}}]}'
        client = FakeClient(replies=[self.DENIAL, good])
        with env(), patch.object(bridge, "_post", side_effect=OSError("down")):
            response = self.complete(opening_turn(), client)
        self.assertEqual(len(client.sent), 2)
        self.assertEqual(response.choices[0].finish_reason, "tool_calls")

    def test_without_a_bridge_an_empty_reply_is_still_a_502(self) -> None:
        with env(CATGPT_BRIDGE_MODEL=""), self.assertRaises(HTTPException) as caught:
            self.complete(opening_turn(), FakeClient(""))
        self.assertEqual(caught.exception.status_code, 502)


if __name__ == "__main__":
    unittest.main()
