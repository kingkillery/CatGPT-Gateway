"""The shape tier of the cleanup spec (src/api/cleanup.py), as tables.

JSON that parses can still be the wrong shape. Each row is one variant a model can emit
and the one canonical call it must become. The properties that matter most are the
negative ones: nothing quoted in prose is run, no tool or argument is invented, and
ambiguity means no call at all.
"""

from __future__ import annotations

import asyncio
import copy
import json
import unittest
from types import SimpleNamespace

from src.api import cleanup, openai_routes
from src.api.openai_schemas import ChatCompletionRequest, ChatMessage, ToolDefinition

TOOLS = [
    ToolDefinition(type="function", function={"name": "read_file", "parameters": {"type": "object"}}),
    ToolDefinition(type="function", function={"name": "list_files", "parameters": {"type": "object"}}),
]
NAMES = [t.function.name for t in TOOLS]
CALL = '{"name":"read_file","arguments":{"path":"a.py"}}'
WANT = [("read_file", {"path": "a.py"})]

# (variant, reply text) -> exactly WANT
CALL_CASES = [
    ("canonical", '{"tool_calls":[%s]}' % CALL),
    ("fence and scraped label", 'Thought for 3s\n\n```json\n{"tool_calls":[%s]}\n```' % CALL),
    ("prose after the canonical envelope", '{"tool_calls":[%s]}\n\nLet me know if you need more.' % CALL),
    ("bare call object", CALL),
    ("singular tool_call wrapper", '{"tool_call":%s}' % CALL),
    ("function_call wrapper", '{"function_call":%s}' % CALL),
    ("top-level list", "[%s]" % CALL),
    ("tool_calls holds one object, not a list", '{"tool_calls":%s}' % CALL),
    ("parameters for arguments", '{"tool_calls":[{"name":"read_file","parameters":{"path":"a.py"}}]}'),
    ("args for arguments", '{"tool_calls":[{"name":"read_file","args":{"path":"a.py"}}]}'),
    ("input for arguments", '{"tool_calls":[{"name":"read_file","input":{"path":"a.py"}}]}'),
    ("tool for name", '{"tool_calls":[{"tool":"read_file","arguments":{"path":"a.py"}}]}'),
    ("function is the name string", '{"tool_calls":[{"function":"read_file","arguments":{"path":"a.py"}}]}'),
    ("namespace prefix on the name", '{"tool_calls":[{"name":"functions.read_file","arguments":{"path":"a.py"}}]}'),
    ("name differs only in case", '{"tool_calls":[{"name":"Read_File","arguments":{"path":"a.py"}}]}'),
    ("name uses a hyphen", '{"tool_calls":[{"name":"read-file","arguments":{"path":"a.py"}}]}'),
    ("tool_calls is a JSON string", json.dumps({"tool_calls": "[%s]" % CALL})),
    ("arguments is a JSON string", '{"tool_calls":[{"name":"read_file","arguments":"{\\"path\\": \\"a.py\\"}"}]}'),
    ("arguments string missing its closer", '{"tool_calls":[{"name":"read_file","arguments":"{\\"path\\": \\"a.py\\""}]}'),
    ("<tool_call> tags", "<tool_call>%s</tool_call>" % CALL),
    ("OpenAI style, arguments as a string", '{"tool_calls":[{"id":"x","type":"function","function":{"name":"read_file","arguments":"{\\"path\\":\\"a.py\\"}"}}]}'),
    ("trailing comma in a bare call", '{"name":"read_file","arguments":{"path":"a.py",},}'),
    ("bare call in a fence", "```json\n%s\n```" % CALL),
]

# (variant, reply text, expected answer text)
ANSWER_CASES = [
    ("capitalised key", '{"Answer":"It is 42."}', "It is 42."),
    ("response key", '{"response":"It is 42."}', "It is 42."),
    ("content key", '{"content":"It is 42."}', "It is 42."),
    ("answer is an object with text", '{"answer":{"text":"It is 42."}}', "It is 42."),
    ("answer is a list of lines", '{"answer":["Line one.","Line two."]}', "Line one.\nLine two."),
    ("canonical answer inside prose still counts", 'Sure. {"answer":"It is 42."}', "It is 42."),
]

# Replies that must NOT become a call.
NOT_A_CALL = [
    ("a call quoted in prose", 'You could run %s to see it.' % CALL),
    ("a list quoted in prose", 'For example: [%s]' % CALL),
    ("a tool that was not offered", '{"tool_calls":[{"name":"rm_rf","arguments":{}}]}'),
    ("a bare call to a tool that was not offered", '{"name":"rm_rf","arguments":{}}'),
    ("arguments that are a list", '{"tool_calls":[{"name":"read_file","arguments":["a.py"]}]}'),
    ("arguments with an unterminated string", '{"tool_calls":[{"name":"read_file","arguments":"{\\"path\\": \\"a.py"}]}'),
    ("an ordinary JSON answer with a name field", '{"name":"Bob","age":3}'),
    ("an empty call list", '{"tool_calls":[]}'),
    ("prose", "The file is empty."),
]

# Replies that must NOT be unwrapped as an answer.
NOT_AN_ANSWER = [
    ("extra keys", '{"text":"hi","lang":"en"}'),
    ("looks like a call", '{"name":"read_file","content":"x"}'),
    ("a wrapper key beside an alias answer", '{"tool_calls":[],"response":"x"}'),
    ("an example inside prose", 'Send this payload: {"text":"hi"} and wait.'),
    ("a number", '{"answer":42}'),
    ("prose", "plain text"),
]


def parsed(reply: str):
    calls = openai_routes._parse_tool_calls(reply, TOOLS)
    return None if not calls else [(c.function.name, json.loads(c.function.arguments)) for c in calls]


class CallShapeTableTest(unittest.TestCase):
    def test_every_variant_becomes_the_one_canonical_call(self) -> None:
        for name, reply in CALL_CASES:
            with self.subTest(name):
                self.assertEqual(parsed(reply), WANT)

    def test_nothing_that_should_not_run_becomes_a_call(self) -> None:
        for name, reply in NOT_A_CALL:
            with self.subTest(name):
                self.assertIsNone(parsed(reply))

    def test_every_answer_variant_is_unwrapped(self) -> None:
        for name, reply, want in ANSWER_CASES:
            with self.subTest(name):
                self.assertEqual(openai_routes._unwrap_answer(reply), want)

    def test_replies_that_are_not_plain_answers_are_left_alone(self) -> None:
        for name, reply in NOT_AN_ANSWER:
            with self.subTest(name):
                self.assertIsNone(openai_routes._unwrap_answer(reply))


class NoGuessingTest(unittest.TestCase):
    def test_a_partial_list_keeps_only_the_offered_tools(self) -> None:
        reply = '{"tool_calls":[{"name":"rm_rf","arguments":{}},%s]}' % CALL
        self.assertEqual(parsed(reply), WANT)

    def test_ambiguous_names_resolve_to_nothing(self) -> None:
        both = ["read_file", "readFile"]
        self.assertEqual(cleanup.resolve_tool_name("readFile", both), "readFile")  # exact wins
        self.assertIsNone(cleanup.resolve_tool_name("READ_FILE", both))  # two candidates: no guess

    def test_null_arguments_mean_no_arguments_not_a_failure(self) -> None:
        self.assertEqual(parsed('{"tool_calls":[{"name":"list_files","arguments":null}]}'), [("list_files", {})])

    def test_an_alias_never_loses_the_arguments_it_carries(self) -> None:
        # The old parser accepted this call and dropped {"path": ...}, running read_file with no path.
        got = parsed('{"tool_calls":[{"name":"read_file","parameters":{"path":"a.py"}}]}')
        self.assertEqual(got, [("read_file", {"path": "a.py"})])


class PurityTest(unittest.TestCase):
    def test_normalize_calls_is_pure_and_deterministic(self) -> None:
        value = {"tool_calls": [{"name": "Functions.Read_File", "parameters": {"path": "a.py", "n": [1, 2]}}]}
        before = copy.deepcopy(value)
        first = cleanup.normalize_calls(value, NAMES)
        self.assertEqual(value, before)  # input untouched
        self.assertEqual(first, cleanup.normalize_calls(value, NAMES))
        self.assertEqual(first, [{"name": "read_file", "arguments": {"path": "a.py", "n": [1, 2]}}])

    def test_the_canonical_form_is_a_fixed_point(self) -> None:
        canonical = {"tool_calls": [{"name": "read_file", "arguments": {"path": "a.py"}}]}
        once = cleanup.normalize_calls(canonical, NAMES)
        self.assertEqual(cleanup.normalize_calls({"tool_calls": once}, NAMES), once)

    def test_reply_json_demands_the_entire_reply(self) -> None:
        self.assertEqual(cleanup.reply_json('Thought for 2s\n```json\n{"a": 1}\n```'), {"a": 1})
        self.assertIsNone(cleanup.reply_json('prose {"a": 1}'))
        self.assertIsNone(cleanup.reply_json('{"a": 1} then prose'))

    def test_the_spec_documents_every_step(self) -> None:
        names = [name for name, _ in cleanup.SHAPE_SPEC]
        self.assertEqual(len(names), len(set(names)))
        for name, why in cleanup.SHAPE_SPEC:
            self.assertTrue(name and len(why) > 10, name)


class FakeClient:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.page = SimpleNamespace(url="https://chatgpt.com/?temporary-chat=true")

    async def new_chat(self) -> None:
        self.page.url = "https://chatgpt.com/?temporary-chat=true"

    async def send_message(self, text, image_paths=None, file_paths=None):
        return SimpleNamespace(message=self.reply)


class EndToEndTest(unittest.TestCase):
    def setUp(self) -> None:
        saved = (openai_routes._client, openai_routes._pool,
                 openai_routes._last_response_time, openai_routes._lock)
        openai_routes._pool, openai_routes._last_response_time, openai_routes._lock = None, 0.0, None
        self.addCleanup(lambda: (setattr(openai_routes, "_client", saved[0]),
                                 setattr(openai_routes, "_pool", saved[1]),
                                 setattr(openai_routes, "_last_response_time", saved[2]),
                                 setattr(openai_routes, "_lock", saved[3])))

    def complete(self, reply: str):
        openai_routes.set_openai_client(FakeClient(reply))
        request = ChatCompletionRequest(model="m", tools=TOOLS, messages=[ChatMessage(role="user", content="Open a.py")])
        return asyncio.run(openai_routes._run_completion(request)).choices[0]

    def test_a_variant_reply_becomes_a_tool_call_with_its_arguments(self) -> None:
        choice = self.complete('Thought for 4s\n\n{"tool":"functions.read_file","parameters":{"path":"a.py"}}')
        self.assertEqual(choice.finish_reason, "tool_calls")
        self.assertEqual(json.loads(choice.message.tool_calls[0].function.arguments), {"path": "a.py"})

    def test_a_variant_answer_reaches_the_client_as_plain_text(self) -> None:
        choice = self.complete('{"response":"The file is empty."}')
        self.assertEqual((choice.finish_reason, choice.message.content), ("stop", "The file is empty."))

    def test_a_call_quoted_in_prose_is_returned_as_text_not_run(self) -> None:
        reply = "You could call %s to see it." % CALL
        choice = self.complete(reply)
        self.assertEqual(choice.finish_reason, "stop")
        self.assertIsNone(choice.message.tool_calls)
        self.assertEqual(choice.message.content, reply)


if __name__ == "__main__":
    unittest.main()
