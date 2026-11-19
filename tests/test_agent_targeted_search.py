import contextlib
import io
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from classes import agent, rag


class AgentTargetedSearchTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            {
                "source": "returns.md",
                "text": "return window",
                "tokens": set(rag.tokenize("return window")),
            },
            {
                "source": "delivery.md",
                "text": "delivery schedule",
                "tokens": set(rag.tokenize("delivery schedule")),
            },
        ]
        self.frequencies = rag.build_document_frequencies(self.documents)

    def test_search_tool_has_optional_sources_parameter(self):
        tools = agent.build_tools(
            knowledge_available=True,
            search_allowed=True,
            action_allowed=False,
        )
        search_tool = tools[0]["function"]
        parameters = search_tool["parameters"]

        self.assertIn("sources", parameters["properties"])
        self.assertEqual(parameters["required"], ["query"])

    def test_empty_sources_searches_entire_knowledge_base(self):
        for sources in (None, []):
            with self.subTest(sources=sources):
                result = agent.search_knowledge_base(
                    "return window",
                    self.documents,
                    self.frequencies,
                    sources=sources,
                )

                self.assertEqual(result["sources"], [])
                self.assertEqual(result["missing_sources"], [])
                self.assertEqual(
                    [item["source"] for item in result["results"]],
                    ["returns.md"],
                )

    def test_search_is_limited_to_requested_documents(self):
        with patch.object(agent, "retrieve", return_value=[]) as retrieve:
            result = agent.search_knowledge_base(
                "return window",
                self.documents,
                self.frequencies,
                sources=["delivery.md"],
            )

        self.assertEqual(result["sources"], ["delivery.md"])
        self.assertEqual(result["missing_sources"], [])
        self.assertEqual(result["results"], [])
        self.assertEqual(retrieve.call_args.args[1], [self.documents[1]])

    def test_missing_documents_do_not_fall_back_to_full_search(self):
        result = agent.search_knowledge_base(
            "return window",
            self.documents,
            self.frequencies,
            sources=["missing.md"],
        )

        self.assertEqual(result["sources"], ["missing.md"])
        self.assertEqual(result["missing_sources"], ["missing.md"])
        self.assertEqual(result["results"], [])

    def test_target_sources_are_normalized_and_deduplicated(self):
        result = agent.search_knowledge_base(
            "return window",
            self.documents,
            self.frequencies,
            sources=[" returns.md ", "returns.md", "", None],
        )

        self.assertEqual(result["sources"], ["returns.md"])
        self.assertEqual(
            [item["source"] for item in result["results"]],
            ["returns.md"],
        )

    def test_debug_displays_requested_missing_and_found_sources(self):
        search_result = {
            "sources": ["returns.md", "missing.md"],
            "missing_sources": ["missing.md"],
            "results": [{"source": "returns.md"}],
        }

        original_debug = agent.DEBUG
        agent.DEBUG = True
        try:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                agent._debug_search_scope(search_result)
        finally:
            agent.DEBUG = original_debug

        debug_output = output.getvalue()
        self.assertIn("Запрошенные источники", debug_output)
        self.assertIn("Отсутствующие источники", debug_output)
        self.assertIn("Найденные источники", debug_output)
        self.assertIn("missing.md", debug_output)

    def test_agent_turn_passes_model_selected_sources_to_search(self):
        tool_call = SimpleNamespace(
            id="search_call",
            function=SimpleNamespace(
                name=agent.SEARCH_TOOL_NAME,
                arguments=json.dumps({
                    "query": "return window",
                    "sources": ["returns.md"],
                }),
            ),
        )
        responses = [
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=None,
                tool_calls=[tool_call],
            ))]),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content="Ответ\nUSED_SOURCES: []",
                tool_calls=[],
            ))]),
        ]
        search_result = {
            "query": "return window",
            "sources": ["returns.md"],
            "missing_sources": [],
            "results": [],
        }

        with (
            patch.object(
                agent,
                "decide_tool_permissions",
                return_value={"search": True, "action": False},
            ),
            patch.object(agent, "find_business_context", return_value="context"),
            patch.object(agent, "complete_messages", side_effect=responses),
            patch.object(
                agent,
                "search_knowledge_base",
                return_value=search_result,
            ) as search,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            turn = agent.run_agent_turn(
                [],
                "Вопрос",
                self.documents,
                self.frequencies,
            )

        self.assertEqual(turn["type"], "answer")
        search.assert_called_once_with(
            "return window",
            self.documents,
            self.frequencies,
            sources=["returns.md"],
        )


if __name__ == "__main__":
    unittest.main()
