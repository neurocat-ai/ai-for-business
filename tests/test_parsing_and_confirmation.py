import contextlib
import io
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from classes import agent, rag


class UsedSourcesParsingTests(unittest.TestCase):
    def setUp(self):
        self.results = [
            {"source": "rules.md", "text": "Правила"},
            {"source": "policy.md", "text": "Политика"},
        ]

    def test_extracts_sources_from_separate_line(self):
        answer, sources = rag.parse_answer_and_sources(
            'Ответ клиенту.\nUSED_SOURCES: ["rules.md"]',
            self.results,
        )

        self.assertEqual(answer, "Ответ клиенту.")
        self.assertEqual(sources, ["rules.md"])

    def test_extracts_sources_immediately_after_answer(self):
        answer, sources = rag.parse_answer_and_sources(
            'Ответ клиенту.USED_SOURCES: ["rules.md", "unknown.md"]',
            self.results,
        )

        self.assertEqual(answer, "Ответ клиенту.")
        self.assertEqual(sources, ["rules.md"])

    def test_agent_uses_same_parser_and_source_validation(self):
        answer, sources = agent.parse_answer_and_sources(
            'Ответ USED_SOURCES: ["policy.md", "policy.md", "unknown.md"]',
            self.results,
        )

        self.assertEqual(answer, "Ответ")
        self.assertEqual(sources, ["policy.md"])

    def test_removes_malformed_service_block(self):
        answer, sources = rag.parse_answer_and_sources(
            "ОтветUSED_SOURCES: not-json",
            self.results,
        )

        self.assertEqual(answer, "Ответ")
        self.assertEqual(sources, [])


class ConfirmationParsingTests(unittest.TestCase):
    def test_accepts_exact_phrase_after_normalization(self):
        self.assertTrue(agent.is_yes("  ДА!!!  "))
        self.assertTrue(agent.is_yes("подтверждаю.   "))
        self.assertTrue(agent.is_no("  НЕ   НАДО... "))

    def test_rejects_ambiguous_prefixes(self):
        self.assertFalse(agent.is_yes("да но не создавай"))
        self.assertFalse(agent.is_yes("подтверждаю, но позже"))
        self.assertFalse(agent.is_no("нет, создавай"))

    def test_ambiguous_answer_keeps_confirmation_pending(self):
        pending = {"arguments": {"order_id": "12345"}}
        history = []

        with (
            patch.object(agent, "execute_tool") as execute_tool,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = agent._handle_pending(
                "да но не создавай",
                pending,
                history,
                agent.get_case_config(),
            )

        self.assertIs(result, pending)
        self.assertEqual(history, [])
        execute_tool.assert_not_called()


if __name__ == "__main__":
    unittest.main()
