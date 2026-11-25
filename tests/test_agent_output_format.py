import contextlib
import io
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from classes import agent


def response(content=None, tool_call=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=content,
        tool_calls=[] if tool_call is None else [tool_call],
    ))])


def search_call(call_id, query):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=agent.SEARCH_TOOL_NAME,
            arguments=json.dumps({"query": query}),
        ),
    )


class AgentOutputFormatTests(unittest.TestCase):
    def test_answer_without_tools_keeps_normal_format(self):
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            agent._print_answer("Обычный ответ")

        self.assertEqual(output.getvalue(), "\nИИ:\nОбычный ответ\n\n")

    def test_tool_calls_and_final_answer_share_one_ai_block(self):
        responses = [
            response(tool_call=search_call("call_1", "первый запрос")),
            response(tool_call=search_call("call_2", "второй запрос")),
            response("Итоговый ответ\nUSED_SOURCES: []"),
        ]
        search_result = {
            "query": "",
            "sources": [],
            "missing_sources": [],
            "results": [],
        }
        output = io.StringIO()

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
            ),
            contextlib.redirect_stdout(output),
        ):
            turn = agent.run_agent_turn([], "Вопрос", [{}], {})
            agent._print_answer(
                turn["answer"],
                turn["used_sources"],
                response_started=turn["response_started"],
            )

        rendered = output.getvalue()
        self.assertEqual(rendered.count("ИИ:"), 1)
        self.assertEqual(rendered.count("Инструмент: search_knowledge_base"), 2)
        self.assertIn(
            "ИИ:\n"
            "Инструмент: search_knowledge_base\n"
            "Инструмент: search_knowledge_base\n\n"
            "Итоговый ответ",
            rendered,
        )
        self.assertLess(rendered.index("ИИ:"), rendered.index("Инструмент:"))
        self.assertLess(rendered.rindex("Инструмент:"), rendered.index("Итоговый ответ"))

    def test_confirmed_action_result_has_header_before_tool(self):
        arguments = {
            "order_id": "12345",
            "item": "Лампа",
            "reason": "Повреждение",
            "comment": "Трещина",
        }
        output = io.StringIO()

        with (
            patch.object(agent, "DATASET", "demo"),
            patch.object(
                agent,
                "execute_tool",
                return_value={"status": "tool_error"},
            ),
            contextlib.redirect_stdout(output),
        ):
            agent._handle_pending(
                "да",
                {"arguments": arguments},
                [],
                agent.get_case_config(),
            )

        rendered = output.getvalue()
        self.assertEqual(rendered.count("ИИ:"), 1)
        self.assertIn(
            "ИИ:\nИнструмент: create_return_request\n\n",
            rendered,
        )
        self.assertLess(rendered.index("ИИ:"), rendered.index("Инструмент:"))
        self.assertIn("технической ошибки", rendered)


if __name__ == "__main__":
    unittest.main()
