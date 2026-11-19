import contextlib
import io
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from classes import agent


DATASET_ARGUMENTS = {
    "demo": {
        "order_id": "12345",
        "item": "Лампа",
        "reason": "Повреждение",
        "comment": "Трещина на плафоне",
    },
    "training": {
        "lead_id": "LEAD-101",
        "company": "Компания",
        "contact": "client@example.com",
        "use_case": "Автоматизация поддержки",
    },
    "own": {
        "subject": "Тестовая заявка",
        "description": "Описание заявки",
    },
}


class ToolErrorSimulationTests(unittest.TestCase):
    def test_simulation_is_disabled_by_default(self):
        self.assertFalse(agent.SIMULATE_TOOL_ERROR)

    def test_confirmed_action_returns_standard_error_without_writes(self):
        for dataset, arguments in DATASET_ARGUMENTS.items():
            with self.subTest(dataset=dataset):
                history = []
                pending = {"arguments": dict(arguments)}

                with (
                    patch.object(agent, "DATASET", dataset),
                    patch.object(agent, "SIMULATE_TOOL_ERROR", True),
                    patch.object(agent, "create_return_request") as demo_write,
                    patch.object(agent, "create_demo_request") as training_write,
                    patch.object(agent, "create_own_request") as own_write,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    expected_answer = agent.build_tool_result_answer(
                        {"status": "tool_error"}
                    )
                    result = agent._handle_pending(
                        "да",
                        pending,
                        history,
                        agent.get_case_config(),
                    )

                self.assertIsNone(result)
                self.assertEqual(history[-1], {
                    "role": "assistant",
                    "content": expected_answer,
                })
                demo_write.assert_not_called()
                training_write.assert_not_called()
                own_write.assert_not_called()

    def test_disabled_simulation_uses_normal_dataset_tool(self):
        tool_functions = {
            "demo": "create_return_request",
            "training": "create_demo_request",
            "own": "create_own_request",
        }
        for dataset, arguments in DATASET_ARGUMENTS.items():
            with self.subTest(dataset=dataset):
                function_name = tool_functions[dataset]
                expected_result = {"status": "created", "request_id": "TEST-1"}

                with (
                    patch.object(agent, "DATASET", dataset),
                    patch.object(agent, "SIMULATE_TOOL_ERROR", False),
                    patch.object(
                        agent,
                        function_name,
                        return_value=expected_result,
                    ) as dataset_tool,
                ):
                    result = agent.execute_tool(dict(arguments))

                self.assertEqual(result, expected_result)
                dataset_tool.assert_called_once()


if __name__ == "__main__":
    unittest.main()
