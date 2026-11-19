import contextlib
import io
import os
import unittest

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from classes import agent, rag


EXPECTED_TITLES = {
    "demo": "Demo: Клиентская поддержка интернет-магазина",
    "training": "Training: Продажа онлайн-сервиса",
    "own": "Own: Собственный кейс",
}


class StartupTitlesTests(unittest.TestCase):
    def test_shared_dataset_titles(self):
        self.assertEqual(rag.DATASET_TITLES, EXPECTED_TITLES)
        self.assertIs(agent.DATASET_TITLES, rag.DATASET_TITLES)

    def test_rag_and_agent_headers_use_same_title_format(self):
        for tool_name in ("RAG", "Agent"):
            for dataset, title in EXPECTED_TITLES.items():
                with self.subTest(tool=tool_name, dataset=dataset):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        rag.print_startup_header(tool_name, dataset)

                    self.assertEqual(
                        output.getvalue().splitlines(),
                        [f"{tool_name} запущен.", title],
                    )

    def test_agent_case_configs_reference_shared_titles(self):
        for dataset, title in EXPECTED_TITLES.items():
            with self.subTest(dataset=dataset):
                self.assertEqual(agent.CASE_CONFIGS[dataset]["title"], title)


if __name__ == "__main__":
    unittest.main()
