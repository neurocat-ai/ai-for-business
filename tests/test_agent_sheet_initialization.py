import contextlib
import copy
import io
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from classes import agent


class FakeWorksheet:
    def __init__(self, rows=None):
        self.rows = copy.deepcopy(rows or [])
        self.appended_rows = []

    def row_values(self, row_number):
        if row_number != 1 or not self.rows:
            return []
        return list(self.rows[0])

    def update(self, values, range_name):
        if range_name == "A1":
            self.rows = copy.deepcopy(values)
            return
        if range_name == "A2":
            self.rows = [self.rows[0], *copy.deepcopy(values)]
            return
        raise AssertionError(f"Неожиданный диапазон: {range_name}")

    def get_all_values(self):
        return copy.deepcopy(self.rows)

    def get_all_records(self):
        if not self.rows:
            return []
        headers = self.rows[0]
        return [
            dict(zip(headers, row))
            for row in self.rows[1:]
        ]

    def append_row(self, row):
        self.appended_rows.append(list(row))
        self.rows.append(list(row))


class AgentSheetInitializationTests(unittest.TestCase):
    def test_empty_demo_and_training_tabs_receive_headers_and_six_seeds(self):
        for dataset in ("demo", "training"):
            with self.subTest(dataset=dataset):
                worksheet = FakeWorksheet()
                config = agent.get_case_config(dataset)

                with patch.object(
                    agent,
                    "get_case_worksheet",
                    return_value=worksheet,
                ):
                    agent.prepare_case_worksheet(config)

                self.assertEqual(worksheet.rows[0], config["headers"])
                self.assertEqual(len(worksheet.rows[1:]), 6)

    def test_empty_own_tab_receives_headers_without_seed_rows(self):
        worksheet = FakeWorksheet()
        config = agent.get_case_config("own")

        with patch.object(
            agent,
            "get_case_worksheet",
            return_value=worksheet,
        ):
            agent.prepare_case_worksheet(config)

        self.assertEqual(worksheet.rows, [config["headers"]])

    def test_repeated_start_does_not_duplicate_seed_rows(self):
        worksheet = FakeWorksheet()
        config = agent.get_case_config("demo")

        with patch.object(
            agent,
            "get_case_worksheet",
            return_value=worksheet,
        ):
            agent.prepare_case_worksheet(config)
            first_state = copy.deepcopy(worksheet.rows)
            agent.prepare_case_worksheet(config)

        self.assertEqual(worksheet.rows, first_state)
        self.assertEqual(len(worksheet.rows[1:]), 6)

    def test_correct_headers_without_rows_restore_demo_and_training_seeds(self):
        for dataset in ("demo", "training"):
            with self.subTest(dataset=dataset):
                config = agent.get_case_config(dataset)
                worksheet = FakeWorksheet([config["headers"]])

                with patch.object(
                    agent,
                    "get_case_worksheet",
                    return_value=worksheet,
                ):
                    agent.prepare_case_worksheet(config)

                self.assertEqual(worksheet.rows[0], config["headers"])
                self.assertEqual(
                    worksheet.rows[1:],
                    agent.load_seed_rows(config),
                )

    def test_existing_requests_are_not_changed_on_start(self):
        config = agent.get_case_config("demo")
        existing = [
            config["headers"],
            *agent.load_seed_rows(config),
            [
                "RET-9999", "15451", "Товар", "Причина", "Комментарий",
                "2026-11-25T10:00:00", "open",
            ],
        ]
        worksheet = FakeWorksheet(existing)

        with patch.object(
            agent,
            "get_case_worksheet",
            return_value=worksheet,
        ):
            agent.prepare_case_worksheet(config)

        self.assertEqual(worksheet.rows, existing)

    def test_data_below_empty_header_stops_initialization_without_changes(self):
        config = agent.get_case_config("demo")
        existing = [[] for _ in range(29)]
        existing.append(["RET-9999", "existing data"])
        worksheet = FakeWorksheet(existing)

        with (
            patch.object(
                agent,
                "get_case_worksheet",
                return_value=worksheet,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "заполнена некорректно",
            ),
        ):
            agent.prepare_case_worksheet(config)

        self.assertEqual(worksheet.rows, existing)

    def test_arbitrary_row_with_correct_headers_is_rejected_unchanged(self):
        config = agent.get_case_config("demo")
        existing = [
            config["headers"],
            *agent.load_seed_rows(config),
        ]
        existing.insert(2, ["произвольное значение"])
        worksheet = FakeWorksheet(existing)

        with (
            patch.object(
                agent,
                "get_case_worksheet",
                return_value=worksheet,
            ),
            self.assertRaisesRegex(RuntimeError, "заполнена некорректно"),
        ):
            agent.prepare_case_worksheet(config)

        self.assertEqual(worksheet.rows, existing)

    def test_missing_seed_request_is_rejected_for_demo_and_training(self):
        for dataset in ("demo", "training"):
            with self.subTest(dataset=dataset):
                config = agent.get_case_config(dataset)
                existing = [
                    config["headers"],
                    *agent.load_seed_rows(config)[:-1],
                ]
                worksheet = FakeWorksheet(existing)

                with (
                    patch.object(
                        agent,
                        "get_case_worksheet",
                        return_value=worksheet,
                    ),
                    self.assertRaisesRegex(
                        RuntimeError,
                        "заполнена некорректно",
                    ),
                ):
                    agent.prepare_case_worksheet(config)

                self.assertEqual(worksheet.rows, existing)

    def test_missing_or_duplicate_request_id_is_rejected(self):
        config = agent.get_case_config("demo")
        seeds = agent.load_seed_rows(config)
        invalid_rows = (
            [config["headers"], *seeds, ["", *seeds[0][1:]]],
            [config["headers"], *seeds, list(seeds[0])],
        )

        for existing in invalid_rows:
            with self.subTest(existing=existing[-1][0]):
                worksheet = FakeWorksheet(existing)
                with (
                    patch.object(
                        agent,
                        "get_case_worksheet",
                        return_value=worksheet,
                    ),
                    self.assertRaisesRegex(
                        RuntimeError,
                        "заполнена некорректно",
                    ),
                ):
                    agent.prepare_case_worksheet(config)

                self.assertEqual(worksheet.rows, existing)

    def test_own_headers_and_registered_request_are_valid(self):
        config = agent.get_case_config("own")
        valid_states = (
            [config["headers"]],
            [
                config["headers"],
                [
                    "OWN-3001", "Тема", "Описание", "", "",
                    "2026-11-25T10:00:00", "requested",
                ],
            ],
        )

        for existing in valid_states:
            with self.subTest(rows=len(existing) - 1):
                worksheet = FakeWorksheet(existing)
                with patch.object(
                    agent,
                    "get_case_worksheet",
                    return_value=worksheet,
                ):
                    agent.prepare_case_worksheet(config)

                self.assertEqual(worksheet.rows, existing)

    def test_main_prepares_only_the_selected_dataset_tab(self):
        worksheets = {
            dataset: FakeWorksheet()
            for dataset in ("demo", "training", "own")
        }

        for dataset in worksheets:
            with self.subTest(dataset=dataset):
                for worksheet in worksheets.values():
                    worksheet.rows = []

                session = MagicMock()
                session.prompt.return_value = "/exit"
                output = io.StringIO()
                with (
                    patch.object(agent, "DATASET", dataset),
                    patch.object(
                        agent,
                        "get_worksheet",
                        side_effect=lambda selected: worksheets[selected],
                    ),
                    patch.object(
                        agent,
                        "_load_optional_knowledge_base",
                        return_value=([], {}),
                    ),
                    patch.object(agent, "PromptSession", return_value=session),
                    patch.object(agent, "clear_terminal"),
                    contextlib.redirect_stdout(output),
                ):
                    agent.main()

                selected = worksheets[dataset]
                self.assertEqual(
                    selected.rows[0],
                    agent.get_case_config(dataset)["headers"],
                )
                for other_dataset, worksheet in worksheets.items():
                    if other_dataset != dataset:
                        self.assertEqual(worksheet.rows, [])

    def test_confirmed_request_appends_exactly_one_row(self):
        config = agent.get_case_config("demo")
        worksheet = FakeWorksheet([
            config["headers"],
            *agent.load_seed_rows(config),
        ])
        arguments = {
            "order_id": "15451",
            "item": "Настольная лампа Lumo",
            "reason": "Повреждение",
            "comment": "Трещина",
        }

        with (
            patch.object(agent, "DATASET", "demo"),
            patch.object(
                agent,
                "get_case_worksheet",
                return_value=worksheet,
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = agent._handle_pending(
                "да",
                {"arguments": arguments},
                [],
                config,
            )

        self.assertIsNone(result)
        self.assertEqual(len(worksheet.appended_rows), 1)
        self.assertEqual(worksheet.appended_rows[0][1], "15451")

    def test_refused_request_does_not_append_rows(self):
        worksheet = FakeWorksheet()
        config = agent.get_case_config("demo")

        with (
            patch.object(agent, "DATASET", "demo"),
            patch.object(
                agent,
                "get_case_worksheet",
                return_value=worksheet,
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = agent._handle_pending(
                "нет",
                {"arguments": {"order_id": "15451"}},
                [],
                config,
            )

        self.assertIsNone(result)
        self.assertEqual(worksheet.appended_rows, [])

    def test_invalid_headers_stop_main_and_preserve_existing_data(self):
        worksheet = FakeWorksheet([
            ["wrong", "headers"],
            ["existing", "data"],
        ])
        initial_state = copy.deepcopy(worksheet.rows)
        output = io.StringIO()

        with (
            patch.object(agent, "get_worksheet", return_value=worksheet),
            patch.object(agent, "PromptSession") as prompt_session,
            patch.object(agent, "clear_terminal"),
            contextlib.redirect_stdout(output),
        ):
            agent.main()

        prompt_session.assert_not_called()
        self.assertEqual(worksheet.rows, initial_state)
        self.assertIn("Не удалось подготовить Google Sheets", output.getvalue())
        self.assertIn("заполнена некорректно", output.getvalue())

    def test_missing_or_unavailable_sheet_stops_main_with_clear_error(self):
        output = io.StringIO()

        with (
            patch.object(
                agent,
                "prepare_case_worksheet",
                side_effect=RuntimeError("Вкладка 'Demo' недоступна."),
            ),
            patch.object(agent, "PromptSession") as prompt_session,
            patch.object(agent, "clear_terminal"),
            contextlib.redirect_stdout(output),
        ):
            agent.main()

        prompt_session.assert_not_called()
        self.assertIn("Не удалось подготовить Google Sheets", output.getvalue())
        self.assertIn("Вкладка 'Demo' недоступна", output.getvalue())
        self.assertIn("Запуск Agent остановлен", output.getvalue())

    def test_connection_message_precedes_sheet_access_and_normal_interface(self):
        output = io.StringIO()
        session = MagicMock()
        session.prompt.return_value = "/exit"
        output_seen_during_access = []

        def prepare(_config):
            output_seen_during_access.append(output.getvalue())

        with (
            patch.object(agent, "prepare_case_worksheet", side_effect=prepare),
            patch.object(
                agent,
                "_load_optional_knowledge_base",
                return_value=([], {}),
            ),
            patch.object(agent, "PromptSession", return_value=session),
            patch.object(agent, "clear_terminal"),
            contextlib.redirect_stdout(output),
        ):
            agent.main()

        connection_message = "Подключение к Google Sheets..."
        self.assertIn(connection_message, output_seen_during_access[0])
        rendered = output.getvalue()
        self.assertLess(
            rendered.index(connection_message),
            rendered.index("\r\033[2K"),
        )
        self.assertLess(
            rendered.index("\r\033[2K"),
            rendered.index("Agent запущен."),
        )
        self.assertNotIn("Таблица готова.", rendered)


if __name__ == "__main__":
    unittest.main()
