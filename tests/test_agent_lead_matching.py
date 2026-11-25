import contextlib
import copy
import io
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from classes import agent


class FakeWorksheet:
    def __init__(self, rows):
        self.rows = copy.deepcopy(rows)
        self.appended_rows = []

    def get_all_records(self):
        headers = self.rows[0]
        return [dict(zip(headers, row)) for row in self.rows[1:]]

    def append_row(self, row):
        self.appended_rows.append(list(row))
        self.rows.append(list(row))


class TrainingLeadMatchingTests(unittest.TestCase):
    def match(self, text, history=None):
        return agent.match_training_lead(history or [], text)

    def test_matches_name_and_exact_company(self):
        match = self.match(
            "Здравствуйте, я   анна   из nOrDlAb. Хотим посмотреть демо "
            "автоматизации IT-заявок"
        )

        self.assertEqual(match["status"], "matched")
        self.assertEqual(match["record"]["lead_id"], "LEAD-3001")

    def test_matches_name_and_email(self):
        match = self.match(
            "Здравствуйте, меня зовут Анна. Моя почта "
            "anna@nordlab.example. Хотим демо"
        )

        self.assertEqual(match["status"], "matched")
        self.assertEqual(match["record"]["lead_id"], "LEAD-3001")

    def test_email_without_name_matches_unique_lead(self):
        match = self.match(
            "Здравствуйте, моя почта anna@nordlab.example. Хотим демо"
        )

        self.assertEqual(match["status"], "matched")
        self.assertEqual(match["record"]["lead_id"], "LEAD-3001")

    def test_name_only_requests_company_or_email_without_lead_id(self):
        match = self.match("Здравствуйте, меня зовут Сергей")
        with patch.object(agent, "ENABLE_LEAD_MATCHING", True):
            context = agent.find_training_context(
                [],
                "Здравствуйте, меня зовут Сергей",
            )

        self.assertEqual(match["status"], "insufficient")
        self.assertIn("компании или email", context)
        self.assertNotIn("lead_id", match["message"])

    def test_company_without_name_matches_unique_lead(self):
        match = self.match("Здравствуйте, компания NordLab")
        with patch.object(agent, "ENABLE_LEAD_MATCHING", True):
            context = agent.find_training_context(
                [],
                "Здравствуйте, компания NordLab",
            )

        self.assertEqual(match["status"], "matched")
        self.assertEqual(match["record"]["lead_id"], "LEAD-3001")
        self.assertIn("contact: anna@nordlab.example", context)

    def test_uses_name_from_history_and_company_from_current_message(self):
        history = [
            {"role": "user", "content": "Здравствуйте, меня зовут Анна"},
            {"role": "assistant", "content": "Уточните компанию"},
        ]
        match = self.match("Компания NordLab", history)

        self.assertEqual(match["status"], "matched")
        self.assertEqual(match["record"]["lead_id"], "LEAD-3001")

    def test_uses_email_from_history_and_name_from_current_message(self):
        history = [
            {
                "role": "user",
                "content": "Моя почта anna@nordlab.example",
            },
            {"role": "assistant", "content": "Как к вам обращаться?"},
        ]
        match = self.match("Анна", history)

        self.assertEqual(match["status"], "matched")
        self.assertEqual(match["record"]["lead_id"], "LEAD-3001")

    def test_unknown_email_does_not_match(self):
        match = self.match("Моя почта nobody@unknown.example")

        self.assertEqual(match["status"], "conflict")
        self.assertNotIn("record", match)

    def test_only_greenfarm_matches_exact_company(self):
        match = self.match("Компания GreenFarm")

        self.assertEqual(match["status"], "matched")
        self.assertEqual(match["record"]["lead_id"], "LEAD-3009")

    def test_only_email_matches_exact_contact(self):
        match = self.match("mikhail@vector.example")

        self.assertEqual(match["status"], "matched")
        self.assertEqual(match["record"]["lead_id"], "LEAD-3002")

    def test_sergey_and_brighthr_match_consistently(self):
        match = self.match("Я Сергей из BrightHR")

        self.assertEqual(match["status"], "matched")
        self.assertEqual(match["record"]["lead_id"], "LEAD-3010")

    def test_name_from_another_lead_conflicts_with_email(self):
        match = self.match(
            "Меня зовут Анна, почта mikhail@vector.example"
        )

        self.assertEqual(match["status"], "conflict")
        self.assertNotIn("record", match)

    def test_email_from_another_lead_conflicts_with_company(self):
        match = self.match(
            "Я Петр из NordLab, почта mikhail@vector.example"
        )

        self.assertEqual(match["status"], "conflict")
        self.assertNotIn("record", match)

    def test_name_from_another_lead_conflicts_with_company(self):
        match = self.match("Я Ольга из NordLab")

        self.assertEqual(match["status"], "conflict")
        self.assertNotIn("record", match)
        self.assertIn("имя и компания", match["message"])

    def test_explicit_lead_id_still_matches_and_checks_other_data(self):
        matched = self.match("Мой номер LEAD-3001")
        conflict = self.match("LEAD-3001, компания Vector Retail")

        self.assertEqual(matched["status"], "matched")
        self.assertEqual(matched["record"]["lead_id"], "LEAD-3001")
        self.assertEqual(conflict["status"], "conflict")

    def test_exact_company_does_not_confuse_nordlab_and_nordlab_2(self):
        match = self.match("Я Ольга из NordLab 2")

        self.assertEqual(match["status"], "matched")
        self.assertEqual(match["record"]["lead_id"], "LEAD-3011")

    def test_confirmed_action_creates_one_request_with_matched_lead(self):
        config = agent.get_case_config("training")
        worksheet = FakeWorksheet([
            config["headers"],
            *agent.load_seed_rows(config),
        ])
        user_input = (
            "Я Анна из NordLab, anna@nordlab.example. "
            "Нужно демо автоматизации IT-заявок"
        )
        arguments = {
            "lead_id": "LEAD-3001",
            "company": "NordLab",
            "contact": "anna@nordlab.example",
            "use_case": "Автоматизация IT-заявок",
            "preferred_time": "",
            "comment": "",
        }

        with (
            patch.object(agent, "DATASET", "training"),
            patch.object(agent, "ENABLE_LEAD_MATCHING", True),
            patch.object(
                agent,
                "get_case_worksheet",
                return_value=worksheet,
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            validated = agent.validate_training_lead_match(
                arguments,
                [],
                user_input,
            )
            result = agent._handle_pending(
                "да",
                {"arguments": validated},
                [],
                config,
            )

        self.assertIsNone(result)
        self.assertEqual(len(worksheet.appended_rows), 1)
        self.assertEqual(worksheet.appended_rows[0][1:4], [
            "LEAD-3001", "NordLab", "anna@nordlab.example",
        ])
        self.assertEqual(worksheet.appended_rows[0][6], "")

    def test_comment_is_not_generated_or_modified_by_matching(self):
        arguments = {
            "lead_id": "LEAD-3001",
            "company": "NordLab",
            "contact": "anna@nordlab.example",
            "use_case": "Автоматизация IT-заявок",
            "preferred_time": "",
            "comment": "Нужна интеграция с почтой",
        }

        with (
            patch.object(agent, "DATASET", "training"),
            patch.object(agent, "ENABLE_LEAD_MATCHING", True),
        ):
            validated = agent.validate_training_lead_match(
                arguments,
                [],
                "Я Анна из NordLab, anna@nordlab.example",
            )

        self.assertEqual(validated["comment"], arguments["comment"])

    def test_comment_survives_tool_arguments_confirmation_and_sheet_write(self):
        config = agent.get_case_config("training")
        worksheet = FakeWorksheet([
            config["headers"],
            *agent.load_seed_rows(config),
        ])
        comment = "На встрече будет руководитель отдела закупок"
        arguments = {
            "lead_id": "LEAD-3005",
            "company": "BuildPro",
            "contact": "olga@buildpro.example",
            "use_case": "Автоматизация заявок на закупку",
            "preferred_time": "",
            "comment": comment,
        }
        tool_call = SimpleNamespace(function=SimpleNamespace(
            name="create_demo_request",
            arguments=json.dumps(arguments, ensure_ascii=False),
        ))

        with (
            patch.object(agent, "DATASET", "training"),
            patch.object(agent, "ENABLE_LEAD_MATCHING", True),
            patch.object(agent, "get_case_worksheet", return_value=worksheet),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            parsed = agent.parse_tool_call(tool_call)
            validated = agent.validate_tool_arguments(parsed)
            validated = agent.validate_training_lead_match(
                validated,
                [],
                "Я Ольга из BuildPro. На встрече будет руководитель "
                "отдела закупок",
            )
            confirmation = agent.make_confirmation(validated)
            pending = agent._handle_pending(
                "да",
                {"arguments": validated},
                [],
                config,
            )

        self.assertIn(f"Комментарий: {comment}", confirmation)
        self.assertIsNone(pending)
        self.assertEqual(len(worksheet.appended_rows), 1)
        self.assertEqual(worksheet.appended_rows[0][6], comment)

    def test_training_tool_describes_meaningful_comment(self):
        with patch.object(agent, "DATASET", "training"):
            tool = agent.build_tools()[0]

        description = tool["function"]["parameters"]["properties"][
            "comment"
        ]["description"]
        self.assertIn("участники предстоящей встречи", description)
        self.assertIn("не представленные в других полях", description)

    def test_company_only_can_register_with_unique_match(self):
        arguments = {
            "lead_id": "LEAD-3001",
            "company": "NordLab",
            "contact": "anna@nordlab.example",
            "use_case": "Демо",
            "comment": "",
        }

        with (
            patch.object(agent, "DATASET", "training"),
            patch.object(agent, "ENABLE_LEAD_MATCHING", True),
        ):
            validated = agent.validate_training_lead_match(
                arguments,
                [],
                "Компания NordLab",
            )

        self.assertEqual(validated["lead_id"], "LEAD-3001")
        self.assertEqual(validated["contact"], "anna@nordlab.example")

    def test_mismatched_or_unconfirmed_lead_blocks_action(self):
        arguments = {
            "lead_id": "LEAD-3002",
            "company": "Vector Retail",
            "contact": "mikhail@vector.example",
            "use_case": "Демо",
        }

        with (
            patch.object(agent, "DATASET", "training"),
            patch.object(agent, "ENABLE_LEAD_MATCHING", True),
        ):
            with self.assertRaises(ValueError):
                agent.validate_training_lead_match(
                    dict(arguments),
                    [],
                    "Я Анна из NordLab",
                )
            with self.assertRaises(ValueError):
                agent.validate_training_lead_match(
                    dict(arguments),
                    [],
                    "Меня зовут Анна",
                )

    def test_existing_duplicate_protection_remains_active(self):
        config = agent.get_case_config("training")
        worksheet = FakeWorksheet([
            config["headers"],
            *agent.load_seed_rows(config),
            [
                "DEMO-9999", "LEAD-3001", "NordLab",
                "anna@nordlab.example", "Сценарий", "", "",
                "2026-11-25T10:00:00", "requested",
            ],
        ])

        with patch.object(
            agent,
            "get_case_worksheet",
            return_value=worksheet,
        ):
            result = agent.create_demo_request(
                "LEAD-3001",
                "NordLab",
                "anna@nordlab.example",
                "Автоматизация IT-заявок",
            )

        self.assertEqual(result, {
            "status": "already_exists",
            "request_id": "DEMO-9999",
        })
        self.assertEqual(worksheet.appended_rows, [])

    def test_disabled_matching_keeps_previous_training_behavior(self):
        arguments = {
            "lead_id": "LEAD-3001",
            "company": "NordLab",
            "contact": "anna@nordlab.example",
            "use_case": "Демо",
        }

        with (
            patch.object(agent, "DATASET", "training"),
            patch.object(agent, "ENABLE_LEAD_MATCHING", False),
        ):
            context = agent.find_training_context([], "LEAD-3001")
            validated = agent.validate_training_lead_match(
                arguments,
                [],
                "LEAD-3001",
            )
            missing_context = agent.find_training_context([], "Хотим демо")

        self.assertIn("lead_id: LEAD-3001", context)
        self.assertIn("сопоставление по имени, компании и email", missing_context)
        self.assertIn("не вызывайте инструмент регистрации", missing_context)
        self.assertNotIn("запросите", missing_context.casefold())
        self.assertIs(validated, arguments)

    def test_disabled_matching_blocks_action_without_explicit_lead_id(self):
        arguments = {
            "lead_id": "LEAD-3001",
            "company": "NordLab",
            "contact": "anna@nordlab.example",
            "use_case": "Демо",
        }

        with (
            patch.object(agent, "DATASET", "training"),
            patch.object(agent, "ENABLE_LEAD_MATCHING", False),
        ):
            with self.assertRaisesRegex(ValueError, "не определена"):
                agent.validate_training_lead_match(
                    dict(arguments),
                    [],
                    "Я Анна из NordLab",
                )
            with self.assertRaisesRegex(ValueError, "не соответствуют"):
                agent.validate_training_lead_match(
                    dict(arguments),
                    [],
                    "LEAD-3002",
                )

    def test_medcore_and_enterprise_context_keep_exact_company_sizes(self):
        with patch.object(agent, "ENABLE_LEAD_MATCHING", True):
            medcore = agent.find_training_context([], "Я Сергей из MedCore")
            enterprise = agent.find_training_context(
                [],
                "Я Дмитрий из MegaHolding",
            )

        self.assertIn("lead_id: LEAD-3004", medcore)
        self.assertIn("company_size: 1300", medcore)
        self.assertIn("lead_id: LEAD-3042", enterprise)
        self.assertIn("company_size: 6200", enterprise)

    def test_service_instructions_require_literal_rules_and_no_fake_handoff(self):
        prompt = agent.build_system_prompt("company_size: 1300")

        self.assertIn("Не расширяй числовые пороги", prompt)
        self.assertIn("передача фактически не выполнялась", prompt)
        self.assertIn("числовой порог считается выполненным", agent.ACTION_VALIDATION_INSTRUCTION)
        self.assertIn("программного инструмента передачи нет", agent.ACTION_VALIDATION_INSTRUCTION)

    def test_matching_does_not_affect_demo_or_own(self):
        arguments = {"lead_id": "invented"}
        for dataset in ("demo", "own"):
            with (
                self.subTest(dataset=dataset),
                patch.object(agent, "DATASET", dataset),
                patch.object(agent, "ENABLE_LEAD_MATCHING", True),
            ):
                validated = agent.validate_training_lead_match(
                    arguments,
                    [],
                    "Анна из NordLab",
                )
                self.assertIs(validated, arguments)


if __name__ == "__main__":
    unittest.main()
