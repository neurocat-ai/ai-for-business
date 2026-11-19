import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from classes import rag


class EmptyKnowledgeBaseTests(unittest.TestCase):
    def test_empty_knowledge_base_is_detected_for_every_dataset(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)

            for dataset in ("demo", "training", "own"):
                with self.subTest(dataset=dataset):
                    knowledge_base_dir = (
                        root / "data" / dataset / "03_knowledge_base"
                    )
                    knowledge_base_dir.mkdir(parents=True)

                    with (
                        patch.object(rag, "ROOT", root),
                        self.assertRaises(rag.EmptyKnowledgeBaseError) as raised,
                    ):
                        rag.load_knowledge_base(dataset)

                    self.assertEqual(
                        raised.exception.knowledge_base_dir,
                        knowledge_base_dir,
                    )

    def test_main_prints_message_and_finishes_without_traceback(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            knowledge_base_dir = root / "data" / "demo" / "03_knowledge_base"
            knowledge_base_dir.mkdir(parents=True)
            output = io.StringIO()

            with (
                patch.object(rag, "ROOT", root),
                patch.object(rag, "DATASET", "demo"),
                patch.object(rag, "clear_terminal"),
                contextlib.redirect_stdout(output),
            ):
                result = rag.main()

            self.assertIsNone(result)
            self.assertEqual(
                output.getvalue().strip(),
                "База знаний пуста. Добавьте хотя бы один .md-файл "
                f"в папку {knowledge_base_dir} и повторите запуск.",
            )

    def test_main_does_not_hide_missing_directory_error(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            with (
                patch.object(rag, "ROOT", Path(temporary_dir)),
                patch.object(rag, "DATASET", "training"),
                patch.object(rag, "clear_terminal"),
                self.assertRaises(FileNotFoundError),
            ):
                rag.main()


if __name__ == "__main__":
    unittest.main()
