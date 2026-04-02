import unittest
from unittest.mock import patch
import time

from src.core import mcq_generator


class TestLatencyConstraints(unittest.TestCase):
    def test_extract_json_rejects_non_json_wrapper(self) -> None:
        with self.assertRaises(ValueError):
            mcq_generator._extract_json("Sure, here it is:\n{\"question\": \"x\"}")

    def test_llm_created_with_hard_generation_limits(self) -> None:
        with (
            patch("src.core.mcq_generator.LLM_PROVIDER", "ollama"),
            patch("src.core.mcq_generator.OllamaLLM") as mock_llm,
        ):
            mcq_generator._get_llm()
            kwargs = mock_llm.call_args.kwargs
            self.assertIn("num_predict", kwargs)
            self.assertIn("num_ctx", kwargs)
            self.assertIn("stop", kwargs)

    def test_generate_quiz_uses_parallel_generation(self) -> None:
        with (
            patch("src.core.mcq_generator.collection_size", return_value=10),
            patch("src.core.mcq_generator.list_unique_values", return_value=["source-a"]),
            patch("src.core.mcq_generator.LLM_PROVIDER", "ollama"),
            patch("src.core.mcq_generator._get_llm"),
        ):
            generator = mcq_generator.MCQGenerator()
            call_count = {"value": 0}

            def _slow_generate_question(source_tags, used_topics=None):
                _ = source_tags, used_topics
                time.sleep(0.1)
                call_count["value"] += 1
                idx = call_count["value"]
                return mcq_generator.MCQuestion(
                    question=f"Q{idx}",
                    options={"A": "a", "B": "b", "C": "c", "D": "d"},
                    correct_option="A",
                    explanation="because",
                    source="source-a",
                    chapter="1",
                    source_text="doc",
                    topic=f"topic-{idx}",
                )

            with patch.object(generator, "generate_question", side_effect=_slow_generate_question):
                start = time.perf_counter()
                questions = generator.generate_quiz(source_tags=["source-a"], num_questions=4)
                elapsed = time.perf_counter() - start

            self.assertEqual(len(questions), 4)
            self.assertLess(elapsed, 0.35)

    def test_openai_batch_generation_uses_single_llm_call(self) -> None:
        with (
            patch("src.core.mcq_generator.collection_size", return_value=10),
            patch("src.core.mcq_generator.list_unique_values", return_value=["source-a"]),
            patch("src.core.mcq_generator.LLM_PROVIDER", "openai"),
            patch("src.core.mcq_generator._get_llm"),
            patch("src.core.mcq_generator.retrieve_chunks", return_value=[
                {"document": "doc", "source": "source-a", "chapter": "1"}
            ]),
        ):
            generator = mcq_generator.MCQGenerator()

            class _Resp:
                content = (
                    '{"questions": ['
                    '{"question":"Q1","options":{"A":"a","B":"b","C":"c","D":"d"},'
                    '"correct_option":"A","explanation":"e1","context_index":1},'
                    '{"question":"Q2","options":{"A":"a","B":"b","C":"c","D":"d"},'
                    '"correct_option":"B","explanation":"e2","context_index":2},'
                    '{"question":"Q3","options":{"A":"a","B":"b","C":"c","D":"d"},'
                    '"correct_option":"C","explanation":"e3","context_index":3}'
                    ']}'
                )

            generator._llm.invoke.return_value = _Resp()
            questions = generator.generate_quiz(source_tags=["source-a"], num_questions=3)

            self.assertEqual(len(questions), 3)
            self.assertEqual(generator._llm.invoke.call_count, 1)

    def test_generate_question_retries_when_output_not_grounded_in_source(self) -> None:
        with (
            patch("src.core.mcq_generator.LLM_PROVIDER", "ollama"),
            patch("src.core.mcq_generator._get_llm"),
            patch(
                "src.core.mcq_generator.retrieve_chunks",
                return_value=[
                    {
                        "document": (
                            "The Money Laundering Regulations require firms to keep "
                            "customer due diligence records for five years."
                        ),
                        "source": "source-a",
                        "chapter": "2",
                    }
                ],
            ),
        ):
            generator = mcq_generator.MCQGenerator()

            class _Resp:
                def __init__(self, content: str) -> None:
                    self.content = content

            generator._llm.invoke.side_effect = [
                _Resp(
                    '{"question":"What color is the compliance logo?",'
                    '"options":{"A":"Blue","B":"Green","C":"Red","D":"Orange"},'
                    '"correct_option":"A","explanation":"The logo policy states it is blue."}'
                ),
                _Resp(
                    '{"question":"For how long must CDD records be retained?",'
                    '"options":{"A":"1 year","B":"3 years","C":"5 years","D":"10 years"},'
                    '"correct_option":"C",'
                    '"explanation":"The regulations state CDD records must be kept for five years.",'
                    '"evidence_quote":"keep customer due diligence records for five years"}'
                ),
            ]

            question = generator.generate_question(source_tags=["source-a"])

            self.assertEqual(question.correct_option, "C")
            self.assertEqual(generator._llm.invoke.call_count, 2)


if __name__ == "__main__":
    unittest.main()
