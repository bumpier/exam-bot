import unittest
from types import SimpleNamespace

from src.core.session_manager import SessionManager


def _q(label: str):
    return SimpleNamespace(
        question=f"Question {label}",
        options={"A": "a", "B": "b", "C": "c", "D": "d"},
        correct_option="A",
        explanation="because",
        source="source-a",
        chapter="1",
        source_text="source text",
        topic="topic",
    )


class TestSessionManagerDisplayQuestion(unittest.TestCase):
    def test_question_for_display_uses_last_record_after_submit(self) -> None:
        session = SessionManager(questions=[_q("1"), _q("2")])
        record = session.record_answer("A")

        shown = session.question_for_display(answer_submitted=True, last_record=record)

        self.assertIsNotNone(shown)
        self.assertEqual(shown.question, "Question 1")
        self.assertEqual(session.current_question.question, "Question 2")

    def test_question_for_display_uses_current_when_not_submitted(self) -> None:
        session = SessionManager(questions=[_q("1"), _q("2")])

        shown = session.question_for_display(answer_submitted=False, last_record=None)

        self.assertIsNotNone(shown)
        self.assertEqual(shown.question, "Question 1")


if __name__ == "__main__":
    unittest.main()
