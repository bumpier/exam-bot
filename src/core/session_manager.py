"""
Quiz session state management.

SessionManager tracks the user's progress through a fixed-length quiz,
records correct/incorrect answers, and produces a gap analysis report
at the end of each session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.mcq_generator import MCQuestion


class SessionState(str, Enum):
    SELECTION = "selection"
    EXAM = "exam"
    RESULTS = "results"


@dataclass
class AnswerRecord:
    question_index: int
    question_text: str
    selected_option: str           # "A", "B", "C", or "D"
    correct_option: str
    is_correct: bool
    source: str
    chapter: str
    explanation: str
    source_text: str               # The chunk text used to generate the question


@dataclass
class GapItem:
    source: str
    chapter: str
    missed_count: int
    total_asked: int
    recommendation: str


@dataclass
class SessionManager:
    """Manages one quiz session from start to gap analysis."""

    questions: list["MCQuestion"] = field(default_factory=list)
    answers: list[AnswerRecord] = field(default_factory=list)
    current_index: int = 0
    selected_books: list[str] = field(default_factory=list)  # display names
    selected_source_tags: list[str] = field(default_factory=list)

    # -----------------------------------------------------------------------
    # Progress helpers
    # -----------------------------------------------------------------------

    @property
    def total_questions(self) -> int:
        return len(self.questions)

    @property
    def is_complete(self) -> bool:
        return self.current_index >= self.total_questions

    @property
    def current_question(self) -> "MCQuestion | None":
        if self.is_complete:
            return None
        return self.questions[self.current_index]

    def question_for_display(
        self,
        answer_submitted: bool,
        last_record: AnswerRecord | None = None,
    ) -> "MCQuestion | None":
        """Return the question that should be displayed in the exam UI.

        When feedback is being shown after submission, the UI should keep showing
        the just-answered question from last_record.question_index, even though
        current_index already points at the next question.
        """
        if answer_submitted and last_record is not None:
            idx = last_record.question_index
            if 0 <= idx < self.total_questions:
                return self.questions[idx]
        return self.current_question

    @property
    def answered_count(self) -> int:
        return len(self.answers)

    # -----------------------------------------------------------------------
    # Answer recording
    # -----------------------------------------------------------------------

    def record_answer(self, selected_option: str) -> AnswerRecord:
        """Record the user's answer for the current question and advance.

        Args:
            selected_option: One of "A", "B", "C", "D".

        Returns:
            The AnswerRecord for the question just answered.
        """
        if self.is_complete:
            raise RuntimeError("Quiz is already complete — no more answers to record.")

        q = self.questions[self.current_index]
        is_correct = selected_option.upper() == q.correct_option.upper()

        record = AnswerRecord(
            question_index=self.current_index,
            question_text=q.question,
            selected_option=selected_option.upper(),
            correct_option=q.correct_option.upper(),
            is_correct=is_correct,
            source=q.source,
            chapter=q.chapter,
            explanation=q.explanation,
            source_text=q.source_text,
        )
        self.answers.append(record)
        self.current_index += 1
        return record

    # -----------------------------------------------------------------------
    # Scoring
    # -----------------------------------------------------------------------

    @property
    def score(self) -> int:
        """Number of correct answers so far."""
        return sum(1 for a in self.answers if a.is_correct)

    @property
    def score_percentage(self) -> float:
        if not self.answers:
            return 0.0
        return round((self.score / len(self.answers)) * 100, 1)

    # -----------------------------------------------------------------------
    # Gap analysis
    # -----------------------------------------------------------------------

    def get_gap_analysis(self) -> list[GapItem]:
        """Return a list of GapItems for concepts/chapters the user struggled with.

        A "gap" is any (source, chapter) pair where the user got at least one
        question wrong. Items are sorted by missed_count descending.
        """
        from collections import defaultdict

        # Track correct and total per (source, chapter)
        totals: dict[tuple[str, str], int] = defaultdict(int)
        misses: dict[tuple[str, str], int] = defaultdict(int)

        for a in self.answers:
            key = (a.source, a.chapter)
            totals[key] += 1
            if not a.is_correct:
                misses[key] += 1

        gaps: list[GapItem] = []
        for key, missed in misses.items():
            if missed == 0:
                continue
            source, chapter = key
            total = totals[key]
            chapter_label = f"Chapter: {chapter}" if chapter else "General section"
            recommendation = (
                f"Review {chapter_label} in '{source}'. "
                f"You missed {missed}/{total} question(s) from this area."
            )
            gaps.append(
                GapItem(
                    source=source,
                    chapter=chapter,
                    missed_count=missed,
                    total_asked=total,
                    recommendation=recommendation,
                )
            )

        return sorted(gaps, key=lambda g: g.missed_count, reverse=True)

    def get_summary(self) -> dict:
        """Return a plain-dict summary suitable for display."""
        return {
            "total_questions": self.total_questions,
            "answered": self.answered_count,
            "correct": self.score,
            "score_percentage": self.score_percentage,
            "selected_books": self.selected_books,
            "gaps": self.get_gap_analysis(),
        }
