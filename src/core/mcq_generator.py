"""
MCQ generation engine.

Uses LangChain with either Ollama (local) or OpenAI (ChatGPT) to produce
compliance-focused multiple-choice questions grounded in retrieved ChromaDB context.
"""

from __future__ import annotations

import json
import random
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any

from langchain_ollama import OllamaLLM
from pydantic import BaseModel, field_validator

from src.config import (
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_STOP,
    OPENAI_API_KEY,
    OPENAI_MAX_TOKENS,
    OPENAI_MODEL,
    OPENAI_TIMEOUT_SECONDS,
    RETRIEVAL_TOP_K,
)
from src.db.chroma_client import collection_size, list_unique_values, retrieve_chunks

DEBUG_LOG_PATH = "/Users/liam/Documents/exam-bot/.cursor/debug-9da7c1.log"
DEBUG_SESSION_ID = "9da7c1"


def _debug_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any],
    run_id: str = "pre-fix",
) -> None:
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

OPTION_LETTERS = ["A", "B", "C", "D"]

COMPLIANCE_TOPICS = [
    "customer due diligence obligations",
    "suspicious transaction reporting",
    "beneficial ownership identification",
    "politically exposed persons (PEP) screening",
    "sanctions list screening procedures",
    "record-keeping requirements",
    "risk-based approach to AML",
    "correspondent banking due diligence",
    "tipping-off prohibition",
    "internal controls and compliance programme",
    "bribery and corruption prevention",
    "gifts and hospitality policies",
    "market abuse and insider trading",
    "best execution obligations",
    "client suitability assessment",
    "transaction monitoring red flags",
    "source of funds verification",
    "enhanced due diligence triggers",
    "simplified due diligence criteria",
    "regulatory reporting deadlines",
]


class MCQuestion(BaseModel):
    """A single multiple-choice compliance question."""

    question: str
    options: dict[str, str]     # {"A": "...", "B": "...", "C": "...", "D": "..."}
    correct_option: str         # "A", "B", "C", or "D"
    explanation: str
    source: str                 # source_tag from ChromaDB metadata
    chapter: str                # chapter tag from ChromaDB metadata
    source_text: str            # the retrieved chunk used to ground this question
    topic: str                  # the compliance topic queried

    @field_validator("correct_option")
    @classmethod
    def validate_correct_option(cls, v: str) -> str:
        upper = v.strip().upper()
        if upper not in OPTION_LETTERS:
            raise ValueError(f"correct_option must be A, B, C, or D — got '{v}'")
        return upper

    @field_validator("options")
    @classmethod
    def validate_options(cls, v: dict) -> dict:
        normalised = {k.strip().upper(): val for k, val in v.items()}
        for letter in OPTION_LETTERS:
            if letter not in normalised:
                raise ValueError(f"options dict is missing key '{letter}'")
        return normalised


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

MCQ_SYSTEM_PROMPT = """Return only one JSON object with keys:
question, options, correct_option, explanation.
Rules:
- Use only SOURCE_TEXT facts.
- options must contain exactly A, B, C, D.
- Exactly one correct option.
- explanation <= 60 words and cite section/article if present.
- No markdown, no prose before/after JSON."""

MCQ_USER_PROMPT = """SOURCE TEXT (from compliance manual):
---
{context}
---

TOPIC: {topic}
Generate one situational MCQ."""

BATCH_MCQ_SYSTEM_PROMPT = """Return only one JSON object:
{
  "questions": [
    {
      "question": "...",
      "options": {"A":"...","B":"...","C":"...","D":"..."},
      "correct_option": "A|B|C|D",
      "explanation": "...",
      "context_index": 1
    }
  ]
}
Rules:
- Generate exactly {num_questions} questions.
- Use only facts from the provided CONTEXTS.
- context_index must reference the context used for that question.
- options must contain exactly A, B, C, D and only one correct option.
- explanation <= 40 words.
- No markdown, no prose outside JSON."""


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def _get_llm() -> Any:
    if LLM_PROVIDER == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing dependency `langchain-openai`. Install it with "
                "`pip install langchain-openai` (or `pip install -r requirements.txt`)."
            ) from exc

        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is missing. Set it in your .env file when using "
                "LLM_PROVIDER=openai."
            )
        return ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0.1,
            max_tokens=OPENAI_MAX_TOKENS,
            timeout=OPENAI_TIMEOUT_SECONDS,
        )

    # Default to Ollama for local/offline usage
    return OllamaLLM(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,      # Lower temperature to reduce meandering output
        format="json",        # Ask Ollama to constrain output to JSON
        num_predict=OLLAMA_NUM_PREDICT,
        num_ctx=OLLAMA_NUM_CTX,
        stop=OLLAMA_STOP,
    )


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict[str, Any]:
    """Parse a strict, single JSON object response from the model."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    if not (cleaned.startswith("{") and cleaned.endswith("}")):
        # region agent log
        _debug_log(
            hypothesis_id="H4",
            location="src/core/mcq_generator.py:_extract_json",
            message="LLM response is not strict JSON object",
            data={
                "starts_with_brace": cleaned.startswith("{"),
                "ends_with_brace": cleaned.endswith("}"),
                "response_preview": cleaned[:250],
            },
        )
        # endregion
        raise ValueError(f"Response is not a strict JSON object:\n{cleaned[:300]}")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # region agent log
        _debug_log(
            hypothesis_id="H4",
            location="src/core/mcq_generator.py:_extract_json",
            message="JSON decode failed",
            data={
                "error": str(exc),
                "response_preview": cleaned[:250],
            },
        )
        # endregion
        raise ValueError(f"Could not parse JSON from LLM response:\n{cleaned[:300]}") from exc


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class MCQGenerator:
    """Generates compliance MCQs using retrieved ChromaDB context + configured LLM."""

    def __init__(self) -> None:
        self._llm = _get_llm()

    def generate_question(
        self,
        source_tags: list[str],
        used_topics: set[str] | None = None,
    ) -> MCQuestion:
        """Generate one MCQuestion grounded in retrieved compliance context.

        Args:
            source_tags:  List of ChromaDB source_tag values to retrieve from.
            used_topics:  Topics already used in this session (for variety).

        Returns:
            A validated MCQuestion dataclass.

        Raises:
            ValueError:   If the LLM returns unparseable or invalid JSON.
            RuntimeError: If no relevant chunks are found in the database.
        """
        if used_topics is None:
            used_topics = set()

        # Pick a topic we haven't used yet in this session
        available = [t for t in COMPLIANCE_TOPICS if t not in used_topics]
        if not available:
            available = COMPLIANCE_TOPICS  # cycle if exhausted
        topic = random.choice(available)

        # Retrieve relevant chunks from ChromaDB
        chunks = retrieve_chunks(
            query=topic,
            source_tags=source_tags,
            top_k=RETRIEVAL_TOP_K,
        )

        if not chunks:
            raise RuntimeError(
                f"No chunks found for source_tags={source_tags}. "
                "Make sure the books are ingested into ChromaDB."
            )

        # Use the single most relevant chunk as context
        best_chunk = chunks[0]
        context = best_chunk["document"]
        source = best_chunk["source"]
        chapter = best_chunk["chapter"]

        # Build the prompt
        full_prompt = (
            MCQ_SYSTEM_PROMPT
            + "\n\n"
            + MCQ_USER_PROMPT.format(context=context, topic=topic)
        )
        response = self._llm.invoke(full_prompt)
        raw_response = response.content if hasattr(response, "content") else str(response)
        # region agent log
        _debug_log(
            hypothesis_id="H4",
            location="src/core/mcq_generator.py:generate_question",
            message="Received raw LLM response",
            data={
                "topic": topic,
                "response_preview": str(raw_response)[:250],
                "response_len": len(str(raw_response)),
            },
        )
        # endregion

        # Parse and validate
        data = _extract_json(raw_response)
        # region agent log
        _debug_log(
            hypothesis_id="H1",
            location="src/core/mcq_generator.py:generate_question",
            message="Parsed LLM payload shape",
            data={
                "keys": list(data.keys()),
                "options_type": type(data.get("options")).__name__,
                "options_preview": str(data.get("options"))[:200],
            },
        )
        # endregion
        # region agent log
        _debug_log(
            hypothesis_id="H2",
            location="src/core/mcq_generator.py:generate_question",
            message="Parsed correct_option payload",
            data={
                "correct_option_type": type(data.get("correct_option")).__name__,
                "correct_option_value": str(data.get("correct_option"))[:120],
            },
        )
        # endregion
        question = MCQuestion(
            question=data["question"],
            options=data["options"],
            correct_option=data["correct_option"],
            explanation=data["explanation"],
            source=source,
            chapter=chapter,
            source_text=context,
            topic=topic,
        )
        return question

    def generate_quiz(
        self,
        source_tags: list[str],
        num_questions: int = 10,
    ) -> list[MCQuestion]:
        """Generate a full quiz of num_questions MCQs.

        Each question targets a different compliance topic for variety.
        Questions that fail to parse are retried up to 2 times before being skipped.

        Args:
            source_tags:   Source tags to retrieve from.
            num_questions: Number of questions to generate.

        Returns:
            List of MCQuestion objects (may be shorter than num_questions if
            retries are exhausted for some questions).
        """
        if collection_size() == 0:
            raise RuntimeError(
                "ChromaDB collection is empty. Run `python scripts/ingest_pdfs.py` "
                "to ingest your PDFs first."
            )

        available_sources = set(list_unique_values("source"))
        missing = [s for s in source_tags if s not in available_sources]
        if missing:
            raise RuntimeError(
                "Selected book(s) not found in the vector database (missing source tags): "
                + ", ".join(missing)
                + ". Run `python scripts/inspect_db.py` to see what sources are available."
            )

        if LLM_PROVIDER == "openai":
            try:
                return self._generate_quiz_batch_openai(
                    source_tags=source_tags,
                    num_questions=num_questions,
                )
            except (ValueError, RuntimeError, KeyError) as exc:
                _debug_log(
                    hypothesis_id="H5",
                    location="src/core/mcq_generator.py:generate_quiz",
                    message="Batch OpenAI generation failed; falling back to parallel single-question path",
                    data={
                        "num_questions_target": num_questions,
                        "source_tags": source_tags,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )

        questions: list[MCQuestion] = []
        max_attempts = num_questions * 3  # generous retry budget
        attempts = 0

        max_workers = max(1, min(num_questions, 8))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            in_flight = set()

            def _submit_one() -> None:
                nonlocal attempts
                if attempts >= max_attempts or len(questions) + len(in_flight) >= num_questions:
                    return
                attempts += 1
                in_flight.add(
                    executor.submit(
                        self.generate_question,
                        source_tags=source_tags,
                        used_topics=None,
                    )
                )

            for _ in range(min(num_questions, max_attempts)):
                _submit_one()

            while in_flight and len(questions) < num_questions:
                done, pending = wait(in_flight, return_when=FIRST_COMPLETED)
                in_flight = set(pending)
                for fut in done:
                    try:
                        q = fut.result()
                        questions.append(q)
                    except (ValueError, RuntimeError, KeyError) as exc:
                        _debug_log(
                            hypothesis_id="H3",
                            location="src/core/mcq_generator.py:generate_quiz",
                            message="Question generation attempt failed",
                            data={
                                "attempt": attempts,
                                "num_questions_target": num_questions,
                                "source_tags": source_tags,
                                "error": str(exc),
                                "error_type": type(exc).__name__,
                            },
                        )
                        print(f"[MCQGenerator] Skipping question (attempt {attempts}): {exc}")

                while len(questions) + len(in_flight) < num_questions and attempts < max_attempts:
                    _submit_one()

        return questions

    def _generate_quiz_batch_openai(
        self,
        source_tags: list[str],
        num_questions: int,
    ) -> list[MCQuestion]:
        topics = [random.choice(COMPLIANCE_TOPICS) for _ in range(num_questions)]
        contexts: list[dict[str, str]] = []
        for topic in topics:
            chunks = retrieve_chunks(
                query=topic,
                source_tags=source_tags,
                top_k=1,
            )
            if not chunks:
                raise RuntimeError(
                    f"No chunks found for source_tags={source_tags}. "
                    "Make sure the books are ingested into ChromaDB."
                )
            best_chunk = chunks[0]
            contexts.append(
                {
                    "topic": topic,
                    "source_text": best_chunk["document"],
                    "source": best_chunk["source"],
                    "chapter": best_chunk["chapter"],
                }
            )

        numbered_contexts = []
        for i, ctx in enumerate(contexts, start=1):
            numbered_contexts.append(
                f"[{i}] TOPIC: {ctx['topic']}\nSOURCE_TEXT:\n{ctx['source_text']}"
            )
        user_prompt = "CONTEXTS:\n\n" + "\n\n".join(numbered_contexts)
        batch_system_prompt = BATCH_MCQ_SYSTEM_PROMPT.replace(
            "{num_questions}", str(num_questions)
        )
        full_prompt = batch_system_prompt + "\n\n" + user_prompt
        response = self._llm.invoke(full_prompt)
        raw_response = response.content if hasattr(response, "content") else str(response)
        data = _extract_json(raw_response)
        items = data.get("questions")
        if not isinstance(items, list):
            raise ValueError("Batch response missing 'questions' list.")
        if len(items) != num_questions:
            raise ValueError(
                f"Batch response returned {len(items)} questions; expected {num_questions}."
            )

        questions: list[MCQuestion] = []
        for item in items:
            ctx_index = int(item["context_index"])
            if ctx_index < 1 or ctx_index > len(contexts):
                raise ValueError(f"context_index out of range: {ctx_index}")
            ctx = contexts[ctx_index - 1]
            questions.append(
                MCQuestion(
                    question=item["question"],
                    options=item["options"],
                    correct_option=item["correct_option"],
                    explanation=item["explanation"],
                    source=ctx["source"],
                    chapter=ctx["chapter"],
                    source_text=ctx["source_text"],
                    topic=ctx["topic"],
                )
            )
        return questions
