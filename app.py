"""
Financial Compliance Exam Bot - Streamlit Application

Three UI modes driven by st.session_state.mode:
  selection  ->  choose books and number of questions
  exam       ->  answer MCQs one by one with immediate feedback
  results    ->  final score and gap analysis report
"""

from __future__ import annotations

import streamlit as st

from src.config import (
    APP_LOGIN_PASSWORD,
    APP_LOGIN_USERNAME,
    DEFAULT_QUESTION_COUNT,
    LLM_PROVIDER,
    MAX_QUESTION_COUNT,
    MIN_QUESTION_COUNT,
    OLLAMA_MODEL,
    OPENAI_MODEL,
    load_books_config,
    DATA_DIR,
)
from src.core.auth import credentials_configured, validate_login
from src.core.mcq_generator import MCQGenerator
from src.core.pdf_reader import (
    build_inline_pdf_data_url,
    list_pdf_files,
    read_pdf_bytes,
)
from src.core.session_manager import SessionManager, SessionState
from src.db.chroma_client import collection_size, list_unique_values

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Compliance Exam Bot",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .stAlert { border-radius: 8px; }
    div[data-testid="stRadio"] label { font-size: 1rem; line-height: 1.6; }
    .score-badge {
        display: inline-block;
        padding: 0.4em 1em;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.25rem;
        margin-bottom: 0.5rem;
    }
    .score-pass { background: #d1fae5; color: #065f46; }
    .score-fail { background: #fee2e2; color: #991b1b; }
    .q-pill {
        background: #eff6ff;
        color: #1d4ed8;
        padding: 0.2em 0.7em;
        border-radius: 6px;
        font-size: 0.9rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------


def _init_state() -> None:
    defaults: dict = {
        "mode": SessionState.SELECTION,
        "session": None,
        "generator": None,
        "answer_submitted": False,
        "last_record": None,
        "authenticated": False,
        "auth_user": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


_init_state()


def _reset() -> None:
    for key in ["session", "generator", "answer_submitted", "last_record", "_retake"]:
        st.session_state.pop(key, None)
    st.session_state.mode = SessionState.SELECTION
    _init_state()


def _logout() -> None:
    _reset()
    st.session_state.authenticated = False
    st.session_state.auth_user = None


def _generate_quiz_with_progress(
    generator: MCQGenerator, source_tags: list[str], num_questions: int
):
    """Generate quiz questions one-by-one with visible progress."""
    progress = st.progress(0.0)
    status = st.empty()
    questions = []
    used_topics: set[str] = set()

    for i in range(num_questions):
        status.info(f"Generating question {i + 1}/{num_questions}…")
        q = generator.generate_question(source_tags=source_tags, used_topics=used_topics)
        questions.append(q)
        used_topics.add(q.topic)
        progress.progress((i + 1) / num_questions)

    status.success(f"Generated {len(questions)} question(s).")
    return questions


# ---------------------------------------------------------------------------
# Mode: LOGIN
# ---------------------------------------------------------------------------


def render_login() -> None:
    st.title("🔐 Login Required")
    st.markdown("Sign in to access the Compliance Exam Bot.")

    if not credentials_configured(APP_LOGIN_USERNAME, APP_LOGIN_PASSWORD):
        st.error(
            "Login credentials are not configured. "
            "Set `APP_LOGIN_USERNAME` and `APP_LOGIN_PASSWORD` in your `.env` file."
        )
        st.stop()

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_clicked = st.form_submit_button(
            "Login",
            type="primary",
            use_container_width=True,
        )

    if login_clicked:
        is_valid = validate_login(
            input_username=username.strip(),
            input_password=password,
            expected_username=APP_LOGIN_USERNAME,
            expected_password=APP_LOGIN_PASSWORD,
        )
        if is_valid:
            st.session_state.authenticated = True
            st.session_state.auth_user = username.strip()
            st.success("Login successful.")
            st.rerun()
        else:
            st.error("Invalid username or password.")


# ---------------------------------------------------------------------------
# Mode: SELECTION
# ---------------------------------------------------------------------------


def render_selection() -> None:
    st.title("⚖️ Compliance Exam Bot")
    st.markdown(
        "Select the compliance books you want to be tested on, "
        "choose how many questions you want, and start your exam."
    )

    # Database health check
    with st.expander("Database status", expanded=False):
        try:
            size = collection_size()
            loaded_sources = list_unique_values("source")
            if size == 0:
                st.warning(
                    "The vector database is empty.  \n"
                    "Drop PDF files into **./data/** and run:  \n"
                    "```\npython scripts/ingest_pdfs.py\n```"
                )
            else:
                st.success(
                    f"**{size:,}** chunks loaded across "
                    f"**{len(loaded_sources)}** source(s)."
                )
                st.caption("Sources in DB: " + ", ".join(f"`{s}`" for s in loaded_sources))
        except Exception as exc:
            st.error(f"Could not connect to ChromaDB: {exc}")

    st.divider()

    with st.expander("Read PDFs in data/", expanded=False):
        pdf_files = list_pdf_files(DATA_DIR)
        if not pdf_files:
            st.caption("No PDF files found in `data/`.")
        else:
            selected_pdf_name = st.selectbox(
                "Choose a PDF to preview",
                options=[p.name for p in pdf_files],
                key="pdf_preview_select",
            )
            selected_pdf = next(p for p in pdf_files if p.name == selected_pdf_name)
            try:
                pdf_bytes = read_pdf_bytes(selected_pdf)
                st.download_button(
                    label="Download selected PDF",
                    data=pdf_bytes,
                    file_name=selected_pdf.name,
                    mime="application/pdf",
                    use_container_width=True,
                )
                data_url = build_inline_pdf_data_url(selected_pdf)
                st.markdown(
                    (
                        "<iframe "
                        f"src='{data_url}' "
                        "width='100%' "
                        "height='700' "
                        "style='border: 1px solid #ddd; border-radius: 8px;'>"
                        "</iframe>"
                    ),
                    unsafe_allow_html=True,
                )
            except Exception as exc:
                st.error(f"Could not open `{selected_pdf.name}`: {exc}")

    # Load books from config
    books = load_books_config()
    if not books:
        st.error(
            "No books found in **books_config.json**. "
            "Add entries or run the ingestion script first."
        )
        return

    st.subheader("Select books to be tested on")

    try:
        db_sources = set(list_unique_values("source"))
    except Exception:
        db_sources = set()

    available_books = [b for b in books if not db_sources or b["source_tag"] in db_sources]
    unavailable_books = [b for b in books if db_sources and b["source_tag"] not in db_sources]

    if not available_books:
        st.warning(
            "None of the configured books have been ingested yet.  \n"
            "Run `python scripts/ingest_pdfs.py` first, then refresh this page."
        )
        if unavailable_books:
            st.caption(
                "Configured but not in DB: "
                + ", ".join(b["display_name"] for b in unavailable_books)
            )
        return

    selected_display_names: list[str] = []
    for book in available_books:
        col_cb, col_text = st.columns([0.05, 0.95])
        with col_cb:
            checked = st.checkbox(
                label=book["display_name"],
                key=f"book_cb_{book['source_tag']}",
                label_visibility="collapsed",
            )
        with col_text:
            st.markdown(f"**{book['display_name']}**  \n{book.get('description', '')}")
        if checked:
            selected_display_names.append(book["display_name"])

    if unavailable_books:
        with st.expander(f"{len(unavailable_books)} book(s) configured but not yet ingested"):
            for b in unavailable_books:
                st.caption(f"• {b['display_name']} (`{b['source_tag']}`)")

    st.divider()

    st.subheader("Number of questions")
    num_questions = st.slider(
        "Questions",
        min_value=MIN_QUESTION_COUNT,
        max_value=MAX_QUESTION_COUNT,
        value=DEFAULT_QUESTION_COUNT,
        step=1,
        label_visibility="collapsed",
    )

    st.divider()

    can_start = len(selected_display_names) > 0
    start_clicked = st.button(
        "Start Exam →",
        type="primary",
        disabled=not can_start,
        use_container_width=True,
    )

    if not can_start:
        st.caption("Select at least one book to begin.")

    if start_clicked and can_start:
        source_tags = [
            b["source_tag"]
            for b in books
            if b["display_name"] in selected_display_names
        ]
        session = SessionManager(
            selected_books=selected_display_names,
            selected_source_tags=source_tags,
        )
        generator = MCQGenerator()

        st.info(
            "Question generation may take several minutes "
            f"for {num_questions} questions."
        )
        try:
            questions = _generate_quiz_with_progress(
                generator=generator,
                source_tags=source_tags,
                num_questions=num_questions,
            )
        except Exception as exc:
            if LLM_PROVIDER == "openai":
                st.error(
                    f"**Failed to generate questions:** {exc}  \n\n"
                    "Make sure `OPENAI_API_KEY` is set and your model "
                    f"`{OPENAI_MODEL}` is available on your OpenAI account."
                )
            else:
                st.error(
                    f"**Failed to generate questions:** {exc}  \n\n"
                    "Make sure **Ollama is running** (`ollama serve`) and the model "
                    f"`{OLLAMA_MODEL}` is available (`ollama pull {OLLAMA_MODEL}`)."
                )
            return

        if not questions:
            if LLM_PROVIDER == "openai":
                st.error(
                    "Could not generate any questions. "
                    "Check that the selected books are ingested and your OpenAI settings are valid."
                )
            else:
                st.error(
                    "Could not generate any questions. "
                    "Check that the selected books are ingested and Ollama is running."
                )
            return

        session.questions = questions
        st.session_state.session = session
        st.session_state.generator = generator
        st.session_state.mode = SessionState.EXAM
        st.session_state.answer_submitted = False
        st.session_state.last_record = None
        st.rerun()


# ---------------------------------------------------------------------------
# Mode: EXAM
# ---------------------------------------------------------------------------


def render_exam() -> None:
    session: SessionManager = st.session_state.session

    if session.is_complete:
        st.session_state.mode = SessionState.RESULTS
        st.rerun()
        return

    q = session.current_question
    q_num = session.current_index + 1
    total = session.total_questions

    # Header row: question counter + end exam button
    col_left, col_right = st.columns([0.75, 0.25])
    with col_left:
        st.markdown(
            f'<span class="q-pill">Question {q_num} of {total}</span>',
            unsafe_allow_html=True,
        )
    with col_right:
        if st.button("End Exam", type="secondary"):
            st.session_state.mode = SessionState.RESULTS
            st.rerun()

    # Progress bar (advances after submission)
    progress_value = (session.current_index) / total
    st.progress(progress_value)

    st.caption(f"Books: {', '.join(session.selected_books)}")
    st.divider()

    # Question text
    st.markdown(f"### {q.question}")
    st.caption(f"Topic area: *{q.topic}*")
    st.markdown("")

    answer_submitted: bool = st.session_state.answer_submitted

    if not answer_submitted:
        # Radio: no default selection forces a conscious choice
        selected = st.radio(
            "Choose your answer:",
            options=list(q.options.keys()),
            format_func=lambda k: f"{k}.  {q.options[k]}",
            key=f"radio_q{session.current_index}",
            index=None,
        )

        col_submit, _ = st.columns([0.35, 0.65])
        with col_submit:
            submit_clicked = st.button(
                "Submit Answer",
                type="primary",
                disabled=(selected is None),
            )

        if submit_clicked and selected:
            record = session.record_answer(selected)
            st.session_state.answer_submitted = True
            st.session_state.last_record = record
            st.rerun()

    else:
        record = st.session_state.last_record

        # Render all options with colour-coded feedback
        for letter, text in q.options.items():
            is_correct_answer = letter == record.correct_option
            is_user_choice = letter == record.selected_option

            if is_correct_answer and is_user_choice:
                st.success(f"**{letter}.** {text}  — ✓ Your answer (Correct)")
            elif is_correct_answer:
                st.success(f"**{letter}.** {text}  — ✓ Correct answer")
            elif is_user_choice:
                st.error(f"**{letter}.** {text}  — ✗ Your answer (Incorrect)")
            else:
                st.markdown(f"**{letter}.** {text}")

        st.markdown("")

        # Overall verdict
        if record.is_correct:
            st.success("**Correct!** Well done.")
        else:
            st.error(
                f"**Incorrect.** The right answer was **{record.correct_option}**."
            )

        # Explanation panel (auto-expanded)
        with st.expander("📖 Explanation & source reference", expanded=True):
            st.markdown(record.explanation)
            st.divider()
            meta_parts = [f"Source: `{record.source}`"]
            if record.chapter:
                meta_parts.append(f"Chapter: `{record.chapter}`")
            st.caption("  |  ".join(meta_parts))

            with st.expander("View the source text used to generate this question"):
                preview = record.source_text[:900]
                if len(record.source_text) > 900:
                    preview += "\n\n[… truncated …]"
                st.code(preview, language=None)

        st.markdown("")

        # Advance button
        is_last = session.current_index >= session.total_questions
        btn_label = "See Results →" if is_last else "Next Question →"
        if st.button(btn_label, type="primary", use_container_width=True):
            st.session_state.answer_submitted = False
            st.session_state.last_record = None
            if is_last:
                st.session_state.mode = SessionState.RESULTS
            st.rerun()


# ---------------------------------------------------------------------------
# Mode: RESULTS
# ---------------------------------------------------------------------------


def render_results() -> None:
    session: SessionManager = st.session_state.session
    summary = session.get_summary()
    pct = summary["score_percentage"]
    score = summary["correct"]
    total = summary["total_questions"]
    gaps = summary["gaps"]

    st.title("📊 Exam Results")

    pass_threshold = 70.0
    badge_class = "score-pass" if pct >= pass_threshold else "score-fail"
    result_label = "PASS" if pct >= pass_threshold else "FAIL"

    st.markdown(
        f"""
        <div style="text-align:center; padding: 1.5rem 0 1rem;">
            <div class="score-badge {badge_class}">
                {result_label} &mdash; {score}/{total} &nbsp;({pct}%)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Score", f"{pct}%")
    col2.metric("Correct", f"{score} / {total}")
    col3.metric("Incorrect", f"{total - score} / {total}")

    st.caption(f"Books tested: {', '.join(summary['selected_books'])}")
    st.divider()

    # Per-question review (collapsed by default)
    with st.expander("Question-by-question review", expanded=False):
        for answer in session.answers:
            icon = "✅" if answer.is_correct else "❌"
            verdict = "Correct" if answer.is_correct else f"Incorrect (correct: **{answer.correct_option}**)"
            st.markdown(
                f"{icon} **Q{answer.question_index + 1}** — "
                f"{answer.question_text[:130]}…  \n"
                f"Your answer: **{answer.selected_option}** — {verdict}"
            )

    st.divider()

    # Gap analysis
    st.subheader("Gap Analysis")

    if not gaps:
        st.success(
            "Excellent work! You answered every question correctly — "
            "no knowledge gaps identified."
        )
    else:
        st.markdown(
            f"You missed questions in **{len(gaps)}** topic area(s). "
            "Review the following material before retaking the exam:"
        )
        for i, gap in enumerate(gaps):
            chapter_label = f" › {gap.chapter}" if gap.chapter else ""
            with st.container():
                st.markdown(
                    f"**{i + 1}. {gap.source}{chapter_label}** — "
                    f"missed {gap.missed_count} of {gap.total_asked} question(s)"
                )
                st.info(f"💡 {gap.recommendation}")

    st.divider()

    # Action buttons
    col_retake, col_change = st.columns(2)
    with col_retake:
        if st.button("🔁 Retake with same books", use_container_width=True):
            st.session_state["_retake"] = {
                "books": session.selected_books,
                "tags": session.selected_source_tags,
                "count": session.total_questions,
            }
            _reset()
            st.rerun()
    with col_change:
        if st.button("📚 Change books", use_container_width=True, type="primary"):
            _reset()
            st.rerun()

    # Handle retake flow (triggered on next render after _reset clears session)
    if "_retake" in st.session_state:
        retake = st.session_state.pop("_retake")
        new_session = SessionManager(
            selected_books=retake["books"],
            selected_source_tags=retake["tags"],
        )
        generator = MCQGenerator()
        st.info("Regenerating questions; this can take a few minutes.")
        try:
            questions = _generate_quiz_with_progress(
                generator=generator,
                source_tags=retake["tags"],
                num_questions=retake["count"],
            )
        except Exception as exc:
            st.error(f"Failed to regenerate questions: {exc}")
            return
        new_session.questions = questions
        st.session_state.session = new_session
        st.session_state.generator = generator
        st.session_state.mode = SessionState.EXAM
        st.session_state.answer_submitted = False
        st.session_state.last_record = None
        st.rerun()


# ---------------------------------------------------------------------------
# Router — top-level dispatch
# ---------------------------------------------------------------------------

mode = st.session_state.mode

if not st.session_state.authenticated:
    render_login()
else:
    with st.sidebar:
        user_label = st.session_state.auth_user or "authenticated user"
        st.caption(f"Signed in as **{user_label}**")
        if st.button("Log out", use_container_width=True):
            _logout()
            st.rerun()

    if mode == SessionState.SELECTION:
        render_selection()
    elif mode == SessionState.EXAM:
        render_exam()
    elif mode == SessionState.RESULTS:
        render_results()
    else:
        st.error(f"Unknown mode: {mode}")
        if st.button("Reset"):
            _reset()
            st.rerun()
