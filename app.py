import streamlit as st
import json
import ollama

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent_tools import (
    calculator,
    search_pdf,
    summarize_text,
    get_pdf_sources,
    find_sources
)


# =========================
# PAGE SETTINGS
# =========================

st.set_page_config(
    page_title="AI Agent Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Agent Assistant")

st.write(
    "An AI agent that selects the appropriate "
    "tool to solve your request."
)


# =========================
# CHAT HISTORY
# =========================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =========================
# KNOWLEDGE BASE
# =========================

st.subheader("📄 Knowledge Base")

uploaded_files = st.file_uploader(
    "Upload one or more PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    all_chunks = []

    for uploaded_file in uploaded_files:

        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_text(text)

        for chunk in chunks:

            all_chunks.append({
                "source": uploaded_file.name,
                "text": chunk
            })

    with open(
        "rag_data.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    st.success(
        f"✅ {len(uploaded_files)} PDF(s) loaded! "
        f"{len(all_chunks)} chunks created."
    )

    st.write("### 📚 Uploaded Documents")

    for uploaded_file in uploaded_files:

        st.write(
            f"📄 {uploaded_file.name}"
        )


# =========================
# CHAT AREA
# =========================

st.subheader("💬 Ask the Agent")

for message in st.session_state.chat_history:

    if message["role"] == "user":

        st.chat_message(
            "user"
        ).write(
            message["content"]
        )

    else:

        st.chat_message(
            "assistant"
        ).write(
            message["content"]
        )


question = st.chat_input(
    "Ask something..."
)


# =========================
# AGENT PROCESSING
# =========================

if question:

    question = question.strip()

    question_lower = question.lower()


    # =========================
    # SAVE USER QUESTION
    # =========================

    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })


    # =========================
    # CALCULATOR DETECTION
    # =========================

    math_symbols = [
        "+",
        "-",
        "*",
        "/",
        "%"
    ]

    has_number = any(
        character.isdigit()
        for character in question
    )

    has_math_symbol = any(
        symbol in question
        for symbol in math_symbols
    )

    only_math = all(
        character.isdigit()
        or character.isspace()
        or character in "+-*/().%"
        for character in question
    )

    is_math = (
        has_number
        and has_math_symbol
        and only_math
    )


    # =========================
    # SOURCE FINDER DETECTION
    # =========================

    source_request = (
        "which pdf" in question_lower
        or "which document" in question_lower
        or "what pdf" in question_lower
        or "find the source" in question_lower
        or "find source" in question_lower
        or "show the source" in question_lower
        or "show source" in question_lower
        or "where did you find" in question_lower
    )


    # =========================
    # SUMMARIZER DETECTION
    # =========================

    summary_request = (
        "summarize" in question_lower
        or "summarise" in question_lower
        or "summary" in question_lower
        or "give a summary" in question_lower
        or "short summary" in question_lower
        or "brief" in question_lower
    )


    # =========================
    # TOOL SELECTION
    # =========================

    # ---------------------------------
    # CALCULATOR
    # ---------------------------------

    if is_math:

        tool_name = "🔧 Calculator"

        answer = calculator(question)

        sources = []


    # ---------------------------------
    # SOURCE FINDER
    # ---------------------------------

    elif source_request:

        tool_name = "🔍 Source Finder"

        sources = find_sources(question)

        if sources:

            answer = (
                "The relevant information "
                "was found in:\n\n"
            )

            for source in sources:

                answer += (
                    f"📄 {source}\n"
                )

        else:

            answer = (
                "I could not find a relevant "
                "source in the uploaded PDFs."
            )


    # ---------------------------------
    # SUMMARIZER
    # ---------------------------------

    elif summary_request:

        tool_name = "📝 Summarizer"

        context = search_pdf(question)

        answer = summarize_text(context)

        sources = get_pdf_sources(question)


    # ---------------------------------
    # RAG
    # ---------------------------------

    else:

        tool_name = "📚 RAG"

        context = search_pdf(question)

        sources = get_pdf_sources(question)

        prompt = f"""
You are a helpful PDF question-answering assistant.

Answer the question using ONLY the information
provided in the PDF context.

If the answer is not available in the PDF context,
say:

"I could not find this information in the PDF."

PDF CONTEXT:

{context}

QUESTION:

{question}

Give a simple and clear answer.
"""

        response = ollama.chat(
            model="qwen2.5:0.5b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response[
            "message"
        ][
            "content"
        ]


    # =========================
    # AGENT RESULT
    # =========================

    st.info(
        f"🤖 Agent selected: {tool_name}"
    )


    with st.expander(
        "🔎 Agent Activity"
    ):

        st.write(
            "✓ Question received"
        )

        st.write(
            f"✓ Tool selected: {tool_name}"
        )

        st.write(
            "✓ Tool executed successfully"
        )

        st.write(
            "✓ Final answer generated"
        )


    # =========================
    # FINAL ANSWER
    # =========================

    st.chat_message(
        "assistant"
    ).write(answer)


    # =========================
    # SOURCES
    # =========================

    if sources:

        st.subheader("📄 Sources")

        for source in sources:

            st.write(
                f"• {source}"
            )


    # =========================
    # SAVE ANSWER
    # =========================

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })


# =========================
# CLEAR CHAT
# =========================

st.divider()

if st.button("🗑️ Clear Chat"):

    st.session_state.chat_history = []

    st.rerun()