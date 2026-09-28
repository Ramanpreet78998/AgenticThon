import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
import ast
import operator

# PAGE SETTINGS

st.set_page_config(
    page_title="AI Agent Assistant",
    page_icon="🤖",
    layout="wide"
)


# GEMINI CLIENT

@st.cache_resource
def get_gemini_client():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]

        if not api_key:
            return None

        return genai.Client(api_key=api_key)

    except Exception:
        return None


client = get_gemini_client()

MODEL_NAME = "gemini-3.8-flash"


# SESSION STATE

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_data" not in st.session_state:
    st.session_state.rag_data = []


# SAFE CALCULATOR

allowed_operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos
}


def safe_calculate(expression):

    try:

        tree = ast.parse(
            expression,
            mode="eval"
        )

        def evaluate(node):

            if isinstance(node, ast.Constant):

                if isinstance(
                    node.value,
                    (int, float)
                ):
                    return node.value

                raise ValueError(
                    "Invalid number"
                )

            if isinstance(node, ast.BinOp):

                operation = allowed_operators.get(
                    type(node.op)
                )

                if operation is None:
                    raise ValueError(
                        "Invalid operator"
                    )

                return operation(
                    evaluate(node.left),
                    evaluate(node.right)
                )

            if isinstance(node, ast.UnaryOp):

                operation = allowed_operators.get(
                    type(node.op)
                )

                if operation is None:
                    raise ValueError(
                        "Invalid operator"
                    )

                return operation(
                    evaluate(node.operand)
                )

            raise ValueError(
                "Invalid expression"
            )

        return str(
            evaluate(tree.body)
        )

    except Exception:

        return "Invalid calculation."


# PDF RELEVANCE SEARCH

def get_relevant_chunks(
    question,
    data,
    top_k=3
):

    if not data:
        return []

    stop_words = {
        "what",
        "is",
        "the",
        "a",
        "an",
        "are",
        "of",
        "in",
        "on",
        "to",
        "for",
        "and",
        "how",
        "why",
        "does",
        "do",
        "this",
        "that",
        "tell",
        "me",
        "about",
        "which",
        "pdf",
        "contains",
        "information"
    }

    question_words = set(
        question.lower()
        .replace("?", "")
        .replace(",", "")
        .replace(".", "")
        .split()
    )

    important_words = (
        question_words - stop_words
    )

    scored_chunks = []

    for item in data:

        text = item["text"]
        source = item["source"]

        chunk_words = set(
            text.lower()
            .replace(".", " ")
            .replace(",", " ")
            .replace("(", " ")
            .replace(")", " ")
            .split()
        )

        score = len(
            important_words
            & chunk_words
        )

        if score > 0:

            scored_chunks.append(
                (
                    score,
                    source,
                    text
                )
            )

    scored_chunks.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return scored_chunks[:top_k]


# SOURCE FINDER

def find_sources(
    question,
    data
):

    chunks = get_relevant_chunks(
        question,
        data,
        top_k=10
    )

    sources = []

    for score, source, text in chunks:

        if source not in sources:
            sources.append(source)

    return sources


# GEMINI REQUEST

def ask_gemini(prompt):

    if client is None:

        return (
            "Gemini API is not configured.\n\n"
            "Please check that GEMINI_API_KEY "
            "is correctly added in Streamlit Secrets."
        )

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        return response.text

    except Exception as error:

        return (
            "AI request could not be completed.\n\n"
            f"Error: {error}"
        )


# GENERAL AI

def general_ai(question):

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question clearly,
accurately and simply.

Question:
{question}
"""

    return ask_gemini(prompt)


# PDF QUESTION ANSWERING

def answer_from_pdf(
    question,
    data
):

    chunks = get_relevant_chunks(
        question,
        data,
        top_k=5
    )

    if not chunks:

        return (
            "I could not find this information "
            "in the uploaded PDFs."
        ), []

    context_parts = []
    sources = []

    for score, source, text in chunks:

        context_parts.append(text)

        if source not in sources:
            sources.append(source)

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are a PDF question-answering assistant.

Answer ONLY using the PDF context below.

If the answer is not available in the
context, say:

"I could not find this information in the PDF."

PDF CONTEXT:

{context}

QUESTION:

{question}

Give a simple and clear answer.
"""

    answer = ask_gemini(prompt)

    return answer, sources


# PDF SUMMARIZER

def summarize_pdf(
    question,
    data
):

    chunks = get_relevant_chunks(
        question,
        data,
        top_k=8
    )

    if not chunks:

        return (
            "I could not find relevant information "
            "in the uploaded PDFs."
        ), []

    context_parts = []
    sources = []

    for score, source, text in chunks:

        context_parts.append(text)

        if source not in sources:
            sources.append(source)

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
Summarize the following PDF information
in simple and clear language.

Focus on the main points.

PDF INFORMATION:

{context}

USER REQUEST:

{question}

Give a short, useful summary.
"""

    answer = ask_gemini(prompt)

    return answer, sources

# MAIN UI

st.title(
    "🤖 AI Agent Assistant"
)

st.write(
    "An AI agent that understands your request "
    "and selects the appropriate tool."
)

# KNOWLEDGE BASE

st.subheader(
    "📄 Knowledge Base"
)

uploaded_files = st.file_uploader(
    "Upload one or more PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)


if uploaded_files:

    all_chunks = []

    for uploaded_file in uploaded_files:

        try:

            reader = PdfReader(
                uploaded_file
            )

            text = ""

            for page in reader.pages:

                text += (
                    page.extract_text()
                    or ""
                )

            splitter = (
                RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )
            )

            chunks = splitter.split_text(
                text
            )

            for chunk in chunks:

                all_chunks.append(
                    {
                        "source":
                            uploaded_file.name,
                        "text":
                            chunk
                    }
                )

        except Exception as error:

            st.error(
                f"Could not read "
                f"{uploaded_file.name}: "
                f"{error}"
            )

    st.session_state.rag_data = (
        all_chunks
    )

    st.success(
        f"✅ {len(uploaded_files)} PDF(s) loaded! "
        f"{len(all_chunks)} chunks created."
    )

    st.write(
        "### 📚 Uploaded Documents"
    )

    for uploaded_file in uploaded_files:

        st.write(
            f"📄 {uploaded_file.name}"
        )


# CHAT

st.subheader(
    "💬 Ask the Agent"
)


for message in (
    st.session_state.chat_history
):

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


if question:

    question = question.strip()

    question_lower = (
        question.lower()
    )


    # Save user message

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )


    # MATH DETECTION

    math_symbols = [
        "+",
        "-",
        "*",
        "/",
        "%",
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

    # SOURCE REQUEST
  
    source_request = (

        "which pdf"
        in question_lower

        or "which document"
        in question_lower

        or "what pdf"
        in question_lower

        or "find the source"
        in question_lower

        or "find source"
        in question_lower

        or "show the source"
        in question_lower

        or "show source"
        in question_lower

        or "where did you find"
        in question_lower
    )

    # SUMMARY REQUEST
    
    summary_request = (

        "summarize"
        in question_lower

        or "summarise"
        in question_lower

        or "summary"
        in question_lower

        or "give a summary"
        in question_lower

        or "short summary"
        in question_lower

        or "brief"
        in question_lower
    )

    # AGENT ROUTING
    
    if is_math:

        tool_name = (
            "🔧 Calculator"
        )

        answer = safe_calculate(
            question
        )

        sources = []


    elif source_request:

        tool_name = (
            "🔍 Source Finder"
        )

        sources = find_sources(
            question,
            st.session_state.rag_data
        )

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


    elif (
        summary_request
        and st.session_state.rag_data
    ):

        tool_name = (
            "📝 Summarizer"
        )

        answer, sources = summarize_pdf(
            question,
            st.session_state.rag_data
        )


    elif st.session_state.rag_data:

        tool_name = (
            "📚 RAG"
        )

        answer, sources = answer_from_pdf(
            question,
            st.session_state.rag_data
        )


    else:

        tool_name = (
            "🤖 General AI"
        )

        sources = []

        answer = general_ai(
            question
        )


    # SHOW AGENT ACTIVITY

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


# SHOW ANSWER

    st.chat_message(
        "assistant"
    ).write(
        answer
    )


    # SHOW SOURCES
    
    if sources:

        st.subheader(
            "📄 Sources"
        )

        for source in sources:

            st.write(
                f"• {source}"
            )


    # Save assistant message

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


# CLEAR CHAT

st.divider()


if st.button(
    "🗑️ Clear Chat"
):

    st.session_state.chat_history = []

    st.rerun()
