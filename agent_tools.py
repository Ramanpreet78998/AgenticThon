import json
import ollama


# =========================
# CALCULATOR TOOL
# =========================

def calculator(expression):

    try:
        result = eval(expression)
        return str(result)

    except:
        return "Invalid calculation"


# =========================
# LOAD RAG DATA
# =========================

def load_rag_data():

    try:

        with open(
            "rag_data.json",
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except FileNotFoundError:

        return []


# =========================
# FIND RELEVANT CHUNKS
# =========================

def get_relevant_chunks(question, top_k=3):

    data = load_rag_data()

    if not data:
        return []

    stop_words = {
        "what", "is", "the", "a", "an",
        "are", "of", "in", "on", "to",
        "for", "and", "how", "why",
        "does", "do", "this", "that",
        "tell", "me", "about",
        "which", "pdf", "contains",
        "information"
    }

    question_words = set(
        question.lower()
        .replace("?", "")
        .replace(",", "")
        .split()
    )

    important_words = question_words - stop_words

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
            important_words & chunk_words
        )

        if score > 0:

            scored_chunks.append(
                (score, source, text)
            )

    scored_chunks.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return scored_chunks[:top_k]


# =========================
# RAG SEARCH TOOL
# =========================

def search_pdf(question):

    chunks = get_relevant_chunks(question)

    if not chunks:

        return (
            "No relevant information found "
            "in the uploaded PDFs."
        )

    results = []

    for score, source, text in chunks:

        results.append(
            f"[Source: {source}]\n{text}"
        )

    return "\n\n".join(results)


# =========================
# SOURCE FINDER TOOL
# =========================

def find_sources(question):

    chunks = get_relevant_chunks(
        question,
        top_k=10
    )

    if not chunks:

        return []

    sources = []

    for score, source, text in chunks:

        if source not in sources:

            sources.append(source)

    return sources


# =========================
# PDF SOURCE TOOL
# =========================

def get_pdf_sources(question):

    return find_sources(question)


# =========================
# SUMMARIZER TOOL
# =========================

def summarize_text(text):

    prompt = f"""
Summarize the following text in simple
and clear language.

Text:

{text}

Give a short summary containing
the main points.
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

    return response["message"]["content"]