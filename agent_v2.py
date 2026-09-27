import re
from agent_tools import calculator, search_pdf, summarize_text


question = input("Ask something: ")

question_lower = question.lower()


# =========================
# Calculator
# =========================

math_pattern = r"^\s*[\d\s\+\-\*\/\(\)\.\%]+\s*$"

if re.fullmatch(math_pattern, question):

    print("🤖 Agent decision: calculator")

    result = calculator(question)

    print("🔧 Calculator Tool Used")
    print("Answer:", result)


# =========================
# Summarizer
# =========================

elif any(word in question_lower for word in [
    "summarize",
    "summary",
    "short summary",
    "brief"
]):

    print("🤖 Agent decision: summarizer")

    context = search_pdf(question)

    print("📝 Summarizer Tool Used")

    result = summarize_text(context)

    print("🤖 Summary:")
    print(result)


# =========================
# RAG
# =========================

else:

    print("🤖 Agent decision: rag")

    context = search_pdf(question)

    print("📚 RAG Tool Used")

    prompt = f"""
You are a helpful PDF question-answering assistant.

Answer the question using ONLY the PDF information below.

If the answer is not available, say:
"I could not find this information in the PDF."

PDF Context:
{context}

Question:
{question}

Give a simple and clear answer.
"""

    import ollama

    response = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("🤖 Final Answer:")
    print(response["message"]["content"])