# AGENT TOOLS


# CALCULATOR TOOL

def calculator(expression):
    """
    Basic calculator tool.

    Note:
    The main Streamlit app currently uses
    its own safe calculator.
    This function is kept for the
    original agent files.
    """

    try:

        allowed = set(
            "0123456789+-*/().% "
        )

        if not all(
            character in allowed
            for character in expression
        ):
            return "Invalid calculation"

        result = eval(
            expression,
            {
                "__builtins__": {}
            },
            {}
        )

        return str(result)

    except Exception:

        return "Invalid calculation"


# PLACEHOLDER PDF SEARCH

def search_pdf(question):
    """
    Kept for compatibility with
    the original agent files.

    PDF search is now handled directly
    inside app.py using session state.
    """

    return (
        "PDF search is handled by "
        "the Streamlit application."
    )


# SOURCE FINDER

def find_sources(question):
    """
    Compatibility function for
    the original agent files.
    """

    return []


# PDF SOURCE TOOL

def get_pdf_sources(question):
    """
    Compatibility function for
    the original agent files.
    """

    return []


# SUMMARIZER TOOL

def summarize_text(text):
    """
    Compatibility function.

    PDF summarization is now handled
    directly by app.py using Gemini.
    """

    return (
        "PDF summarization is handled "
        "by the Streamlit application."
    )
