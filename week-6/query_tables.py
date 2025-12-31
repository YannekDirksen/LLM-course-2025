from llmsherpa.readers import LayoutPDFReader
from llama_index.llms.ollama import Ollama

# -----------------------------
# Configuration
# -----------------------------

llm = Ollama(model="llama3", request_timeout=60.0)

llmsherpa_api_url = "http://localhost:5010/api/parseDocument?renderFormat=all"
pdf_url = (
    "https://s206.q4cdn.com/479360582/files/doc_financials/2024/q1/"
    "2024q1-alphabet-earnings-release-pdf.pdf"
)

# -----------------------------
# Read PDF via llmsherpa (Docker)
# -----------------------------

pdf_reader = LayoutPDFReader(llmsherpa_api_url)
doc = pdf_reader.read_pdf(pdf_url)

# -----------------------------
# Task 1a:
# Load ALL sections that contain tables
# -----------------------------

table_sections = []
for section in doc.sections():
    html = section.to_html(include_children=True, recurse=True)
    if "<table" in html.lower():
        table_sections.append(section)

print(f"Found {len(table_sections)} sections containing tables.")

# -----------------------------
# Helper functions
# -----------------------------

def score_section(section, question: str) -> int:
    """
    Very simple keyword-based relevance scoring.
    This avoids hard-coding section titles while
    still picking the most relevant table.
    """
    q = question.lower()
    title = (section.title or "").lower()
    html = section.to_html(include_children=True, recurse=True).lower()

    keywords = [
        "revenue", "revenues",
        "operating", "operating income",
        "margin",
        "net income",
        "cost", "expenses"
    ]

    score = 0
    for kw in keywords:
        if kw in q and kw in title:
            score += 5
        if kw in q and kw in html:
            score += 1

    return score


def pick_best_table_section(sections, question: str):
    scored = [(score_section(s, question), s) for s in sections]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def ask(question: str):
    """
    Ask a natural-language question over the most
    relevant table section.
    """
    best_section = pick_best_table_section(table_sections, question)
    context = best_section.to_html(include_children=True, recurse=True)

    prompt = (
        "You are given ONE HTML table extracted from a PDF.\n"
        "Answer ONLY the question using ONLY this table.\n"
        "If a calculation is required, show the formula and the exact numbers used.\n"
        "If the table does not contain the needed values, say 'Not found in this table'.\n\n"
        f"QUESTION: {question}\n\n"
        f"TABLE:\n{context}"
    )

    response = llm.complete(prompt)

    print("\n" + "=" * 80)
    print("QUESTION:", question)
    print("SECTION TITLE:", best_section.title)
    print(response.text)


# -----------------------------
# Task 1b:
# Test reasoning with table data
# -----------------------------

ask(
    "What was Google's operating margin for Q1 2024? "
    "If the margin is not directly stated, compute it as "
    "operating income divided by revenues."
)

ask(
    "What percentage of revenues is net income in Q1 2024? "
    "Show the calculation."
)
