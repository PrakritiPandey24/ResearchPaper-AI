
from datetime import datetime
from pathlib import Path
import re

from rag_pipeline import answer_question


PROJECT_DIR = Path(__file__).resolve().parent
REPORT_PATH = PROJECT_DIR / "evaluation_results.md"


# Each group contains alternative terms that can indicate
# whether an expected concept appears in an answer.
TESTS = [
    {
        "id": "T1",
        "category": "Answerable",
        "question": "What retrieval model does the paper use, and how does it work?",
        "groups": [
            ["DPR", "Dense Passage Retriever"],
            ["query encoder", "query representation", "query vector"],
            ["document encoder", "document representation", "document vector"],
        ],
    },
    {
        "id": "T2",
        "category": "Answerable",
        "question": "How do RAG-Sequence and RAG-Token differ?",
        "groups": [
            ["RAG-Sequence"],
            ["RAG-Token"],
            ["same retrieved document", "same document for the entire", "same document"],
            ["different document", "different documents"],
        ],
    },
    {
        "id": "T3",
        "category": "Answerable",
        "question": "Which generator model do the authors use?",
        "groups": [
            ["BART"],
        ],
    },
    {
        "id": "T4",
        "category": "Answerable",
        "question": "Which datasets were used to train the DPR retriever?",
        "groups": [
            ["TriviaQA"],
            ["Natural Questions"],
        ],
    },
    {
        "id": "T5",
        "category": "Answerable",
        "question": "What is Maximum Inner Product Search and what is it used for?",
        "groups": [
            ["MIPS", "Maximum Inner Product Search"],
            ["retrieve", "search"],
            ["document", "documents"],
        ],
    },
    {
        "id": "T6",
        "category": "Answerable",
        "question": "When do RAG-Sequence and RAG-Token become equivalent?",
        "groups": [
            ["sequence of length one", "target sequence of length one", "length-one sequence"],
            ["classification", "target class"],
        ],
    },
    {
        "id": "T7",
        "category": "Unanswerable",
        "question": "What is the lead author's personal phone number?",
    },
    {
        "id": "T8",
        "category": "Unanswerable",
        "question": "What exact F1 score did the authors achieve on an MRI diagnosis benchmark?",
    },
    {
        "id": "T9",
        "category": "Unanswerable",
        "question": "Give me a chocolate cake recipe.",
    },
]


ABSTENTION_PHRASES = [
    "not provided",
    "does not provide",
    "doesn't provide",
    "not stated",
    "not specified",
    "not mentioned",
    "does not mention",
    "not in the paper",
    "not included in the paper",
    "does not contain",
    "doesn't contain",
    "does not discuss",
    "doesn't discuss",
    "does not address",
    "doesn't address",
    "not enough information",
    "insufficient information",
    "cannot determine",
    "can't determine",
    "cannot be determined",
    "unable to determine",
    "cannot answer",
    "can't answer",
    "not available in the paper",
    "outside the scope",
    "unrelated to the paper",
    "not related to the paper",
    "the paper focuses on",
    "insufficient to answer",
    "does not establish this result",
    "no mention of",
]


def matching_groups(text, groups):
    """Count expected concept groups found in a piece of text."""
    text = text.casefold()

    return sum(
        any(term.casefold() in text for term in group)
        for group in groups
    )


def contains_abstention(answer):
    """Check for wording that indicates insufficient evidence."""
    text = answer.casefold()

    return any(
        phrase in text
        for phrase in ABSTENTION_PHRASES
    )


def evaluate_test(test, result):
    answer = result["answer"]
    sources = result.get("sources", [])

    answer_lower = answer.casefold()

    source_text = "\n".join(
        str(source.get("text", ""))
        for source in sources
    )

    pages = sorted({
        str(source.get("page", "?"))
        for source in sources
    })

    lines = [
        f"### {test['id']} — {test['category']}",
        "",
        f"**Question:** {test['question']}",
        "",
    ]

    if test["category"] == "Answerable":
        groups = test["groups"]

        answer_hits = matching_groups(answer, groups)
        evidence_hits = matching_groups(source_text, groups)

        if answer_hits == len(groups):
            answer_status = "PASS — expected terms found"
        elif answer_hits > 0:
            answer_status = "PARTIAL — some expected terms found"
        else:
            answer_status = "FAIL — expected terms not found"

        lines.extend([
            f"- Answer keyword check: **{answer_status}** "
            f"({answer_hits}/{len(groups)} concept groups)",
            f"- Retrieved evidence keyword coverage: "
            f"{evidence_hits}/{len(groups)} concept groups",
            "- Note: These are keyword-based checks, not a "
            "proof of factual correctness.",
        ])

        # Basic check: do explicitly numbered citation pages
        # appear among the retrieved source pages?
        cited_pages = set(
            re.findall(
                r"\b(?:PDF\s+)?(?:page|p)\s*[:.]?\s*(\d+)",
                answer,
                flags=re.IGNORECASE,
            )
        )

        if cited_pages:
            citation_match = cited_pages.issubset(set(pages))
            citation_status = (
                "Cited page numbers appear in retrieved sources"
                if citation_match
                else "Some cited page numbers are not in retrieved sources"
            )
        else:
            citation_status = "No recognizable page citation detected"

        lines.append(f"- Citation consistency check: {citation_status}")
        lines.append(
            "- Manual check: verify that cited passages actually "
            "support the claims."
        )

    else:
        abstained = contains_abstention(answer)

        status = (
            "PASS — insufficient evidence acknowledged"
            if abstained
            else "REVIEW — no recognized abstention wording"
        )

        lines.extend([
            f"- Abstention wording check: **{status}**",
            "- Note: This checks wording only. A manual review "
            "may still be needed to detect unsupported claims.",
        ])

    lines.extend([
        f"- Retrieved source pages: {', '.join(pages) if pages else 'None'}",
        "",
        "**Generated answer:**",
        "",
        answer,
        "",
        "---",
        "",
    ])

    return lines


def main():
    report = [
        "# ResearchPaper AI — Automated Evaluation",
        "",
        f"**Run date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"**Total test cases:** {len(TESTS)}",
        "",
        "This report uses simple keyword and abstention checks. "
        "These checks help identify likely problems but do not "
        "establish factual correctness or citation support.",
        "",
        "## Results",
        "",
    ]

    print(f"Starting {len(TESTS)} evaluation tests...\n")

    for index, test in enumerate(TESTS, start=1):
        print(f"[{index}/{len(TESTS)}] {test['question']}")

        try:
            result = answer_question(test["question"])
            test_report = evaluate_test(test, result)

        except Exception as error:
            print(f"  ERROR: {type(error).__name__}: {error}")

            test_report = [
                f"### {test['id']} — ERROR",
                "",
                f"**Question:** {test['question']}",
                "",
                f"**Error:** `{type(error).__name__}: {error}`",
                "",
                "---",
                "",
            ]

        report.extend(test_report)

        # Save after every test so partial results are not lost.
        REPORT_PATH.write_text(
            "\n".join(report),
            encoding="utf-8",
        )

        print(f"  Report updated: {REPORT_PATH.name}\n")

    print("Evaluation finished.")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
