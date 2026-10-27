import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from chunker import chunk_pdf_pages
from pdf_processor import extract_pdf_pages
from rag_pipeline import answer_question
from vector_store import create_vector_store


PROJECT_DIR = Path(__file__).resolve().parent
CHROMA_DB_DIR = PROJECT_DIR / "chroma_db"


st.set_page_config(
    page_title="ResearchPaper AI",
    page_icon="📚",
    layout="wide",
)

st.title("📚 ResearchPaper AI")
st.caption(
    "Ask questions about a research paper and inspect "
    "the passages used to generate each answer."
)


# Initialize the application's session state.
if "indexed_fingerprint" not in st.session_state:
    st.session_state["indexed_fingerprint"] = None

if "indexed_source" not in st.session_state:
    st.session_state["indexed_source"] = None

if "indexed_page_count" not in st.session_state:
    st.session_state["indexed_page_count"] = 0

if "indexed_chunk_count" not in st.session_state:
    st.session_state["indexed_chunk_count"] = 0

if "messages" not in st.session_state:
    st.session_state["messages"] = []


# Ask the user to upload a PDF.
uploaded_file = st.file_uploader(
    "Upload a research paper",
    type=["pdf"],
    help="Upload one text-based research paper at a time.",
)

if uploaded_file is None:
    st.info("Upload a PDF to start asking questions.")
    st.stop()


# Read the PDF bytes and create a fingerprint.
pdf_bytes = uploaded_file.getvalue()

fingerprint_input = (
    uploaded_file.name.encode("utf-8")
    + b"\0"
    + pdf_bytes
)

file_fingerprint = hashlib.sha256(
    fingerprint_input
).hexdigest()


# Index only when the uploaded file changes.
needs_indexing = (
    file_fingerprint != st.session_state["indexed_fingerprint"]
    or not CHROMA_DB_DIR.exists()
)

if needs_indexing:
    temporary_pdf_path = None

    try:
        with st.spinner(
            "Extracting text, creating chunks, "
            "and indexing your paper..."
        ):
            # Save the uploaded bytes to a temporary PDF.
            with tempfile.NamedTemporaryFile(
                suffix=".pdf",
                delete=False,
            ) as temporary_file:
                temporary_file.write(pdf_bytes)
                temporary_pdf_path = Path(temporary_file.name)

            # Stage 3: extract text page by page.
            pages = extract_pdf_pages(temporary_pdf_path)

            if not pages:
                raise ValueError("The uploaded PDF contains no pages.")

            pages_with_text = [
                page for page in pages
                if str(page["text"]).strip()
            ]

            if not pages_with_text:
                raise ValueError(
                    "No extractable text was found. "
                    "This PDF may require OCR."
                )

            # Stage 4: create overlapping chunks.
            chunks = chunk_pdf_pages(pages)

            if not chunks:
                raise ValueError(
                    "No searchable text chunks could be created."
                )

            # Replace the temporary filename with the actual filename.
            for chunk in chunks:
                chunk["source"] = uploaded_file.name

            # Stage 5: embed the chunks and create the Chroma index.
            create_vector_store(chunks)

        # Update state only after successful indexing.
        st.session_state["indexed_fingerprint"] = file_fingerprint
        st.session_state["indexed_source"] = uploaded_file.name
        st.session_state["indexed_page_count"] = len(pages)
        st.session_state["indexed_chunk_count"] = len(chunks)
        st.session_state["messages"] = []

    except Exception as exc:
        st.error(
            f"Could not process the PDF "
            f"({type(exc).__name__}): {exc}"
        )
        st.stop()

    finally:
        # Remove the temporary PDF after processing.
        if (
            temporary_pdf_path is not None
            and temporary_pdf_path.exists()
        ):
            temporary_pdf_path.unlink()


# Show the active paper's processing information.
st.success(
    f"Indexed: {st.session_state['indexed_source']}"
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Pages",
        st.session_state["indexed_page_count"],
    )

with col2:
    st.metric(
        "Searchable chunks",
        st.session_state["indexed_chunk_count"],
    )


def display_sources(sources: list[dict]) -> None:
    """Display passages retrieved from Chroma."""

    if not sources:
        return

    with st.expander(
        f"Inspect {len(sources)} retrieved source passages"
    ):
        for index, source in enumerate(sources, start=1):
            st.markdown(
                f"**Passage {index}: "
                f"{source.get('source', 'Unknown source')} — "
                f"PDF page {source.get('page', 'Unknown')}**"
            )

            st.caption(
                f"Chunk ID: {source.get('chunk_id', 'Unknown')}"
            )

            st.write(source.get("text", ""))

            if index < len(sources):
                st.divider()


# Restore previous messages after Streamlit reruns.
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            display_sources(message.get("sources", []))


# Accept a new question.
question = st.chat_input(
    "Ask a question about the uploaded paper..."
)

if question:
    # Display and save the user's question.
    st.session_state["messages"].append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve passages and generate an answer.
    with st.chat_message("assistant"):
        try:
            with st.spinner(
                "Searching the paper and generating an answer..."
            ):
                result = answer_question(question)

            answer = result["answer"]
            sources = result["sources"]

            st.markdown(answer)
            display_sources(sources)

            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )

        except Exception as exc:
            # Keep the displayed message useful while logging
            # the underlying error in the terminal.
            print(
                f"Answer generation failed: "
                f"{type(exc).__name__}: {exc}"
            )

            error_message = (
                "I couldn't generate an answer this time. "
                "Check the terminal for the underlying error, "
                "then try again."
            )

            st.error(error_message)

            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                }
            )