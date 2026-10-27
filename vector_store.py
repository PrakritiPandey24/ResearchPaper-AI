import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from chunker import chunk_pdf_pages
from pdf_processor import extract_pdf_pages


# Project paths and database settings
PROJECT_DIR = Path(__file__).resolve().parent
PDF_PATH = PROJECT_DIR / "data" / "test_paper_official.pdf"
CHROMA_DB_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "research_paper"

# Google model used to create text embeddings
EMBEDDING_MODEL = "gemini-embedding-001"


def create_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Configure the Gemini embedding model."""

    load_dotenv(PROJECT_DIR / ".env")

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Check your .env file."
        )

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )

def create_vector_store(
    chunks: list[dict[str, str | int]],
) -> Chroma:
    """Embed the chunks and save them in a local Chroma database."""

    if not chunks:
        raise ValueError("No chunks provided for indexing.")

    embeddings = create_embeddings()

    documents = []
    document_ids = []

    for chunk in chunks:
        # LangChain Document stores text and its metadata together.
        document = Document(
            page_content=str(chunk["text"]),
            metadata={
                "source": str(chunk["source"]),
                "page": int(chunk["page"]),
                "chunk_index": int(chunk["chunk_index"]),
                "chunk_id": int(chunk["chunk_id"]),
            },
        )

        documents.append(document)

        # Use a stable ID based on the source page and chunk.
        document_ids.append(
            f"{chunk['source']}_p{chunk['page']}_c{chunk['chunk_index']}"
        )

    # Remove the old local index so the test creates a clean database.
    # This prevents duplicate chunks when this script is rerun.
    if CHROMA_DB_DIR.exists():
        shutil.rmtree(CHROMA_DB_DIR)

    # Chroma stores the vectors, text, and metadata locally.
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        ids=document_ids,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DB_DIR),
    )

    return vector_store


def load_vector_store() -> Chroma:
    """Open an existing local Chroma database."""

    if not CHROMA_DB_DIR.exists():
        raise FileNotFoundError(
            "Chroma database not found. Index a PDF first."
        )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=create_embeddings(),
        persist_directory=str(CHROMA_DB_DIR),
    )


if __name__ == "__main__":
    print("Extracting PDF text...")

    # Step 1: Extract pages from the test PDF.
    pages = extract_pdf_pages(PDF_PATH)

    print(f"Pages extracted: {len(pages)}")

    # Step 2: Split the pages into overlapping chunks.
    chunks = chunk_pdf_pages(pages)

    print(f"Chunks to index: {len(chunks)}")

    # Step 3: Generate embeddings and create the database.
    print("Generating embeddings and building Chroma index...")

    vector_store = create_vector_store(chunks)

    print("Vector store created successfully.")

    # Step 4: Search the database without generating an AI answer.
    query = "What are RAG-Sequence and RAG-Token, and how do they differ?"

    print(f"\nSearch question: {query}")

    results = vector_store.similarity_search(
        query=query,
        k=4,
    )

    print(f"\nRetrieved passages: {len(results)}")

    for index, document in enumerate(results, start=1):
        print(f"\n--- Result {index} ---")
        print(f"Source: {document.metadata['source']}")
        print(f"Page: {document.metadata['page']}")
        print(f"Chunk ID: {document.metadata['chunk_id']}")
        print(document.page_content[:500])