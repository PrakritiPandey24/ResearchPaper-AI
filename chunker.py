from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from pdf_processor import extract_pdf_pages


# Initial chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def chunk_pdf_pages(
    pages: list[dict[str, str | int]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, str | int]]:
    """
    Split extracted PDF pages into overlapping text chunks.

    Each chunk retains:
    - source: original PDF filename
    - page: original 1-based PDF page number
    - chunk_index: chunk number within that page
    - chunk_id: unique sequential ID within this processing run
    - text: the actual chunk content
    """

    # Validate the chunking settings.
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative "
            "and smaller than chunk_size."
        )

    # Create a splitter that prefers natural text boundaries.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    chunks = []

    # Process each PDF page separately to preserve source pages.
    for page_data in pages:
        page_text = str(page_data["text"]).strip()

        # Empty pages cannot provide useful text chunks.
        if not page_text:
            continue

        page_chunks = splitter.split_text(page_text)

        # Attach the original page metadata to every chunk.
        for chunk_index, chunk_text in enumerate(page_chunks):
            chunks.append(
                {
                    "source": str(page_data["source"]),
                    "page": int(page_data["page"]),
                    "chunk_index": chunk_index,
                    "chunk_id": len(chunks),
                    "text": chunk_text,
                }
            )

    return chunks


if __name__ == "__main__":
    pdf_path = Path("data") / "test_paper_official.pdf"

    # Extract the PDF before splitting its text.
    pages = extract_pdf_pages(pdf_path)

    # Generate the chunks.
    chunks = chunk_pdf_pages(pages)

    print(f"PDF filename: {pdf_path.name}")
    print(f"Pages extracted: {len(pages)}")
    print(f"Chunks generated: {len(chunks)}")
    print(f"Chunk size: {CHUNK_SIZE} characters")
    print(f"Chunk overlap: {CHUNK_OVERLAP} characters")

    # Display a preview of the first five chunks.
    for chunk in chunks[:5]:
        print(
            f"\n--- Chunk {chunk['chunk_id']} "
            f"| Page {chunk['page']} "
            f"| {len(str(chunk['text']))} characters ---"
        )
        print(str(chunk["text"])[:300])

    # Check that each chunk has its source information.
    metadata_complete = all(
        chunk["source"] and
        int(chunk["page"]) >= 1 and
        chunk["text"]
        for chunk in chunks
    )

    print(f"\nAll chunks have valid source metadata: {metadata_complete}")