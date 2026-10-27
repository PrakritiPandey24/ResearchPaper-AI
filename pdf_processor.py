from pathlib import Path

import pymupdf


def extract_pdf_pages(
    pdf_path: str | Path,
) -> list[dict[str, str | int]]:
    """
    Extract text from every page of a PDF.

    Each page is returned with its source filename,
    1-based page number, and extracted text.
    """
    path = Path(pdf_path)

    # 1. Confirm that the PDF exists.
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    extracted_pages = []

    # 2. Open the PDF and process each page.
    with pymupdf.open(str(path)) as document:

        if document.is_encrypted:
            raise ValueError(
                "The PDF is encrypted. Use an unprotected PDF."
            )

        for page_number, page in enumerate(
            document, start=1
        ):
            # 3. Extract readable text from the current page.
            text = page.get_text("text").strip()

            # 4. Keep the text and its source information together.
            extracted_pages.append(
                {
                    "source": path.name,
                    "page": page_number,
                    "text": text,
                }
            )

    return extracted_pages


if __name__ == "__main__":
    # Use the official PDF that we confirmed works.
    pdf_path = Path("data") / "test_paper_official.pdf"

    try:
        pages = extract_pdf_pages(pdf_path)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"PDF processing failed: {error}")

    print(f"PDF filename: {pdf_path.name}")
    print(f"Total pages: {len(pages)}")

    empty_pages = [
        page for page in pages if not page["text"]
    ]

    print(f"Pages with no extractable text: {len(empty_pages)}")

    # Display a preview of the first three pages.
    for page in pages[:3]:
        text = str(page["text"])

        print(
            f"\n--- Page {page['page']} "
            f"({len(text)} characters) ---"
        )

        if text:
            print(text[:500])
        else:
            print("[No text extracted from this page]")

    # Warn if the entire PDF has no extractable text.
    if pages and all(not page["text"] for page in pages):
        print(
            "\nWARNING: No text was extracted. "
            "The PDF may require another extraction method."
        )