from vector_store import load_vector_store


def main():
    # Load the existing Chroma database.
    store = load_vector_store()

    # Ask a question about the research paper.
    query = "What retrieval model does the paper use, and how does it work?"

    # Retrieve the three most relevant passages.
    results = store.similarity_search(query, k=3)

    print(f"Question: {query}")
    print(f"Retrieved passages: {len(results)}")

    # Display each passage and its source information.
    for index, document in enumerate(results, start=1):
        print(f"\n--- Result {index} ---")
        print(f"Source: {document.metadata.get('source', 'Unknown')}")
        print(f"Page: {document.metadata.get('page', 'Unknown')}")
        print(document.page_content[:400])


if __name__ == "__main__":
    main()