# ResearchPaper AI — Project Scope

## 1. Objective
Build a PDF-based research assistant that answers questions about a research paper using retrieved evidence from the uploaded document.

## 2. Target user
A student or researcher who wants to understand a research paper without manually searching every page.

## 3. MVP features
- Upload one text-based PDF at a time.
- Extract text page by page and preserve source metadata.
- Split text into manageable, overlapping chunks.
- Generate embeddings for the chunks.
- Store and search chunks using Chroma.
- Retrieve relevant passages for a user question.
- Use Gemini to generate an answer grounded in the retrieved passages.
- Display supporting passages and PDF page references.
- Handle common input, PDF extraction, and API errors.
- Evaluate the system using a curated question set.

## 4. Technology stack
- Python
- Streamlit
- Gemini API
- LangChain
- Embeddings
- Chroma vector database
- Git for version control

## 5. Out of scope for the initial version
- Searching across multiple PDFs
- Authentication and user accounts
- OCR for scanned PDFs
- Web search
- Advanced conversational memory
- Multi-agent workflows
- Production-scale deployment

## 6. Constraints
- Complete the initial implementation in three days.
- Store API credentials securely and never commit secrets.
- Preserve page metadata throughout the pipeline.
- Avoid unnecessary repeated embedding and indexing.
- Keep retrieval and generation as separate, testable steps.

## 7. Success criteria
The user can upload a research paper, ask questions, retrieve relevant passages, receive an evidence-grounded answer, and inspect supporting source pages. The application should avoid inventing information when the retrieved evidence is insufficient.

## 8. Evaluation
Test direct questions, paraphrased questions, questions requiring multiple passages, and questions that cannot be answered from the paper. Record retrieval relevance, answer correctness, and citation accuracy.
