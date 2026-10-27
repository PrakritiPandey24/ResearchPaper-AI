# ResearchPaper AI — Research Paper Assistant

A PDF-based research assistant that allows users to ask questions about research papers and receive AI-generated answers grounded in retrieved passages from the uploaded document.

Built with Python, Streamlit, Google's Gemini API, LangChain, embeddings, and Chroma.

## Overview

Research papers can be lengthy and difficult to navigate. ResearchPaper AI simplifies this process by allowing users to upload a research paper in PDF format and ask questions about its contents.

The application extracts text from the PDF, divides it into smaller chunks, converts those chunks into embeddings, and stores them in a Chroma vector database. When a user asks a question, the system retrieves relevant passages and provides them to Gemini to generate an evidence-grounded answer.

The application also displays the retrieved source passages and their original PDF page numbers so users can inspect the evidence behind an answer.

## Features

- **PDF upload:** Upload one research paper at a time through a Streamlit interface.
- **Text extraction:** Extract text from PDF pages using PyMuPDF.
- **Text chunking:** Split extracted text into overlapping chunks while preserving source metadata.
- **Semantic retrieval:** Generate embeddings using Google's Gemini embedding model and retrieve relevant passages using Chroma.
- **Retrieval-Augmented Generation (RAG):** Use retrieved passages as context for Gemini's answers.
- **Source references:** Display retrieved passages, source filenames, PDF page numbers, and chunk IDs.
- **Persistent vector storage:** Save the indexed document in a local Chroma database.
- **Evaluation:** Run a predefined set of questions to examine answer quality, retrieval relevance, citations, and handling of insufficient evidence.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | User interface and chat experience |
| PyMuPDF | PDF text extraction |
| LangChain Text Splitters | Text chunking |
| Gemini Embeddings (`gemini-embedding-001`) | Convert text into numerical vectors |
| Chroma | Vector storage and similarity search |
| Gemini API (`gemini-3.8-flash`) | Generate answers from retrieved context |
| Git | Version control |

## How It Works

The application follows a Retrieval-Augmented Generation (RAG) pipeline.

1. **PDF upload:** The user uploads a research paper.
2. **Text extraction:** The application extracts text from each page and preserves its page number.
3. **Chunking:** The extracted text is divided into smaller, overlapping chunks.
4. **Embedding generation:** Each chunk is converted into a numerical vector representing its semantic information.
5. **Vector storage:** The chunks, vectors, and source metadata are stored in Chroma.
6. **Semantic retrieval:** The user's question is embedded and compared with the stored document vectors to retrieve relevant passages.
7. **Answer generation:** Gemini receives the question and retrieved passages and generates an answer using the supplied evidence.
8. **Source inspection:** The interface displays the retrieved passages and their source page numbers.

### Architecture

```text
                 Research Paper PDF
                         |
                         v
                  PDF Extraction
                         |
                         v
                    Text Chunking
                         |
                         v
                 Gemini Embeddings
                         |
                         v
                  Chroma Vector Store
                         |
                  User's Question
                         |
                         v
                 Semantic Retrieval
                         |
                         v
               Retrieved PDF Passages
                         |
                         v
                 Gemini Generation
                         |
                         v
                Answer + Source Passages
                         |
                         v
                  Streamlit Interface
```

## Project Structure

```text
ResearchPaper-AI/
│
├── app.py                    # Streamlit application
├── pdf_processor.py          # PDF text extraction
├── chunker.py                # Text chunking
├── vector_store.py           # Embeddings and Chroma
├── rag_pipeline.py           # Retrieval and Gemini generation
├── test_gemini.py            # Gemini API connectivity test
├── test_retrieval.py         # Semantic retrieval test
├── run_evaluation.py         # Automated evaluation script
├── evaluation_results.md     # Generated evaluation report
│
├── data/                     # Local test PDFs
├── chroma_db/                # Generated vector database
│
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── .gitignore                # Git exclusions
├── PROJECT_SCOPE.md          # Initial project specification
└── README.md                 # Project documentation
```

## Setup and Installation

### 1. Clone or open the repository

Open a terminal in the project root directory.

### 2. Create a Python virtual environment

Python 3.13 is the version used during development.

On Windows PowerShell:

```powershell
py -3.13 -m venv .venv
```

This creates an isolated Python environment for the project.

### 3. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

Activation ensures that Python packages are installed in the project's environment rather than globally.

### 4. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

This installs the libraries required for PDF extraction, embeddings, vector storage, the Gemini API, and the Streamlit interface.

### 5. Configure the Gemini API key

Obtain an API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

Create a `.env` file in the project root with the following content:

```dotenv
GEMINI_API_KEY=your_api_key_here
```

Replace the placeholder with your actual API key.

The application reads the API key from this file. Do not commit `.env` or publish your API key.

### 6. Run the application

```powershell
python -m streamlit run app.py
```

Streamlit will start a local development server. Open the URL shown in the terminal, usually:

`http://localhost:8501`

Upload a text-based research paper PDF and begin asking questions.

## Testing and Evaluation

The project includes scripts to test its main components independently.

### Test Gemini connectivity

```powershell
python test_gemini.py
```

Checks whether the application can communicate with Gemini.

### Test PDF extraction

```powershell
python pdf_processor.py
```

Extracts the configured test PDF's text and displays page-level previews.

### Test chunking

```powershell
python chunker.py
```

Splits the extracted text into chunks and checks whether source metadata is preserved.

### Test embeddings and Chroma retrieval

```powershell
python vector_store.py
```

Creates the local vector index and retrieves relevant passages for a sample question. This script rebuilds the local Chroma index when executed.

### Test semantic retrieval independently

```powershell
python test_retrieval.py
```

Loads the existing index and retrieves passages for a test query.

### Test answer generation

```powershell
python rag_pipeline.py
```

Runs the complete retrieval and answer-generation pipeline outside the Streamlit interface.

### Run automated evaluation

```powershell
python run_evaluation.py
```

Runs nine predefined test questions and writes the results to `evaluation_results.md`.

The evaluation covers answerable questions, retrieval relevance, basic citation consistency, and questions that should not be answered using unsupported information.

The current evaluation uses keyword-based and abstention-phrase checks. These are useful initial checks, but they do not establish factual correctness or prove that every citation supports the associated claim. The results require interpretation and, where appropriate, manual verification.

## Current Limitations

- **PDF extraction quality:** Some PDFs contain complex layouts, mathematical notation, figures, or unusual font encodings that may produce garbled text.
- **Scanned documents:** PDFs without an extractable text layer may require OCR, which is not currently implemented.
- **Single-document workflow:** The application supports one active PDF at a time. Uploading another document replaces the existing local index.
- **Retrieval limitations:** Semantic similarity can return partially relevant passages or miss necessary evidence.
- **Potential hallucinations:** Gemini may misinterpret retrieved text or make unsupported claims despite the grounding instructions.
- **Citation verification:** Source page references are derived from retrieved document metadata, but the current implementation does not independently verify every generated claim against its cited passage.
- **API availability:** Gemini requests may fail temporarily because of network issues, service availability, rate limits, or quota restrictions.

## Future Improvements

- Add OCR support for scanned PDFs.
- Improve retrieval quality and filtering of noisy chunks.
- Strengthen answer grounding and citation verification.
- Expand the evaluation suite with manually verified expected answers and retrieval metrics.
- Support searching across multiple research papers.
- Improve error handling, performance, and the user interface.
- Add automated tests for PDF processing, retrieval, and application behavior.

## Learning Outcomes

This project demonstrates the integration of several Generative AI concepts:

- Retrieval-Augmented Generation (RAG)
- Text embeddings and vector representations
- Semantic similarity search
- Vector databases and metadata
- Prompt design for evidence-grounded generation
- Document processing and chunking
- Basic evaluation of retrieval and generated answers


