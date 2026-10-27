# ResearchPaper AI — Automated Evaluation

**Run date:** 2026-10-27 14:27

**Total test cases:** 9

This report uses simple keyword and abstention checks. These checks help identify likely problems but do not establish factual correctness or citation support.

## Results

### T1 — Answerable

**Question:** What retrieval model does the paper use, and how does it work?

- Answer keyword check: **PASS — expected terms found** (3/3 concept groups)
- Retrieved evidence keyword coverage: 3/3 concept groups
- Note: These are keyword-based checks, not a proof of factual correctness.
- Citation consistency check: Cited page numbers appear in retrieved sources
- Manual check: verify that cited passages actually support the claims.
- Retrieved source pages: 2, 3, 9

**Generated answer:**

Based on the provided passages, the paper uses the **Dense Passage Retriever (DPR)** as its retrieval model (test_paper_official.pdf, PDF page: 2, 3). 

### How DPR Works:

1. **Bi-Encoder Architecture:**  
   DPR uses a bi-encoder structure comprising two separate encoders based on $\text{BERT}_{\text{BASE}}$ (test_paper_official.pdf, PDF page: 3):
   * A **query encoder** ($\text{BERT}_q$) that produces a dense vector representation of the input query, denoted as $q(x)$.
   * A **document encoder** ($\text{BERT}_d$) that produces a dense vector representation of a document, denoted as $d(z)$.

2. **Relevance Scoring:**  
   The retrieval probability $p_\eta(z|x)$ of a document $z$ given an input $x$ is proportional to the inner product (dot product) of their dense representations:
   $$p_\eta(z|x) \propto \exp\left(d(z)^\top q(x)\right)$$
   This means that documents whose vector representations have a higher dot product with the query vector are assigned a higher probability (test_paper_official.pdf, PDF page: 3).

3. **Efficient Search via MIPS:**  
   Finding the top-$k$ documents with the highest prior probability is framed as a **Maximum Inner Product Search (MIPS)** problem (test_paper_official.pdf, PDF page: 2, 3). MIPS allows the model to search the document collection and retrieve top candidate documents approximately in sub-linear time (test_paper_official.pdf, PDF page: 3).

4. **Initialization and Non-Parametric Memory:**  
   The retriever is initialized using a pre-trained DPR bi-encoder that was trained to retrieve documents containing answers to TriviaQA and Natural Questions (test_paper_official.pdf, PDF page: 3). It accesses a non-parametric memory consisting of a dense vector index of Wikipedia (test_paper_official.pdf, PDF page: 2, 3).

---

### T2 — Answerable

**Question:** How do RAG-Sequence and RAG-Token differ?

- Answer keyword check: **PARTIAL — some expected terms found** (2/4 concept groups)
- Retrieved evidence keyword coverage: 4/4 concept groups
- Note: These are keyword-based checks, not a proof of factual correctness.
- Citation consistency check: Cited page numbers appear in retrieved sources
- Manual check: verify that cited passages actually support the claims.
- Retrieved source pages: 3, 8

**Generated answer:**

Based on the provided passages, **RAG-Sequence** and **RAG-Token** differ primarily in how they utilize retrieved documents and marginalize (sum) over them to generate text:

### 1. Document Usage During Generation
* **RAG-Sequence:** Uses the **same** retrieved document to generate the entire output sequence (`test_paper_official.pdf`, p. 3). It treats the document as a single latent variable.
* **RAG-Token:** Can draw a **different** latent document for each individual target token. This allows the generator to synthesize content from multiple retrieved documents across the generated answer (`test_paper_official.pdf`, p. 3).

### 2. Marginalization and Probability Calculation
* **RAG-Sequence:** Retrieves the top-$K$ documents, has the generator produce the complete sequence probability for each document independently, and then marginalizes across the documents (`test_paper_official.pdf`, p. 3):
  $$p_{\text{RAG-Sequence}}(y|x) \approx \sum_{z \in \text{top-}k(p(\cdot|x))} p_\eta(z|x) \prod_{i}^{N} p_\theta(y_i|x, z, y_{1:i-1})$$
* **RAG-Token:** Produces a distribution for the next output token for each retrieved document, marginalizes across the documents for that specific token, and repeats this process token-by-token (`test_paper_official.pdf`, p. 3):
  $$p_{\text{RAG-Token}}(y|x) \approx \prod_{i}^{N} \sum_{z \in \text{top-}k(p(\cdot|x))} p_\eta(z|x) p_\theta(y_i|x, z_i, y_{1:i-1})$$

*(Note: "Marginalization" here refers to summing probabilities across all considered latent documents to obtain the overall probability distribution over the generated words or sequences.)*

---

### 3. Equivalence in Sequence Classification
When applied to sequence classification tasks where the target sequence is of length one, RAG-Sequence and RAG-Token are mathematically equivalent (`test_paper_official.pdf`, p. 3).

---

### 4. Behavior with More Retrieved Documents
* **Open-domain QA performance:** Retrieving more documents at test time monotonically improves Open-domain QA results for RAG-Sequence, whereas performance peaks at 10 retrieved documents for RAG-Token (`test_paper_official.pdf`, p. 8).
* **Generation metrics:** Retrieving more documents leads to a higher Rouge-L score at the expense of Bleu-1 for RAG-Token, but this trade-off is less pronounced for RAG-Sequence (`test_paper_official.pdf`, p. 8).

---

### T3 — Answerable

**Question:** Which generator model do the authors use?

- Answer keyword check: **PASS — expected terms found** (1/1 concept groups)
- Retrieved evidence keyword coverage: 1/1 concept groups
- Note: These are keyword-based checks, not a proof of factual correctness.
- Citation consistency check: Cited page numbers appear in retrieved sources
- Manual check: verify that cited passages actually support the claims.
- Retrieved source pages: 3

**Generated answer:**

The authors use **BART-large** as their generator model (Source: test_paper_official.pdf, PDF page: 3). 

**Key Details:**
* **Architecture:** It is a pre-trained sequence-to-sequence (seq2seq) transformer consisting of 400M parameters (Source: test_paper_official.pdf, PDF page: 3).
* **Pre-training:** It was pre-trained using a denoising objective with various noising functions (Source: test_paper_official.pdf, PDF page: 3).
* **Role in the system:** The authors refer to the BART generator's parameters as the model's "parametric memory" (Source: test_paper_official.pdf, PDF page: 3).

---

### T4 — Answerable

**Question:** Which datasets were used to train the DPR retriever?

- Answer keyword check: **PASS — expected terms found** (2/2 concept groups)
- Retrieved evidence keyword coverage: 2/2 concept groups
- Note: These are keyword-based checks, not a proof of factual correctness.
- Citation consistency check: Cited page numbers appear in retrieved sources
- Manual check: verify that cited passages actually support the claims.
- Retrieved source pages: 14, 3

**Generated answer:**

Based on the provided passages, the DPR retriever was trained to retrieve documents containing answers to questions from the following datasets:

* **TriviaQA** (test_paper_official.pdf, PDF page: 3)
* **Natural Questions** (test_paper_official.pdf, PDF page: 3)

---

### T5 — Answerable

**Question:** What is Maximum Inner Product Search and what is it used for?

- Answer keyword check: **PASS — expected terms found** (3/3 concept groups)
- Retrieved evidence keyword coverage: 3/3 concept groups
- Note: These are keyword-based checks, not a proof of factual correctness.
- Citation consistency check: Cited page numbers appear in retrieved sources
- Manual check: verify that cited passages actually support the claims.
- Retrieved source pages: 12, 2

**Generated answer:**

Based on the provided passages, the available evidence is insufficient to provide a full technical definition or algorithmic explanation of what Maximum Inner Product Search (MIPS) is. 

However, regarding what it is used for, the passages state that:
* For a given query $x$, Maximum Inner Product Search (MIPS) is used to retrieve/find the top-$K$ documents ($z_i$) from the document index (test_paper_official.pdf, PDF page: 2). 
* These retrieved documents are treated as latent variables conditioned on the query and are subsequently used to marginalize over sequence-to-sequence predictions for the final output (test_paper_official.pdf, PDF page: 2).

---

### T6 — Answerable

**Question:** When do RAG-Sequence and RAG-Token become equivalent?

- Answer keyword check: **PASS — expected terms found** (2/2 concept groups)
- Retrieved evidence keyword coverage: 2/2 concept groups
- Note: These are keyword-based checks, not a proof of factual correctness.
- Citation consistency check: Cited page numbers appear in retrieved sources
- Manual check: verify that cited passages actually support the claims.
- Retrieved source pages: 3, 8

**Generated answer:**

Based on the provided passages, **RAG-Sequence** and **RAG-Token** become equivalent when RAG is used for **sequence classification tasks** where the target class is treated as a **target sequence of length one** (i.e., a single token) (Source: test_paper_official.pdf, PDF page: 3).

### Technical Context:
* **RAG-Sequence** generates an entire sequence using the same retrieved document as a latent variable before marginalizing over the top documents (Source: test_paper_official.pdf, PDF page: 3).
* **RAG-Token** can draw a different retrieved document for each individual target token in the sequence (Source: test_paper_official.pdf, PDF page: 3).
* Because of this distinction, when a target sequence has a length of only one token, the marginalization over documents per token (RAG-Token) and over the entire sequence (RAG-Sequence) become identical (Source: test_paper_official.pdf, PDF page: 3).

---

### T7 — Unanswerable

**Question:** What is the lead author's personal phone number?

- Abstention wording check: **PASS — insufficient evidence acknowledged**
- Note: This checks wording only. A manual review may still be needed to detect unsupported claims.
- Retrieved source pages: 10, 11, 2

**Generated answer:**

Based on the supplied passages, the available evidence is insufficient to answer this question. The provided text does not contain the lead author's personal phone number.

---

### T8 — Unanswerable

**Question:** What exact F1 score did the authors achieve on an MRI diagnosis benchmark?

- Abstention wording check: **REVIEW — no recognized abstention wording**
- Note: This checks wording only. A manual review may still be needed to detect unsupported claims.
- Retrieved source pages: 15, 6

**Generated answer:**

Based on the supplied passages, the available evidence is insufficient to answer your question. The passages discuss evaluations on open-domain question answering and text generation/classification benchmarks—such as Natural Questions (NQ), TriviaQA (TQA), WebQuestions (WQ), CuratedTREC (CT), Jeopardy, MS-MARCO, and FEVER (*test_paper_official.pdf*, PDF page 6)—and make no mention of an MRI diagnosis benchmark or an F1 score achieved on one.

---

### T9 — Unanswerable

**Question:** Give me a chocolate cake recipe.

- Abstention wording check: **PASS — insufficient evidence acknowledged**
- Note: This checks wording only. A manual review may still be needed to detect unsupported claims.
- Retrieved source pages: 11, 14, 6, 7

**Generated answer:**

Based on the supplied passages, the available evidence is insufficient to provide a chocolate cake recipe. The retrieved text discusses natural language processing models, evaluation examples, and academic references, and does not contain any cooking or baking instructions.

---
