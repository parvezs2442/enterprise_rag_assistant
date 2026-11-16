# 🚀 Enterprise RAG Assistant

An advanced, production-grade **Retrieval-Augmented Generation (RAG)** pipeline designed for enterprise document question-answering. Built with **Hybrid Search (Dense + BM25)**, **Reciprocal Rank Fusion (RRF)**, **Cross-Encoder Neural Reranking (FlashRank)**, **Query Transformations (Rewriting & HyDE)**, and **Strict Evidence Grounding** with source citations.

---

## 📌 Table of Contents
- [Architecture Overview](#-architecture-overview)
- [Key Features](#-key-features)
- [Full-Stack Web Dashboard](#-full-stack-web-dashboard)
- [REST API Endpoints](#-rest-api-endpoints)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Running the Assistant](#running-the-assistant)
- [CLI Commands & Features](#-cli-commands--features)
- [How It Works (Under the Hood)](#-how-it-works-under-the-hood)
- [Future Roadmap](#-future-roadmap)

---

## 🏛️ Architecture Overview

The system processes enterprise documents (PDFs) into semantic vector chunks and lexical indices, using a multi-stage retrieval and ranking pipeline to deliver grounded, hallucination-free answers:

```
                    ┌─────────────────────────┐
                    │  Enterprise Source PDFs │
                    └────────────┬────────────┘
                                 │
                     [PyPDFLoader + Chunking]
                                 │
                   ┌─────────────┴─────────────┐
                   │                           │
          [Gemini Embeddings]          [BM25 Indexing]
                   │                           │
                   ▼                           ▼
          ┌─────────────────┐         ┌─────────────────┐
          │  Qdrant Vector  │         │  BM25 Lexical   │
          │   Store (Dense) │         │ Retriever (Doc) │
          └────────┬────────┘         └────────┬────────┘
                   │                           │
                   │   User Query / HyDE       │
                   ├───────────────────────────┤
                   ▼                           ▼
          Top-K Dense Chunks           Top-K Lexical Chunks
                   │                           │
                   └─────────────┬─────────────┘
                                 │
                   [Reciprocal Rank Fusion (RRF)]
                                 │
                     Fused Candidate Pool
                                 │
                   [FlashRank Neural Reranker]
                                 │
                      Top-N Re-scored Chunks
                                 │
                     [Strict Grounding Prompt]
                                 │
                     [Google Gemini LLM]
                                 │
                                 ▼
                     Grounded Answer + Citations
```

---

## ✨ Key Features

- **Hybrid Search (Dense + Lexical)**: Combines semantic vector similarity (Qdrant + Gemini Embeddings) with exact keyword matching (BM25) to catch both high-level context and specific technical terms, IDs, and acronyms.
- **Reciprocal Rank Fusion (RRF)**: Merges disparate rankings from dense and sparse retrievers without score scale bias using the proven formula:
  $$\text{RRF Score} = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
- **Cross-Encoder Candidate Reranking**: Uses local `FlashRank` (TinyBERT cross-encoder) to re-evaluate query-passage relevance with full cross-attention, drastically reducing false positives.
- **Query Transformations**:
  - **Query Rewriting**: Expands conversational or ambiguous queries into search-optimized technical keywords.
  - **HyDE (Hypothetical Document Embeddings)**: Generates a hypothetical ideal response before embedding to retrieve documents closer in semantic vector space.
- **Strict Evidence Grounding & Prompt Injection Defense**: Untrusted context is fenced inside secure XML boundaries (`<untrusted_document>`). The system neutralizes instruction overrides and abstains when evidence is missing.
- **Automated Citation Verification Engine**: Answers must tag claims with `[DOC-X]`. A post-generation verification engine detects and flags hallucinated or orphaned citations.
- **Automated Evaluation Suite (LLM-as-a-Judge)**: Includes a versioned golden dataset (`evals/golden_dataset.json`) and benchmark runner (`evaluate.py`) calculating Hit@k, MRR, Faithfulness, Relevance, and Abstention accuracy.
- **Traceable Source Citations**: Every generated fact is referenced with its source document name and page number.
- **Side-by-Side Comparison Tool**: Built-in CLI `compare` mode to evaluate Dense vs. BM25 vs. Hybrid vs. Reranked outputs in real-time.
- **Metadata Filtering**: Query-time filtering by document filename (`source=notes.pdf | <query>`).

---

## 🖥️ Full-Stack Web Dashboard

The web interface (`frontend/`) is built with React 19, TypeScript, and Tailwind CSS v4, providing an intuitive command center for enterprise knowledge:

1. **💬 Assistant (Chat)**: Real-time grounded conversation with interactive citation pills (`[DOC-1] notes.pdf p.2`), collapsible source passage inspectors, and response latency tracking.
2. **⚖️ Search Analytics**: 4-way side-by-side comparison of Vector Semantic Search (Qdrant), Lexical Keyword Search (BM25), Hybrid Fusion (RRF), and Neural Cross-Encoder Reranking (FlashRank).
3. **📊 Quality Benchmarks**: Continuous automated evaluation dashboard reporting Hit Rate @ 4, MRR, Faithfulness (Anti-Hallucination), Answer Relevance, and Safe Abstention.
4. **🛡️ Security Guardrails**: Live adversarial prompt-injection simulation and XML context isolation auditing.
5. **📁 Knowledge Base**: PDF upload and document repository management with automatic chunking and vector indexing.

---

## 🔌 REST API Endpoints

The FastAPI backend (`backend/server.py`) provides fully typed, production-ready REST endpoints:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/chat` | Runs full hybrid retrieval + FlashRank reranking + grounded generation + citations. |
| `POST` | `/api/compare` | Returns side-by-side retrieved passages across all 4 retrieval strategies. |
| `POST` | `/api/upload` | Uploads PDF files and triggers automatic chunking and indexing. |
| `GET` | `/api/documents` | Lists all active ingested PDF documents in the knowledge base. |
| `GET` | `/api/evaluation` | Returns the latest automated benchmark evaluation metrics from disk. |
| `POST` | `/api/evaluation/run` | Triggers a live evaluation run against golden test cases with LLM-as-a-Judge. |
| `POST` | `/api/security-test` | Runs adversarial prompt-injection attack simulation. |
| `GET` | `/api/health` | Health check reporting active models, indexed chunk count, and data paths. |

---

## 🛠️ Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Frontend UI** | React 19, TypeScript, Vite, Tailwind CSS v4, Lucide Icons | Interactive chat, strategy comparison, evaluation dashboard & document management |
| **Backend API** | Python, FastAPI, Uvicorn, Pydantic | Typed REST endpoints, file upload, orchestration & security |
| **LLM Engine** | Google Gemini (`gemini-flash-lite-latest` / `gemini-3.8-flash`) | Context synthesis, answer generation & query transformation |
| **Embedding Model** | Google Gemini Embeddings (`gemini-embedding-001`, 3072-dim) | Dense vector generation |
| **Vector Database** | Qdrant (`qdrant-client`, `langchain-qdrant`) | Local embedded high-performance vector search |
| **Lexical Search** | BM25 (`rank-bm25`, `langchain-community`) | Token-frequency exact keyword retrieval |
| **Neural Reranker** | FlashRank (`flashrank` with TinyBERT) | Cross-encoder candidate re-scoring |
| **Document Processing**| PyPDF (`pypdf`), LangChain Text Splitters | PDF parsing, chunking with overlap |
| **Configuration** | `python-dotenv` | Secure API key & environment management |

---

## 📁 Project Structure

```bash
Enterprise_RAG_Assistant/
├── backend/                      # Production Python Backend
│   ├── app/                      # Enterprise Modular Package
│   │   ├── core/                 # Configuration & settings management
│   │   ├── ingestion/            # PDF parsing, document discovery & chunking
│   │   ├── embeddings/           # Gemini vector embeddings adapter
│   │   ├── retrieval/            # Qdrant (Dense), BM25 (Sparse), RRF, & FlashRank
│   │   ├── query/                # Query rewriting & HyDE generators
│   │   ├── generation/           # Prompt-injection defense, citations, & LLM generator
│   │   ├── evaluation/           # Retrieval metrics, LLM-as-a-Judge, & benchmark runner
│   │   └── pipeline.py           # Unified EnterpriseRAGPipeline orchestrator
│   ├── data/                     # Source enterprise documents (PDFs)
│   │   └── notes.pdf
│   ├── evals/                    # Evaluation resources & benchmark reports
│   │   ├── golden_dataset.json   # Ground-truth test dataset
│   │   └── eval_report.json      # Benchmark evaluation metrics output
│   ├── qdrant_db/                # Local persistent Qdrant vector storage
│   ├── venv/                     # Python virtual environment
│   ├── .env                      # API keys & environment variables (git-ignored)
│   ├── .gitignore                # Git exclusion rules
│   ├── server.py                 # FastAPI REST API server
│   ├── main.py                   # Clean application CLI entrypoint
│   ├── evaluate.py               # Standalone evaluation benchmark entrypoint
│   └── requirements.txt          # Python dependencies
├── frontend/                     # Modern React + TypeScript + Tailwind UI
│   ├── src/
│   │   ├── App.tsx               # Main UI (Chat, Comparison, Eval Dashboard, Security, Docs)
│   │   ├── index.css             # Tailwind CSS styles
│   │   └── main.tsx              # React DOM entrypoint
│   ├── package.json              # Frontend dependencies
│   └── vite.config.ts            # Vite proxy configuration to backend API (:8000)
└── README.md                     # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+ (Tested up to Python 3.14)
- Node.js 18+ and pnpm (or npm)
- Google Gemini API Key ([Get a free key from Google AI Studio](https://aistudio.google.com/))

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/parvezs2442/enterprise_rag_assistant.git
   cd enterprise_rag_assistant
   ```

2. **Backend Setup:**
   - **Windows:**
     ```powershell
     cd backend
     python -m venv venv
     .\venv\Scripts\activate
     pip install -r requirements.txt
     ```
   - **Linux / macOS:**
     ```bash
     cd backend
     python3 -m venv venv
     source venv/bin/activate
     pip install -r requirements.txt
     ```

3. **Frontend Setup:**
   ```bash
   cd ../frontend
   pnpm install
   ```

### Configuration

Create a `.env` file inside the `backend/` directory:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-flash-lite-latest
EMBEDDING_MODEL=models/gemini-embedding-001
```

Place any PDF documents you want to query inside the `backend/data/` folder.

---

## 💻 Running the Assistant

### Option A: Full-Stack Web Application (Recommended)

1. **Start the FastAPI Backend Server:**
   ```powershell
   cd backend
   .\venv\Scripts\activate
   python -m uvicorn server:app --reload --port 8000
   ```
   *The backend REST API and interactive Swagger docs will run at `http://127.0.0.1:8000/docs`.*

2. **Start the React Frontend:**
   ```powershell
   cd frontend
   pnpm dev
   ```
   *Open your browser at `http://localhost:5173` to access the full web UI.*

---

### Option B: Interactive Terminal CLI

Run the direct CLI application:

```powershell
cd backend
python main.py
```

On first launch, it will:
1. Parse all PDFs in `backend/data/`.
2. Chunk text with recursive character splitting (500 chars, 100 overlap).
3. Compute 3072-dim embeddings and index them in local Qdrant storage.
4. Build the BM25 token index.
5. Initialize FlashRank TinyBERT cross-encoder model.

---

## 🎯 CLI Commands & Features

Once running, you can use these interactive commands:

| Command | Example | Description |
| :--- | :--- | :--- |
| **Standard Query** | `What is RAG and why is it used?` | Runs Hybrid Search + RRF + FlashRank reranking + grounded answer. |
| **Compare Strategies** | `compare What is vector database?` | Evaluates Dense vs BM25 vs Hybrid vs Reranked top results side-by-side. |
| **Query Rewriting** | `rewrite rag kyu use krte h` | Rewrites the query using LLM before search to improve retrieval. |
| **HyDE Search** | `hyde What are chunking strategies?` | Generates a hypothetical passage and searches using its vector. |
| **Metadata Filter** | `source=notes.pdf \| What is BM25?` | Restricts search strictly to chunks originating from `notes.pdf`. |
| **Run Benchmark** | `eval` | Runs automated quality evaluation benchmark against the golden dataset. |
| **Security Test** | `security_test` | Tests adversarial prompt-injection defense mechanisms. |
| **Exit** | `exit` or `quit` | Closes the CLI session. |

---

## 🔍 How It Works (Under the Hood)

### 1. Ingestion & Chunking
Documents are loaded page-by-page. Pages are split into overlapping chunks (500 characters, 100 character overlap) to preserve context across boundaries while avoiding overly long embedding vectors. Each chunk retains metadata: `source`, `page`, and `chunk_id`.

### 2. Dual Retrieval (Hybrid)
- **Dense Vector Search**: Measures cosine distance between query embedding and chunk vectors in Qdrant. Great for synonyms and semantic concepts.
- **BM25 Lexical Search**: Uses token frequency statistics (TF-IDF based) to match exact keywords, model names, and IDs that embeddings might overlook.

### 3. Reciprocal Rank Fusion (RRF)
Combines candidates from both retrieval channels by ranking position rather than raw scores:
```python
scores[doc_id] += 1.0 / (60 + rank + 1)
```

### 4. FlashRank Neural Reranking
Passes top fused candidates through `ms-marco-TinyBERT-L-2-v2`. Unlike bi-encoders (which encode query and passage independently), cross-encoders process query and passage together, catching subtle semantic relationships.

### 5. Grounded Generation & Citation
The top reranked chunks are formatted into structured context blocks. Gemini is prompted with strict instruction:
- Answer *only* using supplied facts.
- Explicitly state if information is missing.
- Include source filename and page numbers for traceable accountability.

---

## 🗺️ Future Roadmap

- [x] **FastAPI REST Endpoints**: Typed REST API routes with CORS, document upload, and Swagger docs.
- [x] **Full-Stack Web Dashboard**: Modern single-page React 19 + TypeScript + Tailwind UI with citation inspection.
- [x] **Automated Evaluation Suite**: Golden dataset benchmark testing with LLM-as-a-Judge (Hit@k, MRR, Faithfulness, Relevance, Safe Abstention).
- [ ] **PostgreSQL Metadata Storage**: Multi-user permissions, persistent conversation history, and role-based access control (RBAC).
- [ ] **Multi-Modal Vision RAG**: Visual table, chart, and diagram extraction from enterprise presentations and scanned PDFs.

---

## 📄 License
This project is licensed under the MIT License.
