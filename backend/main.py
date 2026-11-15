"""
=============================================================================
ENTERPRISE RAG ASSISTANT - PHASE 8: ADVANCED RETRIEVAL ENGINEERING
=============================================================================
Yeh script Phase 1 se Phase 8 tak ka complete, simplified pipeline implement karta hai:
- Phase 1: Environment & Settings (.env, API keys)
- Phase 3: LLM Engineering (Gemini Flash model, Structured Prompts)
- Phase 5: Document Ingestion & Chunking (PyPDFLoader, RecursiveCharacterTextSplitter)
- Phase 6: Vector Embeddings (Gemini Embeddings 3072-dim)
- Phase 7: Vector Database & Citations (Qdrant Vector Store, Page citations)
- Phase 8: Advanced Retrieval:
    1. Lexical Search (BM25 keyword search)
    2. Dense Semantic Search (Qdrant vector search)
    3. Metadata Filtering (Source document filtering)
    4. Reciprocal Rank Fusion (RRF)
    5. Cross-Encoder Reranking (FlashRank)
    6. Query Rewriting / Expansion (LLM query optimization)
    7. HyDE (Hypothetical Document Embeddings)
    8. Strategy Comparison Mode (Compare Dense vs BM25 vs Hybrid vs Rerank)
=============================================================================
"""

import os
import sys
from collections import defaultdict
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from flashrank import Ranker, RerankRequest


# =============================================================================
# 1. CONFIGURATION & ENVIRONMENT SETUP (Phase 1)
# =============================================================================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("ERROR: GOOGLE_API_KEY .env file mein nahi mila! Kripya .env mein key dalein.")

DATA_PATH = "data"
QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "enterprise_rag_notes"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

DENSE_K = 8
BM25_K = 8
RRF_K = 60
RERANK_TOP_N = 4

# Gemini Models
GEMINI_CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
GEMINI_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

# True = Document fir se parse karke Qdrant me store honge
# False = Agar Qdrant me data hai to reuse hoga
REBUILD_INDEX = False


# =============================================================================
# 2. INITIALIZE CLIENTS & EMBEDDING ADAPTER (Phase 3 & Phase 6)
# =============================================================================
genai_client = genai.Client()

class GeminiEmbeddingsAdapter(Embeddings):
    """
    LangChain Embeddings adapter using Google GenAI SDK.
    Batch endpoints bypass karke direct stable embedContent call karta hai.
    """
    def __init__(self, model_name: str = GEMINI_EMBED_MODEL):
        self.client = genai_client
        self.model = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            response = self.client.models.embed_content(
                model=self.model,
                contents=text
            )
            embeddings.append(response.embeddings[0].values)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=text
        )
        return response.embeddings[0].values

embeddings_model = GeminiEmbeddingsAdapter()
reranker_model = Ranker()

print(f"[OK] Gemini SDK initialized (LLM: {GEMINI_CHAT_MODEL}, Embed: {GEMINI_EMBED_MODEL})")
print("[OK] FlashRank local reranker ready.")


# =============================================================================
# 3. DOCUMENT INGESTION & CHUNKING (Phase 5)
# =============================================================================
def load_all_pdfs(data_dir: str = DATA_PATH) -> List[Document]:
    """Folder ke saare PDFs ko load karta hai aur page metadata preserve karta hai."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        raise FileNotFoundError(f"'{data_dir}' folder nahi mila! Folder create kiya gaya hai, kripya PDF dalein.")

    documents: List[Document] = []
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        raise ValueError(f"'{data_dir}' folder mein koi PDF document nahi mila!")

    for filename in pdf_files:
        file_path = os.path.join(data_dir, filename)
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        # Ensure source filename is normalized
        for page in pages:
            page.metadata["source"] = filename
        documents.extend(pages)
        print(f" -> Loaded: {filename} ({len(pages)} pages)")

    return documents


def split_into_chunks(documents: List[Document]) -> List[Document]:
    """Documents ko context-preserving chunks mein break karta hai."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    # Add chunk index to metadata
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
    print(f" -> Chunks created: {len(chunks)} (Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})")
    return chunks


# =============================================================================
# 4. VECTOR DATABASE MANAGEMENT (Phase 7)
# =============================================================================
def get_or_create_vector_store(chunks: List[Document], rebuild: bool = False) -> QdrantVectorStore:
    """Local Qdrant vector database create ya load karta hai."""
    client = QdrantClient(path=QDRANT_PATH)
    exists = client.collection_exists(COLLECTION_NAME)

    if rebuild and exists:
        print(f" -> Rebuilding index: Deleting existing collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)
        exists = False
    client.close()

    if not exists:
        print(f" -> Indexing {len(chunks)} chunks into Qdrant ({QDRANT_PATH})...")
        vectorstore = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            path=QDRANT_PATH,
            collection_name=COLLECTION_NAME
        )
        print(" -> Qdrant collection created successfully!")
    else:
        print(f" -> Loading existing Qdrant collection '{COLLECTION_NAME}'...")
        vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=embeddings_model,
            path=QDRANT_PATH,
            collection_name=COLLECTION_NAME
        )
        print(" -> Qdrant collection loaded.")

    return vectorstore


# =============================================================================
# 5. RETRIEVAL STRATEGIES (Phase 8: Advanced Retrieval Engineering)
# =============================================================================

# --- 5.1 BM25 Keyword Retrieval (Lexical / Exact Matching) ---
def init_bm25_retriever(chunks: List[Document]) -> BM25Retriever:
    """BM25 keyword search index initialize karta hai."""
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = BM25_K
    return retriever


def bm25_search(
    bm25: BM25Retriever,
    query: str,
    source_filter: Optional[str] = None,
    k: int = BM25_K
) -> List[Document]:
    """Keyword / BM25 lexical search run karta hai."""
    bm25.k = k * 2 if source_filter else k
    results = bm25.invoke(query)
    if source_filter:
        results = [
            doc for doc in results
            if os.path.basename(doc.metadata.get("source", "")) == source_filter
        ]
    return results[:k]


# --- 5.2 Dense Semantic Retrieval (Embedding Similarity) ---
def dense_semantic_search(
    vectorstore: QdrantVectorStore,
    query: str,
    source_filter: Optional[str] = None,
    k: int = DENSE_K
) -> List[Document]:
    """Qdrant vector store mein semantic cosine similarity search run karta hai."""
    qdrant_filter = None
    if source_filter:
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.source",
                    match=MatchValue(value=source_filter)
                )
            ]
        )
    return vectorstore.similarity_search(query=query, k=k, filter=qdrant_filter)


# --- 5.3 Reciprocal Rank Fusion (RRF) ---
def reciprocal_rank_fusion(result_lists: List[List[Document]], k: int = RRF_K) -> List[Document]:
    """
    Reciprocal Rank Fusion (RRF) algorithm:
    Alag alag retrievers (BM25 + Semantic) ke rank positions ko combine karke
    robust fused ranking create karta hai: Score = sum(1 / (k + rank + 1))
    """
    scores: Dict[tuple, float] = defaultdict(float)
    doc_lookup: Dict[tuple, Document] = {}

    for doc_list in result_lists:
        for rank, doc in enumerate(doc_list):
            doc_key = (
                doc.metadata.get("source", ""),
                doc.metadata.get("page", 0),
                doc.metadata.get("chunk_id", 0),
                doc.page_content.strip()
            )
            scores[doc_key] += 1.0 / (k + rank + 1)
            doc_lookup[doc_key] = doc

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_lookup[key] for key in sorted_keys]


# --- 5.4 Cross-Encoder Candidate Reranking (FlashRank) ---
def rerank_documents(
    query: str,
    documents: List[Document],
    top_n: int = RERANK_TOP_N
) -> List[Document]:
    """
    Retrieved candidates ko transformer cross-encoder (FlashRank) se rerank karta hai.
    Yeh query aur chunk ke bi-directional attention ko score karta hai.
    """
    if not documents:
        return []

    passages = [
        {
            "id": i,
            "text": doc.page_content,
            "meta": doc.metadata
        }
        for i, doc in enumerate(documents)
    ]

    rerank_request = RerankRequest(query=query, passages=passages)
    ranked_passages = reranker_model.rerank(rerank_request)

    reranked_docs = []
    for p in ranked_passages[:top_n]:
        doc = Document(
            page_content=p["text"],
            metadata={**p["meta"], "rerank_score": round(float(p.get("score", 0.0)), 4)}
        )
        reranked_docs.append(doc)

    return reranked_docs


# --- 5.5 Phase 8 Advanced: Query Rewriting / Query Expansion ---
def rewrite_query_for_retrieval(user_query: str) -> str:
    """
    Agar user query colloquial, incomplete ya ambiguous ho, toh LLM se
    retrieval ke liye optimized technical keywords generate karwata hai.
    """
    prompt = f"""You are a search query optimizer for an enterprise documentation search system.
Given the following user query, output an expanded, optimized search query with technical keywords.
Do NOT answer the question. Only output the rewritten search query in 1 concise line.

User Query: {user_query}
Rewritten Query:"""

    try:
        response = genai_client.models.generate_content(
            model=GEMINI_CHAT_MODEL,
            contents=prompt
        )
        optimized = response.text.strip().replace('"', '')
        return optimized if optimized else user_query
    except Exception:
        return user_query


# --- 5.6 Phase 8 Advanced: HyDE (Hypothetical Document Embeddings) ---
def generate_hyde_passage(query: str) -> str:
    """
    HyDE: Query ka ek hypothetical answer generate karta hai.
    Is answer ka vector actual document chunks ke vector space se match karta hai.
    """
    prompt = f"""Write a short, factual passage from a technical document that directly answers this question:
"{query}"
Do not explain, just write the passage (2-3 sentences)."""

    try:
        response = genai_client.models.generate_content(
            model=GEMINI_CHAT_MODEL,
            contents=prompt
        )
        return response.text.strip()
    except Exception:
        return query


# --- 5.7 Master Hybrid Retrieval Pipeline ---
def advanced_retrieval(
    query: str,
    vectorstore: QdrantVectorStore,
    bm25: BM25Retriever,
    source_filter: Optional[str] = None,
    use_query_rewrite: bool = False,
    use_hyde: bool = False,
    use_reranker: bool = True
) -> List[Document]:
    """
    Complete Phase 8 Retrieval Pipeline:
    1. Optional Query Transformation (Rewriting or HyDE)
    2. BM25 Lexical Search
    3. Dense Semantic Search
    4. Reciprocal Rank Fusion (RRF)
    5. FlashRank Cross-Encoder Reranking
    """
    search_query = query
    if use_query_rewrite:
        search_query = rewrite_query_for_retrieval(query)
        print(f" [Query Rewritten] -> '{search_query}'")

    # Semantic Search target (normal query ya HyDE hypothetical text)
    semantic_target = search_query
    if use_hyde:
        hypo_doc = generate_hyde_passage(search_query)
        semantic_target = hypo_doc
        print(f" [HyDE Passage Generated] -> '{hypo_doc[:80]}...'")

    # Step 1: Lexical (BM25)
    bm25_docs = bm25_search(bm25, search_query, source_filter=source_filter, k=BM25_K)

    # Step 2: Dense (Vector)
    dense_docs = dense_semantic_search(vectorstore, semantic_target, source_filter=source_filter, k=DENSE_K)

    # Step 3: Reciprocal Rank Fusion (RRF)
    fused_docs = reciprocal_rank_fusion([dense_docs, bm25_docs])

    # Step 4: Reranking
    if use_reranker:
        final_docs = rerank_documents(query=search_query, documents=fused_docs, top_n=RERANK_TOP_N)
    else:
        final_docs = fused_docs[:RERANK_TOP_N]

    return final_docs


# =============================================================================
# 6. CONTEXT FORMATTING & GENERATION (Phase 3 & 7)
# =============================================================================
def format_context(documents: List[Document]) -> str:
    """Retrieved chunks ko LLM ke liye clean structured context mein format karta hai."""
    context_blocks = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 0) + 1
        score_info = f" | Rerank Score: {doc.metadata['rerank_score']}" if "rerank_score" in doc.metadata else ""
        context_blocks.append(
            f"--- Document Snippet [{i}] (File: {source}, Page: {page}{score_info}) ---\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n".join(context_blocks)


def generate_grounded_answer(query: str, documents: List[Document]) -> str:
    """
    Strict evidence-grounded answer generator.
    Agar context me answer nahi hai toh explicitly mana karta hai (Hallucination prevention).
    """
    if not documents:
        return "I could not find any relevant information in the uploaded documents to answer your question."

    context_str = format_context(documents)

    system_prompt = """You are an Enterprise RAG Assistant.
Answer the user's question using ONLY the facts directly mentioned in the provided Context.
Rules:
1. Strict Grounding: Do NOT assume, extrapolate, or use outside knowledge.
2. If the context does not contain the answer, reply exactly:
   "I cannot answer this question based on the provided documents as the necessary information is not mentioned."
3. Cite sources: Always reference the document name and page number for each claim.
4. Keep the answer clear, structured, and professional."""

    user_prompt = f"""Context:
{context_str}

Question: {query}

Answer:"""

    response = genai_client.models.generate_content(
        model=GEMINI_CHAT_MODEL,
        contents=f"{system_prompt}\n\n{user_prompt}"
    )
    return response.text.strip()


# =============================================================================
# 7. PHASE 8 COMPARISON / EXPERIMENTATION MODE
# =============================================================================
def compare_retrieval_strategies(query: str, vectorstore: QdrantVectorStore, bm25: BM25Retriever):
    """
    Phase 8 Learning Objective:
    Compare Baseline Dense vs BM25 vs Hybrid vs Reranked side-by-side!
    """
    print("\n" + "="*80)
    print(f"PHASE 8 RETRIEVAL EXPERIMENT COMPARISON FOR: '{query}'")
    print("="*80)

    # 1. Dense only
    dense_res = dense_semantic_search(vectorstore, query, k=3)
    print("\n[Strategy 1: Dense Semantic Search (Vector only - Top 3)]")
    for i, d in enumerate(dense_res, 1):
        print(f"  {i}. [Page {d.metadata.get('page',0)+1}] {d.page_content[:110].strip()}...")

    # 2. BM25 only
    bm25_res = bm25_search(bm25, query, k=3)
    print("\n[Strategy 2: Lexical Search (BM25 only - Top 3)]")
    for i, d in enumerate(bm25_res, 1):
        print(f"  {i}. [Page {d.metadata.get('page',0)+1}] {d.page_content[:110].strip()}...")

    # 3. Hybrid (Dense + BM25 with RRF)
    rrf_res = reciprocal_rank_fusion([dense_res, bm25_res])[:3]
    print("\n[Strategy 3: Hybrid Search (BM25 + Dense + RRF - Top 3)]")
    for i, d in enumerate(rrf_res, 1):
        print(f"  {i}. [Page {d.metadata.get('page',0)+1}] {d.page_content[:110].strip()}...")

    # 4. Hybrid + Reranker
    reranked = rerank_documents(query, rrf_res, top_n=3)
    print("\n[Strategy 4: Hybrid + FlashRank Reranker (Top 3)]")
    for i, d in enumerate(reranked, 1):
        score = d.metadata.get('rerank_score', 'N/A')
        print(f"  {i}. [Score: {score} | Page {d.metadata.get('page',0)+1}] {d.page_content[:110].strip()}...")

    print("="*80)


# =============================================================================
# 8. INTERACTIVE CLI CHAT APPLICATION
# =============================================================================
def run_cli():
    print("\n" + "="*60)
    print("      ENTERPRISE RAG ASSISTANT - PHASE 8 PIPELINE")
    print("="*60)

    # 1. Ingestion
    print("\n[Phase 5] Loading and splitting documents...")
    docs = load_all_pdfs(DATA_PATH)
    chunks = split_into_chunks(docs)

    # 2. Vector Store (Phase 7)
    print("\n[Phase 7] Initializing Vector Store...")
    vectorstore = get_or_create_vector_store(chunks, rebuild=REBUILD_INDEX)

    # 3. BM25 Index (Phase 8)
    print("\n[Phase 8] Initializing BM25 index...")
    bm25 = init_bm25_retriever(chunks)

    print("\n" + "-"*60)
    print("System Features Enabled:")
    print(" • BM25 Lexical Keyword Search")
    print(" • Qdrant Semantic Vector Search")
    print(" • Reciprocal Rank Fusion (RRF)")
    print(" • FlashRank Neural Reranking")
    print(" • LLM Query Rewriting & HyDE support")
    print(" • Source Document Metadata Filtering")
    print(" • Grounded Citations (Source + Page)")
    print("-"*60)
    print("Special Commands:")
    print(" • compare <query>  : Compare Dense vs BM25 vs Hybrid vs Rerank")
    print(" • rewrite <query>  : Use LLM Query Rewriting before search")
    print(" • hyde <query>     : Use Hypothetical Document Embedding (HyDE)")
    print(" • source=file.pdf | <query> : Search only within that document")
    print(" • exit             : Quit the assistant")
    print("-"*60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        # Comparison mode
        if user_input.startswith("compare "):
            query = user_input.replace("compare ", "").strip()
            compare_retrieval_strategies(query, vectorstore, bm25)
            continue

        # Modes & filters
        use_rewrite = False
        use_hyde = False
        source_filter = None
        query = user_input

        if query.startswith("rewrite "):
            use_rewrite = True
            query = query.replace("rewrite ", "").strip()
        elif query.startswith("hyde "):
            use_hyde = True
            query = query.replace("hyde ", "").strip()

        if query.startswith("source=") and "|" in query:
            parts = query.split("|", 1)
            source_filter = parts[0].replace("source=", "").strip()
            query = parts[1].strip()

        print(f"\n[Retrieving evidence for: '{query}'...]")
        retrieved_docs = advanced_retrieval(
            query=query,
            vectorstore=vectorstore,
            bm25=bm25,
            source_filter=source_filter,
            use_query_rewrite=use_rewrite,
            use_hyde=use_hyde,
            use_reranker=True
        )

        print(f" -> Found {len(retrieved_docs)} top reranked chunks.")
        print("[Generating grounded answer with Gemini...]")
        answer = generate_grounded_answer(query, retrieved_docs)

        print("\n" + "="*50 + " AI ANSWER " + "="*50)
        print(answer)
        print("="*111)

        print("\nSources Cited:")
        seen = set()
        for doc in retrieved_docs:
            src = doc.metadata.get("source", "Unknown")
            pg = doc.metadata.get("page", 0) + 1
            score = doc.metadata.get("rerank_score", "")
            key = (src, pg)
            if key not in seen:
                score_str = f" (Rerank Score: {score})" if score != "" else ""
                print(f"  • {src} - Page {pg}{score_str}")
                seen.add(key)


if __name__ == "__main__":
    run_cli()
