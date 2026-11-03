
import os
from collections import defaultdict

from dotenv import load_dotenv

# Document Loading
from langchain_community.document_loaders import PyPDFLoader

# Text Splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Gemini
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

# Qdrant
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue
)

# BM25
from langchain_community.retrievers import BM25Retriever

# Reranking
from langchain_community.document_compressors import FlashrankRerank

# Prompt and Output
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==========================================
# 1. CONFIGURATION
# ==========================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")

DATA_PATH = "data"
QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "advanced_rag"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

DENSE_K = 10
BM25_K = 10
RERANK_TOP_N = 5
RRF_K = 60

# True = Rebuild Qdrant index
# False = Reuse existing index
REBUILD_INDEX = True


# ==========================================
# 2. INITIALIZE MODELS
# ==========================================

embeddings = GoogleGenerativeAIEmbeddings(
    model=os.getenv(
        "EMBEDDING_MODEL",
        "models/gemini-embedding-001"
    )
)

llm = ChatGoogleGenerativeAI(
    model=os.getenv(
        "GEMINI_MODEL",
        "gemini-2.5-flash"
    ),
    temperature=0
)

print("Gemini models initialized!")


# ==========================================
# 3. DOCUMENT LOADING
# ==========================================

def load_documents():

    documents = []

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Data folder not found!")

    for filename in os.listdir(DATA_PATH):

        if filename.lower().endswith(".pdf"):

            file_path = os.path.join(DATA_PATH, filename)

            loader = PyPDFLoader(file_path)

            pdf_documents = loader.load()

            documents.extend(pdf_documents)

            print(f"Loaded PDF: {filename}")

    if not documents:
        raise ValueError("No PDF documents found!")

    print("Total pages loaded:", len(documents))

    return documents


# ==========================================
# 4. TEXT CHUNKING
# ==========================================

def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = text_splitter.split_documents(documents)

    print("Total chunks created:", len(chunks))

    return chunks


# ==========================================
# 5. QDRANT VECTOR DATABASE
# ==========================================

def create_vector_store(chunks):

    client = QdrantClient(path=QDRANT_PATH)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.close()

    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME
    )

    print("Documents stored in Qdrant!")

    return vectorstore


def load_vector_store():

    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME
    )

    print("Existing Qdrant collection loaded!")

    return vectorstore


# ==========================================
# 6. BM25 KEYWORD RETRIEVAL
# ==========================================

def create_bm25_retriever(chunks):

    bm25_retriever = BM25Retriever.from_documents(
        chunks
    )

    bm25_retriever.k = BM25_K

    print("BM25 retriever initialized!")

    return bm25_retriever


# ==========================================
# 7. SEMANTIC SEARCH
# ==========================================

def semantic_search(
    vectorstore,
    query,
    source_filter=None
):

    search_filter = None

    if source_filter:

        search_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.source",
                    match=MatchValue(value=source_filter)
                )
            ]
        )

    results = vectorstore.similarity_search(
        query=query,
        k=DENSE_K,
        filter=search_filter
    )

    return results


# ==========================================
# 8. RECIPROCAL RANK FUSION (RRF)
# ==========================================

def reciprocal_rank_fusion(
    result_lists,
    k=RRF_K
):

    scores = defaultdict(float)
    documents = {}

    for result_list in result_lists:

        for rank, doc in enumerate(result_list):

            doc_id = (
                doc.metadata.get("source", ""),
                doc.metadata.get("page", 0),
                doc.page_content
            )

            scores[doc_id] += 1 / (k + rank + 1)

            documents[doc_id] = doc

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    return [
        documents[doc_id]
        for doc_id in ranked_ids
    ]


# ==========================================
# 9. HYBRID SEARCH
# ==========================================

def hybrid_search(
    bm25_retriever,
    vectorstore,
    query,
    source_filter=None
):

    # BM25 keyword search
    bm25_results = bm25_retriever.invoke(query)

    # Filter BM25 results if source is selected
    if source_filter:

        bm25_results = [
            doc for doc in bm25_results
            if os.path.basename(
                doc.metadata.get("source", "")
            ) == source_filter
        ]

    # Semantic vector search
    semantic_results = semantic_search(
        vectorstore=vectorstore,
        query=query,
        source_filter=source_filter
    )

    print("BM25 results:", len(bm25_results))
    print("Semantic results:", len(semantic_results))

    # Combine both result lists
    fused_results = reciprocal_rank_fusion(
        [
            bm25_results,
            semantic_results
        ]
    )

    print("RRF results:", len(fused_results))

    return fused_results


# ==========================================
# 10. RERANKING
# ==========================================

reranker = FlashrankRerank(
    top_n=RERANK_TOP_N
)


def rerank_documents(query, documents):

    if not documents:
        return []

    reranked_documents = reranker.compress_documents(
        documents=documents,
        query=query
    )

    print("Reranked documents:", len(reranked_documents))

    return reranked_documents


# ==========================================
# 11. CONTEXT BUILDING
# ==========================================

def build_context(documents):

    context_parts = []

    for doc in documents:

        source = os.path.basename(
            doc.metadata.get("source", "Unknown")
        )

        page = doc.metadata.get("page", 0) + 1

        context_parts.append(
            f"""
Source: {source}
Page: {page}

Content:
{doc.page_content}
"""
        )

    return "\n\n".join(context_parts)


# ==========================================
# 12. SOURCE CITATIONS
# ==========================================

def display_sources(documents):

    print("\n========== SOURCES ==========")

    seen_sources = set()

    for doc in documents:

        source = os.path.basename(
            doc.metadata.get("source", "Unknown")
        )

        page = doc.metadata.get("page", 0) + 1

        citation = (source, page)

        if citation not in seen_sources:

            print(f"- {source} | Page {page}")

            seen_sources.add(citation)


# ==========================================
# 13. RAG PROMPT
# ==========================================

prompt = ChatPromptTemplate.from_template(
    """
You are an Enterprise Document Assistant.

Answer the user's question using ONLY the provided context.

Instructions:

1. Do not make up information.
2. Do not use outside knowledge.
3. If the answer is not present in the context, say:
   "I don't know based on the provided documents."
4. Give a clear and structured answer.
5. Mention source filename and page number when relevant.

Context:
{context}

Question:
{question}

Answer:
"""
)

rag_chain = prompt | llm | StrOutputParser()


# ==========================================
# 14. COMPLETE ADVANCED RAG PIPELINE
# ==========================================

def advanced_rag(
    query,
    bm25_retriever,
    vectorstore,
    source_filter=None
):

    # Step 1: Hybrid Retrieval
    print("\n[1] Hybrid Retrieval")

    hybrid_results = hybrid_search(
        bm25_retriever=bm25_retriever,
        vectorstore=vectorstore,
        query=query,
        source_filter=source_filter
    )

    if not hybrid_results:
        return "No relevant documents found."

    # Step 2: Reranking
    print("\n[2] Reranking")

    reranked_results = rerank_documents(
        query=query,
        documents=hybrid_results
    )

    if not reranked_results:
        return "No relevant documents found."

    # Step 3: Context Building
    print("\n[3] Building Context")

    context = build_context(reranked_results)

    # Step 4: LLM Generation
    print("\n[4] Generating Answer")

    response = rag_chain.invoke({
        "context": context,
        "question": query
    })

    # Step 5: Citations
    display_sources(reranked_results)

    return response


# ==========================================
# 15. MAIN APPLICATION
# ==========================================

def main():

    # Load documents
    documents = load_documents()

    # Create chunks
    chunks = split_documents(documents)

    # Check Qdrant collection
    client = QdrantClient(path=QDRANT_PATH)

    collection_exists = client.collection_exists(
        collection_name=COLLECTION_NAME
    )

    client.close()

    # Create or load vector database
    if REBUILD_INDEX or not collection_exists:

        print("\nCreating Qdrant index...")

        vectorstore = create_vector_store(chunks)

    else:

        vectorstore = load_vector_store()

    # Initialize BM25
    bm25_retriever = create_bm25_retriever(chunks)

    print("\n====================================")
    print("         ADVANCED RAG SYSTEM")
    print("====================================")

    print("Hybrid Search: Enabled")
    print("BM25: Enabled")
    print("Semantic Search: Enabled")
    print("RRF: Enabled")
    print("Metadata Filtering: Enabled")
    print("Reranking: Enabled")
    print("Citations: Enabled")

    print("\nType 'exit' to quit.")

    # Chat loop
    while True:

        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        source_filter = None
        query = user_input

        # Optional source filter
        # Example:
        # source=notes.pdf | What are embeddings?

        if user_input.startswith("source=") and "|" in user_input:

            filter_part, query = user_input.split("|", 1)

            source_filter = filter_part.replace(
                "source=", ""
            ).strip()

            query = query.strip()

        try:

            answer = advanced_rag(
                query=query,
                bm25_retriever=bm25_retriever,
                vectorstore=vectorstore,
                source_filter=source_filter
            )

            print("\n========== AI ANSWER ==========")
            print(answer)

        except Exception as error:

            print("\nSomething went wrong:", error)


# ==========================================
# 16. APPLICATION ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()
