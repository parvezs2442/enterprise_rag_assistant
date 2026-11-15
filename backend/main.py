"""
=============================================================================
ENTERPRISE RAG ASSISTANT - PRODUCTION APPLICATION ENTRYPOINT
=============================================================================
Clean, modular CLI interface using EnterpriseRAGPipeline (app package).
=============================================================================
"""

import sys
from app.pipeline import EnterpriseRAGPipeline
from app.evaluation.runner import run_evaluation_benchmark

def main():
    print("\n" + "="*70)
    print("      ENTERPRISE RAG ASSISTANT - PRODUCTION SYSTEM")
    print("="*70)

    # Initialize modular RAG pipeline
    pipeline = EnterpriseRAGPipeline()
    pipeline.initialize()

    print("-" * 70)
    print("Commands:")
    print(" • <question>               : Run Hybrid Retrieval + FlashRank + Grounded Citations")
    print(" • compare <query>          : Compare Dense vs BM25 vs Hybrid vs Reranker")
    print(" • rewrite <query>          : Optimize and expand query before search")
    print(" • hyde <query>             : Use Hypothetical Document Embeddings (HyDE)")
    print(" • source=file.pdf | <query>: Restrict search to specific document")
    print(" • eval                     : Run Phase 10 automated evaluation benchmark")
    print(" • security_test            : Run Phase 9 prompt-injection defense audit")
    print(" • exit                     : Quit")
    print("-" * 70)

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

        # Command: Evaluation Suite (Phase 10)
        if user_input.lower() == "eval":
            run_evaluation_benchmark(pipeline)
            continue

        # Command: Security Audit (Phase 9)
        if user_input.lower() == "security_test":
            pipeline.run_security_audit()
            continue

        # Command: Retrieval Comparison (Phase 8)
        if user_input.startswith("compare "):
            query = user_input.replace("compare ", "").strip()
            pipeline.compare_strategies(query)
            continue

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
        retrieved_docs = pipeline.retrieve(
            query=query,
            source_filter=source_filter,
            use_rewrite=use_rewrite,
            use_hyde=use_hyde,
            use_reranker=True
        )

        print(f" -> Found {len(retrieved_docs)} top reranked chunks.")
        print("[Generating grounded answer with Gemini...]")
        answer = pipeline.generate_answer(query, retrieved_docs)

        # Citation Verification (Phase 9)
        citation_info = pipeline.verify_citations(answer, retrieved_docs)

        print("\n" + "="*50 + " AI ANSWER " + "="*50)
        print(answer)
        print("="*111)

        print("\n[Phase 9 Citation Verification]")
        if citation_info["is_citation_valid"] and citation_info["valid_citations"]:
            print(f"  All citations valid: {citation_info['valid_citations']} (Precision: 100%)")
            for cid in citation_info["valid_citations"]:
                meta = citation_info["metadata_map"].get(cid, {})
                print(f"   [{cid}] -> {meta.get('source')} (Page {meta.get('page')}) | Rerank Score: {meta.get('score')}")
        elif citation_info["hallucinated_citations"]:
            print(f" ⚠️ WARNING: Hallucinated citations detected: {citation_info['hallucinated_citations']}")
        else:
            print("  ℹ️ No citations required (General/Abstention answer).")

if __name__ == "__main__":
    main()
