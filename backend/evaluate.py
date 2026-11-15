"""
=============================================================================
ENTERPRISE RAG ASSISTANT - STANDALONE EVALUATION ENTRYPOINT
=============================================================================
Direct CLI runner for Phase 10 evaluation benchmarks.
=============================================================================
"""

import sys
from app.pipeline import EnterpriseRAGPipeline
from app.evaluation.runner import run_evaluation_benchmark

def main():
    print("\n[Enterprise RAG] Initializing pipeline for automated evaluation...")
    pipeline = EnterpriseRAGPipeline()
    pipeline.initialize()
    run_evaluation_benchmark(pipeline, max_cases=3)

if __name__ == "__main__":
    main()
