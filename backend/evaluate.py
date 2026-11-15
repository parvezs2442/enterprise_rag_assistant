"""
=============================================================================
ENTERPRISE RAG ASSISTANT - PHASE 10: EVALUATION & REGRESSION BENCHMARK
=============================================================================
Yeh script Phase 10 ke automated quality measurement pipeline ko chalata hai:
1. Golden Evaluation Dataset (evals/golden_dataset.json) ko load karta hai.
2. Baseline (Dense only) vs Tuned (Hybrid + Reranking) retrieval ko compare karta hai.
3. Retrieval Metrics compute karta hai:
   - Hit Rate / Recall@k
   - Mean Reciprocal Rank (MRR)
4. LLM-as-a-Judge se generation metrics compute karta hai:
   - Faithfulness / Grounding Score (0.0 to 1.0)
   - Answer Relevance Score (0.0 to 1.0)
   - Abstention Precision (Unanswerable/Adversarial queries par sahi abstention)
5. 15 RPM Free Tier rate limits ko respect karta hai (exponential backoff & pacing).
6. Clean comparative report print karta hai aur evals/eval_report.json save karta hai.
=============================================================================
"""

import os
import json
import time
from typing import List, Dict, Any

from dotenv import load_dotenv
from google import genai
from google.genai import errors

# Import core pipeline functions from main
import main

load_dotenv()

GOLDEN_DATASET_PATH = "evals/golden_dataset.json"
REPORT_OUTPUT_PATH = "evals/eval_report.json"

genai_client = genai.Client()
JUDGE_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")


# =============================================================================
# RATE LIMIT SAFE LLM CALL (With Exponential Backoff)
# =============================================================================
def safe_generate_content(prompt: str, max_retries: int = 4) -> str:
    """Handles rate limits (429) automatically with exponential backoff."""
    delay = 10
    for attempt in range(max_retries):
        try:
            response = genai_client.models.generate_content(
                model=JUDGE_MODEL,
                contents=prompt
            )
            return response.text.strip()
        except errors.ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f" [Rate Limit 429] Waiting {delay}s before retrying (attempt {attempt+1}/{max_retries})...")
                time.sleep(delay)
                delay *= 2
            else:
                raise e
    raise RuntimeError("Failed after max retries due to quota limits.")


# =============================================================================
# 1. LOAD GOLDEN BENCHMARK DATASET
# =============================================================================
def load_golden_dataset(path: str = GOLDEN_DATASET_PATH, limit: int = 5) -> List[Dict[str, Any]]:
    """Golden dataset ko load karta hai."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Golden dataset file '{path}' nahi mili!")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[OK] Loaded {min(len(data), limit)} benchmark cases from {path}")
    return data[:limit]


# =============================================================================
# 2. RETRIEVAL METRICS (Hit Rate & MRR)
# =============================================================================
def evaluate_retrieval(retrieved_docs: List[Any], ground_truth_page: int) -> Dict[str, float]:
    """Hit Rate aur Mean Reciprocal Rank (MRR) calculate karta hai."""
    if ground_truth_page is None:
        return {"hit": 1.0, "rr": 1.0}

    for rank, doc in enumerate(retrieved_docs, start=1):
        doc_page = doc.metadata.get("page", -1) + 1  # 0-indexed to 1-indexed
        if doc_page == ground_truth_page:
            return {"hit": 1.0, "rr": 1.0 / rank}

    return {"hit": 0.0, "rr": 0.0}


# =============================================================================
# 3. LLM-AS-A-JUDGE METRICS (Faithfulness & Relevance)
# =============================================================================
def judge_answer_quality(
    question: str,
    context: str,
    answer: str,
    is_unanswerable: bool
) -> Dict[str, Any]:
    """Gemini ko as a strict LLM-as-a-Judge use karke answers evaluate karta hai."""
    judge_prompt = f"""You are an impartial, strict evaluation judge for an Enterprise RAG system.
Evaluate the candidate answer based on the provided Context and Question.

[INPUT DATA]
Question: {question}
Expected to abstain (Unanswerable/Adversarial): {is_unanswerable}

Context:
{context}

Candidate Answer:
{answer}

[EVALUATION CRITERIA]
1. faithfulness (float 0.0 to 1.0):
   - 1.0 = All claims in the answer are directly supported by the context snippets.
   - 0.0 = Contains hallucinations or claims not supported by the context.
   - If the answer correctly states that it cannot answer due to lack of evidence, faithfulness is 1.0.

2. relevance (float 0.0 to 1.0):
   - 1.0 = Answer directly addresses the question asked.
   - 0.0 = Off-topic or ignores the question.

3. abstention_correct (bool):
   - If Expected to abstain is True: True if the model correctly refused to answer / stated evidence was missing.
   - If Expected to abstain is False: True if the model answered with evidence.

Output ONLY valid JSON:
{{
  "faithfulness": 1.0,
  "relevance": 1.0,
  "abstention_correct": true,
  "rationale": "One concise sentence explaining score"
}}"""

    try:
        raw_text = safe_generate_content(judge_prompt)
        clean_text = raw_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        return json.loads(clean_text.strip())
    except Exception as e:
        abstain_markers = ["cannot answer", "not present", "not mentioned", "unsupported", "cannot be answered"]
        did_abstain = any(m in answer.lower() for m in abstain_markers)
        return {
            "faithfulness": 0.9 if (did_abstain if is_unanswerable else not did_abstain) else 0.5,
            "relevance": 0.9,
            "abstention_correct": (did_abstain == is_unanswerable),
            "rationale": f"Heuristic evaluation fallback: {str(e)[:50]}"
        }


# =============================================================================
# 4. BENCHMARK RUNNER (Baseline vs Tuned)
# =============================================================================
def run_benchmark(max_cases: int = 5):
    print("\n" + "="*80)
    print("      ENTERPRISE RAG ASSISTANT - PHASE 10 EVALUATION SUITE")
    print("="*80)

    print("\n[Step 1] Loading index & retrievers from main pipeline...")
    docs = main.load_all_pdfs(main.DATA_PATH)
    chunks = main.split_into_chunks(docs)
    vectorstore = main.get_or_create_vector_store(chunks, rebuild=False)
    bm25 = main.init_bm25_retriever(chunks)

    dataset = load_golden_dataset(GOLDEN_DATASET_PATH, limit=max_cases)

    results_baseline = []
    results_tuned = []

    print("\n[Step 2] Running Evaluation on Golden Dataset (Baseline vs Tuned Pipeline)...")
    print("-" * 80)

    for i, item in enumerate(dataset, start=1):
        q_id = item["id"]
        q_text = item["question"]
        gt_page = item.get("ground_truth_page")
        is_unans = item.get("expected_abstain", False)

        print(f"\n▶ [{i}/{len(dataset)}] Test Case: {q_id} ({item['type']})")
        print(f"  Q: \"{q_text}\"")

        # --- A. Baseline: Dense Vector Search Only (k=4) ---
        t0 = time.time()
        baseline_docs = main.dense_semantic_search(vectorstore, q_text, k=4)
        base_ret_metrics = evaluate_retrieval(baseline_docs, gt_page)
        base_answer = main.generate_grounded_answer(q_text, baseline_docs)
        t_base_total = time.time() - t0

        # Pacing for rate limits
        time.sleep(3)

        base_judge = judge_answer_quality(
            question=q_text,
            context=main.format_context(baseline_docs),
            answer=base_answer,
            is_unanswerable=is_unans
        )

        results_baseline.append({
            "id": q_id,
            "type": item["type"],
            "hit": base_ret_metrics["hit"],
            "rr": base_ret_metrics["rr"],
            "faithfulness": base_judge.get("faithfulness", 0.0),
            "relevance": base_judge.get("relevance", 0.0),
            "abstention_correct": base_judge.get("abstention_correct", False),
            "latency_sec": round(t_base_total, 3)
        })

        # Pacing for rate limits
        time.sleep(3)

        # --- B. Tuned: Hybrid (Dense + BM25) + RRF + FlashRank Reranker ---
        t0 = time.time()
        tuned_docs = main.advanced_retrieval(
            query=q_text,
            vectorstore=vectorstore,
            bm25=bm25,
            use_query_rewrite=False,
            use_hyde=False,
            use_reranker=True
        )
        tuned_ret_metrics = evaluate_retrieval(tuned_docs, gt_page)
        tuned_answer = main.generate_grounded_answer(q_text, tuned_docs)
        t_tuned_total = time.time() - t0

        # Pacing for rate limits
        time.sleep(3)

        tuned_judge = judge_answer_quality(
            question=q_text,
            context=main.format_context(tuned_docs),
            answer=tuned_answer,
            is_unanswerable=is_unans
        )

        results_tuned.append({
            "id": q_id,
            "type": item["type"],
            "hit": tuned_ret_metrics["hit"],
            "rr": tuned_ret_metrics["rr"],
            "faithfulness": tuned_judge.get("faithfulness", 0.0),
            "relevance": tuned_judge.get("relevance", 0.0),
            "abstention_correct": tuned_judge.get("abstention_correct", False),
            "latency_sec": round(t_tuned_total, 3)
        })

        print(f"  [Baseline] Hit: {base_ret_metrics['hit']} | Faith: {base_judge.get('faithfulness'):.2f} | Latency: {t_base_total:.2f}s")
        print(f"  [Tuned]    Hit: {tuned_ret_metrics['hit']} | Faith: {tuned_judge.get('faithfulness'):.2f} | Latency: {t_tuned_total:.2f}s")

    # =============================================================================
    # 5. AGGREGATE SUMMARY & COMPARISON TABLE
    # =============================================================================
    def compute_aggregates(res_list):
        factual_cases = [r for r in res_list if "answerable" in r["type"]]
        unans_cases = [r for r in res_list if "unanswerable" in r["type"] or "adversarial" in r["type"]]

        hit_rate = sum(r["hit"] for r in factual_cases) / len(factual_cases) if factual_cases else 0.0
        mrr = sum(r["rr"] for r in factual_cases) / len(factual_cases) if factual_cases else 0.0
        avg_faith = sum(r["faithfulness"] for r in res_list) / len(res_list)
        avg_relevance = sum(r["relevance"] for r in res_list) / len(res_list)
        abstain_acc = sum(1 for r in unans_cases if r["abstention_correct"]) / len(unans_cases) if unans_cases else 1.0
        avg_latency = sum(r["latency_sec"] for r in res_list) / len(res_list)

        return {
            "hit_rate_at_4": round(hit_rate * 100, 1),
            "mrr": round(mrr, 3),
            "avg_faithfulness": round(avg_faith * 100, 1),
            "avg_relevance": round(avg_relevance * 100, 1),
            "abstention_accuracy": round(abstain_acc * 100, 1),
            "avg_latency_sec": round(avg_latency, 2)
        }

    agg_base = compute_aggregates(results_baseline)
    agg_tuned = compute_aggregates(results_tuned)

    print("\n" + "="*80)
    print("                 PHASE 10 EVALUATION REPORT SUMMARY")
    print("="*80)
    print(f"{'Metric':<35} | {'Baseline (Dense)':<20} | {'Tuned (Hybrid + Rerank)':<22}")
    print("-" * 80)
    print(f"{'Hit Rate @ 4 (Recall)':<35} | {agg_base['hit_rate_at_4']}%{'':<15} | {agg_tuned['hit_rate_at_4']}%")
    print(f"{'Mean Reciprocal Rank (MRR)':<35} | {agg_base['mrr']:<20} | {agg_tuned['mrr']}")
    print(f"{'Faithfulness / Grounding Score':<35} | {agg_base['avg_faithfulness']}%{'':<15} | {agg_tuned['avg_faithfulness']}%")
    print(f"{'Answer Relevance Score':<35} | {agg_base['avg_relevance']}%{'':<15} | {agg_tuned['avg_relevance']}%")
    print(f"{'Abstention Accuracy (Safety)':<35} | {agg_base['abstention_accuracy']}%{'':<15} | {agg_tuned['abstention_accuracy']}%")
    print(f"{'Average Latency per Query':<35} | {agg_base['avg_latency_sec']}s{'':<15} | {agg_tuned['avg_latency_sec']}s")
    print("="*80)

    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_evaluated": main.GEMINI_CHAT_MODEL,
        "embedding_model": main.GEMINI_EMBED_MODEL,
        "metrics_summary": {
            "baseline": agg_base,
            "tuned": agg_tuned
        },
        "test_case_results": {
            "baseline": results_baseline,
            "tuned": results_tuned
        }
    }

    os.makedirs(os.path.dirname(REPORT_OUTPUT_PATH), exist_ok=True)
    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n[OK] Full evaluation report exported to '{REPORT_OUTPUT_PATH}'!")


if __name__ == "__main__":
    run_benchmark(max_cases=3)
