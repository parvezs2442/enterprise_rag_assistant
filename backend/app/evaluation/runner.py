import os
import json
import time
from typing import List, Dict, Any
from app.core.config import settings
from app.evaluation.metrics import evaluate_retrieval_ranking
from app.evaluation.judge import judge_answer_faithfulness_and_relevance
from app.generation.security import format_context_with_fences

GOLDEN_DATASET_PATH = "evals/golden_dataset.json"
REPORT_OUTPUT_PATH = "evals/eval_report.json"

def run_evaluation_benchmark(pipeline, max_cases: int = 4):
    """Evaluation Suite runner."""
    print("\n" + "="*80)
    print("      ENTERPRISE RAG ASSISTANT - QUALITY EVALUATION SUITE")
    print("="*80)

    if not os.path.exists(GOLDEN_DATASET_PATH):
        raise FileNotFoundError(f"Golden dataset file '{GOLDEN_DATASET_PATH}' nahi mili!")

    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)[:max_cases]

    print(f"[OK] Loaded {len(dataset)} test cases from {GOLDEN_DATASET_PATH}")
    print("\n[Running Evaluation: Baseline (Dense) vs Tuned (Hybrid + Reranker)]")
    print("-" * 80)

    results_baseline = []
    results_tuned = []

    for i, item in enumerate(dataset, start=1):
        q_id = item["id"]
        q_text = item["question"]
        gt_page = item.get("ground_truth_page")
        is_unans = item.get("expected_abstain", False)

        print(f"\n▶ [{i}/{len(dataset)}] Test Case: {q_id} ({item['type']})")
        print(f"  Q: \"{q_text}\"")

        # --- Baseline ---
        t0 = time.time()
        base_docs = pipeline.dense_search(q_text, k=4)
        base_ret = evaluate_retrieval_ranking(base_docs, gt_page)
        base_ans = pipeline.generate_answer(q_text, base_docs)
        t_base = time.time() - t0

        time.sleep(3)
        base_judge = judge_answer_faithfulness_and_relevance(
            question=q_text,
            context=format_context_with_fences(base_docs),
            answer=base_ans,
            is_unanswerable=is_unans
        )

        results_baseline.append({
            "id": q_id,
            "type": item["type"],
            "hit": base_ret["hit"],
            "rr": base_ret["rr"],
            "faithfulness": base_judge.get("faithfulness", 0.0),
            "relevance": base_judge.get("relevance", 0.0),
            "abstention_correct": base_judge.get("abstention_correct", False),
            "latency_sec": round(t_base, 3)
        })

        time.sleep(3)

        # --- Tuned ---
        t0 = time.time()
        tuned_docs = pipeline.retrieve(q_text, use_reranker=True)
        tuned_ret = evaluate_retrieval_ranking(tuned_docs, gt_page)
        tuned_ans = pipeline.generate_answer(q_text, tuned_docs)
        t_tuned = time.time() - t0

        time.sleep(3)
        tuned_judge = judge_answer_faithfulness_and_relevance(
            question=q_text,
            context=format_context_with_fences(tuned_docs),
            answer=tuned_ans,
            is_unanswerable=is_unans
        )

        results_tuned.append({
            "id": q_id,
            "type": item["type"],
            "hit": tuned_ret["hit"],
            "rr": tuned_ret["rr"],
            "faithfulness": tuned_judge.get("faithfulness", 0.0),
            "relevance": tuned_judge.get("relevance", 0.0),
            "abstention_correct": tuned_judge.get("abstention_correct", False),
            "latency_sec": round(t_tuned, 3)
        })

        print(f"  [Baseline] Hit: {base_ret['hit']} | Faith: {base_judge.get('faithfulness'):.2f} | Latency: {t_base:.2f}s")
        print(f"  [Tuned]    Hit: {tuned_ret['hit']} | Faith: {tuned_judge.get('faithfulness'):.2f} | Latency: {t_tuned:.2f}s")

    def aggregate(res_list):
        fac = [r for r in res_list if "answerable" in r["type"]]
        unans = [r for r in res_list if "unanswerable" in r["type"] or "adversarial" in r["type"]]
        hit = sum(r["hit"] for r in fac) / len(fac) if fac else 0.0
        mrr = sum(r["rr"] for r in fac) / len(fac) if fac else 0.0
        faith = sum(r["faithfulness"] for r in res_list) / len(res_list)
        rel = sum(r["relevance"] for r in res_list) / len(res_list)
        abs_acc = sum(1 for r in unans if r["abstention_correct"]) / len(unans) if unans else 1.0
        lat = sum(r["latency_sec"] for r in res_list) / len(res_list)
        return {
            "hit_rate_at_4": round(hit * 100, 1),
            "mrr": round(mrr, 3),
            "avg_faithfulness": round(faith * 100, 1),
            "avg_relevance": round(rel * 100, 1),
            "abstention_accuracy": round(abs_acc * 100, 1),
            "avg_latency_sec": round(lat, 2)
        }

    agg_base = aggregate(results_baseline)
    agg_tuned = aggregate(results_tuned)

    print("\n" + "="*80)
    print("                 BENCHMARK EVALUATION REPORT SUMMARY")
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
        "model_evaluated": settings.gemini_chat_model,
        "embedding_model": settings.gemini_embed_model,
        "metrics_summary": {"baseline": agg_base, "tuned": agg_tuned},
        "test_case_results": {"baseline": results_baseline, "tuned": results_tuned}
    }

    os.makedirs(os.path.dirname(REPORT_OUTPUT_PATH), exist_ok=True)
    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n[OK] Full evaluation report exported to '{REPORT_OUTPUT_PATH}'!")
