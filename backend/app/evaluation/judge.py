import json
import time
from typing import Dict, Any
from google import genai
from google.genai import errors
from app.core.config import settings

def safe_generate_evaluation(prompt: str, max_retries: int = 4) -> str:
    """Handles rate limits (429) automatically with exponential backoff."""
    client = genai.Client()
    delay = 10
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=settings.gemini_chat_model,
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

def judge_answer_faithfulness_and_relevance(
    question: str,
    context: str,
    answer: str,
    is_unanswerable: bool
) -> Dict[str, Any]:
    """LLM-as-a-Judge scoring engine."""
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
        raw_text = safe_generate_evaluation(judge_prompt)
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
