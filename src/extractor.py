# src/extractor.py
import requests
import json
import os

JUDGE_SYSTEM_PROMPT = """You are an expert medical information extraction evaluator.

Your task is to evaluate the quality of structured medical information extracted from clinical transcriptions.

You will receive:
1. Original medical transcription
2. Extracted structured data (JSON)

--- SCORING INSTRUCTIONS ---

For EACH field (symptoms, diagnoses, medications, procedures), compute a field_score in [0.0, 1.0] as follows:

  Let N = total number of items that should have been extracted (present in text)
  Let C = number of correctly extracted items (appear in text)
  Let M = number of missing items (present in text but not extracted)
  Let H = number of hallucinated items (extracted but NOT supported by text)

  correctness  = C / (C + H)   if (C + H) > 0, else 1.0
  completeness = C / (C + M)   if (C + M) > 0, else 1.0
  field_score  = (correctness + completeness) / 2

Then compute:
  overall_score = average of all four field_scores

--- OUTPUT FORMAT ---

Output ONLY valid JSON with no additional text:
{
  "field_scores": {
    "symptoms": 0.85,
    "diagnoses": 0.90,
    "medications": 0.75,
    "procedures": 0.95
  },
  "overall_score": 0.86,
  "missing_items": {
    "symptoms": ["item present in text but not extracted"],
    "diagnoses": [],
    "medications": [],
    "procedures": []
  },
  "hallucinations": {
    "symptoms": ["item extracted but not supported by text"],
    "diagnoses": [],
    "medications": [],
    "procedures": []
  },
  "explanation": "One sentence per field describing the main source of error, or 'No errors.' if field_score is 1.0."
}"""


def extract_medical_entities(transcription_text, model="llama3.2:3b", host="http://localhost:11434"):
    """
    Extracts medical entities from transcription using local LLM via Ollama.
    """
    prompt = f"""You are a medical information extraction system.
Extract entities from the medical transcription below.
Return ONLY a valid JSON object.

JSON format:
{{
    "diagnoses": ["list of diagnoses"],
    "medications": ["list of medications"],
    "symptoms": ["list of symptoms"],
    "procedures": ["list of procedures"]
}}

Medical transcription:
{transcription_text[:1000]}

JSON:"""

    response = requests.post(
        f"{host}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    raw = response.json()['response'].strip()
    try:
        start = raw.find('{')
        end = raw.rfind('}') + 1
        return json.loads(raw[start:end])
    except:
        return {"raw_response": raw}


def create_judge_prompt(transcription, extracted_json):
    return f"""Evaluate this medical information extraction.

ORIGINAL TRANSCRIPTION:
---
{transcription[:1000]}
---

EXTRACTED DATA:
---
{json.dumps(extracted_json, indent=2)}
---

Return ONLY valid JSON following the scoring schema exactly."""


def evaluate_extraction(transcription, extracted_json, model="llama3.2:3b", host="http://localhost:11434"):
    """
    Uses LLM as judge to evaluate extraction quality.
    """
    judge_prompt = create_judge_prompt(transcription, extracted_json)
    response = requests.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": JUDGE_SYSTEM_PROMPT + "\n\n" + judge_prompt,
            "stream": False
        }
    )
    raw = response.json()['response'].strip()
    try:
        start = raw.find('{')
        end = raw.rfind('}') + 1
        return json.loads(raw[start:end])
    except:
        return {"raw_response": raw}

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_JUDGE_MODEL = "openai/gpt-oss-120b"  # adressing self-evaluation bias


def _reasoning_effort_for_model(model):
    if model.startswith("openai/gpt-oss"):
        return "low"   # gpt-oss models only accept low/medium/high, no "none"
    if model.startswith("qwen/"):
        return "none"  # qwen models accept "none" to fully disable reasoning
    return None


def evaluate_extraction_groq(transcription, extracted_json, model=GROQ_JUDGE_MODEL, use_json_response_format=True):
    """
    Uses a stronger, heterogeneous LLM (via Groq API) as judge.
    Different model family/architecture than the local extractor -> addresses self-evaluation bias.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Run: export GROQ_API_KEY=your_key")

    judge_prompt = create_judge_prompt(transcription, extracted_json)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": judge_prompt}
        ],
        "temperature": 0,
        "max_completion_tokens": 2048,  # safety margin, in case some reasoning still slips through
    }
    if use_json_response_format:
        payload["response_format"] = {"type": "json_object"}

    reasoning_effort = _reasoning_effort_for_model(model)
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        if use_json_response_format:
            return evaluate_extraction_groq(transcription, extracted_json, model=model,
                                             use_json_response_format=False)
        return {"error": "Groq API request failed", "overall_score": None}

    raw = response.json()["choices"][0]["message"]["content"].strip()

    try:
        start = raw.find('{')
        end = raw.rfind('}') + 1
        return json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"raw_response": raw, "overall_score": None}

PANEL_MODELS = ["openai/gpt-oss-120b", "qwen/qwen3.6-27b"]


def evaluate_extraction_panel(transcription, extracted_json, panel_models=None,
                               include_self_eval_baseline=True,
                               local_model="llama3.2:3b", local_host="http://localhost:11434"):
    """
    Evaluates extraction using a panel of heterogeneous (cross-vendor) judges via Groq.
    The same-model self-evaluation score is computed separately as a baseline for
    comparison and is NOT included in the panel aggregate, since it is exactly the
    biased reference point the panel is meant to correct for.
    """
    panel_models = panel_models or PANEL_MODELS
    panel_results = {
        f"groq_{m}": evaluate_extraction_groq(transcription, extracted_json, model=m)
        for m in panel_models
    }

    panel_scores = [r.get("overall_score") for r in panel_results.values()
                     if isinstance(r.get("overall_score"), (int, float))]

    mean_score = sum(panel_scores) / len(panel_scores) if panel_scores else None
    std_score = (
        (sum((s - mean_score) ** 2 for s in panel_scores) / len(panel_scores)) ** 0.5
        if len(panel_scores) > 1 else 0.0
    )

    summary = {
        "panel_individual_scores": {k: v.get("overall_score") for k, v in panel_results.items()},
        "panel_mean": mean_score,
        "panel_std": std_score,  # simple agreement measure - lower std = judges agree more
    }

    if include_self_eval_baseline:
        self_eval = evaluate_extraction(transcription, extracted_json, model=local_model, host=local_host)
        summary["self_eval_score"] = self_eval.get("overall_score")

    return panel_results, summary

def run_extraction_pipeline(df, n_samples=5, model="llama3.2:3b", host="http://localhost:11434",
                             judge="groq", judge_model=None, panel_models=None):
    """
    judge: "groq" (single stronger judge), "ollama" (local baseline),
           or "panel" (heterogeneous panel + self-eval baseline for comparison)
    """
    samples = df.sample(n=n_samples, random_state=42)
    results = []

    for idx, row in samples.iterrows():
        print(f"\n{'='*50}")
        print(f"Specialty: {row['medical_specialty']}")
        print(f"{'='*50}")

        extracted = extract_medical_entities(row['transcription'], model=model, host=host)
        print("Extracted:")
        print(json.dumps(extracted, indent=2))

        if judge == "panel":
            panel_results, panel_summary = evaluate_extraction_panel(
                row['transcription'], extracted, panel_models=panel_models
            )
            print(f"\nPanel scores: {panel_summary['panel_individual_scores']}")
            print(f"Panel mean: {panel_summary['panel_mean']:.3f} | std: {panel_summary['panel_std']:.3f}")
            print(f"Self-eval baseline: {panel_summary.get('self_eval_score')}")
            evaluation = {"overall_score": panel_summary["panel_mean"], "panel_summary": panel_summary}
        elif judge == "groq":
            evaluation = evaluate_extraction_groq(row['transcription'], extracted, model=judge_model or GROQ_JUDGE_MODEL)
        else:
            evaluation = evaluate_extraction(row['transcription'], extracted, model=model, host=host)

        print(f"\nJudge Score: {evaluation.get('overall_score', 'N/A')}")
        results.append({'specialty': row['medical_specialty'], 'extracted': extracted, 'evaluation': evaluation})

    scores = [r['evaluation'].get('overall_score', 0) for r in results
              if isinstance(r['evaluation'].get('overall_score'), float)]
    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"\n=== AVERAGE JUDGE SCORE: {avg_score:.3f} ===")

    return results