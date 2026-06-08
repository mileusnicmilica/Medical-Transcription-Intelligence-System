# src/extractor.py
import requests
import json

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
{transcription[:3000]}
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


def run_extraction_pipeline(df, n_samples=5, model="llama3.2:3b", host="http://localhost:11434"):
    """
    Runs extraction and evaluation on n_samples from the dataframe.
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

        evaluation = evaluate_extraction(row['transcription'], extracted, model=model, host=host)
        print(f"\nJudge Score: {evaluation.get('overall_score', 'N/A')}")

        results.append({
            'specialty': row['medical_specialty'],
            'extracted': extracted,
            'evaluation': evaluation
        })

    scores = [r['evaluation'].get('overall_score', 0) for r in results
              if isinstance(r['evaluation'].get('overall_score'), float)]
    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"\n=== AVERAGE JUDGE SCORE: {avg_score:.3f} ===")

    return results