"""
intelligence_groq.py — Agentic AI layer using Groq (Open Source LLMs)
=====================================================================
Replaces Claude API with Groq (LLaMA 3.3 / Mixtral).

Features:
  1. explain_prediction
  2. analyze_batch
  3. detect_evasion

All methods are async and return structured JSON-ready dicts.
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

GROQ_MODEL     = "llama-3.3-70b-versatile"
GROQ_URL       = "https://api.groq.com/openai/v1/chat/completions"
MAX_TOKENS     = 1024
REQUEST_TIMEOUT = 30.0


MODEL_DISPLAY = {
    "mnb_tfidf":  "Multinomial Naïve Bayes (TF-IDF)",
    "mnb_count":  "Multinomial Naïve Bayes (Count Vectorizer)",
    "simplernn":  "Simple RNN (GloVe embeddings)",
    "lstm":       "LSTM (GloVe embeddings)",
    "gru":        "GRU (GloVe embeddings)",
    "bilstm":     "Bidirectional LSTM (GloVe embeddings)",
}


class Intelligence:
    """Agentic AI module — Groq-backed explainability."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("GROQ_API_KEY", "")
        if not self._api_key:
            log.warning("GROQ_API_KEY not set. Returning placeholder responses.")

        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    # ── private ─────────────────────────────────────────

    async def _call(self, system: str, user: str) -> str:
        if not self._api_key:
            return json.dumps({
                "error": "GROQ_API_KEY not configured",
                "hint": "Add GROQ_API_KEY=... to .env"
            })

        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": 0.2,  # IMPORTANT for JSON stability
            "max_tokens": MAX_TOKENS,
        }

        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                resp = await client.post(GROQ_URL, headers=self._headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]

        except httpx.HTTPStatusError as e:
            log.error("Groq API HTTP error: %s — %s", e.response.status_code, e.response.text)
            return json.dumps({"error": f"API error {e.response.status_code}"})

        except Exception as e:
            log.error("Groq API error: %s", e)
            return json.dumps({"error": str(e)})

    @staticmethod
    def _parse_json_response(raw: str) -> dict:
        cleaned = raw.strip()

        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {
                "raw_response": raw,
                "parse_error": "Invalid JSON"
            }

    # ── public methods (unchanged logic) ────────────────

    async def explain_prediction(
        self,
        text: str,
        label: str,
        probability: float,
        model_name: str = "bilstm",
    ) -> dict:

        model_display = MODEL_DISPLAY.get(model_name, model_name)
        confidence_pct = round(
            probability * 100 if label == "spam" else (1 - probability) * 100, 1
        )

        system = (
            "You are an expert in spam detection and NLP. "
            "Return ONLY valid JSON."
        )

        user = f"""
Message: "{text}"
Classification: {label}
Confidence: {confidence_pct}%

Return JSON:
{{
  "verdict": "...",
  "confidence_level": "...",
  "key_signals": ["..."],
  "reasoning": "...",
  "spam_category": "...",
  "tip": "..."
}}
"""

        raw = await self._call(system, user)
        result = self._parse_json_response(raw)

        result.update({
            "text": text,
            "label": label,
            "probability": probability,
            "model": model_name,
        })

        return result

    async def analyze_batch(
        self,
        results: List[dict],
        model_name: str = "bilstm",
    ) -> dict:

        spam_msgs = [r for r in results if r.get("label") == "spam"]
        ham_msgs  = [r for r in results if r.get("label") == "ham"]

        if not results:
            return {"error": "No data"}

        spam_sample = spam_msgs[:20]
        spam_texts  = [r["text"] for r in spam_sample]

        system = "You are a spam analyst. Return ONLY JSON."

        user = f"""
Spam messages:
{spam_texts}

Return JSON:
{{
  "threat_level": "...",
  "categories": [],
  "top_patterns": [],
  "summary": "...",
  "recommendation": "..."
}}
"""

        raw = await self._call(system, user)
        result = self._parse_json_response(raw)
        result["model"] = model_name

        return result

    async def detect_evasion(self, text: str) -> dict:

        system = "You are a cybersecurity expert. Return ONLY JSON."

        user = f"""
Message: "{text}"

Return JSON:
{{
  "evasion_risk": "...",
  "is_likely_spam": true,
  "evasion_techniques": [],
  "red_flags": [],
  "reasoning": "...",
  "recommended_action": "..."
}}
"""

        raw = await self._call(system, user)
        result = self._parse_json_response(raw)
        result["text"] = text

        return result