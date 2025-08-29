# app.py — Nykra CAC/Ad Spend backend (GPT-5 Mini)

import os, re
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from openai import OpenAI

# ---- Config ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")  # default to GPT-5 Mini
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---- Payload (matches your front-end field names) ----
class AnalyzePayload(BaseModel):
    name: str
    email: EmailStr
    business_url: Optional[str] = None
    business_description: Optional[str] = None
    offer: Optional[str] = None
    offer_details: Optional[str] = None
    item_price: Optional[str] = None        # text like "$10,000"
    expected_profit: Optional[str] = None   # text like "$300"
    recurring: Optional[str] = None         # "yes / no / sometimes"
    current_ad_spend: Optional[str] = None  # weekly, text like "$500"
    ad_platforms: Optional[str] = None
    struggle: Optional[str] = None

def _to_number(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    cleaned = re.sub(r"[^\d.]", "", s)
    try:
        return float(cleaned) if cleaned else None
    except:
        return None

def _build_prompt(p: AnalyzePayload) -> str:
    price = _to_number(p.item_price)
    profit = _to_number(p.expected_profit)
    spend_week = _to_number(p.current_ad_spend)

    return f"""
You are a friendly senior growth strategist for small businesses. Use plain English.

USER INPUT
- Name: {p.name}
- Email: {p.email}
- Business URL: {p.business_url or "N/A"}
- Business description: {p.business_description or "N/A"}
- Advertising: {p.offer or "N/A"}
- Details: {p.offer_details or "N/A"}
- Price / avg sale value: {price if price is not None else "N/A"}
- Expected profit per sale: {profit if profit is not None else "N/A"}
- Do customers come back?: {p.recurring or "N/A"}
- Current WEEKLY ad spend: {spend_week if spend_week is not None else "N/A"}
- Platforms: {p.ad_platforms or "N/A"}
- Biggest struggle: {p.struggle or "N/A"}

TASK
Create a concise, structured plan with sections:
1) Quick Summary (2–3 bullets).
2) What to know (define CAC = ad spend ÷ new customers; ROAS = revenue ÷ ad spend) in one short paragraph.
3) Verdict: are ads worth running now? Why?
4) Budget & Duration:
   - Give weekly and monthly spend ranges using their numbers if possible.
   - Recommend burst vs always-on and why.
5) Targeting hints: audiences/keywords/geo based on their description.
6) Simple math example using any numbers they gave (state assumptions if something is missing).
7) Funnel fixes & workarounds: upsells/bundles, small price lift, landing page improvements, retargeting, offer tweaks.
8) 4-week action plan with thresholds (Week 1–2 test; Week 3–4 scale/pullback with clear CAC/ROAS rules).
9) Soft CTA (1–2 sentences) offering help to implement.

RULES
- Beginner-friendly; minimal jargon.
- If numbers are missing, make small, explicit assumptions.
- Use short paragraphs and bullets; no tables.
- Keep currency generic using the $ sign.
"""

def _as_html(text: str) -> str:
    safe = (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
    )
    return f'<div class="nykra-calc-result" style="line-height:1.55;color:#e8eaed">{safe}</div>'

@app.post("/analyze")
def analyze(payload: AnalyzePayload):
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are Nykra Studio's helpful growth strategist. Be clear, actionable, kind."},
                {"role": "user", "content": _build_prompt(payload)},
            ],
            temperature=0.35,
        )
        text = completion.choices[0].message.content.strip()
        return {"result_html": _as_html(text), "plain_text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")
