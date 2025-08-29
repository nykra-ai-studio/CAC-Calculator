# app.py — Nykra CAC/Ad Spend backend (Beginner-First, Hardwired Opening)

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

# ---- Payload (matches your front-end names; location added) ----
class AnalyzePayload(BaseModel):
    name: str
    email: EmailStr
    business_url: Optional[str] = None
    business_location: Optional[str] = None     # NEW (ok if None)
    business_description: Optional[str] = None
    offer: Optional[str] = None                 # "Product, Service, ..." (joined)
    offer_details: Optional[str] = None         # unused by current UI; kept for compat
    item_price: Optional[str] = None            # "$10,000"
    expected_profit: Optional[str] = None       # kept for compat
    recurring: Optional[str] = None             # kept for compat
    current_ad_spend: Optional[str] = None      # weekly "$500"
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
    spend_week = _to_number(p.current_ad_spend)

    return f"""
You are a friendly senior growth strategist for beginners. Output must be crystal clear and short-paragraph/bulleted.

USER INPUT
- Name: {p.name}
- Email: {p.email}
- Business URL: {p.business_url or "N/A"}
- Location: {p.business_location or "N/A"}
- What they are advertising (type): {p.offer or "N/A"}
- Business description: {p.business_description or "N/A"}
- Average sale value: {price if price is not None else "N/A"}
- Current WEEKLY ad spend: {spend_week if spend_week is not None else "N/A"}
- Platforms (user plan): {p.ad_platforms or "N/A"}
- Biggest struggle: {p.struggle or "N/A"}

ABSOLUTE TONE RULES
- Speak to a total beginner. No jargon. If a term appears, define it simply in the same line.
- Keep lists short. Do not overwhelm.
- Use sensible default assumptions when data is missing, and say the assumption plainly.

HARDWIRED OPENING (must be the very first section; keep it tight and numbers-filled):
1) One paragraph: “This plan helps you decide ad spend and if ads are worth it.”
   - Define: CAC = how much you spend to win 1 customer (roughly cost-per-click ÷ conversion-rate). ROAS = revenue ÷ ad spend (how many dollars you get back per $1).
   - Pick ONE {p.business_location or "generic"}-sensible CPC assumption **in $** based on the channel most likely for their business:
       * If they mention Google or high-intent search/service: CPC ~ $1.5–$6 → choose a single number (e.g., $3).
       * If Meta/Instagram/ecom/awareness: CPC ~ $0.8–$3 → choose a single number (e.g., $1.50).
       * If nothing given, choose a conservative middle (e.g., $2.00) and say it's a guess we’ll refine.
   - Use a default 1% conversion rate if unknown.
   - Compute example CAC = CPC / 1% (e.g., $2.00 / 0.01 = $200 per customer).
   - If average sale value is provided, compute example ROAS on one sale at that CAC (ROAS = price ÷ CAC). Say “worth it” if ROAS > 1; else “not worth it yet” and state why.
   - If price missing, assume a modest price for their category and show the same math, clearly labeled “assumption”.

TARGET AUDIENCE (very short)
- Give 3–5 precise targets only (demographics, interests, job titles or keywords) tailored to the business and {p.business_location or "their region"}.
- End this section with: “If interested, Nykra’s AI Ad Optimiser auto-tunes your targeting after every click to lower your costs over time.”

BUDGET & DURATION (simple)
- Give a weekly and monthly spend range.
- Recommend “always-on baseline” vs “short burst” depending on offer type (high-ticket/limited spots → bursts; services/subscriptions → always-on).
- If they entered a current weekly spend, anchor advice to it.

PLAN (4 weeks, beginner-proof)
- Week 1–2 (Test): exact daily budget, 2–3 creative ideas, 1 landing-page tweak, simple success targets: CAC goal and ROAS goal.
- Week 3–4 (Decide): 
   * If CAC ≤ target and ROAS ≥ target → scale budget by 20–50% on winners.
   * If close to targets → keep spend flat, fix 1–2 bottlenecks (ad, page, offer), retest.
   * If far from targets → pause losers, keep only the best; try one new angle and one new audience.
- Spell out targets as numbers (e.g., “Goal: CAC ≤ $X; ROAS ≥ Y”).

INDUSTRY-SPECIFIC NOTES
- If high-ticket (e.g., ${price if price else ">$1,000"}): fewer weeks, higher daily budget, stronger proof (case studies, testimonials, guarantees), and explain that big-ticket buyers need more touches.
- If service with repeat customers: explain lifetime value (LTV) in one line; ok to tolerate higher CAC initially.
- If product low-ticket: emphasize bundles, AOV boosts, email capture, retargeting.

RESPOND TO THEIR STRUGGLE (one short paragraph)
- Acknowledge the struggle they wrote, give 2 concrete tips tailored to their type and {p.business_location or "region"} (e.g., “ads didn’t work before → go bigger for a short burst or focus on high-intent search first; tiny budgets + broad awareness = wasted spend”).

CLOSE (soft)
- 1–2 sentences: offer help to set up the initial structure and connect Nykra’s AI Ad Optimiser to keep improving targeting and cost per result.
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
                {"role": "system", "content": "You are Nykra Studio's helpful growth strategist. Be clear, concrete, and beginner-first. Always produce numbers and small steps."},
                {"role": "user", "content": _build_prompt(payload)},
            ]
        )
        text = completion.choices[0].message.content.strip()
        return {"result_html": _as_html(text), "plain_text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

