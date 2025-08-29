# app.py — Nykra CAC/Ad Spend backend (Beginner-first, hardwired opening)

import os, re, statistics
from typing import Optional, List
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

# ---- Payload (matches your front-end) ----
class AnalyzePayload(BaseModel):
    name: str
    email: EmailStr
    business_url: Optional[str] = None
    business_location: Optional[str] = None
    business_description: Optional[str] = None
    offer: Optional[str] = None                 # "Product, Service, Experience, Subscription"
    offer_details: Optional[str] = None         # kept for compatibility
    item_price: Optional[str] = None            # now free-text (price or description)
    expected_profit: Optional[str] = None
    recurring: Optional[str] = None
    current_ad_spend: Optional[str] = None      # weekly (free-text)
    ad_platforms: Optional[str] = None
    struggle: Optional[str] = None

# ---- Small helpers ----
def _money_numbers(text: Optional[str]) -> List[float]:
    """Extracts numeric $ amounts from messy text/ranges like `$100–$300` or `1,200`."""
    if not text:
        return []
    # capture numbers with optional thousands and decimals
    nums = re.findall(r'(?<![A-Za-z])(?:\$?\s*)(\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)(?!%)', text)
    out = []
    for n in nums:
        try:
            out.append(float(n.replace(",", "").replace(" ", "")))
        except:
            pass
    return out

def _representative_price(text: Optional[str]) -> Optional[float]:
    """Choose a single representative price from list (median of found numbers)."""
    vals = _money_numbers(text)
    if not vals:
        return None
    try:
        return float(statistics.median(vals))
    except statistics.StatisticsError:
        return float(vals[0])

def _fmt_money(x: float) -> str:
    return f"${x:,.0f}"

def _guess_cpc(offer: Optional[str], platforms: Optional[str], desc: Optional[str]) -> float:
    """Very simple CPC guesser. Services/search → ~3.0; else social/awareness → ~1.5."""
    s = " ".join([offer or "", platforms or "", desc or ""]).lower()
    # high-intent/service signals → Google-ish CPC
    service_signals = ["service", "google", "search", "plumb", "electric", "therap", "dent", "law", "consult", "solar", "repair", "roof"]
    if any(k in s for k in service_signals):
        return 3.0
    # otherwise lean social
    return 1.5

def _to_float(text: Optional[str]) -> Optional[float]:
    nums = _money_numbers(text)
    return nums[0] if nums else None

def _build_opening(payload: AnalyzePayload) -> str:
    """Hardwired beginner opening with CAC & ROAS using 1% CR + CPC guess."""
    cpc = _guess_cpc(payload.offer, payload.ad_platforms, payload.business_description)
    conv = 0.01  # 1% default
    clicks_needed = int(round(1 / conv))  # 100 clicks
    cac = cpc / conv  # CPC / 1% = CPC * 100

    price_val = _representative_price(payload.item_price)
    roas_line = ""
    verdict_line = ""
    if price_val:
        roas = price_val / cac if cac > 0 else 0
        margin = price_val - cac
        sign = "profit" if margin >= 0 else "loss"
        roas_line = f"At an average sale of {_fmt_money(price_val)}, ROAS ≈ {roas:,.2f}x and profit after ads ≈ {_fmt_money(abs(margin))} {sign} per sale."
        verdict_line = "→ That means ads are likely worth running." if roas >= 1 else "→ That means ads are not worth it yet. Consider raising price, bundling/upsells, or improving the page before scaling."
    else:
        roas_line = "If your average sale is higher than that CAC, ads can work; if it’s lower, improve the offer/AOV or the funnel before scaling."
        verdict_line = ""

    loc = payload.business_location or "your region"
    return (
        "Simple check — CAC & ROAS\n"
        f"This plan helps you decide ad spend and whether ads are worth it. "
        f"CAC (customer acquisition cost) is what you spend to win 1 customer. "
        f"We’ll assume a 1% conversion rate (about {clicks_needed} clicks per sale) and a sensible click cost for {loc}. "
        f"With an estimated CPC of {_fmt_money(cpc)}, you’d need ~{clicks_needed} clicks, costing about {_fmt_money(cac)} for one sale.\n"
        "ROAS (return on ad spend) is revenue ÷ ad spend.\n"
        f"{roas_line} {verdict_line}"
    )

def _build_prompt(p: AnalyzePayload) -> str:
    # Build the deterministic opening
    opening = _build_opening(p)

    return f"""
You are a friendly senior growth strategist for total beginners. Keep it concise, plain-English, and confidence-building.
Use short paragraphs and short bulleted lists (3–5 bullets). No tables. Always include concrete numbers.

CONTEXT FROM USER
- Name: {p.name}
- Email: {p.email}
- Business URL: {p.business_url or "N/A"}
- Location: {p.business_location or "N/A"}
- What they’re advertising: {p.offer or "N/A"}
- Description: {p.business_description or "N/A"}
- Average sale value (free-text): {p.item_price or "N/A"}
- Current WEEKLY ad spend: {p.current_ad_spend or "N/A"}
- Planned platforms: {p.ad_platforms or "N/A"}
- Biggest struggle: {p.struggle or "N/A"}

BEGIN OUTPUT NOW
# Opening (keep EXACTLY two short paragraphs)
{opening}

# Target audience (3–5 items only)
List a few **precise** targets tailored to their business and location (demographics, interests, job titles, or search keywords). 
End this section with this exact sentence:
"If interested, Nykra’s AI Ad Optimiser will automate your targeting so it gets sharper with every click, lowering costs over time."

# Budget & duration (very simple)
- Give a weekly and monthly spend range.
- Recommend "always-on baseline" vs "1–2 week bursts" (high-ticket/limited spots → bursts; services/subscriptions → always-on).
- If they gave a current weekly spend, anchor your suggestion to it.

# 4-week plan (beginner-proof)
- **Week 1–2 – Test:** daily budget, 2–3 creative ideas, 1 landing-page fix, simple success targets with numbers (e.g., "Goal: CAC ≤ $X; ROAS ≥ Y").
- **Week 3–4 – Decide:** clear branching:
  - If hitting targets → scale 20–50% on winners, widen audiences/keywords a little.
  - If close to targets → keep spend flat, fix the bottleneck (ad, page, or offer), retest.
  - If missing badly → pause losers, keep the best one, try one new angle + one new audience.

# Notes by situation
- **High-ticket (>$1k):** fewer weeks, higher daily budget, strong proof (testimonials/case studies), explain that buyers need more touches.
- **Services with repeat customers:** explain LTV in one line; okay to tolerate higher CAC initially because repeats cover it later.
- **Low-ticket product:** emphasize bundles, AOV boosts, email capture + retargeting.

# Address their struggle (1 short paragraph)
Acknowledge their specific struggle and give 2 practical tips that match their business type and location.

# Soft close (1 sentence)
Offer help to set up the structure and connect Nykra’s AI Ad Optimiser so the cost per result keeps dropping.
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
