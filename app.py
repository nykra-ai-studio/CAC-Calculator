# app.py — Nykra CAC/Ad Spend backend (platform-aware CAC/ROAS + beginner-first)

import os, re, statistics
from typing import Optional, List, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from openai import OpenAI

# ---- Config ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
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

# ---- Payload (matches front-end) ----
class AnalyzePayload(BaseModel):
    name: str
    email: EmailStr
    business_url: Optional[str] = None
    business_location: Optional[str] = None
    business_description: Optional[str] = None
    offer: Optional[str] = None                 # "Product, Service, Experience, Subscription"
    offer_details: Optional[str] = None
    item_price: Optional[str] = None            # free-text; we’ll parse $ amounts
    expected_profit: Optional[str] = None
    recurring: Optional[str] = None
    current_ad_spend: Optional[str] = None      # weekly; free-text
    ad_platforms: Optional[str] = None          # e.g., "Meta, Google"
    struggle: Optional[str] = None

# ---------- helpers ----------
def _money_numbers(text: Optional[str]) -> List[float]:
    if not text:
        return []
    nums = re.findall(r'(?<![A-Za-z])(?:\$?\s*)(\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)(?!%)', text)
    out = []
    for n in nums:
        try:
            out.append(float(n.replace(",", "").replace(" ", "")))
        except:
            pass
    return out

def _representative_price(text: Optional[str]) -> Optional[float]:
    vals = _money_numbers(text)
    if not vals:
        return None
    try:
        return float(statistics.median(vals))
    except statistics.StatisticsError:
        return float(vals[0])

def _fmt_money(x: float) -> str:
    return f"${x:,.0f}"

def _ticket_size(price_val: Optional[float], offer: Optional[str], desc: Optional[str]) -> str:
    """Rough ticket classifier: 'high', 'mid', 'low'."""
    s = " ".join([offer or "", desc or ""]).lower()
    if price_val and price_val >= 1000:
        return "high"
    if any(k in s for k in ["retreat", "coaching", "consult", "agency", "b2b", "enterprise", "setup"]):
        return "high"
    if price_val and price_val <= 50:
        return "low"
    return "mid"

def _is_service(offer: Optional[str], desc: Optional[str]) -> bool:
    s = " ".join([offer or "", desc or ""]).lower()
    service_signals = ["service", "consult", "agency", "therap", "dent", "law", "plumb", "electric", "roof", "install", "setup", "maintenance", "retainer"]
    return any(k in s for k in service_signals)

def _platforms_list(ad_platforms: Optional[str]) -> List[str]:
    if not ad_platforms:
        return []
    text = ad_platforms.lower()
    plats = []
    if any(k in text for k in ["meta", "facebook", "instagram"]):
        plats.append("meta")
    if any(k in text for k in ["google", "search"]):
        plats.append("google")
    # de-dupe while preserving order
    seen = set(); ordered=[]
    for p in plats:
        if p not in seen:
            seen.add(p); ordered.append(p)
    return ordered

def _assumptions_for(platform: str, ticket: str, is_service: bool) -> Tuple[float, float]:
    """
    Return (CPC in $, conversion rate as fraction).
    Heuristics chosen to be realistic but beginner-friendly.
    """
    if platform == "google":
        if is_service:
            # high-intent services often expensive; conversion higher than social
            return (6.0 if ticket == "high" else 3.0, 0.008 if ticket == "high" else 0.015)  # 0.8% / 1.5%
        else:
            return (2.0 if ticket == "low" else 3.0, 0.02 if ticket == "low" else 0.012)      # 2.0% / 1.2%
    # meta (facebook/instagram)
    if is_service:
        return (2.5 if ticket == "high" else 2.0, 0.004 if ticket == "high" else 0.008)       # 0.4% / 0.8%
    else:
        return (1.2 if ticket == "low" else 1.6, 0.01 if ticket == "low" else 0.012)          # 1.0% / 1.2%

def _platform_opening_lines(p: AnalyzePayload) -> str:
    """Build platform-specific CAC/ROAS lines. If multiple platforms, compare."""
    loc = p.business_location or "your region"
    price_val = _representative_price(p.item_price)
    is_srv = _is_service(p.offer, p.business_description)
    ticket = _ticket_size(price_val, p.offer, p.business_description)

    plats = _platforms_list(p.ad_platforms)
    if not plats:
        # pick a sensible default if none provided
        plats = ["google"] if is_srv else ["meta"]

    lines = []
    verdicts = []
    for plat in plats:
        cpc, conv = _assumptions_for(plat, ticket, is_srv)
        clicks_per_sale = int(round(1/conv)) if conv > 0 else 0
        cac = cpc/conv if conv > 0 else 0
        if price_val:
            roas = price_val / cac if cac > 0 else 0
            worth = roas >= 1.0
            verdict = "worth testing" if worth else "not worth it yet"
            margin = price_val - cac
            lines.append(
                f"- **{plat.title()}** in {loc}: CPC ≈ {_fmt_money(cpc)}, conv ≈ {conv*100:.1f}% "
                f"→ ~{clicks_per_sale} clicks ≈ CAC {_fmt_money(cac)} per sale. "
                f"At avg sale {_fmt_money(price_val)} → ROAS ≈ {roas:,.2f}x ({'+' if margin>=0 else '−'}{_fmt_money(abs(margin))} per sale) → **{verdict}**."
            )
            verdicts.append((plat, worth, roas))
        else:
            lines.append(
                f"- **{plat.title()}** in {loc}: CPC ≈ {_fmt_money(cpc)}, conv ≈ {conv*100:.1f}% "
                f"→ ~{clicks_per_sale} clicks ≈ CAC {_fmt_money(cac)} per sale. "
                f"If your average sale is above that CAC, it can work; below it, improve AOV/funnel first."
            )

    # If both platforms present and price known, add a quick compare sentence
    if len(plats) >= 2 and price_val:
        best = max(verdicts, key=lambda x: x[2])  # highest ROAS
        worst = min(verdicts, key=lambda x: x[2])
        if best[0] != worst[0]:
            lines.append(f"→ Right now **{best[0].title()} looks stronger than {worst[0].title()}** based on these assumptions.")

    caveat = ("Real costs vary by industry, competition and creative. "
              "Expect Google clicks to be higher for high-intent services, and Meta to be cheaper but lower converting early. "
              "Use this as a starting point, then tune with real data.")
    return "Simple check — CAC & ROAS\n" + \
           "This plan helps you decide ad spend and whether ads are worth it. " \
           "CAC is what you spend to win 1 customer. ROAS is revenue ÷ ad spend.\n" + \
           "\n".join(lines) + "\n" + caveat

def _build_prompt(p: AnalyzePayload) -> str:
    opening = _platform_opening_lines(p)

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
# Opening (exactly two short paragraphs max)
{opening}

# Target audience (3–5 items only)
Give a few **precise** targets tailored to their business and location (demographics, interests, job titles, or search keywords). 
End this section with this exact sentence:
"If interested, Nykra’s AI Ad Optimiser will automate your targeting so it gets sharper with every click, lowering costs over time."

# Budget & duration (very simple)
- Give a weekly and monthly spend range.
- Recommend "always-on baseline" vs "1–2 week bursts" (high-ticket/limited spots → bursts; services/subscriptions → always-on).
- If they gave a current weekly spend, anchor your suggestion to it.

# 4-week plan (beginner-proof)
- **Week 1–2 – Test:** daily budget, 2–3 creative ideas, 1 landing-page fix, success targets with numbers (e.g., "Goal: CAC ≤ $X; ROAS ≥ Y").
- **Week 3–4 – Decide:** branching:
  - If hitting targets → scale 20–50% on winners, widen audiences/keywords slightly.
  - If close → keep spend flat, fix bottleneck (ad, page, or offer), retest.
  - If missing badly → pause losers, keep best, try one new angle + one new audience.

# Notes by situation
- **High-ticket (>$1k):** fewer, bigger tests; stronger proof (case studies/testimonials/guarantees). Explain buyers need more touches.
- **Services with repeat customers:** explain LTV briefly; okay to tolerate higher CAC initially because repeats cover it later.
- **Low-ticket product:** emphasize bundles, AOV boosts, email capture + retargeting.

# Address their struggle (1 short paragraph)
Acknowledge their specific struggle and give 2 practical tips matched to their type and location.

# Close (exactly this one sentence)
We can set up the ad structure and connect Nykra’s AI Ad Optimiser so your cost per result keeps dropping.
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
