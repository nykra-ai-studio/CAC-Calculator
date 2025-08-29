# app.py — Nykra CAC/Ad Spend backend
# - Price-aware (setup vs monthly) + recurring_price field
# - Platform-aware CPC/conv (Meta vs Google) by business type/ticket size
# - Beginner-first output with deterministic opening math (no guessing prices)
# - Recurring-aware plan (front-load spend, taper later)

import os, re, statistics
from typing import Optional, List, Tuple, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from openai import OpenAI

# ---- Config ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
LTV_MONTHS = int(os.getenv("LTV_MONTHS", "6"))  # use 6 by default; change to 12 if you prefer

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

# ---- Payload (matches your form; added recurring_price) ----
class AnalyzePayload(BaseModel):
    name: str
    email: EmailStr
    business_url: Optional[str] = None
    business_location: Optional[str] = None
    business_description: Optional[str] = None
    offer: Optional[str] = None                 # "Product, Service, Experience, Subscription"
    offer_details: Optional[str] = None
    item_price: Optional[str] = None            # free-text; may contain setup/monthly phrases
    recurring_price: Optional[str] = None       # NEW: explicit recurring amount text
    expected_profit: Optional[str] = None
    recurring: Optional[str] = None
    current_ad_spend: Optional[str] = None
    ad_platforms: Optional[str] = None
    struggle: Optional[str] = None

# ---------- helpers ----------
_money_re = re.compile(r'\$?\s*(\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)')

def _nums_in(s: Optional[str]) -> List[Tuple[float, int]]:
    if not s:
        return []
    out = []
    for m in _money_re.finditer(s):
        try:
            out.append((float(m.group(1).replace(",", "").replace(" ", "")), m.start()))
        except:
            pass
    return out

def _near(s: str, idx: int, window: int = 22) -> str:
    a = max(0, idx - window); b = min(len(s), idx + window)
    return s[a:b].lower()

def _extract_first_amount(text: Optional[str]) -> Optional[float]:
    """Pull first numeric amount from a simple field like recurring_price."""
    if not text:
        return None
    for val, _ in _nums_in(text):
        return val
    return None

def _parse_price_details(item_price_text: Optional[str], recurring_text: Optional[str]) -> Dict[str, Optional[float]]:
    """
    Determine upfront (setup), monthly (recurring), and avg_sale used for CAC/ROAS-at-purchase.
    Priority:
      1) Use explicit recurring_price field (if present) for monthly.
      2) From item_price_text, label numbers near 'setup/upfront/project/website/refresh' as upfront.
         Label numbers near 'month/mo/monthly/maintenance/retainer' as monthly.
      3) Fallbacks: upfront = largest number >= 200; monthly = median of numbers 30..1500 not equal to upfront.
      4) avg_sale = upfront if present else median of all numbers.
    """
    upfront_keys = ["setup", "upfront", "project", "website", "refresh", "one-off", "once", "initial"]
    monthly_keys = ["per month", "month", "mo", "monthly", "maintenance", "retainer", "subscription"]

    s = item_price_text or ""
    pairs = _nums_in(s)

    upfronts = []
    monthlies = []

    for val, pos in pairs:
        ctx = _near(s, pos)
        if any(k in ctx for k in monthly_keys):
            monthlies.append(val)
        if any(k in ctx for k in upfront_keys):
            upfronts.append(val)

    # explicit recurring overrides inference
    recurring_explicit = _extract_first_amount(recurring_text)
    if recurring_explicit is not None:
        monthlies = [recurring_explicit]

    # Fallback labeling if empty
    if not upfronts:
        upfronts = [v for v, _ in pairs if v >= 200]
    if not monthlies:
        monthlies = [v for v, _ in pairs if 30 <= v <= 1500 and v not in upfronts]

    upfront = max(upfronts) if upfronts else None
    if monthlies:
        try:
            monthly = float(statistics.median(monthlies))
        except statistics.StatisticsError:
            monthly = float(monthlies[0])
    else:
        monthly = None

    if upfront is not None:
        avg_sale = upfront
    else:
        if pairs:
            try:
                avg_sale = float(statistics.median([v for v, _ in pairs]))
            except statistics.StatisticsError:
                avg_sale = pairs[0][0]
        else:
            avg_sale = None

    return {"upfront": upfront, "monthly": monthly, "avg_sale": avg_sale}

def _fmt_money(x: float) -> str:
    return f"${x:,.0f}"

def _is_service(offer: Optional[str], desc: Optional[str]) -> bool:
    s = " ".join([offer or "", desc or ""]).lower()
    return any(k in s for k in [
        "service", "consult", "agency", "therap", "dent", "law", "plumb", "electric",
        "roof", "install", "setup", "maintenance", "retainer", "seo", "website", "coaching"
    ])

def _ticket_size(upfront: Optional[float], offer: Optional[str], desc: Optional[str]) -> str:
    if upfront and upfront >= 1000: return "high"
    s = " ".join([offer or "", desc or ""]).lower()
    if any(k in s for k in ["retreat", "coaching", "consult", "agency", "b2b", "enterprise", "setup", "website", "seo"]):
        return "high"
    if upfront and upfront <= 50: return "low"
    return "mid"

def _platforms_list(ad_platforms: Optional[str]) -> List[str]:
    if not ad_platforms: return []
    t = ad_platforms.lower()
    plats = []
    if any(k in t for k in ["meta", "facebook", "instagram"]): plats.append("meta")
    if any(k in t for k in ["google", "search"]): plats.append("google")
    # de-dupe
    seen=set(); out=[]
    for p in plats:
        if p not in seen: seen.add(p); out.append(p)
    return out

def _assumptions_for(platform: str, ticket: str, is_service: bool) -> Tuple[float, float]:
    """
    Return (CPC $, conversion fraction).
    Conservative but realistic defaults by platform + type.
    """
    if platform == "google":
        if is_service:
            return (8.0 if ticket == "high" else 4.0, 0.010 if ticket == "high" else 0.015)  # 1.0–1.5%
        else:
            return (2.5 if ticket == "low" else 3.5, 0.018 if ticket == "low" else 0.012)
    # meta (facebook/instagram)
    if is_service:
        return (3.0 if ticket == "high" else 2.2, 0.004 if ticket == "high" else 0.008)       # 0.4–0.8%
    else:
        return (1.4 if ticket == "low" else 1.8, 0.010 if ticket == "low" else 0.012)

def _platform_opening_lines(p: AnalyzePayload) -> str:
    loc = p.business_location or "your region"
    price_info = _parse_price_details(p.item_price, p.recurring_price)
    upfront = price_info["upfront"]
    monthly = price_info["monthly"]
    avg_sale = price_info["avg_sale"]

    is_srv = _is_service(p.offer, p.business_description)
    ticket = _ticket_size(upfront or avg_sale, p.offer, p.business_description)

    plats = _platforms_list(p.ad_platforms)
    if not plats:
        plats = ["google"] if is_srv else ["meta"]

    lines = []
    verdicts = []
    for plat in plats:
        cpc, conv = _assumptions_for(plat, ticket, is_srv)
        clicks_per_sale = int(round(1/conv)) if conv > 0 else 0
        cac = cpc/conv if conv > 0 else 0

        # Use upfront (setup) if present; else avg_sale; never invent values.
        sale_price = upfront or avg_sale
        if sale_price:
            roas = sale_price / cac if cac else 0
            margin = sale_price - cac if cac else 0
            worth = roas >= 1.0
            sale_label = "setup price" if upfront else "average sale"
            lines.append(
                f"- **{plat.title()}** in {loc}: CPC ≈ {_fmt_money(cpc)}, conv ≈ {conv*100:.1f}% "
                f"→ ~{clicks_per_sale} clicks ≈ CAC {_fmt_money(cac)} per sale. "
                f"At {sale_label} {_fmt_money(sale_price)} → ROAS ≈ {roas:,.2f}x "
                f"({'+' if margin>=0 else '−'}{_fmt_money(abs(margin))} per sale) → **{'worth testing' if worth else 'not worth it yet'}**."
            )
            verdicts.append((plat, roas))
        else:
            lines.append(
                f"- **{plat.title()}** in {loc}: CPC ≈ {_fmt_money(cpc)}, conv ≈ {conv*100:.1f}% "
                f"→ ~{clicks_per_sale} clicks ≈ CAC {_fmt_money(cac)} per sale. "
                f"(Enter an average sale or setup price to judge ROAS precisely.)"
            )

        # LTV view only if we actually have a monthly amount
        if monthly and sale_price and cac:
            ltv = sale_price + LTV_MONTHS * monthly
            roas_ltv = ltv / cac
            lines.append(f"  ↳ {LTV_MONTHS}-month LTV view: {_fmt_money(ltv)} / {_fmt_money(cac)} ≈ {roas_ltv:,.2f}x (tolerates higher CAC).")

    if len(verdicts) >= 2:
        best = max(verdicts, key=lambda x: x[1]); worst = min(verdicts, key=lambda x: x[1])
        if best[0] != worst[0]:
            lines.append(f"→ Right now **{best[0].title()} looks stronger than {worst[0].title()}** under these assumptions.")

    # Echo what we actually used (transparency; no hallucinated prices)
    echo_bits = []
    if upfront is not None: echo_bits.append(f"setup detected: {_fmt_money(upfront)}")
    if monthly is not None: echo_bits.append(f"recurring detected: {_fmt_money(monthly)}/mo")
    if not echo_bits and avg_sale is not None: echo_bits.append(f"average sale used: {_fmt_money(avg_sale)}")
    echo = f" ({' | '.join(echo_bits)})" if echo_bits else ""

    caveat = ("Real costs vary by industry, competition and creative. "
              "Google is pricier but higher intent; Meta is cheaper but converts lower at first. "
              "Treat this as a starting point and adjust with real data.")
    header = "Simple check — CAC & ROAS\n" \
             "This plan helps you decide ad spend and whether ads are worth it. " \
             "CAC is what you spend to win 1 customer. ROAS is revenue ÷ ad spend." + echo

    return header + "\n" + "\n".join(lines) + "\n" + caveat

def _build_prompt(p: AnalyzePayload) -> str:
    opening = _platform_opening_lines(p)

    # Recurring-aware tip injected for the plan section
    recurring_note = "If you have recurring revenue, expect higher spend in weeks 1–2, then taper as retention compounds." \
                     if (p.recurring_price or ("month" in (p.item_price or "").lower())) else \
                     "For one-off sales, concentrate spend into shorter bursts around proven creatives."

    return f"""
You are a friendly senior growth strategist for total beginners. Be concise, concrete, and avoid jargon. Short paragraphs, 3–5 bullets max. No tables.

CONTEXT
- Name: {p.name}
- Email: {p.email}
- URL: {p.business_url or "N/A"}
- Location: {p.business_location or "N/A"}
- Advertising type: {p.offer or "N/A"}
- Description: {p.business_description or "N/A"}
- Price text: {p.item_price or "N/A"}
- Recurring price: {p.recurring_price or "N/A"}
- Weekly ad spend: {p.current_ad_spend or "N/A"}
- Platforms: {p.ad_platforms or "N/A"}
- Biggest struggle: {p.struggle or "N/A"}

BEGIN OUTPUT
# Opening (two short paragraphs max)
{opening}

# Target audience (3–5 items only)
List a few precise targets tailored to their business and location (demographics, interests, job titles, or search keywords).
End this section with exactly:
"If interested, Nykra’s AI Ad Optimiser will automate your targeting so it gets sharper with every click, lowering costs over time."

# Budget & duration (simple)
- Give weekly and monthly ranges.
- Choose always-on baseline vs 1–2 week bursts (high-ticket/limited spots → bursts; services/subscriptions → always-on).
- Anchor to any weekly spend they gave.

# 4-week plan (beginner-proof)
{recurring_note}
- **Week 1–2 – Test:** daily budget, 2–3 creative ideas, one landing-page fix, success targets with numbers (e.g., "Goal: CAC ≤ $X; ROAS ≥ Y").
- **Week 3–4 – Decide:**
  - If hitting targets → scale 20–50% on winners.
  - If close → keep spend flat, fix the bottleneck (ad, page, or offer), retest 1 week.
  - If missing badly → pause losers, keep the best, try one new angle + one new audience.

# Notes by situation
- **High-ticket services (setups, websites, SEO):** fewer, bigger tests; strong proof (case studies, guarantee). Buyers need multiple touches.
- **Services with repeat revenue (maintenance/retainer/subscription):** explain LTV briefly; okay to tolerate higher CAC early because monthly revenue compounds value.
- **Low-ticket product:** emphasize bundles/AOV, capture emails, retargeting.

# Address their struggle (1 short paragraph)
Acknowledge their specific struggle and give 2 concrete tips matched to their type and location.

# Close (exactly this sentence)
We can set up the ad structure and connect Nykra’s AI Ad Optimiser so your cost per result keeps dropping.
"""

def _as_html(text: str) -> str:
    safe = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>"))
    return f'<div class="nykra-calc-result" style="line-height:1.55;color:#e8eaed">{safe}</div>'

@app.post("/analyze")
def analyze(payload: AnalyzePayload):
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are Nykra Studio's helpful growth strategist. Be clear, concrete, and beginner-first. Use numbers and small steps. Never invent prices."},
                {"role": "user", "content": _build_prompt(payload)},
            ]
        )
        text = completion.choices[0].message.content.strip()
        return {"result_html": _as_html(text), "plain_text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")
