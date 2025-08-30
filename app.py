# app.py — Nykra CPA/CAC Calculator (deterministic math + archetype plans)
# - Section 1 (ROAS per platform) computed here (no model math)
# - GPT writes Sections 2–5 using our numbers & guardrails
# - Sends lead (name, email, website, source=Calculator) to n8n in a BackgroundTask

import os, re, statistics, math
from typing import Optional, List, Tuple, Dict

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr

from openai import OpenAI
import httpx

# ---------- Config ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
LTV_MONTHS = int(os.getenv("LTV_MONTHS", "6"))         # change to 12 if you prefer

# Define both webhook URLs
N8N_WEBHOOK_TEST_URL = "https://nykrastudio.app.n8n.cloud/webhook-test/nykra-cac-intake"
N8N_WEBHOOK_PROD_URL = "https://nykrastudio.app.n8n.cloud/webhook/nykra-cac-intake"

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

@app.get("/")
def root():
    return {"ok": True, "service": "nykra-cac"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- Payload ----------
class AnalyzePayload(BaseModel):
    name: str
    email: EmailStr
    business_url: Optional[str] = None
    business_location: Optional[str] = None
    business_description: Optional[str] = None
    offer: Optional[str] = None                  # "Product, Service, Experience, Subscription" (comma-joined)
    offer_details: Optional[str] = None
    item_price: Optional[str] = None             # free text (may include multiple numbers)
    recurring_price: Optional[str] = None        # explicit recurring amount text
    expected_profit: Optional[str] = None
    recurring: Optional[str] = None
    current_ad_spend: Optional[str] = None
    ad_platforms: Optional[str] = None           # e.g., "Meta, Google, TikTok"
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
    if not text:
        return None
    for val, _ in _nums_in(text):
        return val
    return None

def _parse_price_details(item_price_text: Optional[str], recurring_text: Optional[str]) -> Dict[str, Optional[float]]:
    """
    Determine upfront (setup), monthly (recurring), and avg_sale for ROAS-at-purchase.
    Priority:
      - avg_sale: directly use the first number from item_price_text
      - upfront: numbers labelled near setup/upfront/project/website/refresh/initial; fallback largest >= 200
      - monthly: explicit recurring_price if given; else labelled near month/mo/maintenance/retainer/subscription; fallback 30..1500 not equal to upfront
    """
    # First, extract the main sale value directly from item_price_text
    avg_sale = None
    if item_price_text:
        nums = _nums_in(item_price_text)
        if nums:
            # Use the first number found as the primary sale value
            avg_sale = nums[0][0]

    upfront_keys = ["setup", "upfront", "project", "website", "refresh", "initial", "one-off", "once"]
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

    # Fallbacks
    if not upfronts and pairs:
        upfronts = [v for v, _ in pairs if v >= 200]
    if not monthlies and pairs:
        monthlies = [v for v, _ in pairs if 30 <= v <= 1500 and v not in upfronts]

    upfront = max(upfronts) if upfronts else None
    if monthlies:
        try:
            monthly = float(statistics.median(monthlies))
        except statistics.StatisticsError:
            monthly = float(monthlies[0]) if monthlies else None
    else:
        monthly = None

    # If avg_sale is still None, fall back to upfront
    if avg_sale is None and upfront is not None:
        avg_sale = upfront
    
    return {"upfront": upfront, "monthly": monthly, "avg_sale": avg_sale}

def _fmt_money(x: float) -> str:
    if x is None or math.isinf(x) or math.isnan(x):
        return "$0"
    if x >= 1000:
        return "${:,.0f}".format(x)
    return "${:,.0f}".format(x) if float(int(x)) == x else "${:,.1f}".format(x)

def _platforms_list(ad_platforms: Optional[str]) -> List[str]:
    if not ad_platforms: return []
    t = ad_platforms.lower()
    plats = []
    if any(k in t for k in ["meta", "facebook", "instagram"]): plats.append("meta")
    if any(k in t for k in ["google", "search"]): plats.append("google")
    if "tiktok" in t: plats.append("tiktok")
    out, seen = [], set()
    for p in plats:
        if p not in seen:
            out.append(p); seen.add(p)
    return out

def _is_service(offer: Optional[str], desc: Optional[str]) -> bool:
    s = " ".join([offer or "", desc or ""]).lower()
    return any(k in s for k in [
        "service", "consult", "agency", "therapy", "therap", "dent", "law", "plumb",
        "electric", "roof", "install", "setup", "maintenance", "retainer", "seo",
        "website", "coaching", "psycholog", "marketing"
    ])

def _is_finite(offer: Optional[str], desc: Optional[str]) -> bool:
    s = " ".join([offer or "", desc or ""]).lower()
    return any(k in s for k in ["retreat", "cohort", "event", "workshop", "bootcamp", "tickets", "limited spots", "limited seats"])

def _ticket_band(upfront: Optional[float], desc: Optional[str]) -> str:
    if upfront is None:
        s = (desc or "").lower()
        if any(k in s for k in ["enterprise", "premium", "custom build", "solar", "sauna", "renovation", "website"]):
            return "high"
        return "mid"
    if upfront < 200: return "low"
    if upfront < 1000: return "mid"
    if upfront < 5000: return "high"
    return "premium"

def _assumptions_for(platform: str, is_service: bool, ticket: str, finite: bool, has_recurring: bool) -> Tuple[float, float]:
    """Return (CPC $, conversion fraction). Conservative, static heuristics (no GPT)."""
    if platform == "google":
        if finite:
            return (4.0, 0.012)
        if is_service:
            if ticket in ("high", "premium"): return (8.0, 0.015)
            if ticket == "mid": return (4.5, 0.015)
            return (3.0, 0.012)
        if ticket == "low": return (2.0, 0.012)
        return (3.0, 0.012)

    if platform == "meta":
        if finite: return (2.5, 0.012)
        if is_service:
            if ticket in ("high", "premium"): return (3.2, 0.004)
            return (2.2, 0.008)
        if ticket == "low": return (1.2, 0.010)
        return (1.8, 0.012)

    if platform == "tiktok":
        if finite: return (1.6, 0.010)
        if is_service:
            if ticket in ("high", "premium"): return (2.0, 0.0035)
            return (1.6, 0.006)
        if ticket == "low": return (0.9, 0.008)
        return (1.2, 0.009)

    return (2.0, 0.010)

def _best_intent_platform(is_service: bool, ticket: str, finite: bool) -> str:
    if finite:
        return "meta"
    if is_service or ticket in ("high", "premium"):
        return "google"
    return "meta"

def _safe_float(s: Optional[str]) -> Optional[float]:
    if not s: return None
    try:
        vals = _nums_in(s)
        return vals[0][0] if vals else None
    except: return None

# ---------- core builders ----------
def compute_platform_rows(p: AnalyzePayload) -> Tuple[str, Dict]:
    """Compute Section 1 lines and return (html_lines, meta dict with numbers for later sections)."""
    price_info = _parse_price_details(p.item_price, p.recurring_price)
    upfront = price_info["upfront"]
    monthly = price_info["monthly"]
    avg_sale = price_info["avg_sale"]
    
    # Ensure we're using the correct sale price - prioritize avg_sale
    sale_price = avg_sale  # This should be the first number from item_price_text
    
    has_recurring = monthly is not None

    plats = _platforms_list(p.ad_platforms)
    is_srv = _is_service(p.offer, p.business_description)
    finite = _is_finite(p.offer, p.business_description)
    ticket = _ticket_band(sale_price, p.business_description)

    if not plats:
        plats = ["google"] if is_srv or ticket in ("high", "premium") else ["meta"]

    rows = []
    meta = {"platforms": []}
    industry_type = "your industry"

    for plat in plats:
        cpc, conv = _assumptions_for(plat, is_srv, ticket, finite, has_recurring)
        clicks_per_sale = int(round(1 / conv)) if conv > 0 else 0
        cac = (cpc / conv) if conv > 0 else None

        roas = (sale_price / cac) if (sale_price and cac) else None
        ltv = (sale_price + LTV_MONTHS * monthly) if (sale_price and monthly is not None) else None
        ltv_roas = (ltv / cac) if (ltv and cac) else None

        verdict = None
        if roas is not None: verdict = "worth testing" if roas >= 1.0 else "not worth it yet"

        line = f"- <b>{plat.title()}</b> in {industry_type}: CPC ≈ {_fmt_money(cpc)}, conv ≈ {conv*100:.1f}% → ~{clicks_per_sale} clicks ≈ CAC {_fmt_money(cac)} per sale."
        if sale_price:
            # Use sale_price directly, not upfront
            line += f" At sale value {_fmt_money(sale_price)} → ROAS ≈ {roas:,.2f}x"
            if verdict:
                margin = sale_price - (cac or 0)
                sign = "+" if margin >= 0 else "−"
                line += f" ({sign}{_fmt_money(abs(margin))} per sale) → <b>{verdict}</b>."
        else:
            line += " Add an average sale or setup price to judge ROAS precisely."

        if ltv_roas is not None:
            line += f"<br/>  ↳ {LTV_MONTHS}-month LTV view: {_fmt_money(ltv)} / {_fmt_money(cac)} ≈ {ltv_roas:,.2f}x."

        rows.append(line)

        meta["platforms"].append({
            "name": plat,
            "cpc": cpc,
            "conv": conv,
            "clicks_per_sale": clicks_per_sale,
            "cac": cac,
            "roas": roas,
            "ltv": ltv,
            "ltv_roas": ltv_roas,
            "verdict": verdict,
        })

    if len(meta["platforms"]) >= 2 and all(pl["roas"] is not None for pl in meta["platforms"]):
        best = max(meta["platforms"], key=lambda d: d["roas"])
        worst = min(meta["platforms"], key=lambda d: d["roas"])
        if best["name"] != worst["name"]:
            rows.append(f"→ Right now <b>{best['name'].title()}</b> looks stronger than {worst['name'].title()} under these assumptions.")

    meta["price_used"] = {"upfront": upfront, "monthly": monthly, "sale_price": sale_price}
    lines_html = "<br/>".join(rows)
    return lines_html, meta

def choose_archetype(p: AnalyzePayload, meta: Dict) -> Dict:
    pi = meta.get("price_used", {})
    sale_price = pi.get("sale_price")
    monthly = pi.get("monthly")

    is_srv = _is_service(p.offer, p.business_description)
    finite = _is_finite(p.offer, p.business_description)
    ticket = _ticket_band(sale_price, p.business_description)
    has_recurring = monthly is not None

    if finite:
        archetype = "finite"
        curve = [0.30, 0.40, 0.30, 0.00]
    elif (is_srv and (ticket in ("high", "premium")) and not has_recurring):
        archetype = "high_durable"
        curve = [0.25, 0.25, 0.25, 0.25]
    elif (is_srv and has_recurring):
        archetype = "service_recurring"
        curve = [0.15, 0.35, 0.35, 0.15]
    elif (ticket == "low"):
        archetype = "ecom_low"
        curve = [0.25, 0.25, 0.25, 0.25]
    else:
        archetype = "generic_mid"
        curve = [0.25, 0.25, 0.25, 0.25]

    # Budget guidance (aim for ~8 CACs total per platform in first month)
    min_per_plat = []
    for pl in meta["platforms"]:
        cac = pl["cac"] or 300.0
        min_per_plat.append(max(cac * 8.0, 300.0))
    baseline_weekly = sum(min_per_plat) / 4.0
    weekly_low = max(200.0, baseline_weekly * 0.7)
    weekly_high = baseline_weekly * 1.3

    likely = _best_intent_platform(is_srv, ticket, finite)
    chosen = [pl["name"] for pl in meta["platforms"]] or [likely]
    if len(chosen) == 1:
        split = {chosen[0]: 1.0}
    else:
        if likely in chosen:
            others = [c for c in chosen if c != likely]
            if others:
                rem = 1.0 - 0.60
                per = rem / len(others)
                split = {likely: 0.60}
                for o in others: split[o] = per
            else:
                split = {likely: 1.0}
        else:
            per = 1.0 / len(chosen)
            split = {c: per for c in chosen}

    return {
        "archetype": archetype,
        "curve": curve,
        "weekly_range": (weekly_low, weekly_high),
        "platform_split": split
    }

def section1_html(p: AnalyzePayload) -> Tuple[str, Dict]:
    header = "<div style='font-weight:700;margin:2px 0 8px'>1) Return on Ad Spend</div>"
    intro = ("We'll use realistic CPC and conversion assumptions for your business & platforms, then calculate CAC (cost to win 1 customer) and ROAS. "
             "These are starting-point estimates; refine with your real data.")
    rows_html, meta = compute_platform_rows(p)
    html = header + f"<div style='opacity:.9;line-height:1.6'>{intro}</div><div style='margin-top:10px;line-height:1.6'>{rows_html}</div>"
    return html, meta

def _to_html(text: str) -> str:
    return (text.replace("&", "&")
                .replace("<", "<")
                .replace(">", ">")
                .replace("\n", "<br/>"))

def build_prompt_for_sections_2_to_5(p: AnalyzePayload, meta: Dict, plan: Dict) -> str:
    price = meta.get("price_used", {})
    sale_price = price.get("sale_price")
    upfront = price.get("upfront")
    monthly = price.get("monthly")

    nums_lines = []
    for pl in meta["platforms"]:
        sp = f"{pl['name'].title()}: CPC ${pl['cpc']:.2f}, conv {pl['conv']*100:.1f}%, CAC {_fmt_money(pl['cac'])}"
        if pl["roas"] is not None:
            sp += f", ROAS {pl['roas']:.2f}x"
        if pl["ltv_roas"] is not None:
            sp += f", LTV-ROAS {pl['ltv_roas']:.2f}x"
        nums_lines.append(sp)

    curve_pct = [f"{int(x*100)}%" for x in plan["curve"]]
    split_lines = [f"{k.title()} {int(v*100)}%" for k, v in plan["platform_split"].items()]
    wl, wh = plan["weekly_range"]

    return f"""
You are a friendly senior growth strategist for beginners. DO NOT change or invent numbers. Use the numbers and guardrails provided.

BUSINESS CONTEXT
- Name: {p.name}
- URL: {p.business_url or "N/A"}
- Location: {p.business_location or "N/A"}
- Offer: {p.offer or "N/A"}
- Description: {p.business_description or "N/A"}
- Price used: {"setup " + str(int(upfront)) if upfront else ""} {"| monthly " + str(int(monthly)) if monthly else ""} {"| sale " + str(int(sale_price)) if sale_price else ""}
- Platforms: {p.ad_platforms or "N/A"}
- Biggest struggle: {p.struggle or "N/A"}

LOCKED NUMBERS (read-only)
{chr(10).join("- " + x for x in nums_lines)}
Weekly budget guide (total): { _fmt_money(wl) }–{ _fmt_money(wh) }
Spend curve by week: {", ".join(curve_pct)} (Week1→Week4)
Platform split (starting): {", ".join(split_lines)}

ARCHETYPE
- {plan["archetype"]}

OUTPUT — EXACT SECTIONS (do not add extra sections):
2) Target audience
- 1 short paragraph tailored to the business and location.
- Then exactly 5 bullets including a mix of: demographics, psychographics, interests, job titles, or high-intent search keywords.

3) 4-week plan
- Write a clear plan that follows the spend curve and platform split above.
- Use the weekly budget guide for scale (include a sentence with the range).
- Translate the archetype into tactics:
  - finite → urgency, countdowns, raise retargeting in last 5 days, watch seat fill pace
  - high_durable → longer cycle, heavy nurture, measure CPL/qualified calls, expect closes 6–8 weeks
  - service_recurring → week1 low test, weeks2–3 scale, week4 taper as calendar fills; tie CAC to LTV
  - ecom_low → consistent always-on, ROAS guardrails, bundles/AOV lift
  - generic_mid → simple steady plan with optimization
- Include numeric success targets (use existing CAC/ROAS numbers sensibly; do NOT alter them).

4) Address your struggle
- 1 empathetic paragraph. Give 2 practical fixes tied to the struggle.
- End exactly with: "Nykra can optimise your ads by automating the target audience every time an ad gets a click — saving money, cutting conversion costs, and getting you more sales."

5) Summary & notes by situation
- 3–5 bullets maximum, tailored to their type (finite, high-ticket durable, recurring service, low-ticket ecom).
"""

# ---------- n8n lead sender (BackgroundTask) ----------
async def send_lead_to_n8n_async(data: dict):
    urls_to_try = [N8N_WEBHOOK_TEST_URL, N8N_WEBHOOK_PROD_URL]
    success = False
    
    for url in urls_to_try:
        if success:
            break
            
        try:
            async with httpx.AsyncClient(timeout=8.0) as ac:
                r = await ac.post(url, json=data, headers={"Content-Type": "application/json"})
                if r.status_code < 400:
                    success = True
                    break
        except Exception:
            # Continue to next URL if this one fails
            pass

# ---------- Endpoint ----------
@app.post("/analyze")
def analyze(payload: AnalyzePayload, background_tasks: BackgroundTasks):
    try:
        # queue lead capture to n8n (runs after response starts)
        background_tasks.add_task(
            send_lead_to_n8n_async,
            {
                "name": payload.name,
                "email": payload.email,
                "business_url": payload.business_url or "",
                "website": payload.business_url or "",
                "source": "Calculator",
            },
        )

        # Opening line
        opening = "<div style='font-weight:700;margin:2px 0 8px'>Thanks for using the Nykra CPA calculator. Here are your results.</div>"

        # Section 1 (deterministic)
        s1_html, meta = section1_html(payload)

        # Archetype + budgets
        plan = choose_archetype(payload, meta)

        # Model-written Sections 2–5 (no math; strict format)
        prompt = build_prompt_for_sections_2_to_5(payload, meta, plan)
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are Nykra Studio's helpful growth strategist. Be clear, concrete, and beginner-first. Never change provided numbers."},
                {"role": "user", "content": prompt},
            ],
        )
        s2to5 = completion.choices[0].message.content.strip()

        # Assemble response
        html = (
            f"<div class='nykra-calc-result' style='line-height:1.55;color:#e8eaed'>"
            f"{opening}"
            f"{s1_html}"
            f"<div style='font-weight:700;margin:14px 0 8px'>2–5) Plan & Guidance</div>"
            f"{_to_html(s2to5)}"
            f"</div>"
        )
        return {
            "result_html": html,
            "plain_text": f"Thanks for using the Nykra CPA calculator. Here are your results.\n\n[s1]\n{re.sub('<[^<]+?>','',s1_html)}\n\n[s2-5]\n{s2to5}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")
