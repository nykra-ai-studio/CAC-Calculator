from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Optional, List
import openai
import os
from datetime import datetime
import re
import json

app = FastAPI(title="Nykra CAC Calculator API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

class CalculatorRequest(BaseModel):
    name: str
    email: EmailStr
    business_url: str
    business_description: Optional[str] = None
    offer: Optional[str] = None
    offer_details: Optional[str] = None
    item_price: Optional[str] = None
    expected_profit: Optional[str] = None
    recurring: bool = False
    current_ad_spend: Optional[str] = None
    ad_platforms: Optional[str] = None
    struggle: Optional[str] = None

def extract_price_from_string(price_str: str) -> float:
    """Extract numeric price from string like '$100' or 'bundles $100-$300'"""
    if not price_str:
        return 1000  # default
    
    # Find all numbers in the string
    numbers = re.findall(r'\d+', price_str.replace(',', ''))
    if numbers:
        # If multiple numbers, take the first one or average
        if len(numbers) == 1:
            return float(numbers[0])
        else:
            # For ranges like $100-$300, take the lower end
            return float(numbers[0])
    return 1000  # default

def detect_industry(business_description: str) -> str:
    """Detect industry from business description"""
    business_description = business_description.lower() if business_description else ""
    
    # Healthcare detection
    if any(term in business_description for term in ["psychology", "therapy", "counseling", "mental health", "clinic", "healthcare", "medical", "doctor", "dentist", "chiropractor"]):
        return "healthcare"
    
    # SaaS/Tech detection
    if any(term in business_description for term in ["saas", "software", "tech", "app", "platform", "subscription software"]):
        return "saas"
    
    # E-commerce detection
    if any(term in business_description for term in ["ecommerce", "e-commerce", "online store", "shop", "retail", "product", "sell products"]):
        return "ecommerce"
    
    # Finance detection
    if any(term in business_description for term in ["finance", "financial", "accounting", "tax", "investment", "banking", "insurance"]):
        return "finance"
    
    # Marketing/Agency detection
    if any(term in business_description for term in ["marketing", "agency", "advertising", "digital marketing", "seo", "ppc"]):
        return "marketing"
    
    # Education detection
    if any(term in business_description for term in ["education", "course", "training", "coaching", "teaching", "workshop", "learning"]):
        return "education"
    
    # Real estate detection
    if any(term in business_description for term in ["real estate", "property", "home", "house", "apartment", "realtor"]):
        return "real_estate"
    
    # Local service detection
    if any(term in business_description for term in ["service", "local business", "contractor", "plumber", "electrician", "cleaning"]):
        return "local_service"
    
    # Default
    return "general"

def get_business_type(business_description: str) -> str:
    """Extract business type for better targeting"""
    business_description = business_description.lower() if business_description else ""
    
    if any(term in business_description for term in ["psychology", "therapy", "counseling", "mental health", "clinic"]):
        return "healthcare service"
    elif "marketing" in business_description or "agency" in business_description:
        return "marketing service"
    elif "coach" in business_description or "consulting" in business_description:
        return "coaching/consulting service"
    elif "retreat" in business_description or "experience" in business_description:
        return "experience"
    elif "product" in business_description or "ecommerce" in business_description:
        return "product business"
    elif "saas" in business_description or "software" in business_description:
        return "software business"
    elif "restaurant" in business_description or "food" in business_description:
        return "food service"
    else:
        return "business"

async def get_gpt_assumptions(platform: str, industry: str, price: float, recurring: bool, business_description: str) -> dict:
    """Use GPT to get more accurate CPC and conversion rate assumptions"""
    try:
        # Fallback values in case API call fails
        fallback = {
            'cpc': 1.50 if platform.lower() in ['meta', 'facebook'] else 2.50,
            'conversion_rate': 0.02,
            'ltv_multiplier': 1
        }
        
        # Detect specific industry
        detected_industry = detect_industry(business_description)
        business_type = get_business_type(business_description)
        
        # Create a detailed prompt for GPT
        prompt = f"""As a digital marketing expert, provide realistic CPC and conversion rate estimates for the following scenario:

Platform: {platform}
Industry: {detected_industry}
Business Type: {business_type}
Product/Service Price: ${price}
Recurring Revenue: {"Yes" if recurring else "No"}
Business Description: {business_description}

"""

        # Add industry-specific context
        if detected_industry == "healthcare":
            prompt += """
Healthcare advertising has specific compliance requirements and typically higher CPCs.
For recurring services like therapy, consider the lifetime value across multiple sessions.
A typical therapy client might attend 8-12 sessions on average.
"""
        elif detected_industry == "saas":
            prompt += """
SaaS businesses typically have longer sales cycles and higher customer acquisition costs,
but benefit from recurring revenue and longer customer lifetimes.
"""
        elif detected_industry == "ecommerce":
            prompt += """
E-commerce typically has lower conversion rates for cold traffic but benefits from 
retargeting and repeat purchases. Consider average order value and purchase frequency.
"""

        prompt += """
Please respond in JSON format only with these values:
{
  "cpc": 0.00,
  "conversion_rate": 0.00,
  "ltv_multiplier": 0
}

Where:
- cpc is the average cost per click in USD (typical range $0.50-$10.00)
- conversion_rate is the decimal probability of conversion (typical range 0.001-0.05)
- ltv_multiplier is how many times the initial price a customer is worth over their lifetime (for recurring services)
"""

        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a digital marketing analytics expert who provides precise numerical estimates."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=150
        )
        
        # Extract JSON from response
        content = response.choices[0].message.content.strip()
        # Find JSON in the response (in case GPT adds explanatory text)
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
            
        result = json.loads(content)
        
        # Validate the results are within reasonable ranges
        cpc = float(result.get('cpc', fallback['cpc']))
        conversion_rate = float(result.get('conversion_rate', fallback['conversion_rate']))
        ltv_multiplier = float(result.get('ltv_multiplier', fallback['ltv_multiplier']))
        
        # Apply sanity checks
        if cpc < 0.1 or cpc > 50:
            cpc = fallback['cpc']
        if conversion_rate < 0.0001 or conversion_rate > 0.2:
            conversion_rate = fallback['conversion_rate']
        if ltv_multiplier < 1:
            ltv_multiplier = max(1, 6 if recurring else 1)  # Default 6 months for recurring
            
        # Apply price-based conversion rate adjustment if GPT didn't account for it
        if price > 5000 and conversion_rate > 0.01:
            conversion_rate = min(conversion_rate, 0.01)
        
        # Set default LTV multipliers by industry if not provided
        if ltv_multiplier <= 1 and recurring:
            if detected_industry == "healthcare":
                ltv_multiplier = 8  # Average therapy client attends ~8 sessions
            elif detected_industry == "saas":
                ltv_multiplier = 12  # 12 month average retention
            else:
                ltv_multiplier = 6  # 6 month default
        
        return {
            'cpc': cpc,
            'conversion_rate': conversion_rate,
            'ltv_multiplier': ltv_multiplier
        }
        
    except Exception as e:
        print(f"GPT API error: {str(e)}")
        # Fallback to the original method if GPT fails
        return get_platform_assumptions(platform, industry)

def get_platform_assumptions(platform: str, industry: str) -> dict:
    """Get CPC and conversion rate assumptions based on platform and industry (fallback method)"""
    platform = platform.lower().strip()
    
    # Base assumptions - these would ideally come from a database
    base_assumptions = {
        'meta': {'cpc': 1.50, 'conversion_rate': 0.02, 'ltv_multiplier': 1},
        'facebook': {'cpc': 1.50, 'conversion_rate': 0.02, 'ltv_multiplier': 1},
        'google': {'cpc': 2.50, 'conversion_rate': 0.035, 'ltv_multiplier': 1},
        'tiktok': {'cpc': 1.20, 'conversion_rate': 0.015, 'ltv_multiplier': 1},
        'linkedin': {'cpc': 5.00, 'conversion_rate': 0.025, 'ltv_multiplier': 1},
        'youtube': {'cpc': 2.00, 'conversion_rate': 0.02, 'ltv_multiplier': 1}
    }
    
    # Industry multipliers (simplified)
    industry_multipliers = {
        'b2b': {'cpc': 1.5, 'conversion': 0.8, 'ltv': 1.5},
        'saas': {'cpc': 2.0, 'conversion': 0.7, 'ltv': 12},
        'ecommerce': {'cpc': 1.0, 'conversion': 1.2, 'ltv': 1.5},
        'service': {'cpc': 1.2, 'conversion': 1.0, 'ltv': 2},
        'healthcare': {'cpc': 2.5, 'conversion': 0.6, 'ltv': 8},
        'finance': {'cpc': 3.0, 'conversion': 0.5, 'ltv': 3},
        'education': {'cpc': 2.0, 'conversion': 0.7, 'ltv': 1},
        'real_estate': {'cpc': 2.5, 'conversion': 0.5, 'ltv': 1},
        'local_service': {'cpc': 1.5, 'conversion': 0.8, 'ltv': 2}
    }
    
    assumptions = base_assumptions.get(platform, base_assumptions['meta'])
    
    # Apply industry multiplier if detected
    for ind, multiplier in industry_multipliers.items():
        if ind in industry.lower():
            assumptions['cpc'] *= multiplier['cpc']
            assumptions['conversion_rate'] *= multiplier['conversion']
            assumptions['ltv_multiplier'] = multiplier['ltv']
            break
    
    return assumptions

async def calculate_roas_analysis(platforms: str, price: float, industry: str, recurring: bool, business_description: str) -> str:
    """Calculate ROAS analysis for given platforms"""
    if not platforms:
        platforms = "Meta"
    
    platform_list = [p.strip() for p in platforms.split(',')]
    analysis = ""
    
    for platform in platform_list:
        # Use GPT for more accurate assumptions
        assumptions = await get_gpt_assumptions(platform, industry, price, recurring, business_description)
        cpc = assumptions['cpc']
        conv_rate = assumptions['conversion_rate']
        ltv_multiplier = assumptions['ltv_multiplier']
        
        # Calculate metrics
        clicks_needed = 1 / conv_rate
        cac = clicks_needed * cpc
        initial_roas = price / cac if cac > 0 else 0
        
        # Determine if it's worth it
        worth_it = "worth it" if initial_roas >= 3 or (recurring and price * ltv_multiplier / cac >= 3) else "not worth it"
        
        # Basic analysis
        analysis += f"{platform.title()} in your industry: CPC ≈ ${cpc:.2f}, conv ≈ {conv_rate*100:.1f}% → ~{clicks_needed:.0f} clicks ≈ CAC ${cac:.0f} per sale. At sale value ${price:.0f} → ROAS ≈ {initial_roas:.1f}x"
        
        # Add LTV analysis for recurring
        if recurring and ltv_multiplier > 1:
            ltv = price * ltv_multiplier
            ltv_roas = ltv / cac if cac > 0 else 0
            analysis += f", LTV-ROAS ≈ {ltv_roas:.1f}x (over {ltv_multiplier:.0f} transactions)"
        
        analysis += f" {worth_it}.\n"
        
        if worth_it == "not worth it" and not (recurring and price * ltv_multiplier / cac >= 3):
            if recurring:
                analysis += "Notes: Consider optimizing your retention strategy to increase customer lifetime value, or bundling services to increase initial transaction value.\n"
            else:
                analysis += "Notes: Consider increasing sale price, creating bundles, or offering subscriptions to improve ROAS.\n"
        
        analysis += "\n"
    
    return analysis.strip()

def generate_target_audience(business_name: str, business_description: str, price: float, recurring: bool) -> str:
    """Generate target audience section"""
    
    # Extract business type for better targeting
    business_type = get_business_type(business_description)
    industry = detect_industry(business_description)
    
    # Industry-specific targeting
    if industry == "healthcare":
        if "psychology" in business_description.lower() or "therapy" in business_description.lower():
            demo = "Adults aged 25-65 seeking mental health support"
            psycho = "Value privacy, professional credentials, and availability"
            interests = "Mental wellness, self-improvement, stress management"
            keywords = "therapy near me, psychologist accepting new patients, anxiety treatment"
            job_titles = "Individuals seeking mental health support, HR managers for EAP programs"
        else:
            demo = "Adults aged 30-65 seeking healthcare services"
            psycho = "Health-conscious, preventative care focused, values expertise"
            interests = "Health & wellness, medical information, preventative care"
            keywords = "healthcare provider near me, specialist accepting new patients"
            job_titles = "Individuals seeking healthcare, HR managers for employee benefits"
    elif industry == "saas":
        demo = "Business professionals aged 30-55 in relevant industries"
        psycho = "Tech-savvy, efficiency-focused, ROI-driven"
        interests = "Business software, productivity tools, industry innovations"
        keywords = "software solution for [problem], [industry] management software"
        job_titles = "CTOs, IT Managers, Department Heads, Operations Directors"
    elif industry == "education":
        demo = "Students, professionals seeking upskilling, or parents of students"
        psycho = "Growth-minded, career-focused, education-valuing"
        interests = "Professional development, certifications, learning platforms"
        keywords = "learn [skill], certification in [field], courses for [profession]"
        job_titles = "Professionals seeking advancement, HR managers, Training coordinators"
    else:
        # Generate audience based on price point and business type
        if price >= 5000:  # High ticket
            demo = "Business owners and executives aged 35–60, revenue $500k–$10M+"
            psycho = "Value premium solutions, have budget for high-ticket items, decision-makers"
            interests = "Premium business solutions, executive resources, industry leadership"
            keywords = "premium [service], executive [solution], best [product] for business"
            job_titles = "C-Suite Executives, Business Owners, Directors, Senior Managers"
        elif price >= 1000:  # Mid ticket
            demo = "Business owners aged 28–55, revenue $100k–$3M"
            psycho = "Value professional services, willing to invest in growth, time-conscious"
            interests = "Business growth, professional services, industry solutions"
            keywords = "professional [service], business [solution], quality [product]"
            job_titles = "Business Owners, Department Heads, Managers, Directors"
        else:  # Lower ticket
            demo = "Small business owners and entrepreneurs aged 25–50, revenue $50k–$500k"
            psycho = "Price-conscious but value-driven, looking for affordable solutions"
            interests = "Small business tools, affordable solutions, DIY approaches"
            keywords = "affordable [service], best value [product], small business [solution]"
            job_titles = "Small Business Owners, Solopreneurs, Startup Founders, Managers"
    
    return f"""For {business_name} {business_type}, focus on {demo.lower()}. These clients {psycho.lower()}.

Demographics: {demo}, primarily English-speaking markets.
Job titles: {job_titles}.
Psychographics: {psycho}, seeking reliable solutions for business growth.
Interests: {interests}.
High-intent keywords: {keywords}."""

def generate_4_week_plan(platforms: str, price: float, recurring: bool, business_description: str) -> str:
    """Generate 4-week advertising plan"""
    
    if not platforms:
        platforms = "Meta"
    
    # Determine business type for strategy
    industry = detect_industry(business_description)
    is_high_ticket = price >= 3000
    is_service = "service" in business_description.lower() or industry in ["healthcare", "local_service"]
    is_ecom = industry == "ecommerce"
    
    # Base weekly budget (adjust based on price point)
    if price >= 5000:
        base_budget = 500
    elif price >= 1000:
        base_budget = 200
    else:
        base_budget = 100
    
    # Industry-specific adjustments
    if industry == "healthcare":
        # Healthcare typically needs more consistent spending
        base_budget = max(base_budget, 150)  # Minimum $150/week for healthcare
    elif industry == "saas":
        # SaaS typically has longer sales cycles
        base_budget = max(base_budget, 200)  # Minimum $200/week for SaaS
    
    plan = "**4-Week Advertising Plan:**\n\n"
    
    # Industry-specific plans
    if industry == "healthcare" and recurring:
        plan += f"**Week 1 (Test):** ${base_budget} - Test messaging and compliance-approved creative\n"
        plan += f"**Week 2 (Refine):** ${base_budget * 1.5} - Optimize targeting and messaging\n"
        plan += f"**Week 3 (Scale):** ${base_budget * 2} - Scale successful campaigns\n"
        plan += f"**Week 4 (Sustain):** ${base_budget * 1.5} - Maintain consistent presence\n"
    elif recurring and not is_high_ticket:
        # Recurring service strategy
        plan += f"**Week 1 (Test):** ${base_budget} - Test audiences and ad creatives\n"
        plan += f"**Week 2 (Scale):** ${base_budget * 2} - Scale winning ads\n"
        plan += f"**Week 3 (Peak):** ${base_budget * 2.5} - Maximum spend on proven campaigns\n"
        plan += f"**Week 4 (Optimize):** ${base_budget * 1.5} - Optimize and maintain momentum\n"
    elif is_high_ticket:
        # High ticket strategy
        plan += f"**Week 1 (Test):** ${base_budget} - Test messaging and audiences carefully\n"
        plan += f"**Week 2 (Scale):** ${base_budget * 3} - Aggressive scaling of working campaigns\n"
        plan += f"**Week 3 (Peak):** ${base_budget * 4} - Maximum investment period\n"
        plan += f"**Week 4 (Sustain):** ${base_budget * 2} - Maintain presence, optimize costs\n"
    elif is_ecom:
        # Ecommerce strategy
        plan += f"**Week 1-2 (Test):** ${base_budget} each week - Test products and audiences\n"
        plan += f"**Week 3-4 (Consistency):** ${base_budget * 1.5} each week - Consistent spend on winners\n"
    else:
        # General service strategy
        plan += f"**Week 1 (Test):** ${base_budget} - Test and validate\n"
        plan += f"**Week 2-3 (Scale):** ${base_budget * 2} each week - Scale successful campaigns\n"
        plan += f"**Week 4 (Maintain):** ${base_budget * 1.5} - Maintain and optimize\n"
    
    # Add platform-specific notes
    platform_list = [p.strip().lower() for p in platforms.split(',')]
    if len(platform_list) > 1:
        plan += f"\n**Platform Split:** Start with {platform_list[0].title()} (70% budget), then add {platform_list[1].title()} (30% budget) in Week 2."
    
    # Add industry-specific notes
    if industry == "healthcare":
        plan += "\n\n**Healthcare Note:** Ensure all ads comply with healthcare advertising policies. Focus on building trust and highlighting credentials."
    elif industry == "saas":
        plan += "\n\n**SaaS Note:** Consider longer sales cycles. Allocate budget for retargeting campaigns to nurture leads."
    
    return plan

def address_struggle(struggle: str, business_description: str, price: float) -> str:
    """Address the user's biggest struggle"""
    
    if not struggle:
        return "Focus on testing small, measuring results, and scaling what works. The key is consistent testing and optimization."
    
    industry = detect_industry(business_description)
    struggle_lower = struggle.lower()
    response = "I hear you — "
    
    if "no sales" in struggle_lower or "not working" in struggle_lower:
        response += "no sales yet can feel discouraging, but it's exactly where testing pays off. We need to confirm offer-market fit and optimize your funnel. "
        response += "Start with a clear, low-friction offer and track your key metrics closely. "
    elif "expensive" in struggle_lower or "budget" in struggle_lower:
        response += "testing can feel expensive, but small, focused tests will save money long-term. "
        response += "Start with minimal budgets to validate your approach before scaling. "
    elif "social media" in struggle_lower or "keep up" in struggle_lower:
        response += "keeping up with social media is overwhelming, but paid ads can be more predictable than organic content. "
        response += "Focus on 1-2 platforms and automate what you can. "
    elif "not sure" in struggle_lower or "don't know" in struggle_lower or "never" in struggle_lower:
        response += "uncertainty is normal when starting with ads. The key is starting small and learning from real data. "
        response += "Begin with conservative budgets and clear success metrics. "
    else:
        response += f"'{struggle}' is a common challenge. "
    
    # Add industry-specific advice
    if industry == "healthcare":
        response += "For healthcare services, focus on building trust and highlighting credentials. "
        response += "Patient testimonials (following privacy guidelines) and clear calls-to-action are particularly effective. "
    elif industry == "saas":
        response += "For SaaS businesses, consider offering free trials or demos to reduce friction. "
        response += "Focus on specific pain points your software solves rather than feature lists. "
    
    response += "Remember, every successful business started with testing and optimization. "
    response += "Follow the 4-week plan above, track your metrics, and adjust based on real performance data. "
    response += "Nykra can optimize your ads by automating the target audience every time an ad gets a click — saving money, cutting conversion costs, and getting you more sales."
    
    return response

@app.post("/analyze")
async def analyze_calculator(request: CalculatorRequest):
    try:
        # Extract price from string
        price = extract_price_from_string(request.item_price) if request.item_price else 1000
        
        # Build the analysis
        result_text = f"Thanks for using the Nykra CPA calculator. Here are your results:\n\n"
        
        # 1. Return on Ad Spend
        result_text += "1. Return on Ad Spend\n"
        result_text += "We'll use realistic CPC and conversion assumptions for your business & platforms, then calculate CAC (cost to win 1 customer) and ROAS. These are starting-point estimates; refine with your real data.\n\n"
        
        roas_analysis = await calculate_roas_analysis(
            request.ad_platforms or "Meta", 
            price, 
            request.business_description or "", 
            request.recurring,
            request.business_description or ""
        )
        result_text += roas_analysis + "\n\n"
        
        # 2. Target audience marketing example
        result_text += "2. Target audience marketing example:\n"
        target_audience = generate_target_audience(
            request.name, 
            request.business_description or "", 
            price, 
            request.recurring
        )
        result_text += target_audience + "\n\n"
        
        # 3. 4-week plan
        result_text += "3. 4-Week Plan:\n"
        week_plan = generate_4_week_plan(
            request.ad_platforms or "Meta", 
            price, 
            request.recurring, 
            request.business_description or ""
        )
        result_text += week_plan + "\n\n"
        
        # 4. Address struggle
        result_text += "4. Address your struggle:\n"
        struggle_response = address_struggle(
            request.struggle or "", 
            request.business_description or "", 
            price
        )
        result_text += struggle_response + "\n\n"
        
        # 5. Summary
        result_text += "5. Summary & Notes:\n"
        if request.recurring:
            result_text += "- For recurring service: focus on lifetime value and customer retention metrics.\n"
        result_text += f"- Expected CAC target: Keep under ${price * 0.3:.0f} for healthy ROAS.\n"
        result_text += "- Track key metrics: Cost per click, conversion rate, and customer acquisition cost.\n"
        result_text += "- Start small, test consistently, and scale what works.\n"
        
        # Convert to HTML for better formatting
        html_result = result_text.replace('\n', '<br>').replace('**', '<strong>').replace('**', '</strong>')
        
        return {
            "success": True,
            "plain_text": result_text,
            "result_html": f'<div style="white-space:pre-wrap;line-height:1.6;color:#e8eaed">{html_result}</div>'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
