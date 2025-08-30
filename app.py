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

async def get_gpt_assumptions(platform: str, industry: str, price: float, recurring: bool) -> dict:
    """Use GPT to get more accurate CPC and conversion rate assumptions"""
    try:
        # Fallback values in case API call fails
        fallback = {
            'cpc': 1.50 if platform.lower() in ['meta', 'facebook'] else 2.50,
            'conversion_rate': 0.02
        }
        
        # Create a detailed prompt for GPT
        prompt = f"""As a digital marketing expert, provide realistic CPC and conversion rate estimates for the following scenario:

Platform: {platform}
Industry: {industry}
Product/Service Price: ${price}
Recurring Revenue: {"Yes" if recurring else "No"}

Please respond in JSON format only with these two values:
{{"cpc": 0.00, "conversion_rate": 0.00}}

Where:
- cpc is the average cost per click in USD (typical range $0.50-$10.00)
- conversion_rate is the decimal probability of conversion (typical range 0.001-0.05)

Consider that:
- Higher-priced items typically have lower conversion rates
- B2B generally has higher CPC than B2C
- Competitive industries like finance have higher CPCs
- Recurring services may have different metrics than one-time purchases
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
        
        # Apply sanity checks
        if cpc < 0.1 or cpc > 50:
            cpc = fallback['cpc']
        if conversion_rate < 0.0001 or conversion_rate > 0.2:
            conversion_rate = fallback['conversion_rate']
            
        # Apply price-based conversion rate adjustment if GPT didn't account for it
        if price > 5000 and conversion_rate > 0.01:
            conversion_rate = min(conversion_rate, 0.01)
        
        return {
            'cpc': cpc,
            'conversion_rate': conversion_rate
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
        'meta': {'cpc': 1.50, 'conversion_rate': 0.02},
        'facebook': {'cpc': 1.50, 'conversion_rate': 0.02},
        'google': {'cpc': 2.50, 'conversion_rate': 0.035},
        'tiktok': {'cpc': 1.20, 'conversion_rate': 0.015},
        'linkedin': {'cpc': 5.00, 'conversion_rate': 0.025},
        'youtube': {'cpc': 2.00, 'conversion_rate': 0.02}
    }
    
    # Industry multipliers (simplified)
    industry_multipliers = {
        'b2b': {'cpc': 1.5, 'conversion': 0.8},
        'saas': {'cpc': 2.0, 'conversion': 0.7},
        'ecommerce': {'cpc': 1.0, 'conversion': 1.2},
        'service': {'cpc': 1.2, 'conversion': 1.0},
        'healthcare': {'cpc': 2.5, 'conversion': 0.6},
        'finance': {'cpc': 3.0, 'conversion': 0.5}
    }
    
    assumptions = base_assumptions.get(platform, base_assumptions['meta'])
    
    # Apply industry multiplier if detected
    for ind, multiplier in industry_multipliers.items():
        if ind in industry.lower():
            assumptions['cpc'] *= multiplier['cpc']
            assumptions['conversion_rate'] *= multiplier['conversion']
            break
    
    return assumptions

async def calculate_roas_analysis(platforms: str, price: float, industry: str, recurring: bool) -> str:
    """Calculate ROAS analysis for given platforms"""
    if not platforms:
        platforms = "Meta"
    
    platform_list = [p.strip() for p in platforms.split(',')]
    analysis = ""
    
    for platform in platform_list:
        # Use GPT for more accurate assumptions
        assumptions = await get_gpt_assumptions(platform, industry, price, recurring)
        cpc = assumptions['cpc']
        conv_rate = assumptions['conversion_rate']
        
        # Calculate metrics
        clicks_needed = 1 / conv_rate
        cac = clicks_needed * cpc
        roas = price / cac if cac > 0 else 0
        
        # Determine if it's worth it
        worth_it = "worth it" if roas >= 3 else "not worth it"
        
        analysis += f"{platform.title()} in your industry: CPC ≈ ${cpc:.2f}, conv ≈ {conv_rate*100:.1f}% → ~{clicks_needed:.0f} clicks ≈ CAC ${cac:.0f} per sale. At sale value ${price:.0f} → ROAS ≈ {roas:.1f}x {worth_it}.\n"
        
        if worth_it == "not worth it":
            analysis += "Notes: Consider increasing sale price, creating bundles, or offering subscriptions to improve ROAS.\n"
        
        analysis += "\n"
    
    return analysis.strip()

def generate_target_audience(business_name: str, business_description: str, price: float, recurring: bool) -> str:
    """Generate target audience section"""
    
    # Extract business type for better targeting
    business_type = "business"
    if "marketing" in business_description.lower() or "agency" in business_description.lower():
        business_type = "marketing service"
    elif "coach" in business_description.lower() or "consulting" in business_description.lower():
        business_type = "coaching/consulting service"
    elif "retreat" in business_description.lower() or "experience" in business_description.lower():
        business_type = "experience"
    elif "product" in business_description.lower() or "ecommerce" in business_description.lower():
        business_type = "product business"
    
    # Generate audience based on price point and business type
    if price >= 5000:  # High ticket
        demo = "Business owners and executives aged 35–60, revenue $500k–$10M+"
        psycho = "Value premium solutions, have budget for high-ticket items, decision-makers"
    elif price >= 1000:  # Mid ticket
        demo = "Business owners aged 28–55, revenue $100k–$3M"
        psycho = "Value professional services, willing to invest in growth, time-conscious"
    else:  # Lower ticket
        demo = "Small business owners and entrepreneurs aged 25–50, revenue $50k–$500k"
        psycho = "Price-conscious but value-driven, looking for affordable solutions"
    
    return f"""For {business_name} {business_type}, focus on {demo.lower()}. These clients {psycho.lower()}.

Demographics: {demo}, primarily English-speaking markets.
Job titles: Founders, CEOs, Business Owners, Managers in relevant industries.
Psychographics: {psycho}, seeking reliable solutions for business growth.
Interests: Industry-specific tools, business growth, digital marketing, efficiency solutions.
High-intent keywords: Related to your specific service/product + "solution", "service", "help", industry-specific terms."""

def generate_4_week_plan(platforms: str, price: float, recurring: bool, business_description: str) -> str:
    """Generate 4-week advertising plan"""
    
    if not platforms:
        platforms = "Meta"
    
    # Determine business type for strategy
    is_high_ticket = price >= 3000
    is_service = "service" in business_description.lower()
    is_ecom = "ecommerce" in business_description.lower() or "product" in business_description.lower()
    
    # Base weekly budget (adjust based on price point)
    if price >= 5000:
        base_budget = 500
    elif price >= 1000:
        base_budget = 200
    else:
        base_budget = 100
    
    plan = "**4-Week Advertising Plan:**\n\n"
    
    if recurring and not is_high_ticket:
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
    
    return plan

def address_struggle(struggle: str, business_description: str, price: float) -> str:
    """Address the user's biggest struggle"""
    
    if not struggle:
        return "Focus on testing small, measuring results, and scaling what works. The key is consistent testing and optimization."
    
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
    elif "not sure" in struggle_lower or "don't know" in struggle_lower:
        response += "uncertainty is normal when starting with ads. The key is starting small and learning from real data. "
        response += "Begin with conservative budgets and clear success metrics. "
    else:
        response += f"'{struggle}' is a common challenge. "
    
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
            request.recurring
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
