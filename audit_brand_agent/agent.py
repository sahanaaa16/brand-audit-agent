from dotenv import load_dotenv
import os
from google.adk.agents import LlmAgent, SequentialAgent, Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.lite_llm import LiteLlm
import re 

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

print(f"OPENAI API KEY loaded: {OPENAI_API_KEY[:10]}...")


openai_model = LiteLlm(model="openai/gpt-4o-mini-search-preview")
greeting_model = LiteLlm(model="openai/gpt-3.5-turbo")

SECTIONS = [
    "Origin Story",
    "Style Mission Vision Major Shifts",
    "Category Overview",
    "Macro Forces",
    "Other Categories Involved In",
    "Revenue Streams",
    "Marketing Strategies",
    "Media Partnerships & Agency Relationships",
    "Ownership Business Structure",
    "Recent Developments & News",
    "Brand Perception",
    "Popular Products",
    "Unexpected Sectors Entered",
    "Brand Opportunities",
    "Competitors",
    "Primary Consumers",
    "Strategic Behavioral Trends",
    "Advertising Spend",
]

BASE_INSTRUCTION = """ 
You are writing a comprehensive brand audit for **{brand}**.

IMPORTANT CITATION REQUIREMENTS:
- For every claim made, include a citation in the format [1], [2], [3], etc.
- Keep track of all sources used and include them in your response
- At the end of your section, include a "Sources Used:" section listing all URLs/sources
- Use primary sources, do NOT Wikipedia
- Include direct links to sources when possible

{section_specific_instructions}

Research Guidelines:
- Use web search to find current, verified information
- Prioritize official brand sources, press releases, financial reports
- Include recent news and developments (within last 12 months)
- Cross-reference information from multiple sources
- If data is unavailable, state "Data unavailable" rather than guessing
- If can not find data search on the web 

Citation Format:
- Use [1], [2], [3] format throughout
- Include source URLs at the end of each section
- Make sure every factual claim has a citation

Your response should be detailed, well-researched, and properly cited.
"""

SECTION_INSTRUCTIONS = {
    "Origin Story": """
Cover:
• How and when the brand was founded [source link]
• Key people involved in the founding [source link]
• The brand's major values [source link]
• Early innovations and first product launches [source link]
If can not find data, search on the web, if still can not find data state "Data Unavailable" [source link]
""",
    "Style Mission Vision Major Shifts": """
Cover:
• Brand's current mission statement [source link]
• Brand's vision statement [source link]
• Evolution of brand positioning over time [source link]
• Key themes in branding, packaging, and campaigns [source link]
If can not find data, search on the web, if still can not find data state "Data Unavailable" [source link]
""",
    "Category Overview": """
Cover:
• List 4-5 major categories the brand operates in [source link]
• Explain what brands in these categories emphasize [source link]
• Market positioning within each category [source link]
If can not find data, search on the web, if still can not find data state "Data Unavailable" [source link]
""",
    "Macro Forces": """
Cover:
• Market size & growth trajectory with specific statistics [source link]
• 3-4 technology/innovation shifts and brand's adaptation [source link]
• Social media presence with direct profile links [source link]
• Cultural expectations (sustainability, DEI) [source link]
If can not find data, search on the web, if still can not find data state "Data Unavailable" [source link]
""",
    "Other Categories Involved In": """
Cover:
• 3-4 adjacent categories the brand operates in [source link]
• When and how they entered each category [source link]
• Success/failure of category expansions [source link]
If can not find data, search on the web, if still can not find data state "Data Unavailable" [source link]
""",
    "Revenue Streams": """
Cover:
• Core and secondary revenue streams [source link]
• Financial performance (past month, year, 5 years) [source link]
• Budget distribution breakdowns if available [source link]
• Sustainability/ESG initiatives [source link]
If can not find data, search on the web, if still can not find data state "Data Unavailable" [source link]
""",
    "Marketing Strategies": """
Cover:
• Successful marketing strategies [source link]
• Failed marketing strategies [source link]
• Recent campaign reactions [source link]
• Political/moral stances [source link]
• CTAs used in marketing [source link]
• Ad budget allocation by channel [source link]
• Influencer strategies [source link]
• Loyalty/CRM programs [source link]
If can not find data, search on the web, if still can not find data state "Data Unavailable" [source link]
""",
    "Media Partnerships & Agency Relationships": """
Cover:
• 4-5 key creative agency partners with website links [source link]
• 5 recent brand partnerships [source link]
• Media buying strategies [source link]
If can not find data, search on the web, if still can not find data state "Data Unavailable" [source link]
""",
    "Ownership Business Structure": """
Cover:
• Parent companies/ownership structure [source link]
• Companies/brands they own [source link]
• 4-5 subsidiary relationships [source link]
• Strategic value of relationships [source link]
If can not find data, search on the web, if still can not find data state "Data Unavailable" [source link]
""",
    "Recent Developments & News": """
Cover:
• 4-5 recent stories from past year with links [source link]
• Major launches, PR moments, controversies [source link]
• Investments and expansions [source link]
If can not find data, search on the web, if still can not find data state "Data Unavailable" [source link]
""",
    "Brand Perception": """
Cover:
• What critics and consumers say [source link]
• 3 major supporters/advocates [source link]
• 3 major critics/detractors [source link]
• Brand sentiment analysis [source link]
If can not find data, search on the web, if still can not find data state "Data Unavailable" [source link]
""",
    "Popular Products": """
Cover:
• 4-5 best-selling products with links [source link]
• Top trending products from past year [source link]
• Product performance metrics [source link]
If can not find data, search on the web, if still can not find data state "Data Unavailable" [source link]
""",
    "Unexpected Sectors Entered": """
Cover:
• 3-4 unexpected sectors entered [source link]
• Success/failure of these ventures [source link]
• Strategic rationale [source link]
If can not find data, search on the web, if still can not find data state "Data Unavailable" [source link]
""",
    "Brand Opportunities": """
Cover:
• 4-5 white-space market opportunities [source link]
• Category issues they could address [source link]
• Innovation gaps [source link]
If can not find data, search on the web, if still can not find data state "Data Unavailable" [source link]
""",
    "Competitors": """
Cover:
• 4-5 primary competitors [source link]
• Competitive positioning [source link]
• 3-4 upstarts/disruptors [source link]
• Market share comparisons [source link]
If can not find data, search on the web, if still can not find data state "Data Unavailable" [source link]
""",
    "Primary Consumers": """
Cover:
• Detailed demographics (age, income, race, gender) [source link]
• Top countries by sales volume [source link]
• Loyalty indicators and metrics [source link]
• Consumer behavior patterns [source link]
If can not find data, search on the web, if still can not find data state "Data Unavailable" [source link]
""",
    "Strategic Behavioral Trends": """
Cover:
• 3 recent marketing strategy changes [source link]
• Effectiveness of new approaches [source link]
• Behavioral shifts in brand strategy [source link]
If can not find data, search on the web, if still can not find data state "Data Unavailable" [source link]
""",
    "Advertising Spend": """
Cover:
• Total annual advertising spend estimate [source link]
• Breakdown by channel (digital, print, TV, influencer, OOH, podcast) [source link]
• Estimated costs per channel [source link]
• Recent spend increases/decreases [source link]
• ROI and effectiveness metrics [source link]
• Compare to industry benchmarks [source link]
If can not find data, search on the web, if still can not find data state "Data Unavailable" [source link]
"""
}


greeting_agent = LlmAgent(
    name="brand_audit_greeting",
    model=greeting_model,
    instruction="""
You are a brand strategy assistant. Your job is to extract a brand name from ANY user message, even if the message contains greetings or extra words.

RULES:
- Look for any well-known company or brand name *anywhere* in the user's message, no matter what else is present.
- Examples:
    - "hello, lululemon" -> extract "lululemon"
    - "can you do a brand audit for nike" -> extract "nike"
    - "apple" -> extract "apple"
    - "good morning! analyze coca cola" -> extract "coca cola"
    - "I want a report on Adidas" -> extract "Adidas"
    - "audit puma" -> extract "puma"
- If you find a brand name, reply to the user with a friendly confirmation and next steps, then set your output_key to ONLY the brand name (all lowercase is fine).
- If there are multiple possible brands, pick the first obvious one.

EXAMPLES:
User: "hello, lululemon"
Assistant: "Great! I'll create a comprehensive brand audit for lululemon..."
output_key (brand): lululemon

User: "Can you analyze Coca Cola?"
Assistant: "Great! I'll create a comprehensive brand audit for Coca Cola..."
output_key (brand): Coca Cola

User: "Hi there!"
Assistant: "Hello! I'm your brand strategy assistant. What brand would you like me to analyze?"
output_key (brand): null

Remember: output_key should be the brand name only, never your whole message. Never ignore a brand just because the user included a greeting or extra words.
    """,
    output_key="brand",
    tools=[]
)


def followup_instruction(ctx: CallbackContext) -> str:
    brand = ctx.state.get("brand")
    # Debug: Check current state
    current_state = {
        "brand": brand,
        "conversation_stage": ctx.state.get("conversation_stage"),
        "final_report": bool(ctx.state.get("final_report"))
    }
    print(f"[DEBUG] Followup agent - Current state: {current_state}")
    
    return f"""
    You are a brand strategy assistant that handles follow-up questions about brand audits that have already been completed.
    
    The brand currently being analyzed is: {brand}
    
    Based on the conversation history and the user's question, provide detailed, well-researched answers about the brand.
    
    IMPORTANT: If the user asks for a DIFFERENT brand audit (e.g., "Audit PepsiCo next" or "Another brand: Pepsi"), 
    respond with: "I'll start a new brand audit for [NEW BRAND]. Let me begin the comprehensive analysis..."
    And set your output_key to "new_audit_requested" with the new brand name.
    
    Otherwise, handle their question as usual—running searches, citing sources, etc.
    
    Use web search to find current, accurate information to answer their specific questions.
    Include proper citations and source links for all claims.
    
    Format your response with proper citations [1], [2], [3] and include source URLs at the end.
    """

followup_agent = LlmAgent(
    name="brand_audit_followup",
    model=LiteLlm(model="openai/gpt-4o"),
    instruction=followup_instruction,
    output_key="followup_response",
    tools=[]
)

def make_worker_for(sections_subset: list[str], idx: int) -> LlmAgent:
    agent_name = f"audit_worker_{idx}" 
    start_section_num = idx * 3 + 1
    
    # Debug: Print worker section assignment
    print(f"[DEBUG] Creating {agent_name} for sections {start_section_num}-{start_section_num + len(sections_subset) - 1}")
    print(f"[DEBUG] Worker {idx} assigned sections: {sections_subset}")
    
    section_list = "\n".join(f"{start_section_num + i}. {s}" for i, s in enumerate(sections_subset))
    
    # Build section-specific instructions
    section_instructions = ""
    for i, section in enumerate(sections_subset):
        section_num = start_section_num + i
        section_instructions += f"\n{section_num}. {section}\n"
        if section in SECTION_INSTRUCTIONS:
            section_instructions += SECTION_INSTRUCTIONS[section] + "\n"
    
    def instruction_provider(ctx: CallbackContext) -> str:
        brand = ctx.state.get("brand")
        if not brand:
            return "Please provide the brand name first."
        
        # Debug: Show what this worker is processing
        print(f"[DEBUG] {agent_name} processing brand: {brand}")
        print(f"[DEBUG] {agent_name} working on sections: {[f'{start_section_num + i}. {s}' for i, s in enumerate(sections_subset)]}")
        
        full_instruction = BASE_INSTRUCTION.format(
            brand=brand,
            section_specific_instructions=section_instructions
        )
        return full_instruction

    return LlmAgent(
        name=agent_name,
        model=openai_model,
        instruction=instruction_provider,
        tools=[],  
        description=f"Research and analyze sections {start_section_num}-{start_section_num + len(sections_subset) - 1} for brand audit",
        output_key=f"chunk_{idx}",
    )

audit_workers = [
    make_worker_for(batch, i)
    for i, batch in enumerate(
        (SECTIONS[i : i + 3] for i in range(0, len(SECTIONS), 3))
    )
]

# Debug: Show all workers created
print(f"[DEBUG] Total audit workers created: {len(audit_workers)}")
for i, worker in enumerate(audit_workers):
    print(f"[DEBUG] Worker {i}: {worker.name} -> output_key: {worker.output_key}")

# Enhanced compilation agent with state management
def compilation_instruction(ctx: CallbackContext) -> str:
    brand = ctx.state.get("brand")
    
    # Debug: Check what chunks are available
    available_chunks = []
    for i in range(len(audit_workers)):
        chunk_key = f"chunk_{i}"
        if ctx.state.get(chunk_key):
            available_chunks.append(chunk_key)
    
    print(f"[DEBUG] Compilation agent - Brand: {brand}")
    print(f"[DEBUG] Compilation agent - Available chunks: {available_chunks}")
    
    return f"""
You are responsible for compiling the final brand audit report for {brand} from all worker chunks.

Your tasks:
1. Combine all chunks into a coherent final report
2. Ensure proper section numbering (1-18)
3. Extract ALL citations from each chunk
4. Create a comprehensive bibliography with all unique sources

CRITICAL BIBLIOGRAPHY REQUIREMENTS:
- Extract every [1], [2], [3] citation from all chunks
- Create a master list of all unique sources
- Renumber citations consecutively (1, 2, 3, etc.)
- Format bibliography as:
  [1] Source Title. Website Name. Date. URL
  [2] Next source...

Format the final output as:
- Title: "{brand} Brand Audit Report"
- All 18 sections in order with proper numbering
- Consistent citation formatting throughout
- Complete bibliography with all sources

Do not skip the bibliography - it's essential for credibility.
"""

compilation_agent = LlmAgent(
    name="compilation_agent",
    model=openai_model,
    instruction=compilation_instruction,
    tools=[],
    output_key="final_report"
)

brand_audit_orchestrator = SequentialAgent(
    name="brand_audit_orchestrator",
    sub_agents=audit_workers + [compilation_agent],  # Add compilation agent to the sequence
    description="Generate comprehensive 18-section brand audit with proper research and citations.",
)

def router_instruction(ctx: CallbackContext):
    # Get current state
    brand = ctx.state.get("brand")
    conversation_stage = ctx.state.get("conversation_stage", "greeting")
    final_report = ctx.state.get("final_report")
    new_audit_requested = ctx.state.get("new_audit_requested")
    
    # Debug: Print current state
    current_state = {
        "brand": brand,
        "conversation_stage": conversation_stage,
        "has_final_report": bool(final_report),
        "new_audit_requested": new_audit_requested
    }
    print(f"[DEBUG] Router - Current state: {current_state}")
    
    # Simple routing logic with state management
    if new_audit_requested:
        print("[DEBUG] Router decision: new_audit_requested -> resetting to greeting")
        # Reset state for new audit
        ctx.state["brand"] = None
        ctx.state["conversation_stage"] = "greeting"
        ctx.state["final_report"] = None
        ctx.state["new_audit_requested"] = None
        return greeting_agent.name
    
    elif not brand:
        print("[DEBUG] Router decision: no brand -> greeting")
        return greeting_agent.name
        
    elif brand and not final_report:
        print("[DEBUG] Router decision: brand exists, no report -> orchestrator")
        # Set stage to indicate audit is in progress
        ctx.state["conversation_stage"] = "auditing"
        return brand_audit_orchestrator.name
        
    else:  # brand exists and final_report exists
        print("[DEBUG] Router decision: brand and report exist -> followup")
        ctx.state["conversation_stage"] = "completed"
        return followup_agent.name

router_agent = Agent(
    name="brand_audit_router",
    model=greeting_model,
    instruction=router_instruction,
    sub_agents=[greeting_agent, brand_audit_orchestrator, followup_agent],
)

root_agent = router_agent