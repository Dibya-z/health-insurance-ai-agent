"""LangChain agent factory: 6 tools + Groq llama-3.3-70b."""

import os

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

from tools.coverage import calculate_claim, check_coverage
from tools.documents import list_required_documents
from tools.hospitals import find_network_hospitals
from tools.policy_search import search_policy_docs
from tools.user import get_user_policy

load_dotenv()

SYSTEM_PROMPT = """You are an Indian health insurance assistant. The user has uploaded their health insurance policy PDF and wants help understanding it.

You have 6 tools. Decision policy:
- "Is X covered?" or coverage of a specific procedure/condition -> check_coverage first.
- Open-ended "why/how/explain" questions about policy wording -> search_policy_docs.
- "How much will I get?" or claim amount questions -> chain: check_coverage -> get_user_policy -> calculate_claim.
- "Where can I go?" or hospital availability -> find_network_hospitals (returns ONLY cashless network).
- "What documents do I need?" or claim paperwork -> list_required_documents.
- For multi-part questions, chain multiple tools as needed.

Hard rules:
- All amounts are in Indian Rupees (INR). Format large numbers with commas (e.g. ₹36,000).
- NEVER invent sub-limits, co-pays, or waiting periods. Only use values returned by tools.
- When you quote the policy, cite the page number from the tool output.
- If check_coverage returns covered=null (no rule found), use search_policy_docs to look up the answer in the policy text.
- If a condition is excluded (covered=false), explain it's not covered and quote the exclusion reason.
- For waiting periods: 0 = no wait, 30 = 30 days initial, 730 = 24 months (specified disease), 1095 = 36 months (PED).

OUTPUT FORMAT (markdown — the UI renders it):
- Lead with a 1-sentence direct answer in **bold**.
- For ANY list of 2+ items (documents, hospitals, conditions, steps), use markdown bullets:
    - Each item on its own line starting with `-` or `*`
    - Use sub-bullets for nested detail
- For claim calculations, use a small table or a key-value bullet list:
    - **Bill amount:** ₹50,000
    - **Sub-limit applied:** ₹40,000
    - **Co-pay (10%):** ₹4,000
    - **Payable to you:** ₹36,000
- Use **bold** for key facts (covered status, amounts, page numbers, waiting periods).
- End with a one-line citation: *Source: policy page X*.
- Keep prose to 1-2 sentences max — let the structure carry the answer.
- NEVER write a long paragraph that contains a comma-separated list. Always break to bullets."""


def build_agent(verbose: bool = True) -> AgentExecutor:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment / .env")

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=api_key)
    tools = [
        check_coverage,
        calculate_claim,
        get_user_policy,
        find_network_hospitals,
        list_required_documents,
        search_policy_docs,
    ]
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=8,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )


if __name__ == "__main__":
    # CLI smoke test — runs the brief's 3 sample flows + 2 more.
    import sys
    from pathlib import Path

    from ingest import pdf_fingerprint
    from tools._context import set_policy, set_user

    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data" / "policy.pdf"
    user_id = sys.argv[2] if len(sys.argv) > 2 else "U1001"

    set_policy(pdf_fingerprint(pdf), pdf)
    set_user(user_id)

    executor = build_agent(verbose=False)

    questions = [
        "Is cataract surgery covered? My bill is Rs 50,000 — how much will I get?",
        "What documents do I need to file a hospitalization reimbursement claim?",
        "I have diabetes — is it covered under my policy?",
        "Find a cashless hospital in Bangalore for ophthalmology.",
        "What's the waiting period for pre-existing diseases?",
    ]
    for q in questions:
        print(f"\n>>> USER: {q}")
        try:
            result = executor.invoke({"input": q})
            print(f"AGENT: {result['output']}")
            steps = result.get("intermediate_steps", [])
            print(f"  ({len(steps)} tool call(s): {[s[0].tool for s in steps]})")
        except Exception as e:
            print(f"ERROR: {e}")
