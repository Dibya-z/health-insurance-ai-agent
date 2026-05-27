import os
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

from tools.coverage import check_coverage, calculate_claim
from tools.hospitals import find_network_hospitals
from tools.user import get_user_policy
from tools.documents import list_required_documents
from tools.policy_search import search_policy_docs

def get_agent_executor(groq_api_key=None):
    if not groq_api_key:
        groq_api_key = os.environ.get("GROQ_API_KEY")
        
    if not groq_api_key:
        raise ValueError("Groq API Key is not set.")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0
    )

    tools = [
        search_policy_docs,
        check_coverage,
        calculate_claim,
        get_user_policy,
        find_network_hospitals,
        list_required_documents
    ]

    system_prompt = "You are a helpful health insurance assistant. Use the tools provided to answer customer queries accurately. Always explain your reasoning clearly and show the steps taken."

    executor = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt
    )
    
    return executor
