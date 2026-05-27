# Health Insurance Chatbot

This project is a conversational AI agent designed to help customers query their health insurance policies. It combines Retrieval-Augmented Generation (RAG) over policy documents with custom tool calling to provide comprehensive answers about coverage, claim estimates, network hospitals, and required documentation.

## Features

- **Policy Document Search (RAG):** Upload your policy PDF and search for specific clauses, waiting periods, and exclusions.
- **Coverage Checker:** Ask if a specific medical condition or surgery is covered.
- **Claim Calculator:** Estimate the expected claim amount based on your policy's sub-limits and co-pay rules.
- **Hospital Network Lookup:** Find cashless network hospitals by city and specialty.
- **Document Checklist:** Get a list of required documents for different types of claims.
- **User Policy Retrieval:** Access user-specific policy details like sum insured, co-pay percentages, and no-claim bonuses.

## Setup Instructions

1. Ensure you have Python installed (preferably 3.10+).
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
5. Get your free Groq API Key from [console.groq.com](https://console.groq.com) and enter it in the app sidebar.
6. Upload a sample health insurance policy PDF to start querying!

## Resume Bullet Point

"Built a multi-tool conversational AI agent for health insurance Q and A using RAG over policy documents combined with 6 custom tools - autonomously checks coverage, calculates claim amounts, and verifies hospital network status. Deployed via Streamlit with a Groq + LangChain backend."
