# 🏥 Health Insurance AI Agent

An advanced, multi-tool conversational AI agent designed to act as a personal health insurance assistant. This project leverages **Agentic RAG** (Retrieval-Augmented Generation) combined with custom deterministic tool calling to provide users with accurate, mathematically sound, and highly personalized answers regarding their insurance policies.

## ✨ Features

Unlike standard RAG applications that only read documents, this agent is equipped with **custom tools** to query live data and perform calculations:

- **📄 Policy Document Search (RAG):** Upload and semantically search complex insurance PDFs for specific clauses, exclusions, and waiting periods.
- **💰 Claim Calculator:** Deterministically calculates expected claim payouts by applying policy co-pays, deductibles, and sub-limits (eliminating LLM math hallucinations).
- **🩺 Coverage Checker:** Verifies if specific medical conditions or surgeries (e.g., Cataract, Knee Replacement) are covered under the policy rules.
- **🏥 Hospital Network Lookup:** Queries local databases to find cashless network hospitals filtered by city and medical specialty.
- **👤 User Profile Integration:** Pulls dynamic user data (like sum insured, no-claim bonuses, and past claims) to provide personalized answers.
- **📋 Document Checklist:** Automatically generates a list of required documents for different types of claim submissions.

## 🛠️ Technology Stack

- **LLM / Brain:** [Groq](https://groq.com/) (LLaMA 3.3 70B) for ultra-fast, high-quality reasoning.
- **Orchestration:** [LangGraph](https://python.langchain.com/v0.1/docs/langgraph/) & [LangChain](https://python.langchain.com/) (ReAct Agent Framework).
- **Vector Database:** [ChromaDB](https://www.trychroma.com/) for local, persistent embedding storage.
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`).
- **Frontend:** [Streamlit](https://streamlit.io/) for an interactive, chat-based UI.
- **Document Parsing:** `pdfplumber` & Recursive Character Chunking.

## 🚀 Getting Started

The core application code is located inside the `health_insurance_bot` directory.

### 1. Prerequisites
Ensure you have Python 3.10+ installed on your system.

### 2. Setup Instructions

Clone the repository and navigate into the bot directory:
```bash
git clone https://github.com/Dibya-z/health-insurance-ai-agent.git
cd health-insurance-ai-agent/health_insurance_bot
```

Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
You need a free Groq API key to run the LLM. 
1. Get a key from [console.groq.com](https://console.groq.com/keys).
2. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
3. Open `.env` and add your API key:
   ```env
   GROQ_API_KEY=gsk_your_api_key_here
   ```

### 4. Run the Application
Launch the Streamlit interface:
```bash
streamlit run app.py
```

## 🏗️ Architecture Note
For a deep dive into how the agent makes decisions, routes tool calls, and handles vector embeddings, please see the `ARCHITECTURE.md` file located inside the `health_insurance_bot/` directory.
