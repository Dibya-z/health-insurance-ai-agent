# Architecture — Health Insurance Chatbot

A single-page reference for the whole system. Read top-to-bottom on day 1; skim by section thereafter.

---

## 1. Mission Statement

A conversational AI agent that lets a customer upload **any** health insurance policy PDF and ask plain-English questions. On upload, the system **auto-extracts structured coverage rules** specific to that policy, then routes user questions through 6 tools (RAG + 5 lookup/calculator tools that read the per-policy rules).

**Multi-policy by design:** every uploaded PDF gets its own ChromaDB collection AND its own auto-generated rules JSON, keyed by SHA-256 of the file. A user uploading a Star Health policy gets Star Health rules; a Niva Bupa upload gets Niva Bupa rules.

**Non-goals (v1):**
- No real insurer API integration (hospitals.csv is a mock cashless network)
- No authentication or persistent user accounts (3 demo profiles in users.json)
- No payment / claim filing
- Single active policy per session (multi-policy compare is a stretch goal)

---

## 2. High-Level Architecture

### 2.1 Onboarding flow (one-time per unique PDF)

```
USER uploads PDF -> app.py
                      │
                      ▼
      ┌───────────────────────────────┐
      │   policy_processor.py         │
      │                               │
      │   1) ingest.py                │
      │      pdfplumber  -> chunks    │
      │      MiniLM      -> vectors   │
      │      ChromaDB    -> collection│
      │              (named by SHA-256)│
      │                               │
      │   2) rules_extractor.py       │
      │      For each canonical cond  │
      │      in conditions_to_check.json:│
      │        a) RAG-retrieve top-5  │
      │        b) Groq LLM extracts   │
      │           {covered, sub_limit,│
      │            co_pay, waiting,   │
      │            evidence, page}    │
      │      -> policies/<hash>/rules.json│
      └───────────────────────────────┘
                      │
                      ▼
              READY FOR CHAT
```

### 2.2 Chat flow (every user turn)

```
┌──────────────────────────────────────────────────────────────┐
│                         BROWSER                              │
│  Streamlit chat UI · PDF uploader · "Show reasoning" toggle  │
└──────────────────────────┬───────────────────────────────────┘
                           │ HTTP
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                 PRESENTATION LAYER (app.py)                  │
│  st.session_state: chat_history, processed_pdf_hash, user_id │
│  tools/_context.CTX = {pdf_hash, rules, user_id}             │
└──────────────────────────┬───────────────────────────────────┘
                           │ executor.invoke({"input": ...})
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                ORCHESTRATION LAYER (agent.py)                │
│  langchain_classic AgentExecutor + create_tool_calling_agent │
│  LLM: Groq llama-3.3-70b-versatile                           │
│  max_iterations=8, return_intermediate_steps=True            │
└──┬─────────┬─────────┬─────────┬──────────┬──────────┬───────┘
   │         │         │         │          │          │
   ▼         ▼         ▼         ▼          ▼          ▼
search    check    calc     get        find       list
_policy  _cover  _claim   _user     _network   _required
_docs    age              _policy   _hospitals _documents
   │       │        │        │          │          │
   ▼       ▼        ▼        ▼          ▼          ▼
Chroma  policies/<hash>/   users   hospitals  documents
collec   rules.json       .json    .csv       _rules.json
tion    (auto-extracted   (mock)   (mock)     (IRDAI std)
        per upload)
```

**Two-stage system:**
- **Onboarding** = expensive (1-2 min): one LLM call per canonical condition (~48 calls), parallelized with ThreadPoolExecutor.
- **Chat** = cheap (~1-3s per turn): tools just read pre-extracted JSON.

---

## 3. Component Responsibilities

### 3.1 Presentation Layer — `app.py`

| Responsibility | Detail |
|---|---|
| Render chat | `st.chat_message("user" \| "agent")` per turn |
| Capture input | `st.chat_input("Ask about your policy...")` |
| Upload PDF | `st.file_uploader(type="pdf")` in sidebar |
| Pick mock user | `st.selectbox` over `users.json` keys |
| Show/hide reasoning | `st.toggle("Show reasoning")` → expands `intermediate_steps` |
| Persist state | `st.session_state` for `chat_history`, `vector_store`, `pdf_hash`, `current_user_id` |

**Why Streamlit?** Zero-config Python-only UI, perfect for an MVP demo. Trade-off: every widget interaction reruns the script top-to-bottom — hence the obsessive use of `st.session_state` to avoid re-ingesting the PDF.

### 3.2 Orchestration Layer — `agent.py`

The agent is a **ReAct-style tool-calling loop** wrapped by `AgentExecutor`. On each user turn:

1. LLM receives: system prompt + tool schemas + chat history + user message.
2. LLM emits a tool call (or a final answer).
3. AgentExecutor runs the tool, captures the result, appends to scratchpad.
4. Loop until LLM returns a final answer or `max_iterations=8` hits.

**Key configuration:**
```python
ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
AgentExecutor(agent, tools, verbose=True, max_iterations=8,
              return_intermediate_steps=True,
              handle_parsing_errors=True)
```

**System prompt outline:**
- Persona: "Indian health insurance assistant. INR. Cite page numbers."
- Decision policy: maps user-intent patterns → which tool(s) to call first.
- Guardrails: never invent sub-limits / co-pays / waiting periods; only use tool outputs.

### 3.3 Tool Layer — `tools/*.py`

Six Python functions, each `@tool`-decorated. Tools are pure: input → output, no global state beyond reading data files.

| Tool | Module | Reads | Returns |
|---|---|---|---|
| `search_policy_docs(query, k=5)` | `policy_search.py` | ChromaDB | `list[{chunk, page, score}]` |
| `check_coverage(condition)` | `coverage.py` | `coverage_rules.json` | `{covered, sub_limit, co_pay, waiting_period_days}` |
| `calculate_claim(condition, bill_amount)` | `coverage.py` | `coverage_rules.json` | `{covered_amount, co_pay, payable_to_user}` |
| `get_user_policy(user_id)` | `user.py` | `users.json` | `{sum_insured, co_pay, no_claim_bonus, policy_start, ...}` |
| `find_network_hospitals(city, specialty)` | `hospitals.py` | `hospitals.csv` | `list[{name, area, rating, cashless: True}]` |
| `list_required_documents(claim_type)` | `documents.py` | `documents_rules.json` | `list[str]` |

**Tool discipline:**
- Each docstring is a *when-to-use* spec (not what-it-does), because LangChain feeds the docstring to the LLM as the tool description.
- All tools have type hints; LangChain extracts them as the JSON schema.
- All tools return JSON-serializable dicts/lists (no custom classes).
- All tools fail loudly on bad input (don't silently return `None`).

### 3.4 Data Layer

| File | Format | Purpose | Origin |
|---|---|---|---|
| `data/policy.pdf` | PDF | Sample policy (HDFC Optima Secure) | Shipped — for "use sample" demo |
| `data/conditions_to_check.json` | dict | Canonical list (~48) of conditions to extract per policy | Hand-authored, insurer-agnostic |
| `data/users.json` | dict | Mock user profiles (3 demo users) | Hand-authored |
| `data/hospitals.csv` | CSV | Mock cashless network (50 rows) | Hand-authored |
| `data/documents_rules.json` | dict | Claim paperwork checklists (IRDAI standard) | Hand-authored |
| `policies/<hash>/policy.pdf` | PDF | Per-upload copy of user PDF | Auto on upload |
| `policies/<hash>/rules.json` | dict | Auto-extracted structured rules | Auto on upload (LLM extraction) |
| `chroma_store/` | ChromaDB | Per-PDF vector collections | Auto on upload |

**Consistency-by-construction:** `rules.json` is generated *from* `policy.pdf` via LLM extraction over RAG hits — they cannot drift. Each rule includes `evidence_quote` and `page_reference` for traceability.

---

## 4. Data Flow — End-to-End Request

### 4.1 PDF upload (one-time per file)

```
User clicks upload
   ↓
app.py: file = st.file_uploader(...)
   ↓
hash = sha256(file.bytes)
   ↓
if hash NOT in chroma_store:
    ingest.py:
        pdfplumber → text per page
        → RecursiveCharacterTextSplitter(chunk_size=500, overlap=75)
        → MiniLM embeddings
        → Chroma.from_documents(persist_directory='chroma_store')
   ↓
st.session_state.vector_store = Chroma(...)
st.session_state.pdf_hash    = hash
```

### 4.2 User question (every turn)

```
User types in st.chat_input
   ↓
app.py: executor.invoke({
    "input": user_msg,
    "chat_history": st.session_state.chat_history,
    "user_id": st.session_state.current_user_id,
})
   ↓
AgentExecutor loop (≤ 8 iterations):
    LLM → tool_call(name, args)
    → tools[name](args)
    → observation
    → LLM (with observation appended)
    ... until LLM returns AgentFinish
   ↓
app.py: st.chat_message("agent").write(response["output"])
        if show_reasoning:
            with st.expander("Reasoning"):
                for step in response["intermediate_steps"]: ...
```

### 4.3 Example: chained tools (3 hops)

User asks: *"I'm getting cataract surgery at Apollo Bangalore next month. How much will I get?"*

```
LLM → check_coverage("cataract surgery")
       ↳ {covered: true, sub_limit: 40000, co_pay: 0.10, waiting_period_days: 0}

LLM → find_network_hospitals(city="Bangalore", specialty="ophthalmology")
       ↳ [{name: "Apollo Hospital", cashless: true, ...}, ...]

LLM → get_user_policy(user_id="U1001")
       ↳ {co_pay: 0.10, sum_insured: 500000, ...}

LLM → calculate_claim(condition="cataract surgery", bill_amount=50000)
       ↳ {covered_amount: 40000, co_pay: 4000, payable_to_user: 36000}

LLM → AgentFinish: "Cataract surgery is covered up to ₹40,000 under your policy.
                    Apollo Bangalore is in your cashless network. With your 10%
                    co-pay, you'll get ~₹36,000 reimbursed (assuming bill ~₹50,000)."
```

---

## 5. RAG Pipeline Detail

| Stage | Library | Config |
|---|---|---|
| Parse | `pdfplumber` | page-level text, retain page number as metadata |
| Chunk | `langchain.text_splitter.RecursiveCharacterTextSplitter` | `chunk_size=500`, `chunk_overlap=75` |
| Embed | `sentence-transformers/all-MiniLM-L6-v2` | local, 384-dim, free |
| Store | `chromadb.PersistentClient` | collection name = SHA-256 of PDF |
| Retrieve | `vector_store.similarity_search_with_score` | `k=5`, score threshold ≥ 0.3 |

**Why these choices:**
- **MiniLM, not OpenAI embeddings:** free, local, no API key for embedding.
- **Chunk 500 tokens:** sweet spot — big enough for context, small enough to score precisely. Bigger chunks dilute scores (per brief's pitfall list).
- **Hash-keyed collections:** re-uploading the same PDF is a no-op; switching PDFs is automatic.
- **ChromaDB over FAISS/Pinecone:** persistent, embedded, zero infra. FAISS lacks built-in persistence; Pinecone needs an API key.

**Optional refinement:** if `top_1` score < 0.3, run a HyDE rewrite (LLM expands query first) and re-search.

---

## 6. Tech Stack & Rationale

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.13 | LangChain ecosystem, ML libs |
| LLM | Groq `llama-3.3-70b-versatile` | Free tier, very fast (~300 tok/s), strong tool-calling |
| Agent framework | LangChain `create_tool_calling_agent` | Native tool-calling, built-in scratchpad |
| Embeddings | `all-MiniLM-L6-v2` | Free, local, no API |
| Vector DB | ChromaDB (persistent) | Embedded, zero infra |
| PDF parsing | `pdfplumber` | Best layout fidelity for tables |
| UI | Streamlit | Python-only, ships with chat primitives |
| Tabular data | pandas | CSV ergonomics |
| Config | python-dotenv | `.env` → `os.environ` |

---

## 7. Module / File Layout

```
health_insurance_bot/
├── app.py                  # Streamlit entry point
├── agent.py                # Agent factory + executor
├── ingest.py               # PDF → ChromaDB pipeline
├── tools/
│   ├── __init__.py
│   ├── policy_search.py    # search_policy_docs
│   ├── coverage.py         # check_coverage, calculate_claim
│   ├── hospitals.py        # find_network_hospitals
│   ├── user.py             # get_user_policy
│   └── documents.py        # list_required_documents
├── data/
│   ├── policy.pdf
│   ├── coverage_rules.json
│   ├── users.json
│   ├── hospitals.csv
│   └── documents_rules.json
├── chroma_store/           # gitignored — generated at runtime
├── .env                    # gitignored — holds GROQ_API_KEY
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── ARCHITECTURE.md         # this file
```

---

## 8. Key Design Decisions (and trade-offs)

| Decision | Rationale | Trade-off |
|---|---|---|
| **Single-process Streamlit app** | Fast to build, easy to demo | Doesn't scale past 1 concurrent user without redesign |
| **Mock data in JSON/CSV, not a real DB** | Zero infra, easy to edit | Unrealistic at scale; replace with Postgres for prod |
| **Tool-calling agent (not ReAct prose)** | Structured calls, fewer parse errors | Requires LLM with native tool support |
| **`max_iterations=8`** | Hard cap prevents infinite loops | Complex chains may truncate (rare in practice) |
| **Tools read files on every call** | Simple, stateless, easy to update data | Tiny perf hit; mitigated by file size (<1 MB) |
| **One vector collection per PDF hash** | Cleanly isolates uploads, idempotent | `chroma_store/` grows over time; cleanup is manual |
| **Synthetic users.json instead of auth** | No login complexity | Cannot demo per-user privacy controls |
| **Verbose mode → expander** | Reasoning is visible, not noisy | Slightly verbose UI |

---

## 9. Error Handling & Edge Cases

| Scenario | Handling |
|---|---|
| User asks before uploading PDF | UI shows hint; `search_policy_docs` returns `[]`; agent falls back to other tools or apologises |
| Condition not in `coverage_rules.json` | `check_coverage` returns `{covered: null, reason: "not found"}`; agent admits uncertainty |
| `find_network_hospitals` returns 0 rows | Agent says "no cashless hospital matches in your area" |
| Bill amount > `sum_insured` | `calculate_claim` clamps `payable` to `sum_insured`; flags this in the result |
| LLM emits malformed tool args | `handle_parsing_errors=True` → AgentExecutor retries once |
| Agent loops past `max_iterations` | AgentExecutor returns last partial answer; UI labels it "incomplete" |
| ChromaDB write fails | `ingest.py` raises; UI shows error in sidebar |
| Groq API quota exceeded | Bubble error to UI; suggest waiting or rotating key |

---

## 10. Security & Privacy

| Concern | Mitigation |
|---|---|
| API key leakage | `.env` excluded by `.gitignore`; key never printed in logs |
| User PDF is sensitive | Stored only in `chroma_store/` on local disk; never uploaded anywhere |
| Prompt injection via PDF text | LLM is read-only over retrievals; tools take typed args, not raw model text |
| Multi-tenant data mixing | Each PDF hash → separate Chroma collection |
| Demo dataset realism | All `users.json` profiles are synthetic, no real PII |

**Note:** for production, swap `users.json` for an authenticated lookup, and add a per-user encryption key for the policy store.

---

## 11. Testing Strategy

| Layer | Test | How |
|---|---|---|
| Tools | Unit | Each `tools/*.py` has a `__main__` block with one assertion |
| RAG | Manual smoke | Run `python ingest.py data/policy.pdf` then query a known phrase |
| Agent | Scenario | CLI harness in `agent.py` runs the 3 sample flows from the brief |
| UI | Manual | Walk through 5 user stories in `streamlit run app.py` |
| End-to-end | Fresh-clone | Clone on a second machine, follow README, confirm chat works |

---

## 12. Extension Points

| Hook | Where to add |
|---|---|
| New tool | Add `@tool def foo(...)` in `tools/`, append to `tools=[...]` in `agent.py` |
| New data source | Add file under `data/`, write loader in the corresponding tool module |
| Different LLM | Swap `ChatGroq(...)` in `agent.py` for `ChatOpenAI` / `ChatAnthropic` etc. |
| Different embeddings | Replace `HuggingFaceEmbeddings(...)` in `ingest.py` |
| Multi-PDF compare | Hold a list of vector_stores in `st.session_state`, route by question |
| Auth | Replace `selectbox` with a Streamlit auth provider; pass `user_id` via session |

---

## 13. Build Phases (recap from plan)

| Phase | Deliverable | Time |
|---|---|---|
| 0 | Setup: folders, venv, deps, .env, git | ~10 min |
| 1 | Data: policy.pdf + 4 mock files | ~15 min |
| 2 | RAG: `ingest.py` + populated `chroma_store/` | ~15 min |
| 3 | Tools: 6 `@tool` functions, each with smoke test | ~25 min |
| 4 | Agent: `agent.py` wiring + CLI harness for 3 flows | ~15 min |
| 5 | UI: `app.py` with chat, upload, reasoning toggle | ~15 min |
| 6 | Polish: README, screenshots, GitHub push | ~10 min |
| 7 (opt) | Stretch: claim tracker / compare / multi-language | varies |

---

## 14. Definition of Done

- [ ] User can upload a PDF and ask 5+ different question types successfully
- [ ] Agent chains 2+ tools when needed (coverage check + claim calculation)
- [ ] Sample claim calculation matches the policy's actual sub-limit and co-pay rules
- [ ] Hospital lookup returns only cashless network hospitals
- [ ] Agent's reasoning is visible (verbose mode or "show reasoning" toggle)
- [ ] App runs end-to-end on a fresh machine via README instructions
