# SETUP — Run on a Fresh Laptop

End-to-end setup guide for cloning the project and running it on a new machine. Tested on macOS / Linux. Windows users should run the same commands in PowerShell / WSL.

---

## 1. Prerequisites

| What | Version | How to check |
|---|---|---|
| Python | 3.10 or newer | `python3 --version` |
| Git | any recent | `git --version` |
| ~3 GB free disk | (for venv + ML models) | — |

If Python is missing: install from [python.org](https://www.python.org/downloads/) or via your package manager (`brew install python` on macOS).

---

## 2. Get the code

```bash
git clone <your-repo-url> health_insurance_bot
cd health_insurance_bot
```

(If you're sharing the folder directly without git, just `cd` into it.)

---

## 3. Create a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows PowerShell
```

You should see `(.venv)` prefix in your shell prompt. **Do this every time** you open a new terminal in this folder.

---

## 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This downloads ~2 GB (PyTorch, sentence-transformers, ChromaDB, langchain, streamlit). Takes 3-5 minutes on first run.

Verify:
```bash
python -c "import streamlit, langchain, chromadb, sentence_transformers, pdfplumber; print('OK')"
```
Should print `OK`.

---

## 5. Get a free Groq API key

1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up (free, no credit card required)
3. Click **API Keys → Create API Key**
4. Copy the key (starts with `gsk_...`)

---

## 6. Configure environment

```bash
cp .env.example .env
```

Open `.env` in any text editor and paste your key:
```
GROQ_API_KEY=gsk_yourActualKeyHere
```

Save. **Never commit `.env`** — it's gitignored.

---

## 7. Process the sample policy (one-time, ~1-2 minutes)

This step parses the included sample PDF, builds the vector index, and auto-extracts coverage rules. It only needs to run once per unique PDF.

```bash
python policy_processor.py data/policy.pdf
```

You should see:
```
=== processing policy.pdf (hash=...) ===
--- step 1: RAG ingestion ---
[1/4] parsing policy.pdf ...
      -> 53 pages
[2/4] chunking (size=500, overlap=75) ...
      -> ~350 chunks
[3/4] embedding with sentence-transformers/all-MiniLM-L6-v2 ...
[4/4] writing to chroma collection ...
done.

--- step 2: rule extraction ---
[extract] 48 conditions to extract via llama-3.3-70b-versatile
  [ 1/48] cataract surgery   covered=Y  wait=730d
  ...
  [48/48] domiciliary hospitalization at home   covered=Y  wait=0d
rules saved -> policies/<hash>/rules.json
```

The first run downloads a ~80 MB embedding model. Subsequent runs reuse it.

---

## 8. Launch the Streamlit app

```bash
streamlit run app.py
```

You should see:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

Open that URL in any browser.

---

## 9. Use it

In the browser:

1. **Sidebar Step 1** — pick a demo user (Asha / Ravi / Meera).
2. **Sidebar Step 2** — either:
   - Click **"...use the sample HDFC Ergo Optima Secure"** for instant demo, OR
   - Drag-drop your own health insurance PDF (will auto-process — adds ~1-2 min on first upload of that PDF).
3. **Main pane** — ask any question:
   - *"Is cataract surgery covered? My bill is Rs 50,000 — how much will I get?"*
   - *"What documents do I need for a hospitalization claim?"*
   - *"Find a cashless hospital in Bangalore for ophthalmology"*
   - *"I have diabetes — is it covered?"*
   - *"What's the waiting period for pre-existing diseases?"*
4. **Toggle "Show reasoning"** in the sidebar to see each tool call the agent makes.

---

## 10. Project layout (for reference)

```
health_insurance_bot/
├── app.py                    # Streamlit chat UI (entry point)
├── agent.py                  # LangChain agent + 6 tools wired
├── ingest.py                 # PDF -> ChromaDB
├── rules_extractor.py        # LLM extraction of structured rules
├── policy_processor.py       # Orchestrator: ingest + extract
├── tools/                    # 6 @tool functions
│   ├── _context.py           # Shared session context
│   ├── coverage.py           # check_coverage, calculate_claim
│   ├── user.py               # get_user_policy
│   ├── hospitals.py          # find_network_hospitals
│   ├── documents.py          # list_required_documents
│   └── policy_search.py      # search_policy_docs (RAG)
├── data/
│   ├── policy.pdf            # Sample HDFC Ergo Optima Secure (53 pages)
│   ├── conditions_to_check.json  # Canonical conditions to extract
│   ├── users.json            # 3 demo profiles
│   ├── hospitals.csv         # 50 mock cashless hospitals
│   └── documents_rules.json  # IRDAI standard claim docs
├── policies/                 # AUTO-GENERATED per-PDF rules (gitignored)
│   └── <pdf_hash>/rules.json
├── chroma_store/             # AUTO-GENERATED vector DB (gitignored)
├── .env                      # YOUR Groq API key (gitignored)
├── .env.example              # Template
├── requirements.txt
├── README.md
├── SETUP.md                  # this file
└── ARCHITECTURE.md           # system design reference
```

---

## 11. Troubleshooting

### "ModuleNotFoundError" when running `streamlit run app.py`
Forgot to activate the venv. Run `source .venv/bin/activate` and try again.

### "GROQ_API_KEY not set"
Check `.env` exists in the project root and contains `GROQ_API_KEY=gsk_...` (no quotes, no spaces).

### Streamlit page loads but agent says "rate limit reached"
Groq free tier has a daily token cap. Either wait ~30 minutes for the quota to reset, or switch to the smaller model: open `agent.py` and change `llama-3.3-70b-versatile` to `llama-3.1-8b-instant` (higher daily limit).

### Re-uploading the same PDF feels slow
It shouldn't — re-uploads are no-ops because we hash the file. If processing re-runs anyway, the file's bytes changed (re-saved/re-exported PDFs will hash differently).

### "No rule found for X"
Either:
- Your PDF didn't include that condition's coverage info, OR
- It wasn't in `data/conditions_to_check.json`. Add it there and re-run `python policy_processor.py <pdf> --force` (the `--force` flag re-extracts).

### Want to use a different policy PDF?
Just upload it via the sidebar — the system auto-extracts new rules for it. The PDF hash keeps everything isolated, so multiple PDFs can coexist.

### Want to wipe everything and start fresh?
```bash
rm -rf chroma_store/ policies/
streamlit run app.py
```

---

## 12. What you need to give the user (recap)

If you're packaging this for someone else, they need exactly **two things from you**:

1. **The code folder** (clone the repo or zip the directory excluding `.venv`, `chroma_store`, `policies`, `.env`).
2. **Their own Groq API key** — they sign up for free at [console.groq.com](https://console.groq.com) and put it in their `.env`.

Everything else (Python deps, ML models, vector DB, rule extraction) installs / generates automatically.

---

## 13. Stopping the app

In the terminal running streamlit, press `Ctrl + C`.

To deactivate the venv: `deactivate`.
