"""Streamlit chat UI for the health insurance agent."""

import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent import build_agent
from ingest import pdf_fingerprint
from policy_processor import process
from tools._context import CTX, set_custom_user, set_policy, set_user

load_dotenv()

st.set_page_config(page_title="Health Insurance Chatbot", page_icon=":hospital:", layout="wide")

ROOT = Path(__file__).parent
POLICIES_DIR = ROOT / "policies"
DATA_DIR = ROOT / "data"
SAMPLE_PDF = DATA_DIR / "policy.pdf"


@st.cache_data
def load_users() -> dict:
    return json.loads((DATA_DIR / "users.json").read_text())


# session state
ss = st.session_state
ss.setdefault("chat_history", [])
ss.setdefault("processed_pdf_hash", None)
ss.setdefault("processed_pdf_path", None)
ss.setdefault("user_id", None)
ss.setdefault("agent", None)
ss.setdefault("show_reasoning", True)


def _ensure_policy_active(pdf_path: Path):
    """Process the PDF if needed, set context, build agent."""
    h = pdf_fingerprint(pdf_path)
    if h != ss.processed_pdf_hash:
        with st.spinner(f"Processing {pdf_path.name}: ingestion + rule extraction (~1-2 min on first upload)..."):
            process(pdf_path)
        ss.processed_pdf_hash = h
        ss.processed_pdf_path = pdf_path
    set_policy(h, pdf_path)
    if ss.agent is None:
        ss.agent = build_agent(verbose=False)


# ===== sidebar =====
with st.sidebar:
    st.title(":hospital: Health Insurance Bot")
    st.caption("LangChain + Groq + ChromaDB")

    st.subheader("1. Your profile")
    profile_mode = st.radio(
        "How do you want to provide your profile?",
        ["Use a demo profile", "Enter my own info"],
        index=0,
        horizontal=True,
    )

    if profile_mode == "Use a demo profile":
        users = load_users()
        options = list(users.keys())
        labels = {uid: f"{u['name']} — {u['city']} (SI ₹{u['sum_insured']:,})" for uid, u in users.items()}
        idx = options.index(ss.user_id) if ss.user_id in options else 0
        selected = st.selectbox("Demo user", options, index=idx, format_func=lambda x: labels[x])
        ss.user_id = selected
        set_user(selected)
    else:
        # Custom user form — pre-fill from session state if previously submitted
        prev = ss.get("custom_user_data", {})
        with st.form("custom_user_form", clear_on_submit=False):
            name = st.text_input("Your name", value=prev.get("name", ""))
            city = st.selectbox(
                "City",
                ["Bangalore", "Mumbai", "Delhi", "Chennai", "Hyderabad", "Pune", "Kolkata"],
                index=["Bangalore", "Mumbai", "Delhi", "Chennai", "Hyderabad", "Pune", "Kolkata"].index(prev.get("city", "Bangalore")),
            )
            sum_insured = st.number_input("Sum insured (₹)", min_value=100_000, max_value=20_000_000, value=int(prev.get("sum_insured", 500_000)), step=100_000)
            co_pay_pct = st.number_input("Co-pay (%)", min_value=0, max_value=50, value=int(round(float(prev.get("co_pay", 0.10)) * 100)))
            ncb_pct = st.number_input("No-claim bonus (%)", min_value=0, max_value=100, value=int(round(float(prev.get("no_claim_bonus", 0)) * 100)))
            policy_start = st.date_input("Policy start date", value=None)
            ped = st.text_input("Pre-existing conditions (comma-separated, optional)", value=", ".join(prev.get("pre_existing", [])))
            submitted = st.form_submit_button("Save my profile", use_container_width=True)
        if submitted and name.strip():
            profile = {
                "user_id": "self",
                "name": name.strip(),
                "city": city,
                "sum_insured": int(sum_insured),
                "co_pay": float(co_pay_pct) / 100.0,
                "no_claim_bonus": float(ncb_pct) / 100.0,
                "policy_start": str(policy_start) if policy_start else None,
                "pre_existing": [p.strip() for p in ped.split(",") if p.strip()],
            }
            ss.custom_user_data = profile
            ss.user_id = "self"
            set_custom_user(profile)
            st.success(f"Saved profile for **{name}** — SI ₹{sum_insured:,}, co-pay {co_pay_pct}%")
        elif ss.get("custom_user_data"):
            # Re-apply previously-saved custom user on re-renders
            set_custom_user(ss.custom_user_data)
            st.caption(f"Active: {ss.custom_user_data['name']} (SI ₹{ss.custom_user_data['sum_insured']:,}, co-pay {int(ss.custom_user_data['co_pay']*100)}%)")
        else:
            st.info("Fill the form and click **Save my profile** to continue.")

    st.subheader("2. Load a policy")
    uploaded = st.file_uploader("Your policy PDF", type="pdf")
    if st.button("...or use the sample HDFC Ergo Optima Secure", use_container_width=True):
        _ensure_policy_active(SAMPLE_PDF)
        st.rerun()

    if uploaded is not None:
        tmp = POLICIES_DIR / "_uploads" / uploaded.name
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(uploaded.getvalue())
        if pdf_fingerprint(tmp) != ss.processed_pdf_hash:
            _ensure_policy_active(tmp)
            st.rerun()

    st.divider()
    ss.show_reasoning = st.toggle("Show reasoning", value=ss.show_reasoning)
    if st.button("Clear chat", use_container_width=True):
        ss.chat_history = []
        st.rerun()

    if ss.processed_pdf_hash:
        st.success("Policy loaded")
        st.caption(f"Hash: `{ss.processed_pdf_hash}`")
        rules_file = POLICIES_DIR / ss.processed_pdf_hash / "rules.json"
        if rules_file.exists():
            r = json.loads(rules_file.read_text())
            st.caption(f"{r['_meta']['num_conditions']} extracted rules")


# ===== main =====
st.title("Ask about your health insurance policy")

if ss.agent is None:
    st.info("← Pick a user and load a policy in the sidebar to start.\n\n"
            "**Tip:** click *use the sample HDFC Ergo Optima Secure* for a quick demo.")
    st.stop()


def render_steps(steps):
    if not steps:
        return
    with st.expander(f"Reasoning — {len(steps)} tool call(s)", expanded=False):
        for i, (action, obs) in enumerate(steps, 1):
            st.markdown(f"**Step {i}: `{action.tool}`**")
            st.json({"input": action.tool_input, "output": obs})


# render history
for turn in ss.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if ss.show_reasoning and turn.get("steps"):
            render_steps(turn["steps"])


# input
if user_msg := st.chat_input("e.g. 'Is cataract surgery covered? Bill is 50000'"):
    ss.chat_history.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = ss.agent.invoke({"input": user_msg})
                answer = result["output"]
                steps = result.get("intermediate_steps", [])
            except Exception as e:
                answer = f"Sorry, I hit an error: `{e}`"
                steps = []
        st.markdown(answer)
        if ss.show_reasoning and steps:
            render_steps(steps)

    ss.chat_history.append({"role": "assistant", "content": answer, "steps": steps})
