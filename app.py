import streamlit as st
import os
import tempfile
import pdfplumber
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from agent import get_agent_executor

# Load environment variables from .env file if it exists
load_dotenv()

st.set_page_config(page_title="Health Insurance Bot", layout="wide")

# Initialize session state for messages and tools
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:
    st.session_state.api_key = os.environ.get("GROQ_API_KEY", "")
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

with st.sidebar:
    st.title("Settings")
    api_key_input = st.text_input("Groq API Key", type="password", value=st.session_state.api_key)
    if api_key_input:
        st.session_state.api_key = api_key_input
        os.environ["GROQ_API_KEY"] = api_key_input
        
    st.subheader("Upload Policy Document")
    uploaded_file = st.file_uploader("Upload a PDF policy document", type="pdf")
    
    if st.button("Process PDF") and uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Extract text using pdfplumber
                text_content = ""
                with pdfplumber.open(tmp_path) as pdf:
                    for page in pdf.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text_content += extracted + "\n"
                            
                # Chunking
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                    length_function=len
                )
                chunks = text_splitter.split_text(text_content)
                docs = [Document(page_content=chunk, metadata={"source": uploaded_file.name}) for chunk in chunks]
                
                # Setup ChromaDB
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                # clear previous db if exists or just append
                if os.path.exists("./chroma_db"):
                    # For simplicity, we just use the existing or create a new one
                    pass
                vector_store = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
                st.session_state.pdf_processed = True
                st.success(f"Processed {len(docs)} chunks from the policy document.")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")

st.title("Health Insurance Chatbot")
st.write("Ask me anything about your policy, coverage, network hospitals, or claims!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("I'm getting cataract surgery. Is it covered?"):
    if not st.session_state.api_key:
        st.error("Please enter your Groq API key in the sidebar.")
    else:
        # Add user message to state and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    executor = get_agent_executor(st.session_state.api_key)
                    
                    show_reasoning = st.sidebar.toggle("Show Agent Reasoning", value=True)
                    
                    chat_history = []
                    for msg in st.session_state.messages:
                        chat_history.append((msg["role"], msg["content"]))
                    
                    callbacks = []
                    if show_reasoning:
                        from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
                        st_callback = StreamlitCallbackHandler(st.container())
                        callbacks.append(st_callback)
                            
                    response = executor.invoke(
                        {"messages": chat_history},
                        {"callbacks": callbacks}
                    )
                    
                    # LangGraph returns a dictionary where "messages" contains the full conversation
                    final_output = response["messages"][-1].content
                    
                    st.markdown(final_output)
                    st.session_state.messages.append({"role": "assistant", "content": final_output})
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
