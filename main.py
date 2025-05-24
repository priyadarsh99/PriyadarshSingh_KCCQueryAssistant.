import os
import shutil
import streamlit as st
from dotenv import load_dotenv
import requests
from serpapi import GoogleSearch
from chromadb.config import Settings
from bs4 import BeautifulSoup
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
serpapi_key = os.getenv("SERPAPI_KEY")  # Add this to your .env
os.environ["HF_TOKEN"] = hf_token

# Embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Chroma DB settings
chroma_settings = Settings(chroma_api_impl="local", anonymized_telemetry=False)

# Streamlit page configuration
st.set_page_config(page_title="KCC Chatbot", page_icon="🤖", layout="wide")
st.markdown("<h1 style='text-align: center; color: green;'>🌾 Kisan Conversational Assistant</h1>", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("🛠️ Configuration")
    api_key = st.text_input("Enter Your GROQ API KEY", type="password")

    st.divider()
    st.subheader("🧠 Chat Sessions")

    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = ["default_session"]

    selected_session = st.radio("Select a session:", st.session_state.chat_sessions, index=0)
    session_id = selected_session

    if st.button("➕ New Chat"):
        new_id = f"session_{len(st.session_state.chat_sessions) + 1}"
        if new_id not in st.session_state.chat_sessions:
            st.session_state.chat_sessions.insert(0, new_id)
        session_id = new_id
        st.experimental_rerun()

    st.divider()
    if st.button("🗑️ Clear Chroma DB"):
        if os.path.exists('./data/chroma_db'):
            shutil.rmtree('./data/chroma_db')
            st.success("✅ Chroma vector store cleared.")
        else:
            st.info("No Chroma DB found to clear.")

# Require API Key
if not api_key:
    st.warning("⚠️ Please enter your GROQ API key to continue.")
    st.stop()

# LLM initialization
llm = ChatGroq(groq_api_key=api_key, model="llama-3.3-70b-versatile")

# Chat history session storage
if "store" not in st.session_state:
    st.session_state.store = {}

# Load vector store
persist_dir = "data/chroma_db"
vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
retriever = vectordb.as_retriever()

# Rewriting follow-up questions
contextualize_q_system_prompt = (
    "You are given a chat history and a follow-up user question that may depend on previous conversation context. "
    "Your task is to rewrite the user's latest question into a standalone version that can be understood without referring to the prior chat. "
    "If the question is already standalone, return it unchanged. Do not answer the question—only reformulate it if necessary."
)

contextualise_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualise_q_prompt
)

# Main QA Prompt
system_prompt = (
    "You are a knowledgeable and helpful AI assistant. "
    "Using only the information provided in the retrieved context below, answer the user's question clearly and accurately. "
    "If the answer is not explicitly stated or cannot be inferred with high confidence from the context, respond with \"I don't know.\" "
    "Keep your answers concise—no more than three sentences.\n\nContext:\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Manage chat history
def get_session_history(_: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# 🔍 Web fallback using SerpAPI
def search_online(query):
    try:
        if not serpapi_key:
            return "❌ SerpAPI key is missing. Please add it to your .env file."

        search = GoogleSearch({
            "q": query,
            "api_key": serpapi_key,
            "num": 3
        })
        results = search.get_dict()

        if "organic_results" not in results or not results["organic_results"]:
            return "🤔 Couldn't find relevant information online."

        top_results = results["organic_results"][:3]
        summary = " ".join([r["title"] + ": " + r.get("snippet", "") for r in top_results])
        return summary.strip()

    except Exception as e:
        return f"🌐 Online search failed: {e}"

# Chat display section
st.subheader("💬 Please Ask your Question...")
st.markdown("You can type a question and the assistant will retrieve context-aware answers.")

# Display chat history
if session_id in st.session_state.store:
    st.markdown("### 📜 Chat History")
    for msg in st.session_state.store[session_id].messages:
        role = msg.type
        icon = "🧑‍💼" if role == "human" else "🤖"
        with st.chat_message("user" if role == "human" else "assistant"):
            st.markdown(f"{icon} {msg.content}")

# Input box
user_input = st.chat_input("Type your question here...")

if user_input:
    session_history = get_session_history(session_id)

    with st.chat_message("user"):
        st.markdown(f"🧑‍💼 {user_input}")
    session_history.add_user_message(user_input)

    with st.spinner("🤔 Thinking..."):
        try:
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            initial_answer = response["answer"].strip()
            st.write("🧠 Answer from context:", initial_answer)

            fallback_triggers = ["i don't know", "i do not know", "cannot be inferred", "not explicitly stated"]
            if any(trigger in initial_answer.lower() for trigger in fallback_triggers) or len(initial_answer) < 20:
                with st.spinner("🔍 Searching online..."):
                    fallback_answer = search_online(user_input)
                    st.write("🌐 Answer from web:", fallback_answer)
                    session_history.add_ai_message(fallback_answer)

                    with st.chat_message("assistant"):
                        st.markdown(f"🤖 {fallback_answer}")
            else:
                session_history.add_ai_message(initial_answer)
                with st.chat_message("assistant"):
                    st.markdown(f"🤖 {initial_answer}")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# Optional: CSS styling
st.markdown("""
    <style>
        .stChatMessage .stMarkdown p {
            font-size: 16px;
            line-height: 1.6;
        }
        .stRadio > label > div[data-testid="stMarkdownContainer"] > p {
            font-size: 16px;
            margin-top: 5px;
        }
        .block-container {
            padding-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)
