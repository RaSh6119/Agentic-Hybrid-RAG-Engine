import streamlit as st
import time
from brain import ask_brain

# --- Page Config ---
st.set_page_config(page_title="Agentic RAG", page_icon="🧠", layout="wide")

st.title("🧠 Agentic Hybrid RAG Engine")
st.markdown("_Graph (Neo4j) + Vector (Qdrant) + Semantic Router + Memory_")

# --- Sidebar: Persona Selection ---
with st.sidebar:
    st.header("👤 Active Persona")
    # You can add more users here if you created them in Neo4j
    selected_user = st.radio("Who are you?", ["Rahul", "Ram"])
    
    st.info(f"**Current Mode:** {selected_user}\nThe engine will adapt answers to this profile.")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []

# --- Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input ---
if prompt := st.chat_input("Ask about Tesla, Microsoft, or AI risks..."):
    # 1. Show User Message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Generate Answer
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner(f"Thinking as {selected_user}..."):
            try:
                # Call the Brain with the selected persona
                response = ask_brain(prompt, user_id=selected_user)
                
                # Simple typewriter effect for the UI
                message_placeholder.markdown(response)
                full_response = response
            except Exception as e:
                st.error(f"Error: {e}")
                full_response = f"Error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": full_response})