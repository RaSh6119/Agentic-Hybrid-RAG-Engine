from core.router import route_question  # <--- CHANGED: Import the correct name
from core.retriever import search_vector, search_graph, get_user_context 
from langchain_openai import ChatOpenAI

# Initialize the Final Answer LLM
llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini")

def ask_brain(question: str, user_id: str = "Alice"):
    """
    The Main Engine:
    1. Fetches User Memory (Persona).
    2. Routes the Question (Graph vs Vector).
    3. Retrieves Data.
    4. Synthesizes a Personalized Answer.
    """
    print(f"\n🧠 PROCESSING for User: {user_id}")
    
    # 1. GET MEMORY (The Twist)
    user_context = get_user_context(user_id)
    print(f"   📄 Context Loaded: {user_context.replace(chr(10), ' ')}") 
    
    # 2. ROUTE & RETRIEVE
    # We call route_question and use .upper() to ensure it matches our check
    route = route_question(question).upper() 
    raw_data = ""
    
    if route == "GRAPH_STORE":
        print(f"   👉 Routing to: Graph Store")
        raw_data = search_graph(query=question)
        if not raw_data or "I don't know" in str(raw_data):
            print("   ⚠️ Graph empty. Fallback to Vector.")
            raw_data = search_vector(query=question)
    else:
        print(f"   👉 Routing to: Vector Store")
        raw_data = search_vector(query=question)

    # 3. SYNTHESIZE ANSWER (The Agentic Part)
    # We combine the User Context + The Retrieved Data into one final prompt
    final_prompt = f"""
    You are a helpful AI Assistant.
    
    {user_context}
    
    DATA RETRIEVED:
    {raw_data}
    
    USER QUESTION: {question}
    
    Answer the question strictly based on the retrieved data, but ADAPT your tone and depth 
    to match the User Profile above.
    """
    
    response = llm.invoke(final_prompt)
    return response.content