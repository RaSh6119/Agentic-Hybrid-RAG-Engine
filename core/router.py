from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# --- THE FIX: Import directly from pydantic ---
from pydantic import BaseModel, Field
from typing import Literal

# 1. Define the Output Structure (The Decision)
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    destination: Literal["vector_store", "graph_store"] = Field(
        ..., 
        description="Choose 'graph_store' for questions about specific entities, relationships, ownership, or roles. Choose 'vector_store' for general summaries, history, or broad concepts."
    )

# 2. The Router Logic
def route_question(question: str):
    print(f"🤔 Routing Question: '{question}'")
    
    # The System Prompt works as the "Brain's Instructions"
    system = """You are an expert at routing user questions to a vectorstore or graph database.
    
    Use the GRAPH_STORE for:
    - Questions about relationships (e.g., "Who is the CEO of X?", "Does A own B?", "How is X connected to Y?")
    - Questions involving specific entities (companies, people) and their connections.
    
    Use the VECTOR_STORE for:
    - Questions asking for summaries (e.g., "Summarize the history of Apple")
    - Broad conceptual questions (e.g., "What is generative AI?", "Risks of cloud computing")
    """
    
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Structured output binding
    structured_router = llm.with_structured_output(RouteQuery)
    
    router = route_prompt | structured_router
    decision = router.invoke({"question": question})
    
    return decision.destination