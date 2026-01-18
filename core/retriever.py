import os
import requests
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import PromptTemplate
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "tech_ecosystem"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"

# --- 1. Vector Search Tool ---
def search_vector(query: str):
    """Searches Qdrant using direct HTTP API."""
    print(f"   [Vector] Searching for: '{query}'")
    embeddings = OpenAIEmbeddings()
    vector = embeddings.embed_query(query)
    
    search_url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search"
    payload = {"vector": vector, "limit": 3, "with_payload": True}
    
    try:
        response = requests.post(search_url, json=payload)
        response.raise_for_status()
        data = response.json()
        results = [item.get("payload", {}).get("page_content", "") for item in data.get("result", []) if item.get("payload")]
        return "\n\n".join(results) if results else "No relevant vector results found."
    except Exception as e:
        return f"Vector Search Error: {e}"

# --- 2. Graph Search Tool ---

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Cypher translator.
**CRITICAL RULES:**
1. **NEVER** use exact matching like `{{id: 'Tesla'}}`.
2. **ALWAYS** use the `CONTAINS` clause with case-insensitive search.
3. **RELATIONSHIPS:** **NEVER** use a colon `:` or type check inside the query.
   - Wrong: `-[r:CEO_OF]-` 
   - Correct: `-[r]-` (Match ANY relationship)
4. **RETURN:** Always return the nodes AND the relationship type.
   - Example: `RETURN p, type(r) as relationship, c`
5. **TARGET FILTERING:** If the question implies a specific entity type (e.g. "Who" -> Person, "Company" -> Company), add that label to the target node to filter out noise.
   - Question: "Who is the CEO of Tesla?"
   - Correct: `MATCH (c:Company)-[r]-(p:Person) WHERE ...`

Schema:
Node properties: [id]
Common Relationships: OWNS, CEO_OF, PARTNERED_WITH, COMPETES_WITH, ACQUIRED, SUBSIDIARY_OF

Question: {question}
Cypher Query:"""

CYPHER_PROMPT = PromptTemplate(
    input_variables=["question"], 
    template=CYPHER_GENERATION_TEMPLATE
)

def search_graph(query: str):
    """Searches Neo4j using a generated Cypher query."""
    print(f"   [Graph] Generating Cypher for: '{query}'")
    
    try:
        graph = Neo4jGraph(
            url=NEO4J_URI, 
            username=NEO4J_USER, 
            password=NEO4J_PASSWORD,
            enhanced_schema=False, 
            refresh_schema=False   
        )
        
        graph.schema = "Node properties: [id]"
        
        chain = GraphCypherQAChain.from_llm(
            ChatOpenAI(temperature=0, model="gpt-4o-mini"), 
            graph=graph, 
            verbose=True,
            allow_dangerous_requests=True,
            cypher_prompt=CYPHER_PROMPT,
            top_k=100  # <--- Safety buffer: Fetch 100 results to catch everything
        )
        
        response = chain.invoke({"query": query})
        return response['result']
        
    except Exception as e:
        return f"Graph Error: {e}"