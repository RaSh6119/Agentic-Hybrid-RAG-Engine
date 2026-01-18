from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv

load_dotenv()

# Connect to Graph
graph = Neo4jGraph(
    url='bolt://localhost:7687', 
    username='neo4j', 
    password='password123', 
    enhanced_schema=False
)

print('\n--- 🔍 SEARCHING FOR TESLA ---')
# We use 'toLower' to find it even if it is 'TESLA' or 'Tesla Inc.'
res = graph.query("MATCH (n) WHERE toLower(n.id) CONTAINS 'tesla' RETURN n.id, labels(n) LIMIT 5")
print(res)

print('\n--- 🔍 SEARCHING FOR META ---')
res = graph.query("MATCH (n) WHERE toLower(n.id) CONTAINS 'meta' RETURN n.id, labels(n) LIMIT 5")
print(res)