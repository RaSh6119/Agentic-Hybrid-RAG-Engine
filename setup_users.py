from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv

load_dotenv()

# Connect to Graph (Blind Mode for speed)
graph = Neo4jGraph(
    url='bolt://localhost:7687', 
    username='neo4j', 
    password='password123',
    enhanced_schema=False,
    refresh_schema=False
)

print("👤 CREATING USER PERSONAS...")

# Cypher query to create users and link them to preferences
create_users_query = """
// 1. Create Rahul (The Technical CTO)
MERGE (u1:User {id: 'Rahul'})
SET u1.role = 'CTO', 
    u1.style = 'Technical, detailed, includes code snippets'
MERGE (t1:Preference {name: 'System Architecture'})
MERGE (t2:Preference {name: 'Python Code'})
MERGE (u1)-[:PREFERS]->(t1)
MERGE (u1)-[:PREFERS]->(t2)

// 2. Create Ram (The Non-Technical CEO)
MERGE (u2:User {id: 'Ram'})
SET u2.role = 'CEO', 
    u2.style = 'Executive summary, concise, focuses on business value'
MERGE (t3:Preference {name: 'Market Risk'})
MERGE (t4:Preference {name: 'ROI Analysis'})
MERGE (u2)-[:PREFERS]->(t3)
MERGE (u2)-[:PREFERS]->(t4)
"""

try:
    graph.query(create_users_query)
    print("   ✅ Users 'Rahul' and 'Ram' created successfully!")
except Exception as e:
    print(f"   ❌ Error creating users: {e}")

# Verify the data exists
print("\n🔍 VERIFYING DATA:")
try:
    res = graph.query("MATCH (u:User)-[:PREFERS]->(p) RETURN u.id, u.role, collect(p.name) as preferences")
    for r in res:
        print(f"   - {r['u.id']} ({r['u.role']}): {r['preferences']}")
except Exception as e:
    print(f"   ⚠️ Verification failed: {e}")