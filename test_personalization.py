from brain import ask_brain

question = "Summarize the history of Microsoft."

print("\n==========================================")
print("🤖 ASKING Rahul (The CTO)")
print("==========================================")
# Rahul should get technical details
res_Rahul = ask_brain(question, user_id="Rahul")
print(f"\n📝 Rahul'S ANSWER:\n{res_Rahul}")

print("\n\n==========================================")
print("💼 ASKING Ram (The CEO)")
print("==========================================")
# Ram should get business/market details
res_Ram = ask_brain(question, user_id="Ram")
print(f"\n📝 Ram'S ANSWER:\n{res_Ram}")