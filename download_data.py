import wikipedia
import os
import time

topics = [
    "Apple Inc.",
    "Microsoft Corporation",
    "Google",
    "Amazon (company)",
    "Meta Platforms",

    # The "AI & Chips" Cluster (Dense Connections)
    "Nvidia",
    "AMD",
    "Intel",
    "Taiwan Semiconductor Manufacturing Company", # TSMC
    "OpenAI",
    "Anthropic",
    "DeepMind",
    
    # The "Hardware & Auto" Cluster
    "Tesla, Inc.",
    "SpaceX",
    "Samsung Electronics",
    "Sony Group",
    "Qualcomm",
    "Arm Holdings",
    
    # The "Enterprise & Cloud" Cluster
    "Oracle Corporation",
    "IBM",
    "Salesforce",
    "Adobe Inc.",
    "Netflix",
    "Uber",
    "Airbnb",
    
    # The "Acquisition Targets" (Good for testing "Who owns X?")
    "LinkedIn",
    "GitHub",
    "Instagram",
    "WhatsApp",
    "YouTube"
]

output_dir = "./data"
os.makedirs(output_dir, exist_ok=True)

print(f"Downloading data for {len(topics)} Wikipedia pages...")

for topic in topics:
    try:
        page = wikipedia.page(topic, auto_suggest=False)
        filename =  topic.replace(" ", "_").replace(",", "").replace("/", "") + ".txt"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(page.content)
        print(f"Saved {filename}")

        time.sleep(1)
    except Exception as e:
        print(f"Error downloading {topic}: {e}")

print("All data downloaded successfully!")