import json
from transformers import pipeline

with open('troubleshooting_knowledge_base.json', 'r') as f:
    knowledge_base = json.load(f)

nlp = pipeline('question-answering')
user_input = input("Please describe your problem: ")

for issue, details in knowledge_base.items():
    if details["sympthom"].lower() in user_input.lower():
        print(f"Possible solution: {details['solution']}")
        break
    else:
        print("No matching issue found in the knowledge base.")

def diagnose_network_issue():
    print("Have you restarted your router?")
    response = input("Yes/No: ").strip().lower()
    if response == "no":
        print("Please restart your router and check again.")
    else:
        print("Try resetting your network settings or contacting your provider.")

if "internet" in user_input.lower():
    diagnose_network_issue()

def automate_fix(issue):
    if issue == "slow_internet":
        print("Resetting network settings...")
        # Simulated network reset
        print("Network settings have been reset. Please check your connection.")
    else:
        print("Automation is not available for this issue.")


if "internet" in user_input.lower():
    automate_fix("slow_internet")

def collect_feedback():
    feedback = input("Did this solution resolve your issue? (Yes/No): ").strip().lower()
    if feedback == "yes":
        print("Great! Your feedback has been recorded.")
    else:
        print("We're sorry the issue persists. We'll improve our solution based on your input.")

collect_feedback()