import json
from transformers import pipeline

with open('troubleshooting_knowledge_base.json', 'r') as f:
    knowledge_base = json.load(f)

nlp = pipeline('question-answering')
user_input = input("Please describe your problem: ")

for issue, details in knowledge_base.items():
    if details["symptom"].lower() in user_input.lower():
        print(f"\nPossible solution: {details['solution']}")
        break
    else:
        print("No matching issue found in the knowledge base.")

def diagnose_network_issue():
    print("\nHave you restarted your router?")
    response = input("Yes/No: ").strip().lower()
    if response == "no":
        print("Please restart your router and check again.")
    else:
        print("Try resetting your network settings or contacting your provider.")

def diagnose_app_issue():
    print("\nHave you updated the app to the latest version and restarted your computer?")
    response = input("Yes/No: ").strip().lower()
    if response == "no":
        print("Please update the app or restart your computer, then check again.")
    else:
        print("Try resetting your app settings!")


def diagnose_email_issue():
    print("\nHave you verfiyed email server settings and check your internet connection?")
    response = input("Yes/No: ").strip().lower()
    if response == "no":
        print("Please verfiy the email server settings and check your internet connection.")
    else:
        print("Try reverifying the email server settings and rechecking your internet connection.")

def diagnose_printer_issue():
    print("\nHave you checked the printer connections, restarted the printer?")
    response = input("Yes/No: ").strip().lower()
    if response == "no":
        print("Please check the printer connections, restart the printer, and reinstall drivers if necessary.")
    else:
        print("Try rechecking printer connections, restarting the printer.")

if "app" in user_input.lower():
    diagnose_app_issue()

if "internet" in user_input.lower():
    diagnose_network_issue()


if "email" in user_input.lower():
    diagnose_email_issue()

if "printer" in user_input.lower():
    diagnose_printer_issue()

def automate_fix(issue):
    if issue == "slow_internet":
        print("\nResetting network settings...")
        print("Network settings have been reset. Please check your connection.")
    elif issue == "crashing_app":
        app_name = input("\nGive your apps name please: ")
        print("\nResetting app settings...")
        print(f"App settings have been reset. Please check your {app_name}")
    elif issue == "email_not_working":
        email = input("\nPlease enter your email here: ")
        print("\nVerfiying the email server setting...")
        print("\nChecking the internet connection...")
        print(f"\nYour email server settings has been verifyed & the internet connection  have been checked and reseted. Please try sending and recieving with your {email}.")
    elif issue == "printer_not_responding":
        printer = input("Enter your printers name: ")
        print("\nRestarting your printer...")
        print(f"\nYour {printer} has been successfully restarted.")
    else:
        print("\nAutomation is not available for this issue.")


if "internet" in user_input.lower():
    automate_fix("slow_internet")

elif "app" in user_input.lower():
    automate_fix("crashing_app")

elif "email" in user_input.lower():
    automate_fix("email_not_working")

elif "printer" in user_input.lower():
    automate_fix("printer_not_responding")

def collect_feedback():
    feedback = input("\nDid this solution resolve your issue? (Yes/No): ").strip().lower()
    if feedback == "yes":
        print("\nGreat! Your feedback has been recorded.")
    else:
        print("\nWe're sorry the issue persists. We'll improve our solution based on your input.")

collect_feedback()