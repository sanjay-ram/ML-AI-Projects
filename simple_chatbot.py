import spacy

nlp = spacy.load('en_core_web_sm')
def respond_text(user_input):
    doc = nlp(user_input)
    for token in doc:
        print(token.text, token.pos_)
    
    if 'hello' in user_input.lower() or 'hi' in user_input.lower():
        response ="Hello there, how can I help you?"
    elif 'how are you' in user_input.lower():
        response = "I'm doing well, thank you. What about you?"
    else:
        for entity in doc.ents:
            if entity.label_ == "PERSON":
                response = f"Ah, your talking about {entity.text}. What about them?"
            elif entity.label_ == "GPE":
                response = f"{entity.text} is an interesting place!"
            else:
                response = "I'm still learning. Can you rephrase that?"
    return response

while True:
    user_input = input("You: ")
    response = respond_text(user_input)
    print(f"Chatbot: {response}")
