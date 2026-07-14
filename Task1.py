import string
import random
from typing import List, Optional

# ==========================================
# PHASE 1: USER INPUT KI SAFAAI (SANITIZATION)
# ==========================================
def sanitize_input(user_input: str) -> str:
    """User ke input se spaces, punctuation khatam karta hai aur lowercase karta hai."""
    # 1. Saare alphabets ko lowercase (choti ABC) mein convert karna
    clean_text = user_input.lower()
    
    # 2. Start aur end se faltu spaces khatam karna
    clean_text = clean_text.strip()
    
    # 3. Punctuation (!, ?, . etc) ko remove karna
    clean_text = "".join(char for char in clean_text if char not in string.punctuation)
    
    return clean_text


# ==========================================
# PHASE 2: USER KI BAAT SAMAJHNA (INTENT MATCHING)
# ==========================================
def match_intent(clean_text: str) -> str:
    """User ke words ko check kar ke uski baat (Intent) samajhta hai."""
    words = clean_text.split()
    
    # Agar user salam ya hello kare
    if any(w in ["hello", "hi", "hey", "greetings", "salam"] for w in words):
        return "greeting"
        
    # Agar user exit karna chahe
    elif any(w in ["exit", "quit", "bye", "shutdown"] for w in words):
        return "exit"
        
    # Agar user madad maange
    elif any(w in ["help", "support", "options"] for w in words):
        return "help"
        
    # UPGRADE 1: Naya Topic - About Us (Company ke baare mein)
    elif any(w in ["about", "decodelabs", "company", "info"] for w in words):
        return "about"
        
    # UPGRADE 1: Naya Topic - Services (Kaam ke baare mein)
    elif any(w in ["services", "products", "work"] for w in words):
        return "services"
        
    # Agar kuch samajh na aaye (Fallback)
    return "unknown"


# ==========================================
# PHASE 3: JAWAB TAYYAR KARNA (RESPONSE DISPATCHER)
# ==========================================
def get_response(intent: str, user_name: Optional[str]) -> str:
    """Intent ke mutabiq random response select karta hai aur user ka naam use karta hai."""
    
    # Har intent ke liye multiple responses taake bot monotonous na lagay
    responses = {
        "help": [
            "I can tell you about our 'services', 'about' our company, or type 'exit' to leave.",
            "Need help? You can ask about what we do, or type 'bye' to quit."
        ],
        "about": [
            "DecodeLabs is an AI firm building safe and smart rule-based programs.",
            "We are DecodeLabs, a team dedicated to cutting-edge AI software integration."
        ],
        "services": [
            "We provide AI Chatbot development, data security guardrails, and software testing.",
            "Our primary services are Custom AI Pipelines and Rule-Based Logic Design."
        ],
        "exit": [
            "Goodbye! Have a great day ahead.",
            "Thank you for chatting with me. Bye!"
        ],
        "unknown": [
            "I didn't quite get that. Could you please try again?",
            "Sorry, that is outside my logic engine. Try typing 'help'."
        ]
    }
    
    # UPGRADE 2: Agar user ka naam save ho chuka hai aur greeting hai, toh personalized response dein
    if intent == "greeting":
        if user_name:
            return f"Hello again, {user_name}! How can I help you today?"
            
        # Agar naam nahi pata, toh standard greeting
        return "Hello! I am your AI assistant. How can I help you today?"
        
    # Baqi intents ke liye randomly ek response choose karna
    return random.choice(responses.get(intent, responses["unknown"]))


# ==========================================
# PHASE 4: MAIN RUNTIME LOOP (THE ENGINE)
# ==========================================
def run_chatbot() -> None:
    """Chatbot ka main loop jo program ko active rakhta hai."""
    
    # State variables (Jo bot chat ke dauran yaad rakhega)
    user_name: Optional[str] = None
    consecutive_unknown_count: int = 0
    awaiting_name: bool = False
    
    print("==========================================")
    print("   SYSTEM ONLINE: DecodeLabs Assistant    ")
    print("==========================================")
    print("Commands: hello, help, about, services, exit\n")
    
    while True:
        # User se input lena
        raw_input = input("You: ")
        
        # Agar user bilkul khali input de, toh dubara poochain
        if not raw_input.strip():
            continue
            
        # UPGRADE 2 (Part A): Agar bot user ke naam ka wait kar raha hai
        if awaiting_name:
            user_name = raw_input.strip()
            awaiting_name = False  # State off kar dein
            print(f"Bot: Thank you, {user_name}! I have saved your name. How can I help you now?\n")
            continue
            
        # Step 1: Input ko clean (sanitize) karna
        clean_text = sanitize_input(raw_input)
        
        # Step 2: Intent match karna
        intent = match_intent(clean_text)
        
        # UPGRADE 2 (Part B): Greeting par naam poochna agar pehle se na pata ho
        if intent == "greeting" and not user_name:
            print("Bot: Welcome! Before we proceed, may I know your name?")
            awaiting_name = True  # Agle loop cycle ke liye state set kar di
            continue
            
        # UPGRADE 3: Error limit (Fallback Counter) check karna
        if intent == "unknown":
            consecutive_unknown_count += 1
            if consecutive_unknown_count >= 3:
                print("Bot: [SYSTEM ALERT] I am having trouble understanding your queries.")
                print("     Would you like to email our support team at support@decodelabs.ai?")
                print("     (Tip: Type 'help' to see valid options.)\n")
                consecutive_unknown_count = 0  # Counter reset kar dein taake loop chalta rahe
                continue
        else:
            # Agar user sahi command likhta hai, toh counter zero ho jata hai
            consecutive_unknown_count = 0
            
        # Step 3: Jawab select karna
        response = get_response(intent, user_name)
        
        # Jawab screen par print karna (Agar user ka naam pata hai toh show karega)
        bot_name = f"Bot ({user_name})" if user_name else "Bot"
        print(f"{bot_name}: {response}\n")
        
        # Agar exit code match ho, toh loop band (break) kar dena
        if intent == "exit":
            break

# Program ko execute karne ke liye
if __name__ == "__main__":
    run_chatbot()