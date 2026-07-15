import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ANSI Color codes for professional terminal UI
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# Richer database with explicit keyword tags for precise mathematical scaling
items_database = [
    {
        "id": 1,
        "title": "Python Basics & Logic Building",
        "tags": "python, programming, loops, logic, scripting, basics, coding, variables",
        "description": "Learn basic programming, variables, loops, conditional statements, and logical thinking using Python scripting."
    },
    {
        "id": 2,
        "title": "Advanced Machine Learning with KNN",
        "tags": "ml, machine learning, knn, classification, standardscaler, f1-score, supervised, model",
        "description": "Master supervised learning algorithms, training-testing split, feature scaling, and evaluation metrics like Confusion Matrix and F1-score."
    },
    {
        "id": 3,
        "title": "Web Development (Frontend & Backend)",
        "tags": "web, frontend, backend, php, html, css, javascript, responsive, website, server",
        "description": "Build modern websites using HTML, CSS, JavaScript, responsive styles, PHP, and server-side configurations."
    },
    {
        "id": 4,
        "title": "Cloud Computing & DevOps Automation",
        "tags": "cloud, devops, aws, automation, server, deployment, pipelines, routing, network",
        "description": "Configure cloud infrastructure, deployment pipelines, virtual networks, automation scripts, and server routing."
    },
    {
        "id": 5,
        "title": "Mobile App Development with Flutter & Dart",
        "tags": "mobile, flutter, dart, cross-platform, app, database, ui, state management, banking app",
        "description": "Develop cross-platform native mobile applications, state management, database connections, and custom UI design using Flutter and Dart."
    },
    {
        "id": 6,
        "title": "Automata Theory & Formal Languages",
        "tags": "automata, dfa, nfa, formal languages, theory, computational, state machine, equivalence",
        "description": "Understand computational logic, DFA to NFA conversions, state machine equivalence, and formal grammatical languages."
    },
    {
        "id": 7,
        "title": "Database Management & SQL Queries",
        "tags": "database, sql, mysql, relational, queries, schema, tables, optimize, index",
        "description": "Design relational database schemas, write structured SQL queries, optimize indexes, and debug table connections."
    }
]

def display_welcome():
    print(f"{Colors.HEADER}{Colors.BOLD}================================================================{Colors.ENDC}")
    print(f"{Colors.GREEN}{Colors.BOLD}          DECODELABS COGNITIVE RECOMMENDATION ENGINE            {Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}================================================================{Colors.ENDC}")
    print(f"{Colors.CYAN}Available specialized tracks in our database:{Colors.ENDC}")
    for idx, item in enumerate(items_database, 1):
        print(f"  {Colors.BOLD}{idx}.{Colors.ENDC} {item['title']}")
    print(f"{Colors.HEADER}----------------------------------------------------------------{Colors.ENDC}")

def main():
    # Clear screen for fresh execution
    os.system('cls' if os.name == 'nt' else 'clear')
    history_logs = []
    
    while True:
        display_welcome()
        
        print(f"\n{Colors.BOLD}[INPUT]{Colors.ENDC} What skills or technologies do you want to learn today?")
        print(f"{Colors.BLUE}Example: 'I want to build a dynamic web database with SQL queries and backend'{Colors.ENDC}")
        user_input = input(f"{Colors.GREEN}Enter your preference:{Colors.ENDC} ").strip()
        
        if not user_input:
            print(f"\n{Colors.FAIL}[ERROR] Input cannot be empty! Please type something.{Colors.ENDC}")
            input("\nPress Enter to try again...")
            os.system('cls' if os.name == 'nt' else 'clear')
            continue
            
        # STEP 2: PROCESS - TF-IDF & Cosine Similarity
        corpus = [f"{item['description']} {item['tags']}" for item in items_database]
        all_docs = corpus + [user_input]
        
        # Penalizing generic high-frequency words and rewarding specific tags
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        tfidf_matrix = vectorizer.fit_transform(all_docs)
        
        item_vectors = tfidf_matrix[:-1]
        user_vector = tfidf_matrix[-1]
        
        similarity_scores = cosine_similarity(user_vector, item_vectors).flatten()
        ranked_indices = np.argsort(similarity_scores)[::-1]
        
        # STEP 3: OUTPUT - Ranked Recommendations
        print(f"\n{Colors.HEADER}{Colors.BOLD}================================================================{Colors.ENDC}")
        print(f"{Colors.GREEN}{Colors.BOLD}                     TOP MATCHED PROGRAMS                       {Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}================================================================{Colors.ENDC}")
        print(f"Based on your profile: {Colors.CYAN}\"{user_input}\"{Colors.ENDC}\n")
        
        top_rec_text = []
        matched_any = False
        
        for rank, idx in enumerate(ranked_indices[:3], 1):
            score = similarity_scores[idx]
            if score > 0.05:  # Strict match threshold
                matched_any = True
                item = items_database[idx]
                match_percentage = score * 100
                color = Colors.GREEN if score > 0.25 else Colors.BLUE
                
                print(f"{Colors.BOLD}Rank {rank}: {item['title']}{Colors.ENDC}")
                print(f"   ↳ {color}Match Confidence: {match_percentage:.2f}%{Colors.ENDC}")
                print(f"   ↳ {Colors.CYAN}Description:{Colors.ENDC} {item['description']}")
                print(f"   ↳ {Colors.WARNING}Core Tags matched:{Colors.ENDC} {item['tags']}\n")
                
                top_rec_text.append(
                    f"Rank {rank}: {item['title']} (Match Confidence: {match_percentage:.2f}%)\n"
                    f"   Description: {item['description']}\n"
                    f"   Core Tags: {item['tags']}\n\n"
                )
        
        if not matched_any:
            no_match_msg = "No strong program matched your query. Try technical keywords like 'Web', 'Python', 'ML', or 'Flutter'.\n"
            print(f"{Colors.FAIL}{no_match_msg}{Colors.ENDC}")
            top_rec_text.append(no_match_msg)
            
        history_logs.append({
            "query": user_input,
            "recommendations": "".join(top_rec_text)
        })
        
        print(f"{Colors.HEADER}----------------------------------------------------------------{Colors.ENDC}")
        choice = input(f"{Colors.BOLD}Do you want to try another query? (y/n):{Colors.ENDC} ").strip().lower()
        if choice != 'y':
            break
            
        os.system('cls' if os.name == 'nt' else 'clear')
        
    # Save search logs history to file
    try:
        os.makedirs("Project3", exist_ok=True)
        with open("Project3/results.txt", "w") as f:
            f.write("================================================================\n")
            f.write("        DECODELABS COGNITIVE RECOMMENDATION LOGS                \n")
            f.write("================================================================\n\n")
            for i, log in enumerate(history_logs, 1):
                f.write(f"--- Session #{i} ---\n")
                f.write(f"User Query: {log['query']}\n\n")
                f.write("Generated Recommendations:\n")
                f.write(log['recommendations'])
                f.write("----------------------------------------------------------------\n\n")
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 Excellent! All search logs saved to 'Project3/results.txt'!{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.FAIL}Error writing results file: {e}{Colors.ENDC}")
        
    print(f"\n{Colors.BLUE}Thank you for using DecodeLabs Recommendation Engine! Goodbye! 👋{Colors.ENDC}\n")

if __name__ == "__main__":
    main()