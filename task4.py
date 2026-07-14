import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Simple Document Corpus (Our Vector Store)
documents = [
    "DevelopersHub Corporation AI/ML Internship started in June 2026.",
    "The advanced tasks deadline is 21st July, 2026.",
    "Interns must complete at least 3 out of 5 advanced tasks to pass.",
    "Python, scikit-learn, LangChain, and Transformers are key tools in this internship."
]

# 2. Simple Vector Retrieval Engine
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)

def retrieve_document(query):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, doc_vectors).flatten()
    best_match_idx = np.argmax(similarities)
    if similarities[best_match_idx] > 0.1:
        return documents[best_match_idx]
    return "No matching internal documents found."

# 3. Context-Aware Conversation Class (Simulating LangChain Memory)
class ContextChatbot:
    def __init__(self):
        self.memory = []

    def chat(self, user_query):
        # Retrieve context from our Vector Store (RAG Step)
        retrieved_context = retrieve_document(user_query)
        
        # Build prompt using memory + retrieved context
        system_prompt = f"[Context: {retrieved_context}] [History: {self.memory[-2:] if self.memory else 'None'}]"
        
        # Simulating LLM response based on context
        response = f"Based on our documents: '{retrieved_context}'. Is there anything else you'd like to know?"
        
        # Save to Memory
        self.memory.append({"user": user_query, "bot": response})
        return response

# Test the Chatbot
bot = ContextChatbot()
print("Chatbot initialized! Type your questions:")
print("User: When is the internship deadline?")
print("Bot:", bot.chat("When is the internship deadline?"))
print("\nUser: What tools are we using?")
print("Bot:", bot.chat("What tools are we using?"))
