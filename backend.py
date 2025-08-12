# backend.py
import os
import pickle
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()

# Load models and KB
Key = os.getenv("api_key")
client = OpenAI(api_key = Key)

with open("classifier.pkl", "rb") as f:
    data = pickle.load(f)

with open("kb.pkl", "rb") as f:
    doc_data = pickle.load(f)

doc_index = doc_data["index"]
doc_chunks = doc_data["documents"]
clf = data["model"]
encoder = data["encoder"]
tag_to_id = data["tag_to_id"]
id_to_tag = data["id_to_tag"]
intents = data["intents"]

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

bot_name = "Mato"
MAX_HISTORY = 6
chat_history = []

base_prompt = """You are the Strategic Business Developer for FastAutomate, creators of the Primius.ai hybrid AI automation platform.
        Your specialization is in identifying and deeply understanding a prospect's or customer's pain points,
        then mapping them to the right solution in the FastAutomate / Primius.ai product suite. 
        You are a trusted advisor who adds measurable value by connecting client challenges to features, workflows, and outcomes that solve them.

        Core Mission
        Diagnose the user's needs through targeted questioning.
        Present accurate, KB-backed solutions from the FastAutomate ecosystem.
        Position solutions in a way that drives adoption, retention, and measurable ROI.
        Maintain strict product boundary rules.
        Product Boundaries
        PrimeLeads → Lead identification, enrichment, scoring, ranking, shortlisting. No outreach.
        PrimeRecruits → Candidate identification, profiling, scoring, shortlisting. No outreach.
        PrimeReachOut → Exclusive owner of outreach, messaging, reply analysis/scoring, and calendar scheduling.
        PrimeVision → Intelligent RPA for document/data workflows.
        PrimeCRM → CRM hygiene, enrichment, and automation.
        Any user request involving sending messages, emailing, contacting, following up, or scheduling must be routed to PrimeReachOut.

        Tone & Style
        Professional, concise, and confident.
        Positive framing, solution-oriented, and helpful.
        Warm and approachable without being casual.
        Avoid jargon unless requested; explain in simple, relevant terms.
        Never speak negatively about competitors.
        Conversational Flow
        Greet & Identify Context: Welcome the user, confirm their role/business context.
        Assess Needs: Ask 2-3 strategic, open-ended questions to clarify their pain points.
        Map to Solutions: Use KB facts to recommend the right product(s) while respecting boundaries.
        Explain Value: Present benefits, relevant features, and potential outcomes.
        Guide Next Steps: Offer actionable paths—demo, workflow run, documentation, PrimeReachOut handoff.
        Escalate if Needed: KB gap → ask clarifying questions → human support if still unclear.
        Proactive Behaviors
        Suggest related KB articles or additional product modules if relevant.
        Anticipate adjacent needs based on the user's industry, role, or workflow.
        Always confirm before triggering PrimeReachOut actions.
        
        Examples
        User: "Find and email 100 high-value leads." Agent: "PrimeLeads can identify and score the high-value leads. Then PrimeReachOut can take over for personalized outreach and scheduling."
        User: "Can you set up interviews with these candidates?" Agent: "PrimeRecruits can shortlist the candidates for you. For contacting and scheduling interviews, we'll pass them to PrimeReachOut."
        Guardrails
        Stay within FastAutomate/Primius.ai domain knowledge.
        Avoid unsupported claims or speculation.
        Never attribute outreach functions to PrimeLeads or PrimeRecruits.
        Always ground answers in KB content before responding.
        """

app = FastAPI()
# Allow your frontend to access the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to ["http://localhost:5500"] for security
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)
class ChatRequest(BaseModel):
    message: str

def format_history(history):
    return "\n".join([f"User: {u}\nBot: {b}" for u, b in history])

@app.post("/chat")
def chat(req: ChatRequest):
    sentence = req.message.lower()
    X = encoder.encode([sentence])
    probs = clf.predict_proba(X)[0]
    pred_id = np.argmax(probs)
    confidence = probs[pred_id]
    tag = id_to_tag[pred_id]

    if confidence > 0.95:
        for intent in intents:
            if intent["tag"] == tag:
                bot_reply = np.random.choice(intent["responses"])
                chat_history.append((sentence, bot_reply))
                if len(chat_history) > MAX_HISTORY:
                    chat_history.pop(0)
                return {"reply": bot_reply}

    # GPT fallback
    query_embedding = embed_model.encode([sentence])
    D, I = doc_index.search(np.array(query_embedding), k=3)
    retrieved_chunks = [doc_chunks[i] for i in I[0]]
    context_text = "\n\n".join(retrieved_chunks)

    history_text = format_history(chat_history)
    prompt = (
        f"{base_prompt}\n{history_text}\n"
        f"Use the following company documents to answer:\n"
        f"{context_text}\nUser: {sentence}\nBot:"
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful company assistant."},
                {"role": "user", "content": prompt}],
        temperature=0.7
    )
    gpt_reply = response.choices[0].message.content.strip()

    followup_prompt = (
        "Based on the conversation so far, suggest one short follow-up question "
        "that will help you better understand the user's needs. "
        "Keep it under 15 words and make sure it is directly relevant to their last message.\n\n"
        f"Conversation so far:\n{format_history(chat_history)}\n"
        f"Last bot reply: {gpt_reply}\n"
        "Follow-up question:"
    )

    follow_up_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful company assistant."},
            {"role": "user", "content": followup_prompt}
        ],
        temperature = 0.7
    )

    follow_up = follow_up_response.choices[0].message.content.strip()
    gpt_reply += f"\n\n{follow_up}"
    print(f"{bot_name}: {gpt_reply}")

    chat_history.append((sentence, gpt_reply))
    if len(chat_history) > MAX_HISTORY:
        chat_history.pop(0)

    return {"reply": gpt_reply}

@app.get("/start_chat")
def start_chat():
    welcome_message = f"{bot_name} is ready to chat! 😊"
    # Optional: reset history when starting a new chat
    chat_history.clear()
    return {"reply": welcome_message}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
