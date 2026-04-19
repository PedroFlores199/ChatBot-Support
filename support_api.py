from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chatbot_support import SupportChatbot

app = FastAPI(
    title="API de soporte tecnico",
    description="Servicio de clasificacion de incidencias mediante PLN",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot = SupportChatbot()

class MessageInput(BaseModel):
    message: str

@app.get("/health")
def health():
    return {
        "estado": "activo",
        "modelo": "SupportChatbot v1.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/chat")
def chat(data: MessageInput):
    result = chatbot.respond(data.message)
    result["timestamp"] = datetime.now().isoformat()
    return result

@app.get("/history")
def history():
    return {
        "total": len(chatbot.history),
        "entradas": chatbot.history[-10:]
    }

@app.get("/evaluation")
def evaluation():
    return {
        "clases": chatbot.classes,
        "reporte": chatbot.test_report
    }
