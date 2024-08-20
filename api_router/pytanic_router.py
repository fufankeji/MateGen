from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str

class KbNameRequest(BaseModel):
    knowledge_base_name: str