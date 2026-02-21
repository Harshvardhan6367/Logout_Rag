from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uuid
import os
from src.config import Config
from src.extractor import PrescriptionExtractor
from src.vector_store import VectorStoreManager
from src.graph import RAGGraph
from src.memory import MemoryManager
from src.utils import setup_logger

logger = setup_logger(__name__)
app = FastAPI(title="Medical Prescription RAG API")

# Initialize components
extractor = PrescriptionExtractor()
vector_store = VectorStoreManager()
rag_graph = RAGGraph().build_graph()
memory = MemoryManager()

class QueryRequest(BaseModel):
    question: str
    session_id: str
    prescription_id: Optional[str] = None
    language: str = "English"

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    decision: Optional[str] = None

@app.post("/upload")
async def upload_prescription(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    upload_path = os.path.join(Config.INPUT_DIR, file.filename)
    
    with open(upload_path, "wb") as f:
        f.write(await file.read())
        
    data = extractor.extract_data(upload_path)
    if not data:
        raise HTTPException(status_code=400, detail="Failed to extract data from prescription")
        
    # Vectorize
    med_details = []
    for med in data.get('medicines', []):
        timing = med.get('timing', {})
        timing_str = f"Morning: {timing.get('morning')}, Afternoon: {timing.get('afternoon')}, Night: {timing.get('night')}, Instruction: {timing.get('instruction')}"
        med_details.append(f"- {med.get('name')} (Qty: {med.get('quantity')}): {timing_str}, Freq: {med.get('frequency')}, Duration: {med.get('duration')}")
    
    context_text = f"Date: {data.get('date')}\n\nMedicines:\n" + "\n".join(med_details)
    vector_store.add_prescription(file_id, [context_text], {"filename": file.name})
    
    return {"file_id": file_id, "filename": file.filename, "extracted_data": data}

@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    try:
        inputs = {
            "question": request.question,
            "prescription_id": request.prescription_id,
            "session_id": request.session_id,
            "language": request.language,
            "context": [],
            "answer": ""
        }
        
        result = rag_graph.invoke(inputs)
        return QueryResponse(
            answer=result["answer"],
            session_id=request.session_id,
            decision=result.get("final_decision")
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
