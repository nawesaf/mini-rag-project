import logging
import uuid
from pathlib import Path

import aiofiles
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from multi_modal_rag import AnswerGenerationError, DocumentNotFoundError, get_answer, run_complete_ingestion_pipeline

logger = logging.getLogger(__name__)

app = FastAPI()



class Question(BaseModel):
    question: str
    
class Response(BaseModel):
    answer: str


@app.post("/document")
async def upload_file(file: UploadFile):
    if (file is None or file.content_type!='application/pdf'):
        raise HTTPException(status_code=415, detail="only pdf files are allowed") 
    
    document_id = str(uuid.uuid4())
    uploads_path = Path('uploads')
    uploads_path.mkdir(parents=True, exist_ok=True)
    document_path = uploads_path / document_id
    document_path.mkdir()
    safe_filename = Path(file.filename or "document.pdf").name
    file_path = document_path / safe_filename
    data = await file.read()
    
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(data)
    
    await run_in_threadpool(run_complete_ingestion_pipeline, str(file_path.resolve()), document_id)
    
    return {"document_id": document_id,
            "status": "ready"}
    
    


@app.get("/document/{document_id}")
def get_status(document_id: str, q: str | None = None):
    return {"document_id": document_id, 
            "filename": q}


@app.post("/document/{document_id}/questions", response_model=Response)
async def ask_question(document_id: str, question: Question):
    if not question.question.strip():
        raise HTTPException(
            status_code=422,
            detail= "The question cannot be empty"
        )
    try:
        answer = await run_in_threadpool(get_answer, question.question, document_id)
        return Response(answer=answer)
    
    except DocumentNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Document not found or not indexed",
        )

    except AnswerGenerationError:
        raise HTTPException(
            status_code=502,
            detail="The language model could not generate an answer",
        )
    except Exception:
        logger.exception(
            "Unexpected error while answering question for %s",
            document_id,
        )
        raise HTTPException(
            status_code=500,
            detail="An unexpected internal error occurred",
        )