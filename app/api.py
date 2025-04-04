from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import tempfile
import uuid
from typing import Optional

from app.core.data_collection.transcriber import Transcriber
from app.core.preprocessing.preprocessor import TextPreprocessor
from app.core.model.openai_interface import GPTInterface
from app.core.few_shot.few_shot_learner import FewShotLearner
from app.core.few_shot.prompt_templates import PromptTemplates
from app.core.formatting.output_formatter import OutputFormatter

app = FastAPI(
    title="Video Recap AI",
    description="API for generating concise summaries of tutorial videos",
    version="1.0.0"
)

# Configuration
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize components
transcriber = Transcriber(use_openai=True, api_key=API_KEY)
preprocessor = TextPreprocessor()
gpt_interface = GPTInterface(api_key=API_KEY, model=MODEL_NAME)
few_shot_learner = FewShotLearner()
output_formatter = OutputFormatter()

# Request models
class TranscriptRequest(BaseModel):
    text: str
    format: Optional[str] = "markdown"

class YouTubeRequest(BaseModel):
    url: str
    format: Optional[str] = "markdown"

class TranscriptResponse(BaseModel):
    summary: str

@app.post("/api/summarize/text", response_model=TranscriptResponse)
async def summarize_text(request: TranscriptRequest):
    """Summarize a text transcript."""
    try:
        # Preprocess the transcript
        cleaned_text = preprocessor.clean_transcript(request.text)
        
        # Create the prompt
        template = PromptTemplates.chapter_summary_template()
        prompt = few_shot_learner.create_prompt(cleaned_text, template=template)
        
        # Generate the summary
        summary = gpt_interface.generate_completion(prompt)
        
        # Format the output
        formatted_summary = output_formatter.format_summary(summary, {"source": "Text transcript"})
        
        return {"summary": formatted_summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
