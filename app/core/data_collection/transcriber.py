# app/core/data_collection/transcriber.py
import os
import whisper
from openai import OpenAI

class Transcriber:
    def __init__(self, model_name="base", use_openai=False, api_key=None):
        """Initialize the transcriber with model specifications."""
        self.use_openai = use_openai
        if use_openai:
            self.client = OpenAI(api_key=api_key)
        else:
            self.model = whisper.load_model(model_name)
        
    def transcribe_local(self, audio_path):
        """Transcribe audio using local Whisper model."""
        if not self.use_openai:
            result = self.model.transcribe(audio_path)
            return result["text"]
        else:
            raise ValueError("Method called with OpenAI configuration. Use transcribe_openai instead.")
    
    def transcribe_openai(self, audio_path):
        """Transcribe audio using OpenAI's API."""
        if self.use_openai:
            with open(audio_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            return transcript.text
        else:
            raise ValueError("Method called with local configuration. Use transcribe_local instead.")
    
    def transcribe(self, audio_path):
        """Transcribe audio using the configured method."""
        if self.use_openai:
            return self.transcribe_openai(audio_path)
        else:
            return self.transcribe_local(audio_path)
