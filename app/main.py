# app/main.py
import os
import argparse
import logging
import tempfile
from pathlib import Path
from dotenv import load_dotenv

from app.core.data_collection.video_collector import VideoCollector
from app.core.data_collection.transcriber import Transcriber
from app.core.preprocessing.preprocessor import TextPreprocessor
from app.core.model.openai_interface import GPTInterface
from app.core.few_shot.few_shot_learner import FewShotLearner
from app.core.few_shot.prompt_templates import PromptTemplates
from app.core.formatting.output_formatter import OutputFormatter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def validate_api_key(api_key):
    """Validate OpenAI API key format"""
    if not api_key or not api_key.startswith("sk-"):
        logger.error("Invalid OpenAI API key format")
        raise ValueError("A valid OpenAI API key is required")

def process_video_input(args, video_collector, transcriber):
    """Process video input (URL or file) and return transcript"""
    try:
        if args.url:
            logger.info(f"Downloading video from URL: {args.url}")
            video_path = video_collector.download_from_youtube(args.url)
        else:
            video_path = args.file

        logger.info(f"Extracting audio from video: {video_path}")
        audio_path = video_collector.extract_audio(video_path)
        
        if not audio_path or not Path(audio_path).exists():
            raise FileNotFoundError("Audio extraction failed")

        logger.info("Transcribing audio content")
        transcript = transcriber.transcribe(audio_path)
        return transcript

    except Exception as e:
        logger.error(f"Video processing failed: {str(e)}")
        raise

def process_transcript_file(transcript_path):
    """Read transcript from text file"""
    try:
        with open(transcript_path, 'r') as f:
            return f.read()
    except IOError as e:
        logger.error(f"Failed to read transcript file: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Video Recap AI - Summarize tutorial videos")
    parser.add_argument("--url", help="YouTube URL of the video to summarize")
    parser.add_argument("--file", help="Path to a local video file to summarize")
    parser.add_argument("--transcript", help="Path to a transcript file to summarize")
    parser.add_argument("--output", default="recap.md", help="Output file for the summary")
    parser.add_argument("--format", choices=["markdown", "html", "json"], default="markdown", 
                       help="Output format")
    parser.add_argument("--model", default="gpt-3.5-turbo", 
                       help="Model to use for summarization")
    parser.add_argument("--api_key", help="OpenAI API key (overrides environment variable)")
    
    args = parser.parse_args()
    
    try:
        # Validate API configuration
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        validate_api_key(api_key)

        # Initialize components
        video_collector = VideoCollector()
        transcriber = Transcriber(use_openai=True, api_key=api_key)
        preprocessor = TextPreprocessor()
        gpt_interface = GPTInterface(api_key=api_key, model=args.model)
        few_shot_learner = FewShotLearner()
        output_formatter = OutputFormatter(format_type=args.format)

        # Add few-shot examples
        example_pairs = [
            {
                "input": "In this tutorial, we're going to be talking about neural networks...",
                "output": "## Introduction to Neural Networks\n- Neural networks are machine learning models inspired by the human brain\n- They consist of interconnected nodes (neurons) that learn patterns in data"
            },
            {
                "input": "Welcome to this Python tutorial. Today we'll be covering the basics of Python programming...",
                "output": "## Introduction to Python\n- Python is a high-level, interpreted programming language\n- Known for readability and simplicity\n- Great for beginners"
            }
        ]
        few_shot_learner.add_examples(example_pairs)

        # Process input and get transcript
        if args.url or args.file:
            transcript = process_video_input(args, video_collector, transcriber)
        elif args.transcript:
            transcript = process_transcript_file(args.transcript)
        else:
            raise ValueError("No input provided. Use --url, --file, or --transcript")

        # Preprocess transcript
        logger.info("Preprocessing transcript")
        cleaned_text = preprocessor.clean_transcript(transcript)
        segments = preprocessor.segment_by_topics(cleaned_text)

        # Generate summary prompt
        logger.info("Creating few-shot prompt")
        template = PromptTemplates.video_recap_template()
        prompt = few_shot_learner.create_prompt(
            "\n".join(segments), 
            template=template
        )

        # Generate summary
        logger.info(f"Generating summary using {args.model}")
        raw_summary = gpt_interface.generate_completion(prompt, max_tokens=1500)

        # Format output
        logger.info(f"Formatting output as {args.format}")
        formatted_summary = output_formatter.format_summary(
            raw_summary, 
            {"source": args.url or args.file or args.transcript}
        )

        # Save output
        with open(args.output, 'w') as f:
            f.write(formatted_summary)
        
        logger.info(f"Successfully generated summary: {args.output}")

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
