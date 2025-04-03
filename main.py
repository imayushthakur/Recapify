# app/main.py
import os
import argparse
import logging
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

def main():
    parser = argparse.ArgumentParser(description="Video Recap AI - Summarize tutorial videos")
    parser.add_argument("--url", help="YouTube URL of the video to summarize")
    parser.add_argument("--file", help="Path to a local video file to summarize")
    parser.add_argument("--transcript", help="Path to a transcript file to summarize")
    parser.add_argument("--output", default="recap.md", help="Output file for the summary")
    parser.add_argument("--format", choices=["markdown", "html", "json"], default="markdown", help="Output format")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="Model to use for summarization")
    parser.add_argument("--api_key", help="OpenAI API key (overrides environment variable)")
    
    args = parser.parse_args()
    
    # Initialize components
    video_collector = VideoCollector()
    transcriber = Transcriber(use_openai=True, api_key=args.api_key)
    preprocessor = TextPreprocessor()
    gpt_interface = GPTInterface(api_key=args.api_key, model=args.model)
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
    
    # Process the input and generate summary
    # Implementation details would continue here...

if __name__ == "__main__":
    main()
