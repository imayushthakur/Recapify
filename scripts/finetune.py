# scripts/finetune.py
import os
import argparse
import json
import logging
from dotenv import load_dotenv
import openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def prepare_training_data(input_file, output_file, format="pairs"):
    """
    Prepare training data for fine-tuning.
    
    Args:
        input_file: Path to the input JSON file containing transcript-summary pairs
        output_file: Path to save the formatted training data
        format: Format of the input data ("pairs" or "objects")
    """
    logger.info(f"Preparing training data from {input_file}")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    training_data = []
    
    if format == "pairs":
        for item in data:
            transcript = item.get("transcript", "")
            summary = item.get("summary", "")
            
            if not transcript or not summary:
                continue
                
            training_example = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that creates concise summaries of tutorial videos."},
                    {"role": "user", "content": f"Please summarize this tutorial transcript: {transcript}"},
                    {"role": "assistant", "content": summary}
                ]
            }
            training_data.append(training_example)
    
    with open(output_file, 'w') as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")
    
    logger.info(f"Training data saved to {output_file}")

def fine_tune_model(training_file, model_suffix, base_model="gpt-3.5-turbo"):
    """
    Fine-tune a model using OpenAI's API.
    
    Args:
        training_file: Path to the training data file
        model_suffix: Suffix to append to the fine-tuned model name
        base_model: Base model to fine-tune
    """
    # Validate API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
        return
    
    openai.api_key = api_key
    
    # Upload the training file
    logger.info(f"Uploading training file: {training_file}")
    try:
        with open(training_file, 'rb') as f:
            response = openai.File.create(
                file=f,
                purpose="fine-tune"
            )
        file_id = response.id
        logger.info(f"File uploaded with ID: {file_id}")
    except Exception as e:
        logger.error(f"Error uploading training file: {e}")
        return
    
    # Start fine-tuning
    logger.info(f"Starting fine-tuning job with base model: {base_model}")
    try:
        response = openai.FineTuningJob.create(
            training_file=file_id,
            model=base_model,
            suffix=model_suffix
        )
        job_id = response.id
        logger.info(f"Fine-tuning job created with ID: {job_id}")
    except Exception as e:
        logger.error(f"Error creating fine-tuning job: {e}")
        return

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model for video recap generation")
    parser.add_argument("--input", required=True, help="Path to input data file")
    parser.add_argument("--output", default="training_data.jsonl", help="Path to save prepared training data")
    parser.add_argument("--format", choices=["pairs", "objects"], default="pairs", help="Format of input data")
    parser.add_argument("--base_model", default="gpt-3.5-turbo", help="Base model to fine-tune")
    parser.add_argument("--model_suffix", default="video-recap", help="Suffix for the fine-tuned model")
    parser.add_argument("--prepare_only", action="store_true", help="Only prepare training data, don't start fine-tuning")
    
    args = parser.parse_args()
    
    # Prepare training data
    prepare_training_data(args.input, args.output, args.format)
    
    # Start fine-tuning if not in prepare-only mode
    if not args.prepare_only:
        fine_tune_model(args.output, args.model_suffix, args.base_model)

if __name__ == "__main__":
    main()
