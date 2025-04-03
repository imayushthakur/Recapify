# app/core/model/openai_interface.py
from openai import OpenAI
import backoff
import time

class GPTInterface:
    def __init__(self, api_key, model="gpt-3.5-turbo"):
        """Initialize the GPT interface with API key and model."""
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    def generate_completion(self, prompt, max_tokens=1000, temperature=0.7):
        """Generate a completion using the GPT model with retry logic."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            time.sleep(2)  # Wait before retry
            raise
