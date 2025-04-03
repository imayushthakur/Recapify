# app/core/few_shot/few_shot_learner.py
class FewShotLearner:
    def __init__(self, examples=None):
        """Initialize the few-shot learner with example pairs."""
        self.examples = examples or []
        
    def add_example(self, input_text, output_text):
        """Add an example pair to the few-shot learner."""
        self.examples.append({"input": input_text, "output": output_text})
        
    def add_examples(self, examples):
        """Add multiple example pairs to the few-shot learner."""
        for example in examples:
            self.add_example(example["input"], example["output"])
    
    def create_prompt(self, input_text, n_shots=3, template=None):
        """Create a few-shot prompt with the given input text."""
        if template is None:
            template = "I want you to summarize video transcripts into concise chapter summaries.\n\nHere are some examples:\n\n{examples}\n\nNow summarize the following transcript:\n{input}"
        
        # Select n_shots random examples if we have more than needed
        import random
        selected_examples = self.examples
        if len(self.examples) > n_shots:
            selected_examples = random.sample(self.examples, n_shots)
            
        examples_text = ""
        for example in selected_examples:
            examples_text += f"Transcript:\n{example['input']}\n\nSummary:\n{example['output']}\n\n---\n\n"
            
        prompt = template.format(examples=examples_text, input=input_text)
        return prompt
