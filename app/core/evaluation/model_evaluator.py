# app/core/evaluation/model_evaluator.py
import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score

class ModelEvaluator:
    def __init__(self, reference_summaries=None):
        """Initialize the model evaluator with reference summaries."""
        self.reference_summaries = reference_summaries or {}
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def add_reference(self, transcript_id, summary):
        """Add a reference summary for a transcript."""
        self.reference_summaries[transcript_id] = summary
        
    def evaluate_rouge(self, transcript_id, generated_summary):
        """Evaluate a generated summary using ROUGE scores."""
        if transcript_id not in self.reference_summaries:
            raise ValueError(f"No reference summary found for transcript ID: {transcript_id}")
            
        reference = self.reference_summaries[transcript_id]
        scores = self.rouge_scorer.score(reference, generated_summary)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
