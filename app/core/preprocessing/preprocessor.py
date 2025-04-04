import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

class TextPreprocessor:
    def __init__(self):
        """Initialize the text preprocessor."""
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('english'))
        
    def clean_transcript(self, text):
        """Clean the transcript by removing timestamps, filler words, etc."""
        # Remove timestamps (e.g., [00:15:30])
        text = re.sub(r'\[\d{2}:\d{2}:\d{2}\]', '', text)
        # Remove speaker identifications (e.g., "Speaker 1: ")
        text = re.sub(r'Speaker \d+: ', '', text)
        # Remove filler words
        filler_words = ['um', 'uh', 'ah', 'like', 'you know', 'actually', 'basically', 'literally']
        for word in filler_words:
            text = re.sub(r'\b' + word + r'\b', '', text, flags=re.IGNORECASE)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def segment_by_topics(self, text, min_sentences=3):
        """Segment the transcript into logical sections based on topics."""
        sentences = sent_tokenize(text)
        segments = []
        current_segment = []
        
        for sentence in sentences:
            current_segment.append(sentence)
            if len(current_segment) >= min_sentences and self._is_segment_break(sentence):
                segments.append(' '.join(current_segment))
                current_segment = []
        
        # Add any remaining sentences
        if current_segment:
            segments.append(' '.join(current_segment))
            
        return segments
    
    def _is_segment_break(self, sentence):
        """Check if a sentence is likely to be a segment break."""
        segment_indicators = ['next', 'now let\'s', 'moving on', 'let\'s talk about', 'in this section', 'chapter']
        return any(indicator in sentence.lower() for indicator in segment_indicators)
