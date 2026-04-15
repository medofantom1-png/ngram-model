import json
from pathlib import Path
import os
from collections import Counter, defaultdict

class NGramModel:
    """Responsibility: building, storing, and exposing n-gram probability tables."""

    def __init__(self, ngram_order, unk_threshold):
        """
        :param ngram_order: The 'n' in n-gram (e.g., 4 for 4-grams).
        :param unk_threshold: Minimum count for a word to stay in the vocab.
        """
        self.ngram_order = int(ngram_order)
        self.unk_threshold = int(unk_threshold)
        self.vocab = set()
        self.model = {} # Structure: {"n_gram": {"context": {"word": probability}}}

    def build_vocab(self, token_file):
        """Build vocabulary and apply UNK threshold to handle rare words."""
        word_counts = Counter()
        with open(token_file, 'r', encoding='utf-8') as f:
            for line in f:
                word_counts.update(line.split())
        
        # Only keep words that appear often enough; others become <UNK>
        self.vocab = {word for word, count in word_counts.items() if count >= self.unk_threshold}
        self.vocab.add("<UNK>")
        print(f"Vocabulary built. Size: {len(self.vocab)}")

    def _replace_unk(self, tokens):
        """Helper to replace out-of-vocabulary words with <UNK>."""
        return [w if w in self.vocab else "<UNK>" for w in tokens]

    def build_counts_and_probabilities(self, token_file):
        """Count n-grams and compute Maximum Likelihood Estimation (MLE) probabilities."""
        # Initialize nested counters for each order from 1 to n
        counts = {f"{i}gram": defaultdict(Counter) for i in range(1, self.ngram_order + 1)}
        
        with open(token_file, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = self._replace_unk(line.split())
                
                # Iterate through each n-gram order
                for n in range(1, self.ngram_order + 1):
                    for i in range(len(tokens) - n + 1):
                        ngram = tokens[i:i+n]
                        context = " ".join(ngram[:-1]) # Words leading up to the target
                        target = ngram[-1]             # The predicted word
                        counts[f"{n}gram"][context][target] += 1

        # Convert raw counts into probabilities (MLE)
        for order, contexts in counts.items():
            self.model[order] = {}
            for context, targets in contexts.items():
                total = sum(targets.values())
                # Store as {context: {word: 0.25, word2: 0.50...}}
                self.model[order][context] = {w: c / total for w, c in targets.items()}

    def lookup(self, context_words):
        """
        Recursive backoff lookup. 
        If a 4-gram isn't found, it looks for a 3-gram, and so on.
        """
        # Map context to vocab/UNK
        context_words = self._replace_unk(context_words)
        
        for order in range(self.ngram_order, 0, -1):
            needed_context_len = order - 1
            # Take the last 'n-1' words from the user's input
            current_context = " ".join(context_words[-needed_context_len:]) if needed_context_len > 0 else ""
            
            if current_context in self.model[f"{order}gram"]:
                return self.model[f"{order}gram"][current_context]
        
        return {}

    def save_model(self, path):
        """Save the probability dictionary as a JSON file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.model, f)
    
    def save_vocab(self, path):
        """Save the vocabulary list as a JSON file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(list(self.vocab), f)

    def load(self, model_path, vocab_path):
        """Load previously trained model and vocabulary."""
        output_path = Path(model_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, 'r', encoding='utf-8') as f:
            self.model = json.load(f)
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = set(json.load(f))