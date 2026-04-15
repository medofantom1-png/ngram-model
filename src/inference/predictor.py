import os

class Predictor:
    """Responsibility: Orchestrating normalization and model lookup."""

    def __init__(self, model, normalizer):
        """
        Initializes the predictor with a trained model and a normalizer.
        :param model: An instance of NGramModel
        :param normalizer: An instance of Normalizer
        """
        self.model = model
        self.normalizer = normalizer

    def predict_next(self, text, k=3):
        """
        Takes raw user input, normalizes it, and returns the top-k 
        most likely next words based on the n-gram model.
        """
        # 1. Clean the user's input text using the same logic used in training
        normalized_text = self.normalizer.normalize(text)
        words = normalized_text.split()
        
        # 2. Extract the context
        # If we have a 4-gram model, we need the last 3 words as context.
        context_size = self.model.ngram_order - 1
        
        if len(words) >= context_size:
            context = words[-context_size:]
        else:
            # If the user typed fewer words than our context window, 
            # we just take everything they typed.
            context = words

        # 3. Perform the lookup in the model (includes backoff logic)
        predictions = self.model.lookup(context)
        
        # 4. Sort and return the top-k results
        # The model returns a dict: { "word": probability }
        sorted_predictions = sorted(
            predictions.items(), 
            key=lambda item: item[1], 
            reverse=True
        )
        
        # Extract just the word string for the top k results
        top_k_words = [word for word, prob in sorted_predictions[:int(k)]]
        
        # If the model finds nothing (e.g., total gibberish), return a helpful message
        if not top_k_words:
            return ["(No prediction found)"]
            
        return top_k_words