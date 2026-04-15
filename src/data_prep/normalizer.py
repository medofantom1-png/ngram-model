import re
from pathlib import Path

class Normalizer:
    """
    Responsibility: Standardizing text across the entire application lifecycle.
    
    This module provides a single source of truth for text cleaning. It 
    ensures that training data and user input at inference time are 
    processed identically to ensure model lookup consistency.
    """

    def __init__(self):
        """
        Initializes the Normalizer with a specific set of words to remove.
        To ensure the model can predict names, we only remove 'the' and 
        formal titles, preserving pronouns and verbs like 'is'.
        """
        self.stopwords = {
            'dr', 'mr', 'mrs', 'ms', 'sir', 'madam', 'prof', 'captain', 
            'colonel', 'lady', 'lord', 'the'
        }

    def load(self, folder_path):
        """
        Reads all text files within a directory and concatenates their content.

        :param folder_path: String or Path to the directory containing .txt files.
        :return: A single string containing all text from the files.
        """
        combined_text = ""
        folder = Path(folder_path)
        paths = list(folder.glob("*.txt"))
        
        for path in paths:
            with open(path, 'r', encoding='utf-8') as f:
                combined_text += f.read() + "\n"
        return combined_text

    def strip_gutenberg(self, text):
        """
        Removes the Project Gutenberg metadata headers and footers.

        :param text: The raw string loaded from a Gutenberg .txt file.
        :return: The string containing only the body of the book.
        """
        start_pattern = r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .* \*\*\*"
        end_pattern = r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .* \*\*\*"
        
        start_match = re.search(start_pattern, text)
        end_match = re.search(end_pattern, text)
        
        start_idx = start_match.end() if start_match else 0
        end_idx = end_match.start() if end_match else len(text)
        
        return text[start_idx:end_idx]

    def normalize(self, text):
        """
        Standardizes text by lowercasing, removing punctuation/numbers, 
        and filtering out specific stopwords.

        :param text: A string of raw text.
        :return: A cleaned, space-delimited string of tokens.
        """
        # 1. Lowercase and remove punctuation/numbers
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # 2. Split into individual words
        words = text.split()
        
        # 3. Remove only the words in self.stopwords
        # This keeps 'my', 'name', and 'is' intact for the N-Gram model.
        filtered_words = [w for w in words if w not in self.stopwords]
        
        return " ".join(filtered_words).strip()

    def sentence_tokenize(self, text):
        """
        Splits a large block of text into individual sentences.

        :param text: The cleaned body of a book.
        :return: A list of individual sentence strings.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s for s in sentences if s.strip()]

    def save(self, sentences, filepath):
        """
        Normalizes a list of sentences and saves them to a file.

        :param sentences: List of raw sentence strings.
        :param filepath: Destination path for the tokenized output.
        :return: None
        """
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                norm_sent = self.normalize(sentence)
                if norm_sent:
                    f.write(norm_sent + "\n")