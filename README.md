# Sherlock Holmes Next-Word Predictor

This project is a next-word prediction system built from scratch using a statistical n-gram language model. By analyzing the prose of four classic Sherlock Holmes novels, the model learns linguistic patterns and word associations to suggest the most probable following words for any given input string. It features a simplified "Stupid Backoff" implementation to handle unseen contexts and Out-of-Vocabulary (OOV) tokens.

## Requirements

- **Python:** 3.10+
- **Environment Manager:** Anaconda / Miniconda (recommended)
- **Dependencies:** All third-party libraries are listed in `requirements.txt`.

## Setup

Follow these steps to set up the development environment:

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd ngram-predictor
