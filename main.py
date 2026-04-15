import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Import custom modules
from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor

def main():
    # 1. Locate and load the .env file from the config directory
    # Using Path ensures the code works on both Windows and Mac/Linux
    base_dir = Path(__file__).resolve().parent
    env_path = base_dir / 'config' / '.env'
    
    if not env_path.exists():
        print(f"ERROR: Configuration file not found at {env_path}")
        print("Please ensure you have created the '.env' file inside the 'config' folder.")
        return

    load_dotenv(dotenv_path=env_path)
    
    # 2. Safety Check: Verify critical variables are actually loaded
    # This prevents the 'NoneType' error by giving you a clear message instead
    ngram_order = os.getenv('NGRAM_ORDER')
    unk_threshold = os.getenv('UNK_THRESHOLD')

    if ngram_order is None or unk_threshold is None:
        print("ERROR: Critical variables missing from config/.env")
        print("Check that NGRAM_ORDER and UNK_THRESHOLD are defined in your .env file.")
        return

    # 3. Setup CLI Arguments
    parser = argparse.ArgumentParser(description="Sherlock Holmes N-Gram Predictor")
    parser.add_argument('--step', choices=['dataprep', 'model', 'inference', 'all'], required=True,
                        help="Which part of the pipeline to run")
    args = parser.parse_args()

    # 4. Instantiate shared objects
    norm = Normalizer()
    model = NGramModel(ngram_order, unk_threshold)

    # --- STEP 1: DATA PREP ---
    if args.step in ['dataprep', 'all']:
        print("\n--- Step 1: Data Preparation ---")
        raw_dir = os.getenv('TRAIN_RAW_DIR')
        output_file = os.getenv('TRAIN_TOKENS')
        
        if not raw_dir:
            print("ERROR: TRAIN_RAW_DIR not defined in .env")
            return

        print(f"Loading files from: {raw_dir}")
        raw_text = norm.load(raw_dir)
        
        if not raw_text.strip():
            print(f"Warning: No text found in {raw_dir}. Ensure your .txt files are in that folder.")
            return

        clean_text = norm.strip_gutenberg(raw_text)
        sentences = norm.sentence_tokenize(clean_text)
        norm.save(sentences, output_file)
        print(f"Success! Processed tokens saved to {output_file}")

    # --- STEP 2: MODEL TRAINING ---
    if args.step in ['model', 'all']:
        print("\n--- Step 2: Model Training ---")
        token_path = os.getenv('TRAIN_TOKENS')
        
        if not token_path or not os.path.exists(token_path):
            print(f"Error: Token file not found. Run --step dataprep first.")
            return

        model.build_vocab(token_path)
        model.build_counts_and_probabilities(token_path)
        model.save_model(os.getenv('MODEL'))
        model.save_vocab(os.getenv('VOCAB'))
        print(f"Success! Model and Vocab saved.")

    # --- STEP 3: INFERENCE ---
    if args.step in ['inference', 'all']:
        print("\n--- Step 3: Interactive Inference ---")
        model_path = os.getenv('MODEL')
        vocab_path = os.getenv('VOCAB')

        if not model_path or not os.path.exists(model_path):
            print(f"Error: Model file not found. Run --step model first.")
            return

        model.load(model_path, vocab_path)
        predictor = Predictor(model, norm)
        
        top_k = int(os.getenv('TOP_K', 3))
        print(f"Ready. Predicting top-{top_k} words. Type 'quit' to exit.")
        
        while True:
            user_input = input("\n> ")
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye.")
                break
            
            if not user_input.strip():
                continue

            predictions = predictor.predict_next(user_input, top_k)
            print(f"Predictions: {predictions}")

if __name__ == "__main__":
    main()