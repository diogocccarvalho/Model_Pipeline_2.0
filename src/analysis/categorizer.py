"""
Module: Categorizer (categorizer.py)
Status: MVP
Model: Zero-Shot Classification (MoritzLaurer/mDeBERTa-v3-base-mnli-xnli)
Input: String or List of strings
Output: Labeled Data (Categories + Confidence Scores)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Union, Tuple

# Ensure the script can find project modules
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

try:
    from transformers import pipeline, Pipeline
    import torch
except ImportError:
    print("Dependencies not installed. Please run: pip install transformers torch sentencepiece")
    sys.exit(1)


def load_taxonomy(taxonomy_path: Path) -> List[str]:
    """Loads the official list of categories from the project's taxonomy file."""
    if not taxonomy_path.exists():
        print(f"Warning: Taxonomy file not found at {taxonomy_path}. Using default list.")
        return ["Saúde", "Educação", "Fiscalidade", "Justiça", "Ambiente", "Habitação", "Economia", "Defesa"]
    
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        return json.load(f)


class Categorizer:
    """
    Categorizes proposals using a Zero-Shot Classification model (NLI-based).
    Supports multi-label classification (one proposal can match multiple topics).
    """

    def __init__(self, 
                 model_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", 
                 taxonomy_path: Path = None):
        """
        Args:
            model_name (str): The NLI model. mDeBERTa-v3 is currently SOTA for multilingual zero-shot.
            taxonomy_path (Path): Path to taxonomy.json.
        """
        if taxonomy_path is None:
            taxonomy_path = project_root / "data/taxonomy.json"
            
        print(f"> Loading Zero-Shot model: {model_name}...")
        
        # Determine device (MPS for Mac, CUDA for NVIDIA, CPU otherwise)
        device = -1
        if torch.cuda.is_available():
            device = 0
            print("> Using CUDA GPU.")
        elif torch.backends.mps.is_available():
            device = "mps" # Requires updated transformers lib, usually safer to leave -1 or use specific torch device
            print("> Using MPS (Apple Silicon).")

        self.pipeline: Pipeline = pipeline("zero-shot-classification", model=model_name, device=device)
        self.labels: List[str] = load_taxonomy(taxonomy_path)
        
        # CRITICAL: This template forces the model to think in Portuguese.
        # Without this, it uses English "This text is about...", which confuses it.
        self.template = "Este texto é sobre {}."
        
        print("> Model loaded successfully.")

    def predict(self, text: str, threshold: float = 0.85) -> Dict[str, Union[List[str], Dict[str, float]]]:
        """
        Predicts categories for a single piece of text.
        """
        if not text or not text.strip():
            return {"categories": ["Uncategorized"], "all_scores": {}}

        # multi_label=True allows probabilities to be independent (e.g. Health 90%, Econ 90%)
        result = self.pipeline(
            text, 
            candidate_labels=self.labels, 
            multi_label=True,
            hypothesis_template=self.template
        )

        # Zip labels and scores together (they come sorted by score)
        predictions = list(zip(result['labels'], result['scores']))
        
        # Filter by threshold (The "Strict" Pass)
        valid_categories = [label for label, score in predictions if score >= threshold]
        scores_map = {label: round(score, 3) for label, score in predictions}

        # Fallback Logic: If nothing met the 85% bar, don't just return nothing.
        # If the top match is > 60%, take it. Otherwise, mark as Uncategorized.
        if not valid_categories:
            top_label, top_score = predictions[0]
            if top_score > 0.60:
                valid_categories = [top_label]
            else:
                valid_categories = ["Uncategorized"]

        return {
            "categories": valid_categories,
            "all_scores": scores_map,
            "top_match": predictions[0] 
        }

    def predict_batch(self, texts: List[str], threshold: float = 0.85) -> List[Dict]:
        """
        Processes a batch of texts. Essential for processing full PDFs efficiently.
        """
        print(f"> Processing batch of {len(texts)} items...")
        
        results = self.pipeline(
            texts, 
            candidate_labels=self.labels, 
            multi_label=True,
            hypothesis_template=self.template
        )

        processed_data = []
        for i, result in enumerate(results):
            predictions = list(zip(result['labels'], result['scores']))
            valid_categories = [label for label, score in predictions if score >= threshold]
            
            # Fallback
            if not valid_categories:
                if predictions[0][1] > 0.60:
                    valid_categories = [predictions[0][0]]
                else:
                    valid_categories = ["Uncategorized"]

            processed_data.append({
                "text": texts[i],
                "categories": valid_categories,
                "top_score": round(predictions[0][1], 3)
            })

        print("> Batch processing complete.")
        return processed_data


def main():
    """CLI Entry point for testing."""
    parser = argparse.ArgumentParser(description="Test the Categorizer Module.")
    parser.add_argument('--test', type=str, help='Text to categorize.')
    parser.add_argument('--threshold', type=float, default=0.85, help='Confidence threshold (0.0 - 1.0)')
    args = parser.parse_args()

    if args.test:
        categorizer = Categorizer() 
        print(f'\n> Input: "{args.test}"')
        
        result = categorizer.predict(args.test, threshold=args.threshold)
        
        print(f"> Assigned Categories: {result['categories']}")
        print(f"> Top Confidence: {result['top_match'][1]:.2%}")
        
        # Debug: Show scores for specific categories of interest
        print("\n--- Full Breakdown ---")
        for label, score in result['all_scores'].items():
            if score > 0.1: # Only show relevant ones
                print(f"{label}: {score:.2f}")
            
    else:
        print("Usage: python src/analysis/categorizer.py --test 'Aumentar o salário mínimo e investir no SNS'")

if __name__ == "__main__":
    main()