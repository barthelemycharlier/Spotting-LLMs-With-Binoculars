import os
import pandas as pd
from tqdm import tqdm

from src.datasets import load_dataset
from src.binocular_scorer import PerplexityCalculator
from src.evaluation import compute_auc, compute_f1, compute_tpr_at_fpr
from src.utils import set_seed

# Set random seed for reproducibility
set_seed(42)

# CONFIGURATION
datasets = ["writing_prompts", "news", "student_essay"]
detectors = ["binoculars"] # list of detectors to run

performer_model = "tiiuae/falcon-7b"
observer_model = "tiiuae/falcon-7b-instruct"

# Output folder for results
results_dir = "results/logs"
os.makedirs(results_dir, exist_ok=True)

# MAIN EXPERIMENT LOOP
def run_experiments():
    # Initialize all detectors
    detector_objs = {}
    if "binoculars" in detectors:
        detector_objs["binoculars"] = PerplexityCalculator(performer_model, observer_model)

    for dataset_name in datasets:
        print(f"\n=== Dataset: {dataset_name} ===")
        df = load_dataset(dataset_name, split="test")

        for det_name, det_obj in detector_objs.items():
            print(f"Running detector: {det_name}")
            scores = []

            for text in tqdm(df["text"]):
                score = det_obj.binoculars_score(text) if det_name == "binoculars" else det_obj.score(text)
                scores.append(score.item() if hasattr(score, "item") else score)

            df_scores = df.copy()
            df_scores["score"] = scores

            # Compute evaluation metrics
            auc = compute_auc(df_scores["label"], df_scores["score"])
            f1 = compute_f1(df_scores["label"], df_scores["score"], threshold=0.5)
            tpr_at_fpr = compute_tpr_at_fpr(df_scores["label"], df_scores["score"], fpr_threshold=0.0001)

            print(f"{det_name} | AUC: {auc:.4f} | F1@0.5: {f1:.4f}, TPR@FPR0.01%: {tpr_at_fpr:.4f}")

            # Save results to CSV
            out_path = os.path.join(results_dir, f"{dataset_name}_{det_name}.csv")
            # save evaluation metrics as header comments
            with open(out_path, "w") as f:
                f.write(f"# AUC: {auc:.4f}\n")
                f.write(f"# F1@0.5: {f1:.4f}\n")
                f.write(f"# TPR@FPR0.01%: {tpr_at_fpr:.4f}\n")
            df_scores.to_csv(out_path, mode='a', index=False)

if __name__ == "__main__":
    run_experiments()
