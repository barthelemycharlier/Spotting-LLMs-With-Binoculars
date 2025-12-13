import os
from tqdm import tqdm
import torch

from src.datasets import load_dataset
from src.binocular_scorer import PerplexityCalculator
from src.evaluation import compute_auc, compute_f1, compute_tpr_at_fpr
from src.utils import set_seed

# -----------------------
# CONFIGURATION
# -----------------------
set_seed(42)

datasets = ["writing_prompts", "student_essay"]
detectors = ["binoculars"]

performer_model = "tiiuae/falcon-7b"
observer_model = "tiiuae/falcon-7b-instruct"

# Choose GPU manually to avoid auto-distribution
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
batch_size = 16        
max_length = 512        # truncate long texts

results_dir = "results/logs"
os.makedirs(results_dir, exist_ok=True)

# -----------------------
# MAIN EXPERIMENT LOOP
# -----------------------
def run_experiments():
    detector_objs = {}
    if "binoculars" in detectors:
        detector_objs["binoculars"] = PerplexityCalculator(
            performer_model, observer_model, device=device, max_length=max_length
        )

    for dataset_name in datasets:
        print(f"\n=== Dataset: {dataset_name} ===")
        df = load_dataset(dataset_name)

        for det_name, det_obj in detector_objs.items():
            print(f"Running detector: {det_name}")
            scores = []

            # batch loop
            texts = df["text"].tolist()
            for i in tqdm(range(0, len(texts), batch_size), desc=f"{dataset_name}-{det_name}"):
                batch_texts = texts[i:i+batch_size]

                # safe forward: split into sub-batches if necessary
                try:
                    batch_scores = det_obj.binoculars_score(batch_texts)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        batch_scores = []
                        for t in batch_texts:
                            score = det_obj.binoculars_score([t])
                            batch_scores.append(score)
                    else:
                        raise e

                # flatten tensors to float
                if isinstance(batch_scores, torch.Tensor) and batch_scores.dim() == 0:
                    batch_scores = [batch_scores.item()]
                elif isinstance(batch_scores, torch.Tensor):
                    batch_scores = [s.item() for s in batch_scores]

                scores.extend(batch_scores)

            df_scores = df.copy()
            df_scores["score"] = scores

            # Compute evaluation metrics
            auc = compute_auc(df_scores["label"], df_scores["score"])
            f1 = compute_f1(df_scores["label"], df_scores["score"], threshold=0.5)
            tpr_at_fpr = compute_tpr_at_fpr(df_scores["label"], df_scores["score"], fpr_threshold=0.0001)

            print(f"{det_name} | AUC: {auc:.4f} | F1@0.5: {f1:.4f}, TPR@FPR0.01%: {tpr_at_fpr:.4f}")

            # Save results
            out_path = os.path.join(results_dir, f"{dataset_name}_{det_name}.csv")
            with open(out_path, "w") as f:
                f.write(f"# AUC: {auc:.4f}\n")
                f.write(f"# F1@0.5: {f1:.4f}\n")
                f.write(f"# TPR@FPR0.01%: {tpr_at_fpr:.4f}\n")
            df_scores.to_csv(out_path, mode='a', index=False)


if __name__ == "__main__":
    run_experiments()
