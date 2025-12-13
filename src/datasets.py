import os
import json
import pandas as pd

# -----------------------------
# Dataset Loaders
# -----------------------------
def load_json_dataset(filename):
    """
    Load a dataset from JSON.
    Expected: list of dicts with keys 'text', 'label' (0=human, 1=machine).
    """
    file_path = filename + ".json"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset JSON must contain 'text' and 'label'")
    return df

def load_csv_dataset(filename):
    """
    Load a dataset from CSV with columns 'text', 'label'.
    """
    file_path = filename + ".csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset CSV must contain 'text' and 'label'")
    return df

def load_dataset(dataset_name):
    """
    Unified loader for supported datasets.
    dataset_name: "writing_prompts", "student_essay"
    """
    dataset_paths = {
        "writing_prompts": "data/ghostbuster_clean/writing_prompts",
        "student_essay": "data/ghostbuster_clean/student_essay"
    }
    
    if dataset_name not in dataset_paths:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    path = dataset_paths[dataset_name]
    
    # Try JSON first, fallback to CSV
    try:
        return load_json_dataset(path)
    except FileNotFoundError:
        return load_csv_dataset(path)

# -----------------------------
# Balanced Sampling
# -----------------------------
def sample_dataset(df, total_examples=300, human_label=0):
    """
    Sample a dataset to have:
    - half examples with label 'human_label'
    - the rest equally distributed among other labels
    """
    # Separate by class
    class_dict = {lbl: df[df['label'] == lbl] for lbl in df['label'].unique()}

    # number of human examples
    num_humans = total_examples // 2

    # number per other class
    other_labels = [l for l in class_dict if l != human_label]
    num_per_other = num_humans // len(other_labels) if other_labels else 0

    sampled = []

    # sample human examples
    if human_label in class_dict:
        sampled.append(class_dict[human_label].sample(n=num_humans, random_state=42))

    # sample other classes
    for lbl in other_labels:
        sampled.append(class_dict[lbl].sample(n=num_per_other, random_state=42))

    # combine and shuffle
    sampled_df = pd.concat(sampled, ignore_index=True).sample(frac=1, random_state=42)

    return sampled_df
