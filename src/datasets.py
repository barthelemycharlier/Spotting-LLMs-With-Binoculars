import os
import json
import pandas as pd

def load_json_dataset(path, split="test"):
    """
    Load a dataset from JSON.
    Expected: list of dicts with keys 'text', 'label' (0=human, 1=machine).
    """
    file_path = os.path.join(path, f"{split}.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset JSON must contain 'text' and 'label'")
    return df

def load_csv_dataset(path, split="test"):
    """
    Load a dataset from CSV with columns 'text', 'label'.
    """
    file_path = os.path.join(path, f"{split}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset CSV must contain 'text' and 'label'")
    return df

def load_dataset(dataset_name, split="test"):
    """
    Unified loader for supported datasets.
    dataset_name: "ghostbuster", "writing_prompts", "news", "student_essay"
    """
    dataset_paths = {
        "ghostbuster": "data/ghostbuster",
        "writing_prompts": "data/writing_prompts",
        "news": "data/news",
        "student_essay": "data/student_essay"
    }
    
    if dataset_name not in dataset_paths:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    path = dataset_paths[dataset_name]
    
    # Try JSON first, fallback to CSV
    try:
        return load_json_dataset(path, split)
    except FileNotFoundError:
        return load_csv_dataset(path, split)
