import os
import json
import pandas as pd

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
