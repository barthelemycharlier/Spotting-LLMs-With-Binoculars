import os
import pandas as pd
from tqdm import tqdm

RAW_ROOT = "data/ghostbuster_raw"
OUT_ROOT = "data/ghostbuster_clean"

DATASETS = {
    "student_essay": "essay",
    "writing_prompts": "wp",
}

os.makedirs(OUT_ROOT, exist_ok=True)


def read_text_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def process_dataset(dataset_name, folder_name):
    rows = []
    dataset_path = os.path.join(RAW_ROOT, folder_name)

    for generator in os.listdir(dataset_path):
        gen_path = os.path.join(dataset_path, generator)
        if not os.path.isdir(gen_path):
            continue

        if generator.lower() in ["prompts"]:
            continue

        label = 1 if generator.lower() == "human" else 0

        for fname in tqdm(os.listdir(gen_path), desc=f"{dataset_name}/{generator}"):
            if not fname.endswith(".txt"):
                continue

            text_path = os.path.join(gen_path, fname)
            text = read_text_file(text_path)

            if len(text) == 0:
                continue

            rows.append({
                "text": text,
                "label": label,
                "source": dataset_name,
                "generator": generator,
            })

    return pd.DataFrame(rows)


def main():
    all_dfs = []

    for dataset_name, folder_name in DATASETS.items():
        print(f"\nProcessing {dataset_name}")
        df = process_dataset(dataset_name, folder_name)

        out_csv = os.path.join(OUT_ROOT, f"{dataset_name}.csv")
        out_json = os.path.join(OUT_ROOT, f"{dataset_name}.json")

        df.to_csv(out_csv, index=False)
        df.to_json(out_json, orient="records", lines=False)

        all_dfs.append(df)

    # Combined dataset
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(os.path.join(OUT_ROOT, "ghostbuster_all.csv"), index=False)

    print("\nGhostbuster datasets prepared successfully.")


if __name__ == "__main__":
    main()
