import os
from huggingface_hub import HfApi
from dotenv import load_dotenv
import polars as pl


def upload_project():
    load_dotenv()
    REPO_ID = "najrum/cf-interactions"
    TOKEN = os.getenv("HF_TOKEN")
    if not TOKEN:
        print("Error: HF_TOKEN not found.")
        return
    api = HfApi(token=TOKEN)

    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)

    print("Uploading interactions folder...")

    df = pl.read_parquet("./data/interactions/fixed.parquet")
    df['has_rating'] = df['rating'].is_not_null()

    users = df.select("user_index").unique()
    train_users = users.sample(fraction=0.9, seed=42)
    test_users = users.filter(~pl.col("user_index").is_in(train_users["user_index"]))
    train = df.filter(pl.col("user_index").is_in(train_users["user_index"]))
    test = df.filter(pl.col("user_index").is_in(test_users["user_index"]))

    train.write_parquet("./data/interactions/train.parquet")
    test.write_parquet("./data/interactions/test.parquet")

    files = {
        "./data/users.csv": "users.csv",
        "./data/tags.json": "tags.json",
        "./data/problems.parquet": "problems.parquet",
        "./data/interactions/train.parquet": "interactions/train.parquet",
        "./data/interactions/test.parquet": "interactions/test.parquet",
        "./data/interactions/fixed.parquet": "interactions/full.parquet",
    }

    for local_path, repo_path in files.items():
        if os.path.exists(local_path):
            print(f"Uploading {repo_path}...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="dataset",
                commit_message=f"Update {repo_path}",
            )
        else:
            print(f"Skipping {repo_path}: File not found.")

    print(f"Upload complete! Check: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    upload_project()
