import os
import shutil
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "snap-stanford/stark"
REPO_TYPE = "dataset"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
HF_CACHE_DIR = DATA_DIR / ".hf_cache"

os.environ["HF_HOME"] = str(HF_CACHE_DIR)


def download_qa_data():
    datasets = ["amazon", "mag", "prime"]
    splits = ["test.index", "test-0.1.index", "train.index", "val.index"]
    qa_csvs = ["stark_qa.csv", "stark_qa_human_generated_eval.csv"]

    for ds in datasets:
        print(f"\n[QA] Downloading QA data for {ds}...")
        qa_dest = DATA_DIR / "qa" / ds
        split_dest = qa_dest / "split"
        stark_qa_dest = qa_dest / "stark_qa"

        split_dest.mkdir(parents=True, exist_ok=True)
        stark_qa_dest.mkdir(parents=True, exist_ok=True)

        for s in splits:
            rel_path = f"qa/{ds}/split/{s}"
            print(f"Downloading {rel_path}...")
            cached_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=rel_path,
                repo_type=REPO_TYPE,
                cache_dir=HF_CACHE_DIR,
            )
            shutil.copy2(cached_path, split_dest / s)

        for q in qa_csvs:
            rel_path = f"qa/{ds}/stark_qa/{q}"
            print(f"Downloading {rel_path}...")
            cached_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=rel_path,
                repo_type=REPO_TYPE,
                cache_dir=HF_CACHE_DIR,
            )
            shutil.copy2(cached_path, stark_qa_dest / q)

    print("\n[QA] All QA datasets downloaded successfully!")


def extract_zip_flat(zip_path: Path, target_dir: Path, strip_prefix: str = "processed/"):
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():
            filename = member.filename
            if filename.startswith(strip_prefix):
                rel_name = filename[len(strip_prefix):]
            else:
                rel_name = filename

            if not rel_name or rel_name.endswith("/"):
                continue

            dest_path = target_dir / rel_name
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as source, open(dest_path, "wb") as target:
                shutil.copyfileobj(source, target)


def download_raw_graph(ds_name: str):
    print(f"\n[Graph] Downloading raw graph for {ds_name}...")
    zip_rel = f"skb/{ds_name}/processed.zip"
    cached_zip = hf_hub_download(
        repo_id=REPO_ID,
        filename=zip_rel,
        repo_type=REPO_TYPE,
        cache_dir=HF_CACHE_DIR,
    )
    raw_graph_dest = DATA_DIR / "raw_graphs" / ds_name
    print(f"[Graph] Extracting {ds_name} processed.zip into {raw_graph_dest}...")
    extract_zip_flat(Path(cached_zip), raw_graph_dest)
    print(f"[Graph] {ds_name} raw graph extracted successfully!")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    download_qa_data()

    for ds in ["prime", "mag", "amazon"]:
        download_raw_graph(ds)

    print("\n=== STaRK Dataset Download Complete ===")


if __name__ == "__main__":
    main()
