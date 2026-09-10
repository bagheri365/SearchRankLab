"""Download and validate the FiQA BEIR dataset."""

from searchranklab.datasets import summarize_dataset, validate_dataset
from searchranklab.datasets.fiqa import download_fiqa, load_fiqa


def main() -> None:
    path = download_fiqa()
    dataset = load_fiqa(path)
    validate_dataset(dataset)
    summary = summarize_dataset(dataset)

    print("FiQA dataset")
    print(f"path: {path}")
    print(f"documents: {summary.documents:,}")
    print(f"queries: {summary.queries:,}")
    print(f"qrels: {summary.qrels:,}")
    print(
        f"unique relevant documents: "
        f"{summary.relevant_documents:,}"
    )


if __name__ == "__main__":
    main()
