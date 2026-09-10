"""Download SciFact, validate it, and print a dataset summary."""

from searchranklab.datasets import (
    download_scifact,
    load_scifact,
    summarize_dataset,
    validate_dataset,
)


def main() -> None:
    path = download_scifact()
    dataset = load_scifact()
    validate_dataset(dataset)
    summary = summarize_dataset(dataset)

    print(f"SciFact path: {path}")
    print(f"documents: {summary.documents:,}")
    print(f"test queries: {summary.queries:,}")
    print(f"qrels: {summary.qrels:,}")
    print(f"unique relevant documents: {summary.relevant_documents:,}")


if __name__ == "__main__":
    main()
