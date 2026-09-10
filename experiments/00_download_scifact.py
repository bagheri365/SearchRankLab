"""Download SciFact and print a small dataset summary."""

from searchranklab.datasets import download_scifact, load_scifact


def main() -> None:
    path = download_scifact()
    dataset = load_scifact()

    print(f"SciFact path: {path}")
    print(f"documents: {len(dataset.corpus):,}")
    print(f"test queries: {len(dataset.queries):,}")
    print(f"qrels: {sum(len(values) for values in dataset.qrels.values()):,}")


if __name__ == "__main__":
    main()
