from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scraper import scrape_article


def sample_articles(input_csv: Path, sample_size: int, seed: int) -> None:
    dataframe = pd.read_csv(input_csv)

    if "url" not in dataframe.columns:
        raise ValueError("Input CSV must contain a 'url' column.")

    sample_count = min(sample_size, len(dataframe))
    sampled = dataframe.sample(n=sample_count, random_state=seed)

    for idx, row in enumerate(sampled.itertuples(index=False), start=1):
        url = str(row.url)
        result = scrape_article(url)

        print(f"[{idx}/{sample_count}] {url}")

        if result.get("error"):
            print(f"title: ERROR")
            print(f"context: {result['error']}")
            print("-" * 120)
            continue

        title = result.get("title", "")
        context = result.get("content", "")

        print(f"title: {title}")
        print(f"context: {context}")
        print("-" * 120)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomly sample AI Times article URLs and print title/content to the terminal."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("aitimes_article_links.csv"),
        help="Input CSV file containing article URLs.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of random articles to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    args = parser.parse_args()

    sample_articles(
        input_csv=args.input,
        sample_size=args.sample_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
