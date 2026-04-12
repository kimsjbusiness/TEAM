from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from scraper import scrape_article


def build_context_csv(
    input_csv: Path,
    output_csv: Path,
    sleep_seconds: float = 0.2,
    save_every: int = 50,
) -> None:
    dataframe = pd.read_csv(input_csv)

    if "url" not in dataframe.columns:
        raise ValueError("Input CSV must contain a 'url' column.")

    rows: list[dict[str, str]] = []

    for idx, url in enumerate(dataframe["url"].dropna(), start=1):
        result = scrape_article(str(url))

        title = result.get("title", "")
        context = result.get("content", "")

        if result.get("error"):
            title = ""
            context = ""

        rows.append(
            {
                "title": title,
                "context": context,
            }
        )

        if idx % save_every == 0:
            pd.DataFrame(rows, columns=["title", "context"]).to_csv(
                output_csv,
                index=False,
                encoding="utf-8-sig",
            )
            print(f"[checkpoint] processed={idx}")

        time.sleep(sleep_seconds)

    pd.DataFrame(rows, columns=["title", "context"]).to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"processed={len(rows)}")
    print(f"saved_to={output_csv.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a title/context CSV from an AI Times article link CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("aitimes_article_links.csv"),
        help="Input CSV file containing article URLs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("aitimes_articles_context.csv"),
        help="Output CSV file for title/context pairs.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Delay between article requests in seconds.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Checkpoint save interval.",
    )
    args = parser.parse_args()

    build_context_csv(
        input_csv=args.input,
        output_csv=args.output,
        sleep_seconds=args.sleep,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
# python crawling/build_aitimes_context_csv.py --input aitimes_article_links.csv --output aitimes_articles_context.csv
