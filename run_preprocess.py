"""run_preprocess.py

New preprocessing entrypoint.

Instead of the older minute-CSV pipeline, this reads raw Excel files from:
  - raw_stock_data/India
  - raw_stock_data/US

and converts each .xlsx to a same-stem .parquet next to it, then writes
processed wide price matrices into data/processed/.
"""

from portfolio_sim.preprocess_xlsx import preprocess_xlsx_pipeline


if __name__ == "__main__":
    summary = preprocess_xlsx_pipeline(
        raw_root="../raw_stock_data",
        processed_dir="data/processed",
        dataset="both",  # india | us | both
        freq="1D",
        # Optional date slicing (ISO):
        # start_date="2015-01-01",
        # end_date="2025-12-31",
        min_coverage_frac=0.5,
        drop_tickers_with_low_coverage=True,
        restrict_to_common_intersection=False,
        overwrite_parquet_cache=True,
        max_workers=8,
    )
    print(summary)

