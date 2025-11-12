# from portfolio_sim.preprocess import preprocess_ohlcv_folder
# if __name__ == "__main__":
#     print(preprocess_ohlcv_folder(freq="1D"))
from portfolio_sim.preprocess import preprocess_pipeline

summary = preprocess_pipeline(
    raw_dir="../raw",
    processed_dir="data/processed",
    freq="1D",
    max_workers=8,
    start_datetime="2015-11-06 10:00:00",   # <-- your requested start
    #end_datetime="2024-11-06 15:45:00",     # <-- your requested end
    min_coverage_frac=0.5,
    drop_tickers_with_low_coverage=True,
    restrict_to_common_intersection=False
)
print(summary)

