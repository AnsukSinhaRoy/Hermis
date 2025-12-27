import pandas as pd
from pathlib import Path

def main():
    script_dir = Path(__file__).resolve().parent

    parquet_path = script_dir / "prices_1D.parquet"
    csv_path = script_dir / "prices_1D.csv"

    print(f"Reading: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    print(f"Writing: {csv_path}")
    df.to_csv(csv_path, index=False)

    print("✅ Conversion complete")

if __name__ == "__main__":
    main()
