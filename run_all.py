"""
run_all.py
----------
Master entry point for the SupeRR Selector Season 2 Clutch Analytics Pipeline.
Usage:
    python run_all.py --data-dir ./data
"""

import argparse
import importlib
import os

DEFAULT_DATA_DIR = (
    "./data"
    if os.path.isdir("./data")
    else r"C:\Users\Asus\Downloads\6af458274c1e8_data_pack\Data Pack"
)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

step2 = importlib.import_module("02_pressure_and_ratings")
step3 = importlib.import_module("03_sensitivity_and_charts")


def main():
    parser = argparse.ArgumentParser(description="SupeRR Selector Season 2 Analytics Pipeline")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Path to folder containing the 6 raw CSVs")
    args = parser.parse_args()

    print("[1/3] Auditing data, reconstructing match states, fitting Win Prob Leverage, and computing Ratings...")
    df, audit_stats, val_table, qual_bat, qual_bowl = step2.run_ratings_pipeline(args.data_dir, OUT_DIR)
    print("Audit & WP Stats:", audit_stats)

    print("[2/3] Running sensitivity & temporal reliability stress tests...")
    sens_df = step3.run_sensitivity_tests(df, qual_bat, qual_bowl, OUT_DIR)

    print("[3/3] Generating publication-ready charts and styled Excel workbook...")
    step3.generate_charts(df, val_table, qual_bat, qual_bowl, OUT_DIR)
    step3.export_styled_excel(audit_stats, val_table, qual_bat, qual_bowl, sens_df, OUT_DIR)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
