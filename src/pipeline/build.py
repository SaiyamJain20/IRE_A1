import argparse
import os
from src.data.cleaner import build_unified_schema
from src.data.language_processing import process_text_features
from src.data.feature_store import save_feature_store
import polars as pl

def main():
    parser = argparse.ArgumentParser(description="Rebuild entire feature store from raw downloaded files.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory where raw data is stored")
    parser.add_argument("--out_dir", type=str, default="feature_store", help="Directory to save the feature store")
    args = parser.parse_args()

    print("Starting data pipeline...")
    # Step 1: Download raw data
    print("Step 1: Downloading and Extracting Data...")
    from src.data.download import download_and_extract
    download_and_extract(args.data_dir)

    print("Step 1.5: Parsing and Cleaning Data (Unified Schema)...")
    data_dict = build_unified_schema(args.data_dir, ebnerd_split="demo")
    
    # Process each dataset
    for dataset_name, data in data_dict.items():
        if data["articles"] is not None:
            print(f"[{dataset_name}] Loaded {data['articles'].shape[0]} articles.")
            
            # Step 2: Language-Aware Processing
            lang = "en" if dataset_name == "MIND" else "da"
            print(f"[{dataset_name}] Step 2: Processing Language-Specific Text ({lang})...")
            data["articles"] = process_text_features(data["articles"], language=lang)
            
        if data["behaviors"] is not None:
            train_df = data["behaviors"].get("train")
            val_df = data["behaviors"].get("val")
            
            if train_df is not None and val_df is not None:
                # Custom Temporal Split: Combine and re-split manually to satisfy Q1 constraints
                combined_df = pl.concat([train_df, val_df])
                
                # Sort by timestamp
                combined_df = combined_df.sort("timestamp")
                
                # Determine the split point (e.g., last 2 days for validation)
                max_time = combined_df.select(pl.max("timestamp")).item()
                from datetime import timedelta
                split_time = max_time - timedelta(days=2)
                
                new_train_df = combined_df.filter(pl.col("timestamp") < split_time)
                new_val_df = combined_df.filter(pl.col("timestamp") >= split_time)
                
                print(f"[{dataset_name}] ✅ Custom Temporal Split Applied: Val contains last 2 days (>= {split_time}).")
                print(f"[{dataset_name}] Step 3: Train: {new_train_df.shape[0]}, Val: {new_val_df.shape[0]}")
                
                data["behaviors_train"] = new_train_df
                data["behaviors_val"] = new_val_df
            
            del data["behaviors"]

    # Step 4: Feature Store Creation
    print("Step 4: Building Feature Store...")
    save_feature_store(data_dict, args.out_dir)
    print(f"Feature store saved to '{args.out_dir}'.")
    
    print("Data pipeline completed successfully!")

if __name__ == "__main__":
    main()
