import os
import polars as pl

def save_feature_store(data_dict, output_dir):
    """
    Saves the cleaned and unified schema data to a structured feature store directory.
    This creates a reusable base for downstream models.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for dataset_name, data in data_dict.items():
        dataset_dir = os.path.join(output_dir, dataset_name)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            
        articles = data.get("articles")
        if articles is not None:
            articles.write_parquet(os.path.join(dataset_dir, "articles.parquet"))
            
        # For behaviors, you might have train/val/test
        for split in ["train", "val", "test"]:
            behaviors = data.get(f"behaviors_{split}")
            if behaviors is not None:
                split_dir = os.path.join(dataset_dir, split)
                if not os.path.exists(split_dir):
                    os.makedirs(split_dir)
                behaviors.write_parquet(os.path.join(split_dir, "behaviors.parquet"))
