import pandas as pd
import numpy as np
from pymongo import MongoClient

class DataHandler:
    def __init__(self):
        self.msme_data = None
        self.target_col = None
        self.engineered_features = []

    def load_csv(self, file_path):
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            df.columns = df.columns.str.strip()

            # --- Step 1: Handle missing values & duplicates ---
            df.drop_duplicates(inplace=True)
            df.fillna(0, inplace=True)

            # --- Step 2: Standardize text columns (district/state names) ---
            for col in df.select_dtypes(include='object').columns:
                df[col] = df[col].astype(str).str.strip().str.title()

            # Find all numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if not numeric_cols:
                return False, "Error: Your CSV does not contain any numeric data."

            # Automatically select the LAST numeric column as target
            self.target_col = numeric_cols[-1]
            df[self.target_col] = pd.to_numeric(df[self.target_col], errors='coerce').fillna(0)

            # --- Step 3: Feature Engineering ---
            df = self._engineer_features(df, numeric_cols)

            self.msme_data = df
            return True, f"Loaded! Auto-selected '{self.target_col}' as analysis target. Features engineered successfully."
        except Exception as e:
            return False, f"File read error: {e}"

    def _engineer_features(self, df, numeric_cols):
        """Create derived features for better ML analysis."""
        self.engineered_features = []

        # MSME Density proxy: if any col suggests count/total
        count_cols = [c for c in numeric_cols if any(k in c.lower() for k in ['count', 'total', 'num', 'no', 'registered'])]
        if count_cols:
            df['MSME_Density'] = df[count_cols[0]]
            self.engineered_features.append('MSME_Density')

        # Growth Rate: if two sequential-value cols exist
        if len(numeric_cols) >= 2:
            c1, c2 = numeric_cols[0], numeric_cols[1]
            # Avoid div-by-zero
            df['Growth_Rate'] = df.apply(
                lambda r: ((r[c2] - r[c1]) / r[c1] * 100) if r[c1] != 0 else 0, axis=1
            ).round(2)
            self.engineered_features.append('Growth_Rate')

        # Sector Diversity Index: std deviation of numeric columns (proxy for variety)
        if len(numeric_cols) >= 3:
            df['Sector_Diversity_Index'] = df[numeric_cols[:5]].std(axis=1).round(2)
            self.engineered_features.append('Sector_Diversity_Index')

        # Economic Activity Score: normalized mean of all numeric cols
        df['Economic_Activity_Score'] = (
            (df[numeric_cols] - df[numeric_cols].min()) /
            (df[numeric_cols].max() - df[numeric_cols].min() + 1e-9)
        ).mean(axis=1).round(4)
        self.engineered_features.append('Economic_Activity_Score')

        return df

    def get_processed_data(self):
        return self.msme_data

    def save_to_mongodb(self, analyzed_df):
        try:
            client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
            client.server_info()

            db = client['ArthVigyanDB']
            collection = db['FinalReports']

            records = analyzed_df.to_dict('records')
            collection.delete_many({})
            collection.insert_many(records)

            return True, f"Successfully saved {len(records)} analyzed records to MongoDB!"
        except Exception as e:
            return False, f"MongoDB Error. Is server running? {e}"