from pathlib import Path
import pandas as pd
import joblib

DATA_PATH = Path("data/creditcard.csv")
MODEL_PATH = Path("models/xgb_model.joblib")
OUT_PATH = Path("models/app_sample.parquet")

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("creditcard.csv not found in data/")

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Class"])
    df["p_fraud"] = model.predict_proba(X)[:, 1]

    fraud = df[df["Class"] == 1]
    nonfraud = df[df["Class"] == 0].sample(n=6000, random_state=42)

    sample = (
        pd.concat([fraud, nonfraud])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(OUT_PATH, index=False)

    print(f"✅ App sample created: {OUT_PATH}")
    print(f"Rows: {len(sample)} | Fraud rows: {sample['Class'].sum()}")

if __name__ == "__main__":
    main()
