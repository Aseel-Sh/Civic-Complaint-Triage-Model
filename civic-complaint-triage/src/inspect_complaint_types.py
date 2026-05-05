from __future__ import annotations

from train_model import load_feature_data


def main() -> None:
    df = load_feature_data()
    if "complaint_type" not in df.columns:
        print("complaint_type column not found in feature data.")
        return

    complaint_types = df["complaint_type"].dropna().astype(str).str.upper()

    print("Top 30 complaint_type values:")
    print(complaint_types.value_counts().head(30).to_string())

    has_illegal_dumping = "ILLEGAL DUMPING" in complaint_types.unique()
    print(f"\nILLEGAL DUMPING exists: {has_illegal_dumping}")

    print("\nComplaint types containing 'DUMP':")
    print(complaint_types[complaint_types.str.contains("DUMP")].value_counts().head(30).to_string())

    print("\nComplaint types containing 'VACANT':")
    print(complaint_types[complaint_types.str.contains("VACANT")].value_counts().head(30).to_string())


if __name__ == "__main__":
    main()
