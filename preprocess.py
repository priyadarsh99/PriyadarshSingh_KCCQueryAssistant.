import pandas as pd
import os
import json

def preprocess_kcc(states=['MH','MP','TN','UP']):
    os.makedirs("data/processed", exist_ok=True)

    all_docs = []
    doc_id = 0

    for state_name in states:
        filepath = f"data/raw_dataset/KCC_{state_name}.csv"
        df = pd.read_csv(filepath)
        df_sample = df.sample(n=10000, random_state=42)

        for i, row in df_sample.iterrows():
            ques_type = str(row.get("QueryType", "")).strip()
            question = str(row.get("QueryText", "")).strip()
            answer = str(row.get("KccAns", "")).strip()
            crop = str(row.get("Crop", "Unknown")).strip()
            state = str(row.get("StateName", state_name.title())).strip()
            district = str(row.get("DistrictName", "Unknown")).title().strip()
            season = str(row.get("Season", "Unknown")).title().strip()
            sector = str(row.get("Sector", "Unknown")).title().strip()
            category = str(row.get("Category", "Unknown")).title().strip()

            if question and answer:
                all_docs.append({
                    "id": f"{state_name}_{i}",
                    "text": f"Q: {question}\nA: {answer}\nQT: {ques_type}",
                    "metadata": {
                        "state": state,
                        "district": district,
                        "season": season,
                        "sector": sector,
                        "category": category,
                        "crop": crop
                    }
                })
            doc_id += 1

    # Save all to JSONL
    with open("data/processed/kcc_docs.jsonl", "w", encoding="utf-8") as f:
        for doc in all_docs:
            f.write(json.dumps(doc) + "\n")

    print(f"Preprocessed {len(all_docs)} docs")

if __name__ == "__main__":
    preprocess_kcc()
