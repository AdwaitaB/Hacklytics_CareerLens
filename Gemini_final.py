import pandas as pd
from google import genai

API_KEY = "AIzaSyBmPKgzAXuRhhOwrzNFEqr-dfpVcAtCMqg"
EXCEL_PATH = "/Users/adwaitaboralkar/Desktop/Hacklytics26_datasets/Unique_jobs.xlsx"

# Fix: header=None tells pandas the first row is data, not a header
df = pd.read_excel(EXCEL_PATH, header=None, names=["job_title", "description"])

print("Excel loaded successfully")
print(df.head())

texts = []
for _, row in df.iterrows():
    text = f"Job title: {row['job_title']}. Description: {row['description']}"
    texts.append(text)

print(f"Prepared {len(texts)} job descriptions for embedding")

client = genai.Client(api_key=API_KEY)

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=texts
)

embeddings = [e.values for e in result.embeddings]

print("Embeddings created successfully")
print(f"Embedding dimension: {len(embeddings[0])}")

df["embedding"] = embeddings

OUTPUT_PATH = "/Users/adwaitaboralkar/Desktop/Hacklytics26_datasets/embedded_jobs.csv"
df.to_csv(OUTPUT_PATH, index=False)

print("Saved embedded data to:", OUTPUT_PATH)