import pandas as pd
import psycopg2
import ast

# Load embedded CSV
df = pd.read_csv("/Users/adwaitaboralkar/Desktop/Hacklytics26_datasets/embedded_jobs.csv")

# Convert string → list
df["embedding"] = df["embedding"].apply(ast.literal_eval)

# Connect to Actian VectorDB
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    user="vector",
    password="vector",
    database="vectordb"
)
cur = conn.cursor()

# Insert rows
for _, row in df.iterrows():
    cur.execute(
        """
        INSERT INTO jobs (job_title, description, embedding)
        VALUES (%s, %s, %s)
        """,
        (row["job_title"], row["description"], row["embedding"])
    )

conn.commit()
cur.close()
conn.close()

print("✅All embeddings stored in Actian VectorDB")