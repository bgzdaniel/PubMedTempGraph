import chromadb
import csv
import ast
from time import time

client = chromadb.PersistentClient(path="data/chroma_store")

try:
    client.delete_collection(name="pubmed_embeddings")
except ValueError:
    pass

collection = client.create_collection(
    name="pubmed_embeddings",
    embedding_function=None,
    metadata={"hnsw:space": "cosine"},
)

duplicate_docs = 0
ids = []
embeddings = []
batch = []
metadatas = []
batch_size = 500
inserted_rows = 0

start_insertion = time()
for year in range(2014, 2025):
    with open(f"data/embeddings/embeddings_{year}.csv", encoding="utf-8") as input_csv:
        reader = csv.DictReader(input_csv)
        for row in reader:
            id = row["doc"]  # set id as the document itself, not important
            year = int(row["year"])

            ids.append(id)
            batch.append(row["doc"])
            embeddings.append(ast.literal_eval(row["embedding"]))
            metadatas.append({"year": year})

            if len(batch) >= batch_size:
                start = time()
                collection.upsert(
                    ids=ids,
                    documents=batch,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
                inserted_rows += len(batch)
                print(f"docs inserted: {inserted_rows}")
                batch = []
                ids = []
                embeddings = []
                metadatas = []

        if batch:
            collection.upsert(
                ids=ids,
                documents=batch,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            inserted_rows += len(batch)
            batch = []
            ids = []
            embeddings = []
            metadatas = []

end_insertion = time()

print("done!")
print(f"inserted in total {inserted_rows} docs")
print(f"insertion duration: {(end_insertion - start_insertion)/60:.2f}min")
