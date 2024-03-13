import chromadb
import csv
import ast
from time import time
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--years", required=True, type=str)
parser.add_argument("--create_new", action="store_true")
args = parser.parse_args()
years = [int(year) for year in args.years.split(",")]

client = chromadb.PersistentClient(path="data/chroma_store")

if args.create_new:
    try:
        client.delete_collection(name="pubmed_embeddings")
    except ValueError:
        pass

    collection = client.create_collection(
        name="pubmed_embeddings",
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )
else:
    collection = client.get_collection(
        name="pubmed_embeddings",
        embedding_function=None,
    )

duplicate_docs = 0
ids = []
embeddings = []
batch = []
metadatas = []
batch_size = 5000
inserted_rows = 0

start_insertion = time()
for year in tqdm(years):
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
                #print(f"docs inserted: {inserted_rows}")
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
