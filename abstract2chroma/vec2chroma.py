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
    for year in years: # avoid duplicates if possible
        collection.delete(
            where={"year": year}
        )

duplicate_docs = 0
ids = []
embeddings = []
documents = []
metadatas = []
batch_size = 5000
inserted_rows = 0

def upsert_with_duplicates(collection, ids, documents, embeddings, metadatas):
    for id, document, embedding, metadata in zip(ids, documents, embeddings, metadatas):
        try:
            collection.upsert(
                ids=[id],
                documents=[document],
                embeddings=[embedding],
                metadatas=[metadata],
            )
        except chromadb.errors.DuplicateIDError:
            duplicate_docs += 1
            continue

def upsert(collection, ids, documents, embeddings, metadatas):
    try:
        collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
    except chromadb.errors.DuplicateIDError:
        upsert_with_duplicates(collection, ids, documents, embeddings, metadatas)

start_insertion = time()
for year in tqdm(years):
    with open(f"data/embeddings/embeddings_{year}.csv", encoding="utf-8") as input_csv:
        reader = csv.DictReader(input_csv)
        for row in reader:
            id = row["doc"]
            doc = row["doc"]
            embedding = ast.literal_eval(row["embedding"])
            year = int(row["year"])

            ids.append(id)
            documents.append(doc)
            embeddings.append(embedding)
            metadatas.append({"year": year})

            if len(documents) >= batch_size:
                start = time()
                upsert(
                    collection=collection,
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
                inserted_rows += len(documents)
                documents = []
                ids = []
                embeddings = []
                metadatas = []

        if documents:
            upsert(
                collection=collection,
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            inserted_rows += len(documents)
            documents = []
            ids = []
            embeddings = []
            metadatas = []

end_insertion = time()

print("done!")
print(f"inserted in total {inserted_rows} docs")
print(f"found {duplicate_docs} duplicates")
print(f"insertion duration: {(end_insertion - start_insertion)/60:.2f}min")
