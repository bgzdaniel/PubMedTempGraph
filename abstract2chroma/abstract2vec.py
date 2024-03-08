import torch
import csv
from time import time
from utils.preprocess_utils import preprocess_row, get_combined_doc, increase_csv_maxsize
from utils.embedding_utils import PubMedBert

increase_csv_maxsize()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

model = PubMedBert(device=device)

input_csv = open("data/studies.csv", encoding="utf-8")
reader = csv.DictReader(input_csv)

output_csv = open("data/embeddings.csv", "w", encoding="utf-8")
writer = csv.DictWriter(output_csv, fieldnames=["year", "doc", "embedding"])
writer.writeheader()

doc_lookup = set()
doc_batch = []
year_batch = []
batch_size = 128 + 64 + 32 # 10GB CUDA MEM is max, use all 10GB for optimal utilization
total_docs_processed = 0
duplicates = 0
start = time()
for row in reader:
    row = preprocess_row(row)
    doc = get_combined_doc(row)
    year = row["Year"]

    if doc in doc_lookup:
        duplicates += 1
        continue

    doc_lookup.add(doc)
    doc_batch.append(doc)
    year_batch.append(year)

    if len(doc_batch) >= batch_size:
        embeddings = model.encode(doc_batch)
        rows_to_write = []
        for year, doc, embedding in zip(year_batch, doc_batch, embeddings):
            row = {"year": year, "doc": doc, "embedding": embedding}
            rows_to_write.append(row)
        writer.writerows(rows_to_write)
        total_docs_processed += batch_size
        print(f"Embedded {total_docs_processed} docs")
        year_batch = []
        doc_batch = []

if doc_batch:
    embeddings = model.encode(doc_batch)
    rows_to_write = []
    for year, doc, embedding in zip(year_batch, doc_batch, embeddings):
        row = {"year": year, "doc": doc, "embedding": embedding}
        rows_to_write.append(row)
    writer.writerows(rows_to_write)
    total_docs_processed += len(doc_batch)
    year_batch = []
    doc_batch = []

end = time()
print(f"done!\nembedded in total {total_docs_processed} docs\ntotal time: {(end-start)/60:.2f}min")
print(f"found {duplicates} duplicate docs")