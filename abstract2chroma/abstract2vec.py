import torch
import csv
from time import time
from utils.preprocess_utils import preprocess_row, get_combined_doc, increase_csv_maxsize
from utils.embedding_utils import PubMedBert

increase_csv_maxsize()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PubMedBert(device=device)

abstract_lookup = set()
doc_batch = []
year_batch = []
batch_size = 128 + 64 + 32 # 10GB CUDA MEM is max, use all 10GB for optimal utilization
total_docs_processed = 0
duplicates = 0
abstracts_missing = 0
invalid_years = 0
start = time()

for year in range(2014, 2025):
    with open(f"data/studies/studies_{year}.csv", encoding="utf-8") as input_csv, \
        open(f"data/embeddings_{year}.csv", "w", encoding="utf-8") as output_csv:
        reader = csv.DictReader(input_csv)
        writer = csv.DictWriter(output_csv, fieldnames=["year", "doc", "embedding"])
        writer.writeheader()
        for row in reader:
            row = preprocess_row(row)

            try:
                year = int(row["Year"])
            except ValueError:
                invalid_years += 1
                continue

            if row["Abstract"] == "NA":
                abstracts_missing += 1
                continue

            if row["Abstract"] in abstract_lookup:
                duplicates += 1
                continue

            abstract_lookup.add(row["Abstract"])

            doc = get_combined_doc(row)

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
                print(f"embedded {total_docs_processed} docs")
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
print(f"found {abstracts_missing} missing abstracts")
print(f"found {invalid_years} years not possible to cast into int")