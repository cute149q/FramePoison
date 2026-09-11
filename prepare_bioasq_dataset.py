import json
import os

import numpy as np
from tqdm import tqdm


def save_to_beir_format_corpus(corpus, output_file):
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    # Json Line format and BEIR _id title text
    with open(output_file, "w") as f:
        for item in tqdm(corpus, desc="Processing items"):
            # Create a dictionary for each item
            item = {
                "_id": item["pmid"],
                "title": item["title"],
                "text": item["abstractText"],
                "metadata": {
                    "meshMajor": item.get("meshMajor", []),
                },
            }
            f.write(json.dumps(item) + "\n")


def save_to_beir_format_queries(queries, output_file):
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    # Json Line format and BEIR _id title text
    with open(output_file, "w") as f:
        for item in tqdm(queries, desc="Processing items"):
            # Create a dictionary for each item
            item = {
                "_id": item["id"],
                "text": item["body"],
                "metadata": {
                    "ideal_answer": item["ideal_answer"],
                    "exact_answer": item.get("exact_answer", []),
                    "type": item["type"],
                    "documents": item["documents"],
                },
            }
            f.write(json.dumps(item) + "\n")


def create_qrels(corpus, queries, output_file):
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    with open(output_file, "w") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for item in tqdm(queries, desc="Processing items"):
            # Create a dictionary for each item
            qid = item["id"]
            # for docs in corpus:
            # assume all documents are relevant
            f.write(f"{qid}\t{corpus[0]['pmid']}\t1\n")


with open("datasets/bioasq/allMeSH_2020.json", "r", encoding="latin-1") as f:
    all_mesh = json.load(f)

print(f"Loaded MeSH terms from allMeSH_2020.json with {len(all_mesh['articles'])} entries.")

# read csv file
manual_add = []
manual_fixes_path = "datasets/bioasq/Manual-fixes-BioASQ-Task8b.csv"
if os.path.exists(manual_fixes_path):
    import pandas as pd

    manual_fixes_df = pd.read_csv(manual_fixes_path, header=None, skip_blank_lines=True)
    # iterate through each row and create a dictionary
    for index, row in manual_fixes_df.iterrows():
        manual_add.append({"pmid": row[0], "title": row[1], "abstractText": row[2], "meshMajor": []})

print(f"Loaded {len(manual_add)} manual fixes from {manual_fixes_path}")

all_mesh["articles"].extend(manual_add)

save_to_beir_format_corpus(all_mesh["articles"], "datasets/bioasq/corpus.jsonl")

# print("Loaded MeSH terms from allMeSH_2020.json" f" with {len(all_mesh)} entries.")

# with open("datasets/bioasq/8B1_golden.json", "r", encoding="latin-1") as f:
#     queries = json.load(f)

# save_to_beir_format_queries(queries["questions"], "datasets/bioasq/queries.jsonl")

# create_qrels(all_mesh['articles'], queries["questions"], "datasets/bioasq/qrels.tsv")
