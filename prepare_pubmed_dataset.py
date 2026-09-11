import os

os.environ["HF_HOME"] = "/storage/shanxie/hf_cache/"

import json

from tqdm import tqdm

from datasets import load_dataset


def save_to_beir_format_corpus(corpus, output_file):
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    # Json Line format and BEIR _id title text
    with open(output_file, "w") as f:
        for item in tqdm(corpus, desc="Processing items"):
            # Create a dictionary for each item
            item = {
                "_id": item["id"],
                "title": item["title"],
                "text": item["content"],
                "metadata": {
                    "PMID": item["PMID"],
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
                "_id": item["pubid"],
                "text": item["question"],
                "metadata": {
                    "long_answer": item["long_answer"],
                    "final_decision": item["final_decision"],
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
            qid = item["_id"]
            # for docs in corpus:
            # assume all documents are relevant
            f.write(f"{qid}\t{corpus[0]['_id']}\t1\n")


# # ds = load_dataset("MedRAG/pubmed")
# ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")

# queries = ds['train']

# save_to_beir_format_queries(queries, "datasets/pubmed/queries.jsonl")
# corpus = ds["train"]

# output_file = "datasets/pubmed/corpus.jsonl"

# save_to_beir_format_corpus(corpus, output_file)

# transfer to BEIR format

with open("./datasets/pubmed/queries.jsonl", "r") as json_file:
    queries = list(json_file)
for i in range(len(queries)):
    queries[i] = json.loads(queries[i])

with open("./datasets/pubmed/corpus.jsonl", "r") as json_file:
    corpus = list(json_file)
for i in range(0, 1):
    corpus[i] = json.loads(corpus[i])

create_qrels(corpus, queries, "datasets/pubmed/qrels/test.tsv")
