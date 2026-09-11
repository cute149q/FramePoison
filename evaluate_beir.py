import argparse
import json
import logging
import os

os.environ["HF_HOME"] = "/storage/shanxie/hf_cache/"


import torch
import transformers
from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.models import DPR
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from src.contriever_src.beir_utils import DenseEncoderModel
from src.contriever_src.contriever import Contriever
from src.models import create_model
from src.utils import generate_similar_queries, load_json

parser = argparse.ArgumentParser(description="test")

parser.add_argument("--model_code", type=str, default="contriever")
parser.add_argument("--score_function", type=str, default="dot", choices=["dot", "cos_sim"])
parser.add_argument("--top_k", type=int, default=100)
parser.add_argument("--dataset", type=str, default="nq", help="BEIR dataset to evaluate")
parser.add_argument("--split", type=str, default="test")

parser.add_argument("--result_output", default="results/beir_results/debug.json", type=str)

parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--per_gpu_batch_size", default=256, type=int, help="Batch size per GPU/CPU for indexing.")
parser.add_argument("--max_length", type=int, default=128)

args = parser.parse_args()

# args.dataset = "trec-covid"
args.dataset = "nfcorpus"
args.split = "train"
args.result_output = f"results/beir_results/{args.dataset}-similar-{args.model_code}-{args.split}.json"

from src.utils import model_code_to_cmodel_name, model_code_to_qmodel_name


def filter_empty_corpus(corpus):
    """
    Filter out empty documents from the corpus.
    """
    empty = 0
    total = 0
    filtered_corpus = {}
    for doc_id, doc in corpus.items():
        total += 1
        if doc and "text" in doc and doc["text"].strip():
            filtered_corpus[doc_id] = doc
        else:
            empty += 1
    print(f"Filtered {empty} empty documents out of {total} total documents.")
    return filtered_corpus


def compress(results):
    for y in results:
        k_old = len(results[y])
        break
    sub_results = {}
    for query_id in results:
        sims = list(results[query_id].items())
        sims.sort(key=lambda x: x[1], reverse=True)
        sub_results[query_id] = {}
        for c_id, s in sims[:2000]:
            sub_results[query_id][c_id] = s
    for y in sub_results:
        k_new = len(sub_results[y])
        break
    logging.info(f"Compressed retrieval results from top-{k_old} to top-{k_new}.")
    return sub_results


#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
#### /print debug information to stdout

logging.info(args)
gen_similar = True
has_similar_queries = True

# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#### Download and load dataset
dataset = args.dataset
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(os.getcwd(), "datasets")
data_path = os.path.join(out_dir, dataset)
if not os.path.exists(data_path):
    data_path = util.download_and_unzip(url, out_dir)
logging.info(data_path)

if args.dataset == "msmarco":
    args.split = "train"
corpus, queries, qrels = GenericDataLoader(data_path).load(split=args.split)

if gen_similar == True:
    model_name = "llama38b"
    model_config_path = f"model_configs/{model_name}_config.json"
    llm = create_model(model_config_path)
    similar_queries = generate_similar_queries(llm, queries, num_queries=5)

    # save similar queries
    path = f"dataset_similar/{args.dataset}/similar_queries/{args.dataset}-{args.split}-{model_name}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(similar_queries, f)

if has_similar_queries:
    # load similar queries
    model_name = "llama38b"
    path = f"dataset_similar/{args.dataset}/similar_queries/{args.dataset}-{args.split}-{model_name}.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Similar queries file not found: {path}")
    similar_queries = load_json(path)
    plain_queries = dict()
    for query_id, sim_queries in similar_queries.items():
        for i, sim_query in enumerate(sim_queries):
            plain_queries[f"{query_id}_similarity_{i}"] = sim_query


corpus = filter_empty_corpus(corpus)

# grp: If you want to use other datasets, you could prepare your dataset as the format of beir, then load it here.

logging.info("Loading model...")
if "contriever" in args.model_code:
    encoder = Contriever.from_pretrained(model_code_to_cmodel_name[args.model_code]).to(device)
    tokenizer = transformers.BertTokenizerFast.from_pretrained(model_code_to_cmodel_name[args.model_code])
    model = DRES(
        DenseEncoderModel(encoder, doc_encoder=encoder, tokenizer=tokenizer), batch_size=args.per_gpu_batch_size
    )
elif "dpr" in args.model_code:
    model = DRES(
        DPR((model_code_to_qmodel_name[args.model_code], model_code_to_cmodel_name[args.model_code])),
        batch_size=args.per_gpu_batch_size,
        corpus_chunk_size=5000,
    )
elif "ance" in args.model_code:
    model = DRES(models.SentenceBERT(model_code_to_cmodel_name[args.model_code]), batch_size=args.per_gpu_batch_size)
else:
    raise NotImplementedError

logging.info(f"model: {model.model}")

retriever = EvaluateRetrieval(
    model, score_function=args.score_function, k_values=[args.top_k]
)  # "cos_sim"  or "dot" for dot-product
# results = retriever.retrieve(corpus, queries)
results = retriever.retrieve(corpus, plain_queries)
logging.info("Printing results to %s" % (args.result_output))
sub_results = compress(results)

with open(args.result_output, "w") as f:
    json.dump(sub_results, f)
