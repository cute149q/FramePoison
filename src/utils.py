import json
import logging
import os
import random
import sys
from collections import defaultdict
from datetime import timedelta
from typing import Callable, TypeVar

T = TypeVar("T")

import time

import numpy as np
import torch
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer

from .contriever_src.contriever import Contriever

model_code_to_qmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp",
}

model_code_to_cmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp",
}


def contriever_get_emb(model, input):
    return model(**input)


def dpr_get_emb(model, input):
    return model(**input).pooler_output


def ance_get_emb(model, input):
    input.pop("token_type_ids", None)
    return model(input)["sentence_embedding"]


def load_models(model_code):
    assert (
        model_code in model_code_to_qmodel_name and model_code in model_code_to_cmodel_name
    ), f"Model code {model_code} not supported!"
    if "contriever" in model_code:
        model = Contriever.from_pretrained(model_code_to_qmodel_name[model_code])
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = model
        tokenizer = AutoTokenizer.from_pretrained(model_code_to_qmodel_name[model_code])
        get_emb = contriever_get_emb
    elif "ance" in model_code:
        model = SentenceTransformer(model_code_to_qmodel_name[model_code])
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = model
        tokenizer = model.tokenizer
        get_emb = ance_get_emb
    else:
        raise NotImplementedError

    return model, c_model, tokenizer, get_emb


def load_beir_datasets(dataset_name, split):
    assert dataset_name in [
        "nq",
        "msmarco",
        "hotpotqa",
        "trec-covid",
        "nfcorpus",
        "trec-covid-v2",
        "trec-covid-beir",
        "pubmed",
        "bioasq",
    ], f"Dataset {dataset_name} not supported!"
    if dataset_name == "msmarco":
        split = "train"
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_name)
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = os.path.join(out_dir, dataset_name)
    if not os.path.exists(data_path):
        data_path = util.download_and_unzip(url, out_dir)
    print(data_path)

    data = GenericDataLoader(data_path)
    if "-train" in data_path:
        split = "train"
    corpus, queries, qrels = data.load(split=split)

    return corpus, queries, qrels


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def extract_json_from_string(json_string: str) -> str:
    json_string = json_string.strip()
    # remove \n in the content
    # json_string = json_string.replace("\n", "")
    return json_string[json_string.find("{") : json_string.rfind("}") + 1]


def save_results(results, dir, file_name="debug"):
    json_dict = json.dumps(results, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    if not os.path.exists(f"results/query_results/{dir}"):
        os.makedirs(f"results/query_results/{dir}", exist_ok=True)
    with open(os.path.join(f"results/query_results/{dir}", f"{file_name}.json"), "w", encoding="utf-8") as f:
        json.dump(dict_from_str, f)


def load_results(file_name):
    with open(os.path.join("results", file_name)) as file:
        results = json.load(file)
    return results


def save_json(results, file_path="debug.json"):
    json_dict = json.dumps(results, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(dict_from_str, f)


def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results


def setup_seeds(seed):
    # seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clean_str(s):
    try:
        s = str(s)
    except:
        print("Error: the output cannot be converted to a string")
    s = s.strip()
    if len(s) > 1 and s[-1] == ".":
        s = s[:-1]
    return s.lower()


def f1_score(precision, recall):
    """
    Calculate the F1 score given precision and recall arrays.

    Args:
    precision (np.array): A 2D array of precision values.
    recall (np.array): A 2D array of recall values.

    Returns:
    np.array: A 2D array of F1 scores.
    """
    f1_scores = np.divide(2 * precision * recall, precision + recall, where=(precision + recall) != 0)

    return f1_scores


def top_k_scores_ids(qrels: dict, k=10):
    """
    Get the top k scores and their corresponding ids from qrels.

    Args:
    k (int): The number of top scores to retrieve.
    qrels (dict): A dictionary containing query ids as keys and a list of tuples (doc_id, score) as values.

    Returns:
    dict: A dictionary with query ids as keys and lists of tuples (doc_id, score) as values.
    """
    top_k_results = defaultdict(list)
    for query_id, doc_scores in qrels.items():
        # Sort the document scores in descending order and get the top k
        sorted_scores = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:k]
        top_k_results[query_id] = sorted_scores

    return top_k_results


def retry(
    func: Callable[[], T],
    success: Callable[[T], bool] = bool,
    timeout: timedelta = timedelta(seconds=1),
    step: timedelta = timedelta(seconds=0.1),
    max_attempts: int = 10,
) -> T:
    global fail
    result = func()
    wait_time = timedelta()
    attempts = 0
    while not success(result) and attempts < max_attempts:
        attempts += 1
        if wait_time >= timeout:
            raise TimeoutError()
        time.sleep(step.total_seconds())
        wait_time += step
        result = func()
    if result == False:
        fail += 1
    return result


def is_json_no_keys(response):
    """Check if the JSON response has no keys, fix it and return the corrected JSON
    Handles responses like:
    {
      "How Should I Take Probiotics for Optimal Benefits",
      "What is the Best Way to Consume Probiotics for Health",
      "How Do I Take Probiotics Effectively for Gut Health",
      "What are the Proper Instructions for Taking Probiotics Supplements",
      "How Can I Get the Most Out of Taking Probiotics Daily"
    }
    """
    if not response:
        return None

    try:
        # Extract JSON-like content from the response
        json_str = extract_json_from_string(response)

        # First try to parse as normal JSON (with key-value pairs)
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and len(parsed) > 0:
                return parsed  # Already has proper key-value format
        except json.JSONDecodeError:
            pass

        # If that fails, try to parse as a JSON array/set without keys
        # Remove outer braces to get the content
        cleaned = json_str.strip()
        if cleaned.startswith("{") and cleaned.endswith("}"):
            # This looks like a set format, convert to array format for parsing
            array_format = "[" + cleaned[1:-1] + "]"
            try:
                # Try to parse as JSON array
                items_list = json.loads(array_format)
                if isinstance(items_list, list):
                    # Convert list to dict with query keys
                    result = {}
                    for i, item in enumerate(items_list, 1):
                        if isinstance(item, str) and item.strip():
                            result[f"query{i}"] = item.strip()
                    return result if result else None
            except json.JSONDecodeError:
                pass

        # If JSON parsing fails, try manual parsing
        # This handles cases where quotes might be malformed
        if cleaned.startswith("{") and cleaned.endswith("}"):
            content = cleaned[1:-1]  # Remove outer braces

            # Split by commas, but be careful with commas inside quotes
            items = []
            current_item = ""
            in_quotes = False

            for char in content:
                if char == '"':
                    in_quotes = not in_quotes
                    current_item += char
                elif char == "," and not in_quotes:
                    if current_item.strip():
                        # Clean and add the item
                        clean_item = current_item.strip()
                        # Remove surrounding quotes if present
                        if clean_item.startswith('"') and clean_item.endswith('"'):
                            clean_item = clean_item[1:-1]
                        items.append(clean_item)
                    current_item = ""
                else:
                    current_item += char

            # Don't forget the last item
            if current_item.strip():
                clean_item = current_item.strip()
                if clean_item.startswith('"') and clean_item.endswith('"'):
                    clean_item = clean_item[1:-1]
                items.append(clean_item)

            # Convert to proper JSON format with keys
            result = {}
            for i, item in enumerate(items, 1):
                if item and item.strip():
                    result[f"query{i}"] = item.strip()

            return result if result else None

        return None

    except Exception as e:
        print(f"Error processing response: {e}")
        print(f"Response was: {response[:200]}...")  # Show first 200 chars
        return None


def generate_similar_queries(model, queries, num_queries=10):
    """
    Generate similar queries by randomly selecting a subset of queries.
    """

    prompt = 'Please help me generate similar queries for the following query: [query]. You can try to change the wording, add synonyms, or paraphrase the query. Make sure the meaning of the generated queries is same to the original query and try to make the generated queries as similar with the original query as possible. Please generate [queries_count] similar queries. Return the queries in a json format. Please use the following format: {"query1": generated query 1 , "query2": generated query 2, ...]. Please do not return any other text, just the json format.'
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)
    generated_queries = {}

    def is_valid_json_response(response):
        """Check if the response is a valid JSON that can be parsed"""
        processed = is_json_no_keys(response)
        return processed is not None and len(processed) > 0

    for query_id, query_text in tqdm(queries.items(), desc="Generating similar queries"):
        generated_queries[query_id] = {}
        generation_prompt = prompt.replace("[query]", query_text).replace("[queries_count]", str(num_queries))

        def generate_query_with_retry():
            """Function to be retried"""
            if hasattr(model, "provider") and model.provider == "gpt":
                return model.query(generation_prompt, return_json=True)
            else:
                return model.pipeline_query(generation_prompt)

        try:
            # Use retry to get a valid JSON response
            raw_response = retry(
                func=generate_query_with_retry,
                success=is_valid_json_response,
                timeout=timedelta(seconds=30),
                step=timedelta(seconds=1),
                max_attempts=5,
            )

            # 直接使用 is_json_no_keys 处理后的结果，不要重新解析
            processed_queries = is_json_no_keys(raw_response)

            if processed_queries:
                generated_queries[query_id] = processed_queries
            else:
                print(f"Failed to extract queries from response for: {query_text}")

        except TimeoutError:
            print(f"Timeout: Failed to generate valid JSON for query: {query_text}")
            continue
        except Exception as e:
            print(f"Error generating queries for: {query_text}, Error: {e}")
            continue

    for query_id, queries in generated_queries.items():
        # need to set keys to query_n
        queries_with_keys = {}
        for i, (key, value) in enumerate(queries.items(), start=1):
            queries_with_keys[f"query{i}"] = value
        generated_queries[query_id] = queries_with_keys

    return generated_queries


def extract_keywords_from_query(model, query):
    """
    Extract keywords from a query string.
    This is a simple implementation that splits the query by spaces and removes common stop words.
    """
    prompt = f"Please extract the keywords from the following query: {query}. Return the keywords in a list format. Please use the following format: [keyword1, keyword2, ...]. Please do not return any other text, just the list format."
    generation_prompt = prompt.replace("[query]", query)

    def is_valid_list_response(response):
        """
        Check if the response is a valid list that can be parsed.
        valid responses are like:
        "['keyword1', 'keyword2', ...]"
        """
        try:
            # Clean the response and extract list content
            response = response.strip()

            # Handle different list formats
            if response.startswith("[") and response.endswith("]"):
                # Try to parse as JSON list
                try:
                    parsed_list = json.loads(response)
                    return isinstance(parsed_list, list) and len(parsed_list) > 0
                except json.JSONDecodeError:
                    pass

            # Try manual parsing for malformed JSON
            content = response[1:-1]  # Remove brackets
            if content.strip():
                return True

            return False
        except Exception:
            return False

    def generate_keywords_with_retry():
        """Function to be retried"""
        return model.pipeline_query(generation_prompt)

    try:
        # Use retry to get a valid list response
        raw_response = retry(
            func=generate_keywords_with_retry,
            success=is_valid_list_response,
            timeout=timedelta(seconds=30),
            step=timedelta(seconds=1),
            max_attempts=5,
        )

        # Process the response to extract keywords
        response = raw_response.strip()

        if response.startswith("[") and response.endswith("]"):
            try:
                # Try to parse as JSON list first
                keywords = json.loads(response)
                if isinstance(keywords, list):
                    return [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
            except json.JSONDecodeError:
                # Manual parsing for malformed JSON
                content = response[1:-1]  # Remove brackets
                keywords = []
                current_item = ""
                in_quotes = False

                for char in content:
                    if char in ['"', "'"]:
                        in_quotes = not in_quotes
                    elif char == "," and not in_quotes:
                        if current_item.strip():
                            # Clean the item
                            clean_item = current_item.strip().strip('"').strip("'")
                            if clean_item:
                                keywords.append(clean_item)
                        current_item = ""
                    else:
                        current_item += char

                # Don't forget the last item
                if current_item.strip():
                    clean_item = current_item.strip().strip('"').strip("'")
                    if clean_item:
                        keywords.append(clean_item)

                return keywords

    except TimeoutError:
        print(f"Timeout: Failed to generate valid list for query: {query}")
        return []
