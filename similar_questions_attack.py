import argparse
import json
import os
import random

os.environ["HF_HOME"] = "/storage/shanxie/hf_cache/"

import numpy as np
import torch
from tqdm import tqdm

from src.attack import Attacker
from src.models import create_model
from src.prompts import wrap_prompt
from src.utils import (
    clean_str,
    extract_keywords_from_query,
    f1_score,
    load_beir_datasets,
    load_json,
    load_models,
    save_results,
    setup_seeds,
)


def generate_similar_target_queries(incorrect_answers, similar_queries, llm):
    """
    Generate M similar target queries from the incorrect answers.
    """
    incorrect_answers_similar = []
    id = 0
    for item in incorrect_answers:
        for key, queries in similar_queries.items():
            if key == item["id"]:
                original_query = item["question"]
                if llm is not None:
                    # Use LLM to generate similar queries
                    keywords = extract_keywords_from_query(llm, original_query)

                for query_id, query in queries.items():
                    incorrect_answer = {
                        "id": str(id),
                        "original_id": item["id"],
                        "original_query": original_query,
                        "similar_query_id": f"{key}_similarity_{int(query_id[-1])-1}",
                        "question": query,
                        "incorrect answer": item["incorrect answer"],
                        "correct answer": item["correct answer"],
                        "adv_texts": item["adv_texts"],
                        "keywords": keywords if llm is not None else [],
                    }

                    incorrect_answers_similar.append(incorrect_answer)
                    id += 1
                break

    return incorrect_answers_similar


def parse_args():
    parser = argparse.ArgumentParser(description="test")

    # Retriever and BEIR datasets
    parser.add_argument("--eval_model_code", type=str, default="contriever")
    parser.add_argument("--eval_dataset", type=str, default="nq", help="BEIR dataset to evaluate")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--orig_beir_results",
        type=str,
        default=None,
        help="Eval results of eval_model on the original beir eval_dataset",
    )
    parser.add_argument("--query_results_dir", type=str, default="main")

    # LLM settings
    parser.add_argument("--model_config_path", default=None, type=str)
    parser.add_argument("--model_name", type=str, default="llama7b")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--use_truth", type=str, default="False")
    parser.add_argument("--gpu_id", type=int, default=0)

    # attack
    parser.add_argument("--attack_method", type=str, default="LM_targeted")
    parser.add_argument("--adv_per_query", type=int, default=5, help="The number of adv texts for each target query.")
    parser.add_argument("--score_function", type=str, default="dot", choices=["dot", "cos_sim"])
    parser.add_argument("--repeat_times", type=int, default=4, help="repeat several times to compute average")
    parser.add_argument("--M", type=int, default=10, help="one of our parameters, the number of target queries")
    parser.add_argument("--seed", type=int, default=12, help="Random seed")
    parser.add_argument("--name", type=str, default="debug", help="Name of log and result.")

    args = parser.parse_args()
    print(args)
    return args


def main():
    args = parse_args()

    args.eval_dataset = "trec-covid"  # "trec-covid"  # For debugging, set to a specific dataset
    # args.eval_dataset = "pubmed"
    args.model_name = "llama38b"
    args.split = "test"  # Set to dev for debugging
    args.top_k = 5
    args.name = f"Similar_queries_llama38b_{args.eval_dataset}_topk_{args.top_k}_{args.split}_adv_gpt4mini_2_similar_questions_obvious"  # Name of the log and result file

    # local mac set device to cpu
    if torch.cuda.is_available():
        # torch.cuda.set_device(args.gpu_id)
        args.gpu_id = "0,1,2,3"  # Set to the GPU ID you want to use
        device = f"cuda:{args.gpu_id}"
    else:
        print("No GPU available, using CPU.")
        device = "cpu"
    # device = 'cuda'
    setup_seeds(args.seed)
    if args.model_config_path == None:
        args.model_config_path = f"model_configs/{args.model_name}_config.json"

    llm = create_model(args.model_config_path)

    # load target queries and answers
    if args.eval_dataset == "msmarco":
        args.split = "train"

    corpus, queries, qrels = load_beir_datasets(args.eval_dataset, args.split)
    incorrect_answers_json = load_json(
        f"results/adv_targeted_results/{args.eval_dataset}_gpt4mini_{args.split}_2_similar_questions_obvious.json"
    )
    incorrect_answers = list(incorrect_answers_json.values())
    similar_queries = load_json(
        f"dataset_similar/{args.eval_dataset}/similar_queries/{args.eval_dataset}-{args.split}-gpt4mini.json"
    )

    incorrect_answers_similar = generate_similar_target_queries(incorrect_answers, similar_queries, llm)

    # load BEIR top_k results

    if args.orig_beir_results is None:
        print(f"Please evaluate on BEIR first -- {args.eval_model_code} on {args.eval_dataset}")
        # Try to get beir eval results from ./beir_results
        print("Now try to get beir eval results from results/beir_results/...")
        if args.split == "test":
            args.orig_beir_results = (
                f"results/beir_results/{args.eval_dataset}-similar-{args.eval_model_code}-test.json"
            )
        elif args.split == "dev":
            args.orig_beir_results = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}-dev.json"
        if args.score_function == "cos_sim":
            args.orig_beir_results = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}-cos.json"
        assert os.path.exists(args.orig_beir_results), f"Failed to get beir_results from {args.orig_beir_results}!"
        print(f"Automatically get beir_resutls from {args.orig_beir_results}.")
    with open(args.orig_beir_results, "r") as f:
        results = json.load(f)
    # assert len(qrels) <= len(results)
    print("Total samples:", len(results))

    if args.use_truth == "True":
        args.attack_method = None

    if args.attack_method not in [None, "None"]:
        # Load retrieval models
        model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)
        model.eval()
        model.to(device)
        c_model.eval()
        c_model.to(device)
        attacker = Attacker(
            args,
            model=model,
            c_model=c_model,
            tokenizer=tokenizer,
            get_emb=get_emb,
            all_adv_texts=incorrect_answers_json,
        )

    all_results = []
    asr_list = []
    ret_list = []

    args.repeat_times = len(incorrect_answers_similar) // args.M

    for iter in range(args.repeat_times):
        print(f"######################## Iter: {iter+1}/{args.repeat_times} #######################")

        target_queries_idx = range(iter * args.M, iter * args.M + args.M)  # 每次选择M个目标查询

        target_queries = [incorrect_answers_similar[idx]["question"] for idx in target_queries_idx]

        if args.attack_method not in [None, "None"]:
            for i in target_queries_idx:
                top1_idx = list(results[incorrect_answers_similar[i]["similar_query_id"]].keys())[0]
                top1_score = results[incorrect_answers_similar[i]["similar_query_id"]][top1_idx]
                target_queries[i - iter * args.M] = {
                    "query": target_queries[i - iter * args.M],
                    "top1_score": top1_score,
                    "id": incorrect_answers_similar[i]["id"],
                    "original_query": incorrect_answers_similar[i]["original_query"],
                    "original_id": incorrect_answers_similar[i]["original_id"],
                    "keywords": incorrect_answers_similar[i]["keywords"],
                }

            # 生成攻击的文本
            adv_text_groups = attacker.get_attack_similar(target_queries)
            adv_text_list = sum(adv_text_groups, [])  # convert 2D array to 1D array
            adv_text_list = list(set(adv_text_list))  # remove duplicates

            adv_input = tokenizer(adv_text_list, padding=True, truncation=True, return_tensors="pt")
            adv_input = {key: value.cuda() for key, value in adv_input.items()}
            with torch.no_grad():
                adv_embs = get_emb(c_model, adv_input)

        asr_cnt = 0
        ret_sublist = []

        iter_results = []
        for i in target_queries_idx:
            iter_idx = i - iter * args.M  # iter index
            print(f"############# Target Question: {iter_idx+1}/{args.M} #############")
            question = incorrect_answers_similar[i]["question"]
            print(f"Question: {question}\n")

            gt_ids = list(qrels[incorrect_answers_similar[i]["original_id"]].keys())
            ground_truth = [corpus[id]["text"] for id in gt_ids]
            incco_ans = incorrect_answers_similar[i]["incorrect answer"]

            if args.use_truth == "True":
                query_prompt = wrap_prompt(question, ground_truth, 4)
                response = llm.pipeline_query(query_prompt)
                print(f"Output: {response}\n\n")
                iter_results.append(
                    {
                        "question": question,
                        "input_prompt": query_prompt,
                        "output": response,
                    }
                )

            else:  # topk
                topk_idx = list(results[incorrect_answers_similar[i]["similar_query_id"]].keys())[: args.top_k]

                # original topk contents 原始top-k检索的结果
                topk_results = [
                    {"score": results[incorrect_answers_similar[i]["similar_query_id"]][idx], "context": corpus[idx]["text"]}
                    for idx in topk_idx
                ]

                if args.attack_method not in [None, "None"]:
                    query_input = tokenizer(question, padding=True, truncation=True, return_tensors="pt")
                    query_input = {key: value.cuda() for key, value in query_input.items()}
                    with torch.no_grad():
                        # 生成query的embedding
                        query_emb = get_emb(model, query_input)
                    for j in range(len(adv_text_list)):
                        adv_emb = adv_embs[j, :].unsqueeze(0)
                        # similarity
                        if args.score_function == "dot":
                            adv_sim = torch.mm(adv_emb, query_emb.T).cpu().item()
                        elif args.score_function == "cos_sim":
                            adv_sim = torch.cosine_similarity(adv_emb, query_emb).cpu().item()

                        topk_results.append({"score": adv_sim, "context": adv_text_list[j]})

                    # 排序，看看有多少攻击的进去了topk
                    topk_results = sorted(topk_results, key=lambda x: float(x["score"]), reverse=True)
                    topk_contents = [topk_results[j]["context"] for j in range(args.top_k)]
                    # tracking the num of adv_text in topk
                    adv_text_set = set(adv_text_groups[iter_idx])

                    cnt_from_adv = sum([i in adv_text_set for i in topk_contents])
                    ret_sublist.append(cnt_from_adv)
                query_prompt = wrap_prompt(question, topk_contents, prompt_id=4)

                response = llm.pipeline_query(query_prompt)

                print(f"Output: {response}\n\n")
                injected_adv = [i for i in topk_contents if i in adv_text_set]
                iter_results.append(
                    {
                        "id": incorrect_answers_similar[i]["id"],
                        "question": question,
                        "original_query": incorrect_answers_similar[i]["original_query"],
                        "original_id": incorrect_answers_similar[i]["original_id"],
                        "similar_query_id": incorrect_answers_similar[i]["similar_query_id"],
                        "injected_adv_num": cnt_from_adv,
                        "injected_adv": injected_adv,
                        "input_prompt": query_prompt,
                        "output_poison": response,
                        "incorrect_answer": incco_ans,
                        "answer": incorrect_answers_similar[i]["correct answer"],
                    }
                )

                if clean_str(incco_ans) in clean_str(response):
                    asr_cnt += 1

        asr_list.append(asr_cnt)
        ret_list.append(ret_sublist)

        all_results.append({f"iter_{iter}": iter_results})
        save_results(all_results, args.query_results_dir, args.name)
        print(f"Saving iter results to results/query_results/{args.query_results_dir}/{args.name}.json")

    asr = np.array(asr_list) / args.M
    asr_mean = round(np.mean(asr), 2)
    ret_precision_array = np.array(ret_list) / args.top_k
    ret_precision_mean = round(np.mean(ret_precision_array), 2)
    ret_recall_array = np.array(ret_list) / args.adv_per_query
    ret_recall_mean = round(np.mean(ret_recall_array), 2)

    ret_f1_array = f1_score(ret_precision_array, ret_recall_array)
    ret_f1_mean = round(np.mean(ret_f1_array), 2)

    print(f"ASR: {asr}")
    print(f"ASR Mean: {asr_mean}\n")

    print(f"Ret: {ret_list}")
    print(f"Precision mean: {ret_precision_mean}")
    print(f"Recall mean: {ret_recall_mean}")
    print(f"F1 mean: {ret_f1_mean}\n")

    print(f"Ending...")


if __name__ == "__main__":
    main()
