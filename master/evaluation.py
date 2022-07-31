import json
import operator
import pickle
import pandas as pd
import numpy as np
from scipy import optimize
import os 
from datasets import DATASET, RESULTS_ROOT


def combine_rank_scores(coeffs, *rank_scores):
    """Combining the rank score of different algorithms"""

    final_score = []
    for scores in zip(*rank_scores):
        combined_score = coeffs @ np.array(scores)
        final_score.append(combined_score)

    return final_score


def cost(coeffs, src_files, bug_reports, *rank_scores):
    """The cost function to be minimized"""

    final_scores = combine_rank_scores(coeffs, *rank_scores)

    mrr = []
    mean_avgp = []

    for i, report in enumerate(bug_reports.items()):

        # Finding source files from the simis indices
        src_ranks, _ = zip(
            *sorted(zip(src_files.keys(), final_scores[i]), key=operator.itemgetter(1), reverse=True))

        # Getting reported fixed files
        fixed_files = report[1].fixed_files

        # Getting the ranks of reported fixed files

        relevant_ranks = []
        for fixed in fixed_files:
            if fixed in src_ranks:
                relevant_ranks.append(src_ranks.index(fixed)+1)
        relevant_ranks = sorted(relevant_ranks)

        # If required fixed files are not in the codebase anymore
        if not relevant_ranks:
            mrr.append(0)
            mean_avgp.append(0)
            continue

        # MRR
        min_rank = relevant_ranks[0]
        mrr.append(1 / min_rank)

        # MAP
        for j, rank in enumerate(relevant_ranks):
            t = len(relevant_ranks[:j + 1]) / rank
            mean_avgp.append(np.mean(t))

    return -1 * (np.mean(mrr) + np.mean(mean_avgp))


def estiamte_params(src_files, bug_reports, *rank_scores):
    """Estimating linear combination parameters"""

    res = optimize.differential_evolution(
        cost, bounds=[(0, 1)] * len(rank_scores),
        args=(src_files, bug_reports, *rank_scores),
        strategy='randtobest1exp', polish=True, seed=458711526
    )

    return res.x.tolist()


path = os.getcwd()
path += "/table.csv"
df = pd.read_csv(path)


def evaluate(src_files, bug_reports, coeffs, *rank_scores):

    final_scores = combine_rank_scores(coeffs, *rank_scores)

    # Writer for the output file
    result_file = open(RESULTS_ROOT / f'{DATASET.name}_output.jsonl', 'w')

    top_n = (1, 5, 10)
    top_n_rank = [0] * len(top_n)
    mrr = []
    mean_avgp = []

    precision_at_n = [[] for _ in top_n]
    recall_at_n = [[] for _ in top_n]
    f_measure_at_n = [[] for _ in top_n]

    for i, (bug_id, report) in enumerate(bug_reports.items()):

        # Finding source codes from the simis indices
        src_ranks, _ = zip(*sorted(zip(src_files.keys(), final_scores[i]),
                                   key=operator.itemgetter(1), reverse=True))

        # Getting reported fixed files
        fixed_files = report.fixed_files

        # Iterating over top n
        for k, rank in enumerate(top_n):
            hit = set(src_ranks[:rank]) & set(fixed_files)

            # Computing top n rank
            if hit:
                top_n_rank[k] += 1

            # Computing precision and recall at n
            if not hit:
                precision_at_n[k].append(0)
            else:
                precision_at_n[k].append(len(hit) / len(src_ranks[:rank]))
            recall_at_n[k].append(len(hit) / len(fixed_files))
            if not (precision_at_n[k][i] + recall_at_n[k][i]):
                f_measure_at_n[k].append(0)
            else:
                f_measure_at_n[k].append(2 * (precision_at_n[k][i] * recall_at_n[k][i])
                                         / (precision_at_n[k][i] + recall_at_n[k][i]))

        # Getting the ranks of reported fixed files
        relevant_ranks = sorted(src_ranks.index(fixed) + 1
                                for fixed in fixed_files if fixed in src_ranks)

        # If required fixed files are not in the codebase anymore
        if not relevant_ranks:
            mrr.append(0)
            mean_avgp.append(0)
            continue

        # MRR
        min_rank = relevant_ranks[0]
        mrr.append(1 / min_rank)

        # MAP

        mean_avgp = []
        for j, rank in enumerate(relevant_ranks):
            t = len(relevant_ranks[:j + 1]) / rank
            mean_avgp.append(np.mean(t))

        result_file.write(json.dumps(
            {'bug_id': bug_id, 'src_ranks': src_ranks}) + '\n'
        )

    result_file.close()
    first = top_n_rank
    second = []
    for x in top_n_rank:
        second.append(x/len(bug_reports))
    third = np.mean(mrr)
    fourth = np.mean(mean_avgp)
    fifth = np.mean(precision_at_n, axis=1).tolist()
    sixth = np.mean(recall_at_n, axis=1).tolist()
    seventh = np.mean(f_measure_at_n, axis=1).tolist()
    return (first, second, third, fourth, fifth, sixth, seventh)


def main():
    with open(DATASET.root / 'preprocessed_src.pickle', 'rb') as file:
        src_files = pickle.load(file)

    with open(DATASET.root / 'preprocessed_reports.pickle', 'rb') as file:
        bug_reports = pickle.load(file)

    with open(DATASET.root / 'token_matching.json', 'r') as file:
        token_matching_score = json.load(file)

    with open(DATASET.root / 'vsm_similarity.json', 'r') as file:
        vsm_similarity_score = json.load(file)

    with open(DATASET.root / 'stack_trace.json', 'r') as file:
        stack_trace_score = json.load(file)

    with open(DATASET.root / 'semantic_similarity.json', 'r') as file:
        semantic_similarity_score = json.load(file)

    with open(DATASET.root / 'fixed_bug_reports.json', 'r') as file:
        fixed_bug_reports_score = json.load(file)

    params = estiamte_params(
        src_files,
        bug_reports,
        vsm_similarity_score,
        token_matching_score,
        fixed_bug_reports_score,
        semantic_similarity_score,
        stack_trace_score
    )

    results = evaluate(
        src_files,
        bug_reports,
        params,
        vsm_similarity_score,
        token_matching_score,
        fixed_bug_reports_score,
        semantic_similarity_score,
        stack_trace_score
    )

    # print(f'{params = }')
    # print('Top N Rank:', results[0])
    # print('Top N Rank %:', results[1])
    print('MRR:', results[2])
    print('MAP:', results[3])
    some = ['Semantic_similarity_score', DATASET.name,
            np.mean(results[2]), np.mean(results[3])]
    df.loc[len(df)] = some

    # print('Precision@N:', results[4])
    # print('Recall@N:', results[5])
    # print('F-measure@N:', results[6])


if __name__ == '__main__':
    main()
