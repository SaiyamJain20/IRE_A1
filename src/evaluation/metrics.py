import numpy as np
from sklearn.metrics import roc_auc_score, ndcg_score as sklearn_ndcg

def calculate_recall(retrieved_items, ground_truth_items, k):
    """Calculate recall@K"""
    if not ground_truth_items:
        return 0.0
    
    retrieved_k = set(retrieved_items[:k])
    gt_set = set(str(x) for x in ground_truth_items)
    
    hits = len(retrieved_k.intersection(gt_set))
    return hits / len(gt_set) if len(gt_set) > 0 else 0.0

def mrr_score(y_true, y_score):
    """
    Calculate Mean Reciprocal Rank (MRR)
    y_true: list of ground truth labels (0 and 1)
    y_score: list of predicted scores
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true) if np.sum(y_true) > 0 else 0.0

def ndcg_k(y_true, y_score, k):
    """
    Calculate nDCG@k
    """
    if np.sum(y_true) == 0:
        return 0.0
    return sklearn_ndcg([y_true], [y_score], k=k)

def auc_score(y_true, y_score):
    """
    Calculate AUC
    """
    if len(np.unique(y_true)) == 1:
        return 0.0 # Cannot compute AUC with only one class
    return roc_auc_score(y_true, y_score)

def intra_list_diversity(recommended_lists, item_embeddings_dict):
    """
    Calculate Average Intra-List Diversity (ILD).
    Based on pairwise cosine distance between recommended items.
    """
    ild_scores = []
    for rec_list in recommended_lists:
        embs = [item_embeddings_dict[item] for item in rec_list if item in item_embeddings_dict]
        if len(embs) < 2:
            continue
        embs = np.array(embs)
        # Normalize for cosine similarity
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1e-9
        embs_norm = embs / norms
        
        sim_matrix = np.dot(embs_norm, embs_norm.T)
        # We want diversity, which is 1 - similarity
        # Upper triangle without diagonal
        idx = np.triu_indices(len(embs_norm), k=1)
        pairwise_div = 1.0 - sim_matrix[idx]
        ild_scores.append(np.mean(pairwise_div))
        
    return np.mean(ild_scores) if ild_scores else 0.0

def novelty(recommended_lists, popularity_dict, total_users):
    """
    Calculate average Novelty of recommendations.
    popularity_dict: mapping from item to number of users who interacted with it
    total_users: total number of users
    """
    novelty_scores = []
    for rec_list in recommended_lists:
        list_novelty = []
        for item in rec_list:
            pop = popularity_dict.get(item, 0)
            p_i = (pop + 1) / (total_users + 1) # smooth to avoid log(0)
            list_novelty.append(-np.log2(p_i))
        if list_novelty:
            novelty_scores.append(np.mean(list_novelty))
            
    return np.mean(novelty_scores) if novelty_scores else 0.0

def coverage(recommended_lists, total_items_count):
    """
    Calculate Catalog Coverage.
    """
    unique_recommended = set()
    for rec_list in recommended_lists:
        unique_recommended.update(rec_list)
        
    return len(unique_recommended) / total_items_count if total_items_count > 0 else 0.0

def bootstrap_ci(metric_fn, y_trues, y_scores, n_bootstraps=1000, ci=95):
    """
    Calculate bootstrap confidence intervals for a metric.
    y_trues: list of lists
    y_scores: list of lists
    """
    n = len(y_trues)
    rng = np.random.RandomState(42)
    bootstrapped_scores = []
    
    for _ in range(n_bootstraps):
        indices = rng.randint(0, n, n)
        y_true_sample = [y_trues[i] for i in indices]
        y_score_sample = [y_scores[i] for i in indices]
        
        # Calculate metric over this bootstrap sample
        # Since metrics like AUC are calculated per impression and then averaged,
        # we average the metric over the sample
        scores = []
        for yt, ys in zip(y_true_sample, y_score_sample):
            score = metric_fn(yt, ys)
            scores.append(score)
        bootstrapped_scores.append(np.mean(scores))
        
    sorted_scores = np.sort(bootstrapped_scores)
    lower_bound = np.percentile(sorted_scores, (100 - ci) / 2.0)
    upper_bound = np.percentile(sorted_scores, 100 - (100 - ci) / 2.0)
    return lower_bound, upper_bound
