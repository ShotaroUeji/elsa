import torch
import numpy as np

def get_retrieval_ranks(sim_matrix: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    類似度行列からText-to-AudioとAudio-to-Textのランクを計算する。
    Args:
        sim_matrix (torch.Tensor): (テキスト数, 音声数) の類似度行列。
                                   対角成分が正解ペアに対応すると仮定。
    Returns:
        torch.Tensor: Text-to-Audioの各クエリに対する正解のランク。
        torch.Tensor: Audio-to-Textの各クエリに対する正解のランク。
    """
    # Text-to-Audio
    t2a_ranks = []
    for i in range(sim_matrix.shape[0]):
        sims_for_query = sim_matrix[i]
        sorted_indices = sims_for_query.argsort(descending=True)
        true_rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        t2a_ranks.append(true_rank)

    # Audio-to-Text
    a2t_ranks = []
    for i in range(sim_matrix.shape[1]):
        sims_for_query = sim_matrix[:, i]
        sorted_indices = sims_for_query.argsort(descending=True)
        true_rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        a2t_ranks.append(true_rank)
        
    return torch.tensor(t2a_ranks), torch.tensor(a2t_ranks)


def calculate_recall_at_k(ranks: torch.Tensor, k_values: list) -> dict:
    """
    ランクのリストから複数のRecall@Kを計算する。
    """
    metrics = {}
    for k in k_values:
        recall = (ranks <= k).sum().item() / len(ranks) * 100
        metrics[f'r@{k}'] = recall
    return metrics

def calculate_median_rank(ranks: torch.Tensor) -> float:
    """
    ランクのリストから中央値ランク(MedR)を計算する。
    """
    return torch.median(ranks).item()