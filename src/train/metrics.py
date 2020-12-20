def mrr(true, pred):
    """
    Note that only the rank of the first relevant answer is considered,
    possible further relevant answers are ignored. If users are interested
    also in further relevant items, mean average precision is a potential
    alternative metric.
    Args:
        true ([List[int]]): true value
        pred ([List[int]]): pred

    Returns:
        [float]: MRR@10
    """
    res = 0.0
    for rank, item in enumerate(pred):
        if item in true:
            res += 1 / (rank + 1)
            break
    return res
