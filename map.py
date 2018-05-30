import numpy as np

def apk(actual, predicted):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual:
            num_hits += 1.0

            score += (num_hits / (i+1.0))
            print(score)

    if not actual:
        return 0.0

    return score / num_hits

def mapk(actual, predicted):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p) for a,p in zip(actual, predicted)])

if __name__ == '__main__':
    a = [1]
    pr = [1,0,1,1,0,0,0,0,0,1,1,0,1]
    len(a)
    ap_value = mapk([a],[pr])
    print(ap_value)
