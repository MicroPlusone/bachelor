def f_beta(y_hat, y_true, beta=0.3, threshold=0.5):

    epsilon = 1e-7
    y_hat = y_hat > threshold
    y_hat = np.int8(y_hat)

    tp = np.sum(y_hat*y_true, axis=0)

    fp = np.sum(y_hat*(1-y_true), axis=0)

    fn = np.sum((1-y_hat)*y_true, axis=0)
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    beta2 = beta ** 2
    
    denom = beta2 * precision + recall
    f_score = (1 + beta2) * precision * recall / denom
    
    return f_score
