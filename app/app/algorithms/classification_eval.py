def classification_eval(classification_output):
    label, probability = classification_output
    classification_score = 0.0
    label_scores = {
        'AK': 0.3,
        'BCC': 0.2,
        'BKL': 0,
        'DF': 0.1,
        'MEL': 0.3,
        'NV': 0.1,
        'SCC': 0.3,
        'UNK': 0,
        'VASC': 0
    }
    classification_score += label_scores[label]

    if label in ['AK', 'MEL', 'SCC', 'BCC']:
        classification_score += 0.7 * probability
    elif label in ['DF', 'NV']:
        classification_score += 0.5 * probability
    elif label in ['BKL', 'UNK', 'VASC']:
        classification_score += 0.3 * (1.0 - probability)

    return classification_score
