def classification_eval(classification_output):
    label, probability = classification_output
    classification_score = 0.0
    label_scores = {
        'AK': 0.5,
        'BCC': 0.4,
        'BKL': 0,
        'DF': 0.2,
        'MEL': 0.5,
        'NV': 0.2,
        'SCC': 0.5,
        'UNK': 0,
        'VASC': 0
    }
    classification_score += label_scores[label]

    if label in ['AK', 'MEL', 'SCC', 'BCC']:
        classification_score += 0.5 * probability
    elif label in ['DF', 'NV']:
        classification_score += 0.5 * probability
    elif label in ['BKL', 'UNK', 'VASC']:
        classification_score += 0.5 * (1.0 - probability)

    return classification_score
