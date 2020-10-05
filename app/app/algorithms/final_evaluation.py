def final_evaluation(A_score, B_score, C_score, D_score, classification_score):
    ## TDS = Total Dermoscopy Score
    # A for Asymmetric
    # B for Border irregularity
    # C for Color
    # D for Diameter (size in mm)
    # + our consideration of image classification
    final_score = 0.0
    TDS = A_score * 1.4 + B_score * 0.1 + C_score * 0.6 + D_score * 0.1 \
          + classification_score * 0.3

    if TDS < 4.75:
        final_score += 0.5 * 4.75 * TDS
    elif 4.75 <= TDS < 5.45:
        final_score += (0.3 * (5.45 - 4.75)) ** 2 * TDS
    else:  # TDS >= 5.45
        final_score += (final_score + 0.2 * (8.0 - 5.45)) ** 3

    return final_score
