# Calculates the shd correct edges - incorrect edges / total
import numpy as np


def structural_hamming_distance(label_edges, predicted_edges):

    total = len(label_edges)
    correct = 0
    wrong = 0
    # Add 1 for correct edges
    for edge in predicted_edges:
        if any(np.all(edge == label) for label in label_edges):
            correct += 1
        # Subtract 1 for wrong edges
        else:
            wrong += 1

    score = correct - wrong
    missing = total - score
    print(f"Correct: {correct} "
          f"Wrong: {wrong} "
          f"Missing: {missing}")
    return 0.5 * (total - score) / total