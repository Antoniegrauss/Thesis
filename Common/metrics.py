import numpy as np


def structural_hamming_distance(label_edges, predicted_edges):

    total = len(label_edges)
    correct = 0
    wrong = 0
    wrong_edges = []
    # Add 1 for correct edges
    for edge in predicted_edges:
        if any(np.all(edge == label) for label in label_edges):
            correct += 1
        # Subtract 1 for wrong edges
        else:
            wrong += 1
            wrong_edges.append(edge)

    missing_edges = []
    for edge in label_edges:
        if edge not in predicted_edges:
            missing_edges.append(edge)

    score = correct - wrong
    missing = total - score
    print(f"Correct: {correct} "
          f"Wrong: {wrong} {wrong_edges} "
          f"Missing: {missing} {missing_edges}")
    return (total - score) / total