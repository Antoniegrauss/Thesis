from os import path

import matplotlib.pyplot as plt
import numpy as np

import light_switches, discovery, interventional_discovery, counterfactual_discovery


# Calculates the shd correct edges - incorrect edges / total
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


# Compares the SHD of
# correct (light_switches)
# PC
# ID
# CD
def compare_shd():
    directory = 'Method_Comparison'
    correct_edges = np.load(path.join('Data', 'light_switches_correct_edges.npy'))
    pc_edges = np.load(path.join('Result', 'light_switches_pc_edges.npy'))
    id_edges = np.load(path.join('Result', 'light_switches_id_edges.npy'))
    cf_edges = np.load(path.join('Result', 'light_switches_cf_edges.npy'))
    print("Normalized SHD with PC is (0=best, 1=worst): \n",
          structural_hamming_distance(correct_edges, predicted_edges=pc_edges))
    print("Normalized SHD distance with ID is (0=best, 1=worst): \n",
          structural_hamming_distance(correct_edges, predicted_edges=id_edges))
    print("Normalized SHD distance with CF is (0=best, 1=worst): \n",
          structural_hamming_distance(correct_edges, predicted_edges=cf_edges))


# Runs all the scripts
# light_switches creates new data
# three discovery modules
def generate_lights_and_discovery():
    # Generate new light switches and connections
    light_switches.main(n_switches=5, n_lights=5, connections_per_switch=2, n_samples=200)
    # Run PC
    discovery.main()
    # Run Interventional Discovery
    interventional_discovery.main()
    # Run Counterfactual Discovery
    counterfactual_discovery.main()


if __name__ == '__main__':

    plot = True

    generate_lights_and_discovery()
    compare_shd()

    if plot:
        plt.show()
