import numpy as np
from matplotlib import pyplot as plt

first_part_path = "Result/scores_4_rooms_"
paths = []
for i in range(10):
    paths.append(first_part_path + str(i) + ".npy")

res = []
for path in paths:
    res.append(np.load(path))

plt.figure()
for i in res:
    score = i
    plt.title("SHD score for environment: 4 rooms, \n Adjusting search space every 50 time steps")
    score_str = str(score[-1, 1])
    if len(score_str) > 3:
        score_str = score_str[:4]
    plt.plot(score[:, 0], score[:, 1], label=("stopping time: " + str(score[-1, 0]) + ", score: " + score_str))
    plt.xlabel("Time [steps]")
    plt.ylabel("Normalized Structural Hamming Distance")
plt.legend()
plt.ylim(0, 1)
plt.xlim(0, 1500)
plt.grid()
plt.show()