import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

"""first_part_path = "scores_2_rooms_"
paths = []
for i in range(10):
    paths.append(first_part_path + str(i) + ".npy")"""

room_2_res = pd.read_csv("Result/Results_table_room_2.csv").to_numpy()
room_4_res = pd.read_csv("Result/Results_table_room_4.csv").to_numpy()
print(room_2_res)
print(room_4_res)

heights_4 = [eval(x) for x in room_4_res[:, 2]]
widths_4 = [int(x)/10 for x in room_4_res[:, 1]]

heights_2 = [eval(x) for x in room_2_res[:, 2]]
widths_2 = [int(x)/10 for x in room_2_res[:, 1]]

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
plt.title("Perfect results for 4 room environment")
plt.scatter(x=room_4_res[:, 1], y=heights_4, label="Fraction of perfect results")
plt.scatter(x=room_4_res[:, 1], y=room_4_res[:, 3], label="Average error (without perfect)")
ax.set_xscale('log')
plt.xlabel("Adjust search space each n time steps")
plt.ylabel("Perfect results [blue] \n NSHD score [orange]")
plt.legend()
ax.set_ylim(0, 1)

plt.grid()

ax_2 = fig.add_subplot(1, 2, 2)
ax_2.set_xscale('log')
plt.title("Perfect results for 2 room environment")
plt.scatter(x=room_2_res[:, 1], y=heights_2, label="Fraction of perfect results")
plt.scatter(x=room_2_res[:, 1], y=room_2_res[:, 3], label="Average error (without perfect)")
plt.xlabel("Adjust search space each n time steps")
plt.ylabel("Perfect results [blue] \n NSHD score [orange]")
plt.legend()
ax_2.set_ylim(0, 1)

plt.grid()


fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.set_xscale('log')
ax.set_title("Stopping times for the 4 room environment")
plt.scatter(x=room_4_res[:, 1], y=room_4_res[:, -2], label="perfect stopping time")
plt.scatter(x=room_4_res[:, 1], y=room_4_res[:, -1], label="imperfect stopping time")
plt.xlabel("Adjust search space each n time steps")
plt.ylabel("Stopping time")
plt.grid()
plt.legend()

ax_2 = fig.add_subplot(1, 2, 2)
ax_2.set_xscale('log')
ax_2.set_title("Stopping times for the 2 room environment")
plt.scatter(x=room_2_res[:, 1], y=room_2_res[:, -2], label="perfect stopping time")
plt.scatter(x=room_2_res[:, 1], y=room_2_res[:, -1], label="imperfect stopping time")
plt.xlabel("Adjust search space each n time steps")
plt.ylabel("Stopping time")
plt.grid()
plt.legend()

plt.show()