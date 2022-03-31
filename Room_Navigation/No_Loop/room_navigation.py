import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import time

# Need to define variables / nodes
# List of waypoint coordinates for room a and one for room b
import pandas as pd

waypoints_a = np.array([[1., 1.],
                        [1., 1.5],
                        [1.5, 1.],
                        [1.5, 1.5],
                        [1.5, 2.],
                        [2., 1.],
                        [2., 2.],
                        [2, 1.5],
                        [1., 2.]])
waypoints_b = np.array([[1., 1.],
                        [1., 1.5],
                        [1.5, 1.],
                        [1.5, 1.5],
                        [1.5, 2.],
                        [2., 1.],
                        [2., 2.],
                        [2, 1.5],
                        [1., 2.]])
waypoints_c = np.array([[1., 1.],
                        [1., 1.5],
                        [1.5, 1.],
                        [1.5, 1.5],
                        [1.5, 2.],
                        [2., 1.],
                        [2., 2.],
                        [2, 1.5],
                        [1., 2.]])
waypoints_d = np.array([[1., 1.],
                        [1., 1.5],
                        [1.5, 1.],
                        [1.5, 1.5],
                        [1.5, 2.],
                        [2., 1.],
                        [2., 2.],
                        [2, 1.5],
                        [1., 2.]])
# Waypoints room b are the same as room a, just a bit further to the right
waypoints_b[:, 0] += 1.5
waypoints_c[:, 1] += 1.5
waypoints_d[:, :] += 1.5

# array with the amount of waypoints per room
room_sizes = np.array([len(waypoints_a), len(waypoints_b), len(waypoints_c), len(waypoints_d)])

# np.split needs the index to split and room_size holds the amount of waypoints per room
# so need to accumulate them in another array
room_sizes_split = np.zeros(len(room_sizes))
room_sum = 0
for x, room in enumerate(room_sizes):
    room_sum += room
    room_sizes_split[x] = room_sum

# array with the room letter for each waypoint
room_array = np.concatenate(([np.repeat('a', room_sizes[0]),
                              np.repeat('b', room_sizes[1]),
                              np.repeat('c', room_sizes[2]),
                              np.repeat('d', room_sizes[3])])).ravel()
print(room_array)

# Create a list of the types of all waypoints per room
types_a = np.zeros(len(waypoints_a)).astype(str)
types_a[4] = 'door'
types_a[7] = 'door'
types_a[3] = 'table'

types_b = np.zeros(len(waypoints_b)).astype(str)
types_b[1] = 'door'
types_b[4] = 'door'
types_b[3] = 'table'

types_c = np.zeros(len(waypoints_c)).astype(str)
types_c[2] = 'door'
types_c[7] = 'door'
types_c[3] = 'table'

types_d = np.zeros(len(waypoints_d)).astype(str)
types_d[1] = 'door'
types_d[2] = 'door'
types_d[3] = 'table'

# The list of waypoints that are connected by doors
door_array = np.array([[7, 10],
                       [4, 20],
                       [13, 29],
                       [25, 28]])
# Make sure doors are two-way
door_array = np.vstack((door_array, np.fliplr(door_array)))

# Combine waypoints into one list containing all
# Everything that is not a door is set to 'wall
all_waypoints = np.vstack((waypoints_a, waypoints_b, waypoints_c, waypoints_d))
all_types = np.hstack((types_a, types_b, types_c, types_d))
all_types[all_types == '0.0'] = 'wall'

# Generate adjacency matrix, which waypoints are connected
adjacency_matrix_waypoints = np.zeros((len(all_waypoints), len(all_waypoints)), dtype=bool)
for x, y in np.ndenumerate(adjacency_matrix_waypoints):
    # Don't connect a waypoint to itself
    if x[0] != x[1]:

        # If both points are in the same room
        if room_array[x[0]] == room_array[x[1]]:
            adjacency_matrix_waypoints[x] = True

        # If waypoints are in different rooms, but connected by a door
        for row in door_array:
            if (x == row).all():
                adjacency_matrix_waypoints[x] = True


# plot all the waypoints as scatterplot
# plot all the connecting lines according to the adjacency_matrix
def plot_waypoints(waypoints, types, adjacency_matrix_waypoints):
    for line_from in range(len(waypoints)):

        # plot the point
        plt.scatter(waypoints[line_from, 0], waypoints[line_from, 1])
        plt.annotate(types[line_from], (waypoints[line_from, 0], waypoints[line_from, 1]))

        # plot the line when there is a path in adjacency_matrix
        lines_to = np.array(range(len(waypoints)))[adjacency_matrix_waypoints[line_from, :]]
        for id in lines_to:
            line_to = id
            plt.plot([waypoints[line_from][0], waypoints[line_to][0]],
                     [waypoints[line_from][1], waypoints[line_to][1]])


# Calls plot_waypoints on all the rooms and plots the robot
def plot_everything(all_waypoints, all_types, adjacency_matrix_waypoints, room_sizes_split, robot_at):
    plot_waypoints(all_waypoints, all_types, adjacency_matrix_waypoints)

    # plot the robot
    plt.scatter(all_waypoints[robot_at][0], all_waypoints[robot_at][1], marker="s", s=200, c="red")


# Generate the possible moves with the adjacency matrix
def possible_moves(all_waypoints, adjacency_matrix_waypoints, robot_at):
    waypoint_ids = np.array(range(all_waypoints.shape[0]))
    adjacent_waypoints = [adjacency_matrix_waypoints[robot_at, :]]

    return waypoint_ids[adjacent_waypoints]


# Chooses a next move, prioritzes moves where the robot has not been much yet
# If there is no move where the robot has been in history (last x moves), return random choice
def choose_next_move(possible_moves, history):
    for move in possible_moves:
        if move not in history:
            return move

    return random.choice(possible_moves)


# Robot starts at random waypoint
robot_at = random.choice(range(all_waypoints.shape[0]))
plt.figure()

# Move to a new waypoint each second
starttime = time.time()
# Record the (waypoint_id, room, type) in data

# record int_data
int_cols = ["waypoint_from", "waypoint_to", "room_from", "room_to", "type_from", "type_to", "changed_rooms"]
# or record obs_data ["waypoint_id", "room", "waypoint_type"]
data = np.zeros(7)


# This loop had the robot move to a new point X times and records the data
# options: plotting, saving data, using deque as moving heuristic
# record the last x moves of history in a deque to stimulate exploration
robot_at_history = deque(maxlen=10)
while True:
    # Plot the whole thing without blocking execution
    # Clear old plot
    plt.clf()
    #plot_everything(all_waypoints, all_types, adjacency_matrix_waypoints, room_sizes, robot_at)
    #plt.show(block=False)
    #plt.pause(0.001)

    # save where the robot came from
    robot_from = robot_at
    robot_at_history.append(robot_at)

    # Choose new waypoint
    moves = possible_moves(all_waypoints, adjacency_matrix_waypoints, robot_at)
    robot_at = choose_next_move(possible_moves=moves, history=robot_at_history)
    print(f"Possible moves: {room_array[moves]}")
    print(f"Moving to waypoint {robot_at}")

    # stack the new data for int_data
    data = np.vstack((data, np.array([robot_from, robot_at,
                                      room_array[robot_from], room_array[robot_at],
                                      all_types[robot_from], all_types[robot_at],
                                     (room_array[robot_from] != room_array[robot_at])])))

    # stack the new data for obs_data
    #data = np.vstack((data, np.array([robot_at,
    #                  room_array[robot_at],
    #                  all_types[robot_at]])))
    print("new data = ", data[-1, :])

    # For delaying the loop while ensuring a certain time per loop
    # time.sleep(0.1 - ((time.time() - starttime) % 0.1))

    # Generate a dataset and save it without the first empty row
    if data.shape[0] > 1000:
        np.save('int_data_small', np.delete(data, 0, 0))

        # Very crude way to convert values to numerics
        # needed to calculate covariances
        data[np.logical_or(data == "a", data == "door")] = 0
        data[np.logical_or(data == "b", data == "table")] = 1
        data[np.logical_or(data == "c", data == "wall")] = 2
        data[data == "d"] = 3
        df = pd.DataFrame(np.delete(data, 0, 0), columns=int_cols)
        df["changed_rooms"] = df["changed_rooms"].astype(bool) * 1
        df.to_csv('room_navigation_int.csv')
        df_cov = df.astype(int).cov()
        df_cov.to_csv('room_navigation_cov.csv', index=False)
        break
