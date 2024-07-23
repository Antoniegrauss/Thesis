import copy
import os

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import PC
import sys

environment_list = []
robot_list = []
reasoner_list = []
env = "4_rooms"

scores = np.zeros(2)

for i in range(1):

    if env == "2_rooms":
        import environment_2_rooms

        environment = environment_2_rooms.generate_new()
    elif env == "4_rooms":
        import environment_4_rooms

        environment = environment_4_rooms.generate_new()

    elif env == "1_rooms":
        import environment_1_rooms

        environment = environment_1_rooms.generate_new()

    else:
        print("Invalid environment")
        exit()

    # Stop the printing
    #sys.stdout = open(os.devnull, 'w')

    thesescores = []
    robot = environment.robot
    reasoner = environment.robot.reasoner
    # Starting values, and save sensor data
    print(f"Starting values are: ", environment.get_values(environment.variable_list))

    # Add a figure and subplots for visualization
    fig = plt.figure(1, figsize=(20, 10))
    ax_ani = fig.add_subplot(1, 2, 1)
    ax_env = fig.add_subplot(1, 2, 2)

    _ = robot.reasoner.planner(animate=True, ax_ani=ax_ani, plot_env=True, ax_env=ax_env)

    thesescores.append(robot.reasoner.shd_score)
    thesescores = np.vstack((np.array([0, 1]), np.array(thesescores).reshape(-1, 2)))
    scores = np.vstack((scores, thesescores[-1, :]))

    name = "scores_4_rooms_" + str(i)
    np.save(name, thesescores)

    sys.stdout = sys.__stdout__
    print("Iteration ", i, " scores: ", thesescores)

    # Enable printing
    sys.stdout = sys.__stdout__

    print(scores[1:, :])


def calculate_scores(scores):
    count = scores.shape[0]-1
    imperfect = np.count_nonzero(scores[1:, 1])
    perfect = count - imperfect
    print("Perfect score: ", perfect, " / ", count)

    non_perfect = scores[scores[:, 1] > 0, :]
    average_error = np.average(non_perfect[:, 1])
    print("Average error: ", average_error)

    average_stop_imperfect = np.average(non_perfect[:, 0])
    print("Average stopping time imperfect: ", average_stop_imperfect)

    perfect_scores = scores[scores[:, 1] == 0, :]
    average_stop_time = np.average(perfect_scores[:, 0])
    print("Average stopping time perfect: ", average_stop_time)


calculate_scores(scores[1:, :])

    # Code block for generating GIES compatible data in a csv
    # Generate data with the exploration based actions/moves
def save_data(environment_1, reasoner_1, robot_1):
    for i in range(300):
        possible_actions = environment_1.possible_actions(robot_1.position)
        possible_moves = environment_1.possible_moves()
        action, move = reasoner_1.choose_action_or_move(possible_actions=possible_actions, possible_moves=possible_moves)
        environment_1.do_actions_move_time(action=action, move=move)
        print(environment_1.time)

    # Transform the data to numeric
    # give the action states also if they are non-observable
    df_data = np.full(reasoner_1.sensor_history[1:, 1:].shape, -1)
    for index, el in np.ndenumerate(reasoner_1.sensor_history[1:, 1:]):
        print(index, el)
        # State of switches does not change apart from actions
        if "switch" in reasoner_1.sensor_history[0, index[1]+1]:
            df_data[index] = reasoner_1.last_observed_sensor_history[index[0]+1, index[1]+1]
        else:
            if type(el) == int:
                df_data[index] = el
            elif "waypoint" in el:
                df_data[index] = (int(el[9:]))
            elif "room" in el:
                df_data[index] = (ord(el[-1]) - 97)
            elif len(el) <= 2:
                df_data[index] = (int(el))

    intervention_targets_2_rooms = [0, 1, 2, 5, 7, 8]
    #intervention_targets_4_rooms = [0, 1, 2, 5, 7, 8, 12, 13, 17, 18]
    df_data = np.hstack((df_data, reasoner_1.action_history[1:, 1].reshape(-1, 1)))

    #None actions are treated as observational data (action index 0)
    # Set action indices
    # 2 rooms
    for row in df_data:
        if row[-1] == "None":
            row[-1] = 1
        elif row[-1] == "switch_1":
            row[-1] = 2
        elif row[-1] == "switch_2":
            row[-1] = 3
        elif row[-1] == "position_robot_1":
            row[-1] = 4
        elif row[-1] == "switch_3":
            row[-1] = 5
        elif row[-1] == "switch_4":
            row[-1] = 6

    # 4 rooms
    # Create lookup table
    count = 0
    lookup = []
    lookup.append(["None", count, 0])
    count += 1
    for id, el in enumerate(reasoner_1.sensor_history[0, :]):
        if "switch" in el or "position" in el:
            lookup.append([el, count, id])
            count += 1
    lookup = np.array(lookup).reshape(-1, 3)
    intervention_targets_4_rooms = lookup[:, 2].reshape(1, -1)[0].tolist()
    for row in df_data:
        id = lookup[:, 0] == row[-1]
        row[-1] = int(lookup[id, 1][0]) + 1

    #df_data = df_data[df_data != -1]
    print(intervention_targets_4_rooms)
    df = pd.DataFrame(df_data, columns=np.append(reasoner_1.last_observed_sensor_history[0, 1:], ["action"]))
    df.to_csv("Dataset_4_rooms_300_samples.csv", index=False)
