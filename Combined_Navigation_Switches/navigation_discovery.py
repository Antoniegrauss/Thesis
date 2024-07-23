import math
import random

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from pgmpy.base import DAG
import numpy as np

# Own files
import room_navigation
import sufficiency_refuter
import metrics


# Find necessary and sufficient causes for the room navigation case
# Actions are moving to a certain waypoint
# Inputs are:
# move_from waypoint type
# move_to waypoint type
# Outputs are waypoint types and current room

# TODO: initialize necessary and sufficient sets

# TODO: choose an action not at random (exploration?)
# Choose an action
def choose_random_action(actions):
    action = random.choice(actions)

    return action


# Get new light and switch states
def new_states(action, robot_at, history, room_array, all_waypoints,
               all_types, adjacency_matrix):
    # Calculate the light states
    history, robot_at, new_data = room_navigation.loop_once(robot_at, history, room_array, all_waypoints,
                                                            all_types, adjacency_matrix, action)

    return history, robot_at, new_data


# Returns the difference between the two arrays
def calculate_changes(old, new):
    changes = np.array([x != y for x, y in zip(old, new)])
    return changes


# Combines old data and new data
def stack_data(data, new_data):
    data = np.vstack((data, new_data.copy()))
    return data


# If the new set is a subset of one of the sets in the set_of_sets
# return the found superset
def subset_in_set_of_sets(set_of_sets, new_set):
    for set_x in set_of_sets:
        if set(new_set).issubset(set(set_x)):
            return set_x
    return {}


# If the new set is a superset of one of the sets in the set_of_sets
# returns the found subset
# can also be the same set
def superset_in_set_of_sets(set_of_sets, new_set):
    for set_x in set_of_sets:
        if set(new_set).issuperset(set(set_x)):
            return set_x
    return {}


# Combines a new set and a set of sets
# add the set if it is not a superset of one already in the set
# new_set is added as a tuple
def combine_sufficient_sets(set_of_sets, new_set):
    print("set_list, ", set_of_sets)
    print("new_set", new_set)

    # Check for empty new set
    if new_set == ():
        return set_of_sets

    # Check for empty set
    if set_of_sets == {}:
        return {new_set}

    subset = superset_in_set_of_sets(set_of_sets, new_set)
    if subset != {}:
        # If there is already a subset
        print("is superset")
        return set_of_sets

    superset = subset_in_set_of_sets(set_of_sets, new_set)
    if superset != {}:
        # If the new set is a subset of one in the list
        # swap the new set with that entry
        print("is subset")
        return_set = {new_set}
        for x in set_of_sets:
            return_set.update([x])
        return_set.remove(superset)
        return return_set

    # If none of the sets are a subset (or the same) as the new set,
    # append it to the list
    return_set = {new_set}
    for x in set_of_sets:
        return_set.update([x])
    print("added set to the list ", return_set)
    return return_set


# TODO: what function to use for updating confidence?
# Updates the confidence 0 .. 1
# according to the new sample value and the total amount of samples
# new_sample should be 1 for positive feedback, -1 for negative feedback
def update_confidence(confidence, new_sample, n_samples, function='square_root'):
    return confidence + (new_sample / (2 ** n_samples))


# Creates first-order logic formulas
# necessary,    not cause -> not effect
def generate_fol_necessary(pn, cause, effect):
    return_str = str(pn) + 'not' + str(cause) + '->not' + str(effect)
    return return_str


# Creates first-order logic formulas
# sufficient,   cause 1...cause n -> effect
def generate_fol_sufficient(causes, effect):
    return_str = ''
    for cause in causes:
        return_str += str(cause) + '^'
    return_str = return_str[-1]
    return_str += '->' + str(effect)
    return return_str


# Generates a DAG object
# and plots it
def generate_graph_and_plot(connections, all_types, save_name='looped_discovery_edges',
                            save=True, plot=True):
    graph = DAG()
    graph.add_edges_from(connections)
    graph_nodes = graph.nodes
    all_types = np.unique(all_types)
    if "no_move" not in graph_nodes:
        graph.add_node("no_move")
    for type_combi in zip(all_types, all_types):
        if type_combi not in graph_nodes:
            graph.add_node(type_combi)
    if save:
        np.save(save_name, graph.edges)
    if plot:
        plt.figure()
        nx.draw_circular(graph, with_labels=True, arrowsize=30, node_size=800, alpha=0.5)
        # plt.show()

    return graph


# Generates the column indexes for a Pandas Dataframe
def get_cols(n_switches, n_lights, heating=False):
    cols = ['action']
    for s in range(n_switches):
        cols.append("s" + str(s))
    if heating:
        cols.append('h')
    for l in range(n_lights):
        cols.append("l" + str(l))
    if heating:
        cols.append('t')

    return cols


# Choose new action (loop back)
def main(n_loop=20, plot_graph=True, plot_robot=True):
    # Setup everything
    all_waypoints, all_types, room_array, door_array = room_navigation.setup_room_navigation()
    adjacency_matrix, history, data, cols, robot_at = room_navigation.setup_loop()
    states = np.array(cols)
    prev_action = robot_at

    # TODO implement confidence
    # The connections are stored in an array with 3 columns
    # col 1 = from, col 2 = to, col 3 = confidence
    # ['from', 'to']
    connections = []

    if plot_robot:
        plt.figure()

    # Loop the discovery
    for n in range(n_loop + 1):
        # Choose a new action and calculate the new states
        possible_actions = room_navigation.possible_moves(all_waypoints, adjacency_matrix, robot_at)
        # Add None action, means standing still
        possible_actions = np.concatenate((possible_actions, [None]))
        action = choose_random_action(possible_actions)
        print("action ", action)
        # Moving from type ..  to type ... as tuple
        if action is not None:
            action_type_from_to = (all_types[prev_action], all_types[action])
            print(f"Moving from waypoint type {action_type_from_to[0]} to {action_type_from_to[1]}: \n")
        else:
            print("Null action, standing still")

        history, robot_at, new_data = room_navigation.loop_once(robot_at, history, room_array, all_waypoints,
                                                                all_types, adjacency_matrix, action)

        # TODO: keep track of whether 'cause' variables change as well
        # Look at the difference with last time step
        if n <= 0:
            state_changes = calculate_changes(data, new_data)
        elif n > 0:
            state_changes = calculate_changes(data[-1, :], new_data)

        # If there are changes in states
        # update Necessary and Sufficient set for that state
        if True in state_changes:
            changed_states = states[state_changes]
            for state in changed_states:
                if action is None:
                    action_type_from_to = "no_move"

                # TODO: write for room navigation (necessity and sufficiency)

                # Add this switch, light pair to connections
                if (action_type_from_to, state) not in connections:
                    print("Add connection ", (action_type_from_to, state))
                    connections.append((action_type_from_to, state))

        data = stack_data(data, new_data)

        if action is not None:
            prev_action = action


        # Plot the robot position
        if plot_robot:
            plt.clf()
            room_navigation.plot_everything(all_waypoints, all_types, adjacency_matrix, robot_at)
            plt.show(block=False)
            plt.pause(0.001)

    df = pd.DataFrame(np.delete(data, 0, 0), columns=cols)

    # TODO: Generate the first-order logic formulas

    print("Final data: \n", df)

    print("connections: \n", connections)
    #print("sufficient_sets \n", sufficient_sets)
    #print("link_confidence \n", link_confidence[:, :, 0])

    if plot_graph:
        # TODO: correct graph?
        # correct_graph = data_generator.generate_graph_and_save(switches, lights, connections_array)
        predicted_graph = generate_graph_and_plot(connections, all_types)
        # shd = metrics.structural_hamming_distance(correct_graph.edges, predicted_graph.edges)
        # print("Normalized Structural Hamming Distance: ", shd)
        plt.show()


if __name__ == '__main__':
    main()
