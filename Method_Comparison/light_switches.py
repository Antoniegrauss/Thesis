from collections import deque

import networkx as nx
import numpy as np
import random
import pandas as pd

# For comparability choose a seed
from matplotlib import pyplot as plt

# random.seed(42)
from pgmpy.base import DAG


# connect each switch to 'x' lights (for now 1)
# keep a list of all the lights and choose 1 light per switch
# remove that light from the list and add it to the connections
def generate_connections(light_switches, lights, connections_per_switch):
    connections_array = np.zeros((len(light_switches), len(lights)))
    for id, switch in enumerate(light_switches):

        # A list of available light to connect to
        all_lights = np.copy(lights)
        for x in range(connections_per_switch):
            # Choose a light to connect to (can not connect to same light twice)
            this_connection = random.choice(all_lights)
            all_lights = all_lights[all_lights != this_connection]
            connections_array[id, this_connection] = 1

    return connections_array


# TODO: implement probabilistic return
# TODO: implement return with sequence
# Function that determines whether lights are on or off
# depending on the switch states
def lights_output(switch_states, connections_array):
    lights = []
    # For each light check whether all the switches are correct
    # subtracting the row from the switch states should not give any -1 values
    for row in connections_array.T:
        if np.any(np.isin(switch_states - row, -1)):
            lights.append(0)
        else:
            lights.append(1)

    return lights


# Returns the switch states after flipping a switch (action)
def switch_states_after_action(action, switches):
    if switches[action] == 0:
        switches[action] = 1
    elif switches[action] == 1:
        switches[action] = 0
    else:
        print("Invalid light switch choice")

    return switches


# loop where the robot chooses an action and result is generated and saved
# in data save these cols
def setup_and_run_loop(n_switches, n_lights, light_switches, lights, connections_per_switch, n_samples):
    cols = ['action']
    for s in range(n_switches):
        cols.append("s" + str(s))
    for l in range(n_lights):
        cols.append("l" + str(l))
    # two_cols = ['action', 's0', 's1', 'l0', 'l1']
    data = np.zeros(1 + len(light_switches) + len(lights))
    # Set all the light switches to their initial position
    switch_states = np.zeros(n_switches).astype(int)

    # Generate the connection matrix (multiple options)
    connections_array = generate_connections(light_switches, lights, connections_per_switch)
    print("Connections are: \n", connections_array)

    # Keep the switch states in a deque
    # use the exploration policy
    max_len = int((2 ** n_switches + 1))
    history_switch_states = deque(maxlen=max_len)
    action = 0
    history_switch_states.append(switch_states)

    generate_data(n_switches, light_switches, switch_states, history_switch_states, cols,
                  connections_array, data, n_samples)

    return connections_array


def choose_action(n_switches, light_switches, switch_states, history_switch_states):
    # Choice of light switch
    # Iterate over shuffled list of switches to stratify the action choice
    # without shuffling its just binary counting
    shuffled_actions = list(range(n_switches))
    random.shuffle(shuffled_actions)

    # Safety for return variables
    action = 0
    future_switch_states = switch_states.copy()

    # Loop over all the actions, see if we can visit a state not in the history
    for i in shuffled_actions:
        switch_states_copy = switch_states.copy()
        future_switch_states = np.array(switch_states_after_action(i, switch_states_copy))
        if not any((future_switch_states == history).all() for history in history_switch_states):
            #print("Switch state not in history found, action: s", i)
            action = i
            break

        # If no new action is found, choose a random action
        if i == n_switches - 1:
            action = random.choice(range(len(light_switches)))
            #print("No new switch state found")
            break

    return action, future_switch_states


# Run the loop, store the data and save the data
def generate_data(n_switches, light_switches, switch_states, history_switch_states, cols,
                  connections_array, data, n_samples):
    while True:

        # Choose a new action and calculate the next steps states
        action, future_switch_states = choose_action(n_switches, light_switches, switch_states, history_switch_states)
        history_switch_states.append(future_switch_states)
        switch_states = switch_states_after_action(action, switch_states)
        #print("Action: flipping switch ", action)

        light_states = np.array(lights_output(switch_states, connections_array))
        new_data = np.insert(np.concatenate((switch_states, light_states)), 0, action)
        #print(new_data)
        data = np.vstack((data, new_data))

        # For delaying the loop while ensuring a certain time per loop
        # time.sleep(1 - ((time.time() - starttime) % 1))
        if data.shape[0] > n_samples:
            np.save('Data/5_light_switches', np.delete(data, 0, 0))
            pd.DataFrame(np.delete(data, 0, 0), columns=cols).to_csv('Data/five_switches.csv')
            break


# Show the correct graph
def generate_graph_and_save(light_switches, lights, connections_array,
                            save_name='light_switches_correct_edges', save=True, plot=True):
    graph = DAG()
    for id, switch in enumerate(light_switches):
        for idd, light in enumerate(lights):
            if connections_array[id, idd] == 1:
                graph.add_edge(u="s" + str(switch), v="l" + str(light))
    if save:
        np.save(save_name, graph.edges)
    if plot:
        plt.figure()
        plt.title("Generated causal links")
        nx.draw_circular(graph, with_labels=True, arrowsize=30, node_size=800, alpha=0.5)
        # plt.show()


def main(n_switches=5, n_lights=5, connections_per_switch=2, n_samples=200):
    # Light switches
    light_switches = np.array(range(n_switches))

    # Lights
    lights = np.array(range(n_lights))

    connections_array = setup_and_run_loop(n_switches, n_lights, light_switches, lights,
                                           connections_per_switch, n_samples)
    generate_graph_and_save(light_switches, lights, connections_array)


if __name__ == '__main__':
    main()
