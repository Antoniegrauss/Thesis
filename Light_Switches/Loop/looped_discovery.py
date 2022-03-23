import math
import random

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from pgmpy.base import DAG
import numpy as np

# Own files
import data_generator
import sufficiency_refuter
import metrics

# TODO:
# Discover causal links from the switches and lights
# One by one (exploit)
# Multiple at once (explore)

# Types of causal links
# PN
# From light x look at change T's
# Look for necessary switches (same T or farther back)
# Store these beliefs
# Test beliefs with actions

# PS
# From stationary data (non change)
# For light x
# Gather the sufficient set of switches
# Test this by turning these switches off while keeping the light on


# Generate logical statements from PN and PS set
# N
# not cause -> not effect
# S
# cause_1 ^ cause_2 ... cause_n -> effect

# Setup the connections
def setup_connections(connection_choices, n_switches=5, n_lights=5, connections_per_switch=2):
    # Array with all the switches (unique integer)
    switches = np.array(range(n_switches))

    # Array with all the lights (unique integer)
    lights = np.array(range(n_lights))

    connections_array, connection_types = data_generator.setup_connections(switches, lights,
                                                                           connections_per_switch,
                                                                           connection_choices)
    return connections_array, connection_types, lights, switches


# TODO: choose an action not at random
# Choose an action
def choose_random_action(actions):
    action = random.choice(actions)

    return action


# Calculates the new room temperature based on the heating temp and the room temp
# If the heating is hotter than inside, increase inside temp
# If heating is not hotter and outside is cooler, decrease inside temp
def calculate_new_temperature(heating, room_temp, outside_temp):
    difference = heating - room_temp
    if difference > 0:
        return room_temp + int(0.5 * difference)
    elif room_temp > outside_temp:
        return room_temp - int(0.5 * (room_temp - outside_temp))


# Get new light and switch states
def new_states(action, switch_states, connections_array, connection_types):
    # Calculate the light states
    next_switch_states = data_generator.switch_states_after_action(action, switch_states)
    next_light_states = data_generator.lights_output(next_switch_states, connections_array, connection_types)

    return next_light_states, next_switch_states


# Returns the difference between the two arrays
def calculate_changes(old, new):
    return new - old


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
def generate_graph_and_plot(connections, connection_types, save_name='looped_discovery_edges',
                            save=True, plot=True):
    graph = DAG()
    graph.add_edges_from(connections)
    if save:
        np.save(save_name, graph.edges)
    if plot:
        plt.figure()
        plt.title(f"Found links: looped discovery \n {connection_types}")
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
def main(n_loop=100, plot=True):
    # Setup everything
    # Switches
    n_switches = 5
    n_lights = 5
    connections_per_switch = 2

    # Heating knob and temps
    heating_knob_settings = [0, 1, 2, 3, 4, 5]
    outside_temp = 10
    inside_temp = 15
    heating = 0
    heating_knob = 0

    # The possible types of connections
    all_choices = np.array([['ON', 'OFF'],
                            ['AND', 'OR', 'XOR', 'NAND', 'NOR', 'NXOR']])
    # Option to restrict the choices a bit for simplicity
    connection_choices = np.array([['ON', 'OFF'],
                                   ['AND', 'OR', 'XOR']])
    connections_array, connection_types, lights, switches = \
        setup_connections(connection_choices,
                          n_switches=n_switches,
                          n_lights=n_lights,
                          connections_per_switch=connections_per_switch)

    # All the switches are off at the start
    switch_states = np.zeros(n_switches)

    # The light states at the start
    light_states = data_generator.lights_output(switch_states, connections_array, connection_types)

    # Create a numpy array for the data
    data = np.zeros(3 + n_switches + n_lights)

    # The connections are stored in an array with 3 columns
    # col 1 = from, col 2 = to, col 3 = confidence
    # ['from', 'to']
    connections = []

    # Gather a list of possible actions
    possible_actions = np.append(switches.copy(), 'heating')

    # Necessary is an array between lights and switches
    # each cell holds:
    # a number between 0 and 1 for the confidence
    # n_samples to update the confidence
    # that that switch is necessary for that light
    # or in logic  not_switch -> not_light
    link_confidence = np.zeros((n_lights, n_switches, 2))

    # Sufficient_sets holds a list of sets (one for each light)
    # a sufficient set will always turn on a light
    # supersets can not be added, since the extra variables are non-necessary
    sufficient_sets = []
    for _ in lights:
        sufficient_sets.append({})

    # Loop the discovery
    for n in range(n_loop + 1):
        # Choose a new action and calculate the new states
        action = choose_random_action(possible_actions)
        print("action ", action)

        # If we act on the heating, do not calculate new switch states etc..
        if action == 'heating':
            heating_knob = random.choice(heating_knob_settings)
            heating = heating_knob * 5
            action = 'h' + str(heating_knob)
            new_inside_temp = calculate_new_temperature(heating, inside_temp, outside_temp)

            new_light_states, new_switch_states = light_states.copy(), switch_states.copy()
            inside_temp = new_inside_temp

        # If we change a switch calculate the new states and do inference on connections
        else:
            action = int(action)
            new_light_states, new_switch_states = new_states(action, switch_states,
                                                             connections_array, connection_types)
            print("Switch states: \n", new_switch_states)
            print("Light states: \n", new_light_states)

            # TODO: keep track of whether 'cause' variables change as well
            # Look at the difference with last time step
            switch_changes = calculate_changes(switch_states, new_switch_states)

            #print("Switch changes: \n", switch_changes)
            light_changes = calculate_changes(light_states, new_light_states)
            #print("Light changes: \n", light_changes)

            # If there are changes in the lights
            # update Necessary and Sufficient set for that light
            if len(np.unique(light_changes)) > 1:
                changed_lights = lights[light_changes != 0]
                #print("changed lights", changed_lights)
                for light in changed_lights:

                    # TODO: sufficient set only works for ON (so no XOR and such)
                    # TODO: implement new sufficiency check algo
                    # Add the current switches to the Sufficient set
                    print("light, ", light, " has changed")
                    # new_set must be immutable set(set()) is not possible
                    new_set = tuple(switches[switch_states == 1])
                    print("sufficient_sets, ", sufficient_sets)
                    if new_set not in sufficient_sets[light]:
                        # Use the sufficiency refuter before combining
                        refuted_sufficiency_set = sufficiency_refuter.reduce_sufficiency_set(new_set, light,
                                                                                             new_light_states.copy(),
                                                                                             new_switch_states.copy(),
                                                                                             connections_array,
                                                                                             connection_types)
                        sufficient_sets[light] = combine_sufficient_sets(sufficient_sets[light], refuted_sufficiency_set)


                    # Update the Necessary of the switch that was just flipped
                    n_samples = link_confidence[light, action, 1] + 1
                    confidence = link_confidence[light, action, 0]
                    # Update the confidence with '1' as value
                    updated_confidence = update_confidence(confidence, 1, n_samples)
                    print(f"Updated confidence {light}, {action} to {updated_confidence}")
                    link_confidence[light, action, :] = [updated_confidence, n_samples]

                    # Add this switch, light pair to connections
                    if ('s' + str(action), 'l' + str(light)) not in connections:
                        connections.append(('s' + str(action), 'l' + str(light)))

        # Save the new data in the light and switches array
        light_states = new_light_states.copy()
        switch_states = new_switch_states.copy()

        # Build an array of [action, switch_states, light_states]
        new_data = np.hstack((action, new_switch_states, heating_knob, new_light_states, inside_temp))
        data = stack_data(data, new_data)

    cols = get_cols(n_switches, n_lights, heating=True)
    df = pd.DataFrame(np.delete(data, 0, 0), columns=cols)

    # TODO: Generate the first-order logic formulas

    print("Final data: \n", df)

    print("connections: \n", connections)
    print("sufficient_sets \n", sufficient_sets)
    print("link_confidence \n", link_confidence[:, :, 0])

    if plot:
        correct_graph = data_generator.generate_graph_and_save(switches, lights, connections_array)
        predicted_graph = generate_graph_and_plot(connections, connection_types)
        shd = metrics.structural_hamming_distance(correct_graph.edges, predicted_graph.edges)
        print("Structural Hamming Distance: ", shd)
        plt.show()


if __name__ == '__main__':
    main()
