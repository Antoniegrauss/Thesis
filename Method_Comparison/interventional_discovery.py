import itertools

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pgmpy.estimators.CITests import chi_square, g_sq
from pgmpy.base import DAG


# Sort by the interventions
# determine conditional independence on this subset of data

# If we flip a switch and a variable is still dependent on it, causal link switch -> variable
# If we flip a switch and a variable is not dependent on it, 2 options (determine with intervention on variable 2)
#   1- no causal link (intervention 2 also finds independence)
#   2- variable -> switch (intervention 2 does find dependence)


# Reads in a file
# finds the intervention targets (unique values in the "action" column)
# Splits the dataframe into 1 per intervention target
def prepare_intervention_data(filename):
    df_switches = pd.read_csv(filename)
    # Drop the index column
    df_switches = df_switches.drop(df_switches.columns[0], axis=1)

    # Split dataframe by action
    # Keep the split dataframe in a list
    intervention_targets = sorted(pd.unique(df_switches['action']).astype(int))
    print("Intervention targets in dataframe: ", intervention_targets)
    df_split_by_interventions = []
    for target in intervention_targets:
        # Take the rows where the intervention target == x
        # and the row before that
        split_df = df_switches[df_switches['action'] == target]
        split_set = set.union(set(split_df.index.values), (set(split_df.index.values + 1)))
        possible_rows = set(df_switches.index)
        split_set = set.intersection(split_set, possible_rows)
        print(f"Rows for intervention {target} are: \n", split_set)

        df_split_by_interventions.append(df_switches.iloc[list(split_set)])

    return df_split_by_interventions, df_switches, intervention_targets


# Check (conditional) dependence between intervention target and other variables
# Find the connections for a single variable
def find_connections_variable(df, target, other_variables, connections, num_conditionals=2,
                              intervention=True):
    print("Testing variable: ", target)

    for variable in other_variables:
        if variable != target and (variable, target) not in connections and (target, variable) not in connections:
            # print("checking variable ", variable)
            if not chi_square(X=target, Y=variable, Z=[], boolean=True, data=df, significance_level=0.01):
                print(target, variable, " dependent")

                # If this data is from an intervention we know the arrow direction
                if intervention:
                    connections.append((target, variable))

                # If this data is from an observation use a double arrow
                if not intervention:
                    connections.append((target, variable))
                    connections.append((variable, target))

                # If dependent, check for conditional independence
                # This checks for indirect effect
                for i in range(num_conditionals):
                    if check_conditional_independence(target, variable, df, other_variables, num_conditionals):
                        if (target, variable) in connections:
                            connections.remove((target, variable))
                        if (variable, target) in connections:
                            connections.remove((variable, target))
                        break

    return connections


# Checks conditional independence on all combinations of (X) elements from the conditional array
# returns true if independence is found
def check_conditional_independence(target, variable, df, conditional_array, amount_conditionals=2):
    # Remove the variables target and variable from the array to avoid duplicates in the independence test
    conditional_array = conditional_array[conditional_array != target]
    conditional_array = conditional_array[conditional_array != variable]

    # Get all possible combinations of conditionals
    for conditional in itertools.combinations(conditional_array, amount_conditionals):
        if chi_square(X=target, Y=variable, Z=conditional, data=df, significance_level=0.01):
            print(target + " and ", variable, " independent, conditional on: ", conditional)
            return True

    return False


# Searches for the connections from the switches to the lights
def find_switch_light_connections(df_split, connections, amount_conditionals=0):
    # For each of the interventions targets (1 df each) find the connections
    # TODO: target should not be just enumerated here
    for target, df in enumerate(df_split):
        target = "s" + str(target)
        other_variables = df.columns.values[1:]
        # Since we are looking at interventions, dependence means a causal link
        connections = (find_connections_variable(df, target, other_variables, connections,
                                                      amount_conditionals, intervention=True))

    return connections


# Check the lights for a set of causes that is both necessary and sufficient
# that means there are no more causes for that light
# since the behaviour is fully defined by the both necessary and sufficient causes
def generate_light_light_connections(connections, df, intervention_targets, amount_conditionals=2):
    lights = df.columns.values[(len(intervention_targets) + 1):]
    for light in lights:
        other_variables = df.columns.values[1:]
        # Search for connections or not??
        this_connections = find_connections_variable(df, light, other_variables, connections,
                                                      amount_conditionals, intervention=False)
        connections.append(this_connections)

    return connections


# Generates a DAG from a connections array (tuples)
def generate_graph_from_connections_save_plot(connections, save_name='Result/light_switches_id_edges',
                                              save=True, plot=True):
    # Set up a DAG from the found dependency relations
    graph = DAG(ebunch=connections)
    # for connection in connections:
    #    graph.add_edge(u=connection[0], v=connection[1])
    if save:
        np.save(save_name, graph.edges)

    if plot:
        plt.figure()
        plt.title("Interventional Discovery")
        nx.draw_circular(graph, with_labels=True, arrowsize=30, node_size=800, alpha=0.5)
        #plt.show()

    return graph


def main():
    connections = []
    df_split, df, intervention_targets = prepare_intervention_data('Data/five_switches.csv')
    connections = find_switch_light_connections(df_split, connections)
    connections = find_switch_light_connections(df_split, connections)
    graph = generate_graph_from_connections_save_plot(connections)


if __name__ == '__main__':
    main()