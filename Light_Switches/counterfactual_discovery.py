import itertools

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pgmpy.estimators.CITests import chi_square, g_sq
from pgmpy.base import DAG


# Sort by the interventions
# determine conditional independence on this subset of data
# compare to PC result


def load_df_and_find_interventions(filename):
    # cols = ['action', 's0', 's1', 's2', 's3', 's4', 'l0', 'l1', 'l2', 'l3', 'l4']
    # short_cols = ['action', 's0', 's1', 'l0', 'l1']
    df_switches = pd.read_csv(filename)
    # Drop the index column
    df_switches = df_switches.drop(df_switches.columns[0], axis=1)

    # Split dataframe by action
    # Keep the split dataframe in a list
    intervention_targets = sorted(pd.unique(df_switches['action']).astype(int))
    print("Intervention targets in dataframe: ", intervention_targets)

    return intervention_targets, df_switches


# Check (conditional) dependence between intervention target and other variables
# Find the connections for a single variable
def find_connections_variable(df, target, other_variables, connections, num_conditionals, intervention):
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
def check_conditional_independence(target, variable, df, conditional_array, amount_conditionals):
    # Remove the variables target and variable from the array to avoid duplicates in the independence test
    conditional_array = conditional_array[conditional_array != target]
    conditional_array = conditional_array[conditional_array != variable]

    # Get all possible combinations of conditionals
    for conditional in itertools.combinations(conditional_array, amount_conditionals):
        if chi_square(X=target, Y=variable, Z=conditional, data=df, significance_level=0.01):
            print(target + " and ", variable, " independent, conditional on: ", conditional)
            return True

    return False


# TODO: change to P(necessary) instead of boolean
# Checks the causes individually whether they are necessary for 'variable'
# using the data from df
# return an array with the necessary causes
def necessary_causes(variable, df, causes):
    # Necessary cause iff P(effect|not_cause) = 0 and
    # P(effect|cause) > 0
    necessary_causes = set()
    for cause in causes:
        sorted_df = df.groupby([variable, cause]).count()
        # Necessary if the light is never on if the cause is 1
        # meaning it does not contain a row [0, 1]
        index_count = sorted_df.index
        check_array = np.array([1.0, 0.0])
        if not any(np.all(row == check_array) for row in index_count):
            necessary_causes.add(cause)

    return necessary_causes


# TODO: change to P(sufficient) instead of boolean
# Calculates the sets of sufficient causes
# a set of causes is sufficient iff P(variable|causes) = 1
# returns empty set is no set is sufficient
def sufficient_causes(variable, df, causes):
    stop = False
    # Start looking for sets with 1 sufficient cause
    # then increase set up to num_causes
    for i in range(1, len(causes) + 1):
        # print(f"Searching light {variable} for sufficient sets of {i} causes")

        if not stop:
            # Generate the possible combinations from the causes
            for cause_set in itertools.combinations(causes, i):
                cols = [variable]
                for cause in cause_set:
                    cols.append(cause)
                sorted_df = df.groupby(cols)

                # Sufficient if the light is always on if the causes are 1
                # meaning it does not contain a row [0, 1, 1, ...]
                # this row means the light is off when the switches (causes) are on
                index_count = sorted_df.count().index
                check_array = np.full(len(cols), 1.0)
                check_array[0] = 0.0
                if not any(np.all(row == check_array) for row in index_count):
                    # print(f"{cause_set} is sufficient for light {light}")
                    # print(sorted_df.count())
                    sufficient_set = cause_set
                    return set(sufficient_set)

    return ()


# Check the lights for a set of causes that is both necessary and sufficient
# that means there are no more causes for that light
# since the behaviour is fully defined by the both necessary and sufficient causes
def discovery_with_counterfactual_inference(df, intervention_targets, connections):
    lights = df.columns.values[(len(intervention_targets) + 1):]
    for light in lights:
        # print(f"Searching light {light} for necessary and sufficient causes")

        causes = ["s" + str(target) for target in intervention_targets]
        sufficient_set = sufficient_causes(light, df, causes)
        # print(f"Causes sufficient for {light}: \n {sufficient_set}")
        necessary_set = necessary_causes(light, df, causes)
        # print(f"Causes necessary for {light}: \n {necessary_set}")

        # If there is a necessary and sufficient set, this light has no more causes
        necessary_and_sufficient = set.intersection(sufficient_set, necessary_set)
        if necessary_and_sufficient == necessary_set:
            print(f"Light {light} has necessary and sufficient set {necessary_and_sufficient}")
            for cause in necessary_and_sufficient:
                # graph.add_edge(u=cause, v=light)
                connections.append((cause, light))
        else:
            print(f"Light {light} does not have a complete N and S set \n"
                  f"Searching for connections")
            other_variables = df.columns.values[1:]
            amount_conditionals = 2
            connections = find_connections_variable(df, light, other_variables, connections,
                                                    amount_conditionals, intervention=False)

    return connections


# Generates a DAG from a connections array (tuples)
def generate_graph_from_connections_save_plot(connections, save_name='light_switches_cf_edges', save=True, plot=True):
    # Set up a DAG from the found dependency relations
    graph = DAG(ebunch=connections)
    # for connection in connections:
    #    graph.add_edge(u=connection[0], v=connection[1])
    if save:
        np.save(save_name, graph.edges)

    if plot:
        plt.figure()
        plt.title("Counterfactual Discovery")
        nx.draw_circular(graph, with_labels=True, arrowsize=30, node_size=800, alpha=0.5)
        #plt.show()

    return graph


def main():
    filename = 'five_switches.csv'

    intervention_targets, df_switches = load_df_and_find_interventions(filename)

    connections = []
    # Set up a DAG from the found dependency relations
    connections = discovery_with_counterfactual_inference(df_switches, intervention_targets, connections)
    graph = generate_graph_from_connections_save_plot(connections)


if __name__ == '__main__':
    main()
