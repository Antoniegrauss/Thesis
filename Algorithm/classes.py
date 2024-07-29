import random
from collections import deque
from pgmpy.estimators import MaximumLikelihoodEstimator
import metrics
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pgmpy.base import DAG
from pgmpy.models import BayesianNetwork


class Robot:

    def __init__(self, name, position, room, reasoner, sensor_data=None, environment=None, coords=[0, 0]):
        self.name = name
        self.position = position
        self.room = room
        self.sensor_data = sensor_data
        self.reasoner = reasoner
        self.environment = environment
        self.coords = coords

    # Performs an action or a movement or both
    # progresses the time
    # calculates next values
    def perform_action(self, action=None, move=None):
        self.environment.perform_action(action=action, move=move)


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


# This function filters a numpy array of variables from a numpy array of filter variables
def filter_variables(variables, filter_variables):
    return_variables = []
    if variables.size > 0:
        if np.array(filter_variables).size > 0:
            for variable in variables:
                filtered = False
                for filter_variable in filter_variables:
                    if filter_variable in variable:
                        filtered = True
                if not filtered:
                    return_variables.append(variable)
    return np.array(return_variables)


class Reasoner:

    def __init__(self, causal_graph, sufficiency_sets=[], necessity_sets=[],
                 action_history=None, sensor_history=None, robot=None):
        """
        @param causal_graph: a Causal Graph object that holds the connections and confidence levels of those
        connections
        @param sufficiency_sets: a set of sets for each effect variable. Each set contains sufficient causes for
        that effect
        @param necessity_sets: a set of sets for each variable. Each set holds causes that are necessary for
        that effect
        @param action_history: 2d numpy array of all T with the actions performed by the Robot
        has a header as the first row
        @param sensor_history: 2d numpy array of all T and the sensor_data at that moment
        """
        self.last_all_try_time = 0
        self.walk_time = 0
        self.causal_graph = causal_graph
        self.sufficiency_sets = sufficiency_sets
        self.necessity_sets = necessity_sets
        self.action_history = action_history
        self.sensor_history = sensor_history
        self.last_observed_sensor_history = sensor_history
        self.robot = robot
        self.duplicates_counter = 0
        self.duplicates_threshold = 50
        self.last_success_time = 0
        self.last_bn_fit = 0
        self.rejected_connections = []
        self.unchecked_connections = []
        self.fully_determined_variables = []
        self.partially_determined_variables = []
        self.non_determined_variables = []
        self.last_checked_variable = deque()
        self.shd_score = []

    # focus discovery on those effects?
    # Controls the flow of the program
    # holds a list of unchecked connections
    # searches for new connections
    # takes a walk to a random point if not discovering new connections for x time steps
    def planner(self, ax_env, ax_ani, animate=False, plot_env=False):
        now = self.robot.environment.time
        while True:

            """# Set amount of max iterations in the loop
            if len(self.non_determined_variables) == 0 and \
                    len(self.partially_determined_variables) == 0 \
                    and self.robot.environment.time > 50:
                print("Time ", now, " fitting bn")
                self.fit_bn(now=now, print_cpds=False)
                self.supergraph_link_types()
                self.update_determined_list()"""

            if len(self.non_determined_variables) == 0 and \
                    len(self.partially_determined_variables) == 0 \
                    and len(self.fully_determined_variables) > 0:
                self.end_loop(ax_env=ax_env, ax_ani=ax_ani, plot_env=plot_env, animate=animate)
                break
            if self.robot.environment.time > 7000:
                self.end_loop(ax_env=ax_env, ax_ani=ax_ani, plot_env=plot_env, animate=animate)
                break

            # The first time step do nothing, to create the sensor history array
            if self.sensor_history is None:
                self.robot.environment.do_actions_move_time(action=None, move=None, animate=animate, ax_ani=ax_ani,
                                                            plot_env=plot_env, ax_env=ax_env)

            # Every 100'th time step update the Bayesian Network of the supergraph
            if now - self.last_bn_fit > 1:
                print("Time ", now, " last fit ", self.last_bn_fit)
                self.fit_bn(now=now, print_cpds=False)
                self.supergraph_link_types()
                self.update_determined_list()
                # self.causal_graph.print_supergraph_cpds()
                print("Time at ", now)
                print("Fully determined variables: \n", self.fully_determined_variables)
                print("Partially determined variables: \n", self.partially_determined_variables)
                print("Non determined variables: \n", self.non_determined_variables)

                # Save the shd score
                correct_graph = self.robot.environment.correct_graph()
                shd = metrics.structural_hamming_distance(correct_graph.edges, self.causal_graph.supergraph.edges)
                self.shd_score.append([now, shd])

            now = self.robot.environment.time
            intervenable_variables = self.robot.environment.pass_intervenable_variables()
            possible_actions = self.robot.environment.possible_actions()
            possible_moves = self.robot.environment.possible_moves()

            # Search for new connections after this
            if self.action_history is not None:
                new_connection_bool = self.search_new_connections(time=now, tested_actions=[],
                                                                  intervenable_variables=intervenable_variables,
                                                                  ax_env=ax_env, ax_ani=ax_ani, animate=animate,
                                                                  plot_env=plot_env)

                # If we find a new connection, save the time stamp
                if new_connection_bool:
                    self.last_success_time = self.robot.environment.time
                    self.decrement_duplicates_counter(20)

                new_connection_bool = self.review_unchecked_connections(possible_actions=possible_actions,
                                                                        possible_moves=possible_moves,
                                                                        ax_env=ax_env, ax_ani=ax_ani, animate=animate,
                                                                        plot_env=plot_env)
                # If we find a new connection, save the time stamp
                if new_connection_bool:
                    self.last_success_time = self.robot.environment.time
                    self.decrement_duplicates_counter(20)

            # Take a walk to explore if nothing interesting happens for x time steps
            if now - self.walk_time > 30 and now - self.last_success_time > 30:
                self.explore(ax_env=ax_env, plot_env=plot_env)
                possible_actions = self.robot.environment.possible_actions()
                self.robot.environment.do_actions_move_time(action=random.choice(possible_actions), ax_env=ax_env,
                                                            plot_env=plot_env)
                self.explore(ax_env=ax_env, plot_env=plot_env)
                self.walk_time = now
            # Choose an action/move by exploration or random if no new states can be visited
            else:
                possible_actions = self.robot.environment.possible_actions()
                possible_moves = self.robot.environment.possible_moves()
                action, move = self.choose_action_or_move(possible_actions=possible_actions,
                                                          possible_moves=possible_moves)
                if action is None and move is None and self.duplicates_counter > 10:
                    self.explore(ax_env=ax_env, plot_env=plot_env)
                    self.walk_time = now
                    '''elif action is None and move is None:
                    move = random.choice(possible_moves)
                    self.robot.environment.do_actions_move_time(action=action, move=move, animate=True, ax_ani=ax_ani,
                                                                plot_env=True, ax_env=ax_env)'''
                else:
                    position_cols = self.sensor_history[0, :] != "position_robot_1"
                    occurrences = self.state_in_history_occurrences(action=action, move=move, cols=position_cols[:])
                    if occurrences > 0:
                        self.duplicates_counter += 1
                        # print(f"State + action duplicates {self.duplicates_counter} (state visited {occurrences} times)")

                        # Switch search strategy if too many duplicates are encountered or
                        # there is no new connection for X time steps
                        if self.duplicates_counter > self.duplicates_threshold or now - self.last_success_time > 50:
                            if now - self.last_all_try_time > 20:
                                self.last_all_try_time = now
                                # Merge the partially and non determined variables
                                effect = None
                                effects_to_try = self.partially_determined_variables + self.non_determined_variables
                                if len(effects_to_try) > 0:
                                    for this_effect in effects_to_try:
                                        # Check whether we already tried this effect or not
                                        if this_effect not in self.last_checked_variable:
                                            effect = this_effect
                                            break
                                    if effect is None:
                                        i = 1
                                        while effect not in effects_to_try and i <= len(self.last_checked_variable):
                                            effect = self.last_checked_variable[-i]
                                            i += 1
                                        self.last_checked_variable.pop()

                                    self.last_checked_variable.appendleft(effect)
                                    self.try_all_causes_for_effect(effect_name=effect, ax_env=ax_env, plot_env=plot_env)
                                    self.decrement_duplicates_counter(decrement=50)

                    elif occurrences == 0:
                        self.decrement_duplicates_counter(decrement=1)
                # If we visit a new state, reset the duplicates_counter
                self.robot.environment.do_actions_move_time(action=action, move=move, animate=animate, ax_ani=ax_ani,
                                                            plot_env=plot_env, ax_env=ax_env)

    # TODO:filter actions if already cause or something
    # Tries all the different causes one by one for this effect
    def try_all_causes_for_effect(self, effect_name, ax_env, plot_env):
        actions = np.unique(self.action_history[1:, 1])
        actions = actions[actions != "None"]
        actions = actions[actions != "position_robot_1"]

        # Gather all the switches that the robot can flip and try them one by one
        res = 0
        for action_name in actions:
            print("Trying all actions for effect ", effect_name)
            if (action_name, effect_name) not in self.causal_graph.supergraph.edges:
                col = self.last_observed_sensor_history[0, :] == action_name
                action_value = self.last_observed_sensor_history[-1, col][0]
                action = (action_name, action_value)
                effect = (effect_name, self.get_last_observed_value_of(variable_name=effect_name))
                res = self.review_new_connection(connection_from=action, connection_to=effect, ax_env=ax_env,
                                                 ax_ani=None, plot_env=plot_env)
                if res == 1:
                    return True

        # If this all fails, switch the known cause of this effect and try again
        if res != 1:
            try:
                causes = self.causal_graph.supergraph.get_cpds(node=effect_name)
            except ValueError:
                print("Try to get causes ValueError")
                return None
            if causes is not None:
                causes = causes.variables
            else:
                return None
            if len(causes) == 2:
                cause = causes[1]
                self.backtrack_to_variable(variable_name=cause)
                for poss_action in self.robot.environment.possible_actions():
                    if poss_action[0] == cause:
                        print("Flipping cause switch ", cause, " for trying all actions for effect ", effect_name)
                        self.robot.environment.do_actions_move_time(action=poss_action, ax_env=ax_env, plot_env=plot_env)

                        for action_name in actions:
                            print("Trying all actions for effect ", effect_name)
                            if (action_name, effect_name) not in self.causal_graph.supergraph.edges:
                                col = self.last_observed_sensor_history[0, :] == action_name
                                action_value = self.last_observed_sensor_history[-1, col][0]
                                action = (action_name, action_value)
                                effect = (effect_name, self.get_last_observed_value_of(variable_name=effect_name))
                                res = self.review_new_connection(connection_from=action, connection_to=effect,
                                                                 ax_env=ax_env, ax_ani=None,
                                                                 plot_env=plot_env)
                                if res == 1:
                                    break

    # Move to a room where the effect is unobservable, try an action and move back
    def try_random_action_for_effect(self, effect, ax_env, plot_env):
        # If we are in the same room, move somewhere else
        if effect in self.robot.environment.pass_observable_variables(self.robot.room):
            self.explore(ax_env=ax_env, plot_env=plot_env)
        # Do a random action
        action = random.choice(self.robot.environment.possible_actions())
        print("Trying random action ", action[0], " for effect ", effect)
        self.robot.environment.do_actions_move_time(action=action, move=None,
                                                    ax_env=ax_env, plot_env=plot_env)
        # Observe the effect again
        self.observe_effect_with_navigation(effect_name=effect, ax_env=ax_env,
                                            plot_env=plot_env)

    # Prints all the data on screen before the loop ends
    # Also fits the BN one last time
    def end_loop(self, ax_ani, ax_env, plot_env=True, animate=True):
        self.fit_bn(now=self.robot.environment.time, print_cpds=False)
        self.supergraph_link_types()
        self.update_determined_list()
        #self.causal_graph.print_supergraph_cpds()
        print("Rejected connections: ", self.rejected_connections)
        print("Unchecked connections: ", self.unchecked_connections)
        print("Fully determined effects: ", self.fully_determined_variables)
        print("Partially determined variables: \n", self.partially_determined_variables)
        print("Non determined variables: \n", self.non_determined_variables)

        if plot_env:
            self.robot.environment.plot_environment_and_animation(block=False, ax_ani=ax_ani, ax_env=ax_env)

            fig = plt.figure()
            ax_gen = fig.add_subplot(1, 2, 1)
            self.causal_graph.draw_supergraph(ax=ax_gen)
            ax_corr = fig.add_subplot(1, 2, 2)
            plt.sca(ax_corr)
            correct_graph = self.robot.environment.correct_graph()
            shd = metrics.structural_hamming_distance(correct_graph.edges, self.causal_graph.supergraph.edges)
            plt.title(f"correct graph \n shd score (0=good 1=bad) = {shd}")

            pos = nx.shell_layout(correct_graph)
            nx.draw_networkx_labels(correct_graph, pos)
            nx.draw_networkx_nodes(correct_graph, pos, node_size=800, alpha=0.5)
            nx.draw_networkx_edges(correct_graph, pos, alpha=0.5)
            plt.show()

    # Uses the cpd values to determine whether any cause, or combination of causes,
    # is sufficient
    # returns these sufficient set(s)
    def determine_sufficiency(self, values, variables, variable_card):
        pass

    # Uses the cpd values to determine whether any combination of the causes is neccessary
    # returns a set of these values
    def determine_necessity(self, values, variables, variable_card):
        pass

    # Calls the fit_supergraph_bn method from CausalGraph
    def fit_bn(self, now, print_cpds=False):
        # Select the rows of non-intervenable variables
        data = self.sensor_history.copy()
        vars = data[0]
        intervenables = self.robot.environment.pass_intervenable_variables()
        non_intervenables = [var not in intervenables for var in vars]
        data = data[:, non_intervenables]

        # Suppose the actions are not changed since the last time we changed them
        # so using the last_observed value is fine
        action_data = self.last_observed_sensor_history[:, [var in intervenables for var in vars]]

        self.causal_graph.fit_data(data=data, actions=action_data)
        if print_cpds:
            self.causal_graph.print_supergraph_cpds()
        self.last_bn_fit = now

    # Takes the cpds from the Bayesian Network and checks the link types (necc, suff, both or neither)
    # TODO: store the link types and apply them in search
    def supergraph_link_types(self):
        cpds = self.causal_graph.supergraph.cpds
        for cpd in cpds:
            # If the effect is fully determined add it to the list for that
            if len(cpd.variables) > 1:
                necc, suff = self.determine_link_type(cpd)
                result = np.array(["Necc", "Suff"])[necc, suff]
                #print("Link type of variable ", cpd.variables[0], " causes ", cpd.variables[1:], " = ", result)
                # Update the link confidence in the graph if N\&S type link
                for cause in cpd.variables[1:]:
                    if len(result) > 0:
                        factor = len(result[0]) / 2.
                    else:
                        factor = 0.1
                    self.causal_graph.update_confidence(connection_to=cpd.variables[0], connection_from=cause,
                                                        supergraph=True, factor=factor)

                if len(result) > 0 and len(result[0]) == 2:
                    if cpd.variables[0] not in self.fully_determined_variables:
                        self.fully_determined_variables.append(cpd.variables[0])
                        # print("Added effect ", cpd.variables[0], " to fully determined list")
                    if cpd.variables[0] in self.partially_determined_variables:
                        self.partially_determined_variables.remove(cpd.variables[0])
                    if cpd.variables[0] in self.non_determined_variables:
                        self.non_determined_variables.remove(cpd.variables[0])

                # Else if the variable is already in the list, but not fully determined
                # remove it from the list
                elif len(result) == 0 or len(result[0]) == 1:
                    if cpd.variables[0] in self.fully_determined_variables:
                        self.fully_determined_variables.remove(cpd.variables[0])
                    if cpd.variables[0] in self.non_determined_variables:
                        self.non_determined_variables.remove(cpd.variables[0])
                    if cpd.variables[0] not in self.partially_determined_variables:
                        self.partially_determined_variables.append(cpd.variables[0])
                        # print("Removed effect ", cpd.variables[0], " from fully determined list")

    # Removes the effects from non_determined_variables that are in either
    # self.fully_determined or self.partially_determined
    def update_determined_list(self):
        intervenable_variables = self.robot.environment.pass_intervenable_variables()
        all_variables = np.array([variable.name for variable in
                         self.robot.environment.pass_observable_variables(room=self.robot.room)])
        all_non_action_variables = filter_variables(variables=all_variables,
                                                    filter_variables=intervenable_variables)
        for effect in all_non_action_variables:
            if effect not in self.non_determined_variables and \
                    effect not in self.partially_determined_variables and \
                    effect not in self.fully_determined_variables:
                self.non_determined_variables.append(effect)

        for effect in self.non_determined_variables:
            if effect in self.fully_determined_variables or effect in self.partially_determined_variables:
                self.non_determined_variables.remove(effect)

        self.causal_graph.fully_determined_list = self.fully_determined_variables

    # Determines the causal link type given a cpd (numpy array
    def determine_link_type(self, cpd):
        necc = True
        suff = True

        cols = []
        values = cpd.values
        if len(cpd.variables) == 2:
            if values.shape[0] == 3:
                values = values[:-1, :]

        elif len(cpd.variables) == 3:
            rows = cpd.variable_card
            values = values.reshape(rows, -1)
            if rows == 3:
                values = values[:-1, :]

        # Remove cols with only unobserved
        cols = []
        for row in np.transpose(values):
            if np.all(row == row[0]):
                cols.append(False)
            else:
                cols.append(True)
        values = values[:, cols]

        # Values should equal 0.0, but floating points so use some margin
        for row in np.transpose(values):
            if not np.any(row == 0.):
                suff = False
                break
        for row in values:
            if not np.any(row == 0.):
                necc = False
                break
        """f necc != suff or not necc and not suff:
            if len(cpd.variables) == 3:
                print("Partial or non determined cpd ")
                print(cpd)
                print("Rearranged values: ", values)
                if "light_9" in cpd.variables:
                    print("9 in vars")"""

        return necc, suff

    # Lowers the duplicates counter
    def decrement_duplicates_counter(self, decrement=10):
        self.duplicates_counter -= decrement
        if self.duplicates_counter < 0:
            self.duplicates_counter = 0

    # Uses the off_and_on_again on a connection that is in self.unchecked_connections
    # action must be reachable from current position
    def review_unchecked_connections(self, possible_actions, possible_moves, ax_ani, ax_env, animate, plot_env):
        new_list = []
        return_bool = False
        for connection in self.unchecked_connections:
            # filter the actions that are already rejected or in the graph
            if connection not in self.rejected_connections and \
                    connection not in self.causal_graph.graph.edges:
                if connection[1][0] in self.fully_determined_variables:
                    self.unchecked_connections.remove(connection)
                    continue
                # Check if we can check the action right now
                if connection[0][0] in possible_actions or connection[0][0] in possible_moves:
                    print("Reviewing unchecked connection: ", connection)
                    prev_value = self.last_different_value(cause=connection[0])
                    off_and_on_again = self.off_and_on_again(action=connection[0], effect=connection[1],
                                                             prev_value=prev_value,
                                                             ax_env=ax_env, ax_ani=ax_ani, animate=animate,
                                                             plot_env=plot_env)
                    # If off_and_on_again is true, add the connection to graph
                    if off_and_on_again == 1:
                        self.causal_graph.add_connection(connection[0], connection[1])
                        self.causal_graph.update_confidence(connection_from=connection[0],
                                                            connection_to=connection[1],
                                                            factor=off_and_on_again)
                        return_bool = True
                        # If off_and_on_again is false, add connection to the rejected connections
                        """elif off_and_on_again == -1:
                            if connection not in self.rejected_connections:
                                self.rejected_connections.append(connection)"""
                    else:
                        if connection not in new_list:
                            new_list.append(connection)
                else:
                    if connection not in new_list:
                        new_list.append(connection)
        self.unchecked_connections = new_list
        return return_bool

    # Checks whether a certain state, action pair was already in history
    # Using the variables in cols, option to filter variables from the search
    # returns the amount of occurrences
    # returns -1 if the new_state has wrong dimensions
    def state_in_history_occurrences(self, action=None, move=None, cols=None):
        if self.sensor_history is not None and self.action_history is not None:

            if cols is None:
                cols = np.full(True, self.sensor_history.shape[0])
            new_state = [self.get_last_observed_value_of(index) for index in self.sensor_history[0, cols]]
            if action is not None:
                new_action = action
            elif move is not None:
                new_action = ["position_robot_1", move]
            else:
                new_action = "None"

            iterator = zip(self.last_observed_sensor_history[:, cols], 
                           self.action_history[1:, :])

            occurrences = 0
            for row, action_row in iterator:
                # Evaluate without the first column (time is unique per row)
                eval_row = np.all(row[1:] == new_state[1:])
                eval_action = np.all(action_row[1:] == np.array(new_action, dtype=object))
                if eval_row and eval_action:
                    occurrences += 1
                    # return occurrences
            return occurrences
        return -1

    # Returns the last observed value of a variable that is different from the current value
    def last_different_value(self, cause):
        # Get the last different observed value of the effect
        if self.sensor_history.size > 1 and not "None" in cause:
            col = self.sensor_history[0, :] == cause[0]
            prev_values = self.sensor_history[1:, col]
            prev_value = prev_values[-2][0]
            i = -2
            while prev_value == cause[1] or prev_value == str(-1):
                i -= 1
                prev_value = prev_values[i][0]
        else:
            prev_value = None

        return prev_value

    # Takes a new connection and applies heuristics to increase or decrease the confidence
    # According to the result, can add the causes to the sufficiency set or necessity set of the effect
    def review_new_connection(self, connection_from, connection_to, ax_env, ax_ani, animate=False, plot_env=False):
        print("Reviewing connection ", connection_from, connection_to)
        cause = connection_from
        effect = connection_to

        prev_value = self.last_different_value(cause=cause)
        if prev_value == cause[1]:
            print("Previous value same")
            return 0

        off_and_on_again = self.off_and_on_again(cause, effect, prev_value=prev_value,
                                                 ax_env=ax_env, ax_ani=ax_ani, animate=animate,
                                                 plot_env=plot_env)
        print("Off and on again  = ", off_and_on_again)
        """if off_and_on_again == -1:
            if (cause, effect) not in self.rejected_connections:
                self.rejected_connections.append((cause, effect))"""
        # If the value is 1, seems like the cause is sufficient
        # together with all the other actions
        if off_and_on_again == 1:
            # Add the connection to the graph
            self.causal_graph.add_connection(connection_from, connection_to)
            self.causal_graph.update_confidence(connection_from=connection_from, connection_to=connection_to,
                                                factor=off_and_on_again)
            if animate:
                self.robot.environment.plot_animated_graph(ax_ani=ax_ani)

        elif off_and_on_again == 0:
            if (cause, effect) not in self.unchecked_connections:
                self.unchecked_connections.append((cause, effect))

        return off_and_on_again

    # Tries to refute a sufficiency set
    # Consisting of cause 1.. cause n
    def refute_sufficiency_set(self):
        pass

    # Tries to perform an action
    # if the action is not in the current room, try navigation to the correct room
    def do_action_with_navigation(self, action_name, action_value, ax_env, time=None, plot_env=False):
        # Check if we can perform the action
        possible_actions = self.robot.environment.possible_actions(self.robot.position)
        new_action = [action_name, action_value]
        for poss_action in possible_actions:
            if np.all(new_action == poss_action):
                self.robot.environment.do_actions_move_time(action=new_action, plot_env=plot_env, ax_env=ax_env)
                return None

        if action_name != "position_robot_1":
            # print("Driving back to reach action ", action)
            moves = self.backtrack_to_variable(action_name)
            if moves is not None:
                # Aggregate moves
                for x, move in enumerate(moves):
                    if move == moves[0]:
                        goto = (self.robot.position.name, move)
                    else:
                        goto = (moves[x - 1], move)
                    self.robot.environment.do_actions_move_time(action=None, move=goto, plot_env=plot_env,
                                                                ax_env=ax_env)
                possible_actions = self.robot.environment.possible_actions(self.robot.position)
                if new_action[0] in possible_actions:
                    self.robot.environment.do_actions_move_time(action=new_action, plot_env=plot_env, ax_env=ax_env)

        elif action_name == "position_robot_1":
            connections = self.robot.position.connections
            if action_value[1] in connections:
                self.robot.environment.do_actions_move_time(move=action_value, plot_env=plot_env, ax_env=ax_env)
            # If there is no direct connection, use backtracking
            else:
                room = self.robot.environment.waypoint_in_which_room(action_value[1])
                waypoint = self.robot.environment.get_waypoint_from_name(waypoint_name=action_value[1])
                moves = self.robot.environment.navigate_from_wp_to_room(wp_from=waypoint, room_to=room)
                for x, move in enumerate(moves):
                    if move == moves[0]:
                        goto = (self.robot.position.name, move)
                    else:
                        goto = (moves[x - 1], move)
                    self.robot.environment.do_actions_move_time(move=goto, plot_env=plot_env, ax_env=ax_env)
                if action_value in connections:
                    self.robot.environment.do_actions_move_time(move=action_value, plot_env=plot_env, ax_env=ax_env)

    # Uses the action history to plan a path back to when a variable was reachable
    # Returns None if path is unreachable or variable is already reachable
    def backtrack_to_variable(self, variable_name):
        # Find out where we need to go
        # which is when the variable was still visible
        i = 1
        var = self.sensor_history[-i, self.get_variable_index(variable_name=variable_name)]
        while str(var) == str(-1):
            i += 1
            var = self.sensor_history[-i, self.get_variable_index(variable_name=variable_name)]

        if i == 1:
            return None

        positions = self.sensor_history[-i:, self.get_variable_index(variable_name="position_robot_1")]
        path = [positions[0]]

        # Remove the duplicates, e.g. the robot standing still
        for position in positions:
            if position != path[-1]:
                path.append(position)
        path = [move for move in reversed(path)]
        path = path[1:]

        return self.remove_loops(path)

    # Removes the loops in a path
    def remove_loops(self, path):
        # Remove bigger loops (no loops if set is as long as path)
        if len(set(path)) == len(path):
            return path
        else:
            unique_moves = []
            new_path = path
            for x, move in enumerate(path):
                if move not in unique_moves:
                    unique_moves.append(move)
                elif move in unique_moves:
                    loop_start = path.index(move)
                    loop_end = x
                    return self.remove_loops(new_path[:loop_start] + new_path[loop_end:])
            # Reverse the path and remove the first move (that's the robot position)
            return path

    # Chooses a random variable that is not visible and moves towards it
    def explore(self, ax_env, plot_env=False):
        last_data = self.sensor_history[-1, 1:]
        variables_not_visible = self.sensor_history[0, 1:][last_data == str(-1)]
        if len(variables_not_visible) > 1:
            variable_name = random.choice(variables_not_visible)
            _ = self.observe_effect_with_navigation(effect_name=variable_name, ax_env=ax_env, plot_env=plot_env)
            print("Taking a walk to ",
                  self.robot.environment.variable_in_which_room(variable_name=variable_name).name)
        else:
            print("No unobserved variables to walk to, taking random move")
            move = random.choice(self.robot.environment.possible_moves())
            self.robot.environment.do_actions_move_time(action=None, move=move, ax_env=ax_env, plot_env=plot_env)

    # Tries to observe an effect
    # with navigation if the effect is only observable in another room
    def observe_effect_with_navigation(self, effect_name, ax_env, plot_env=False):
        # Check if we can observe the variable, if not, drive back to the last waypoint to see its value
        observable_variables = self.robot.environment.pass_observable_variables(self.robot.room)
        observable_variable_names = [observable_variable.name for observable_variable in observable_variables] + [
            "Time"]

        if effect_name not in observable_variable_names:
            # print("Driving back to observe variable ", effect)
            moves = self.backtrack_to_variable(effect_name)
            for x, move in enumerate(moves):
                if move == moves[0]:
                    goto = (self.robot.position.name, move)
                else:
                    goto = (moves[x - 1], move)
                self.robot.environment.do_actions_move_time(action=None, move=goto, plot_env=plot_env,
                                                            ax_env=ax_env)
            return self.get_last_observed_value_of(effect_name)
        return observable_variables[observable_variable_names == effect_name].value

    # Heuristic #1
    # Investigates whether an effect follows the state of an action
    # by resetting and then repeating the action
    # Returns confidence_change, 0 if heuristic failed, 1 if confidence rises, -1 if confidence falls
    def off_and_on_again(self, action, effect, prev_value, ax_env=None, ax_ani=None, animate=False, plot_env=False):
        confidence_change = 0
        time = self.robot.environment.time - 1

        # If the action was a movement, swap the movement for prev_value
        # Prev value is used to "repeat" the action
        action_is_pos = False
        if action[0] == "position_robot_1":
            prev_value = (action[1][1], action[1][0])
            action_is_pos = True
        action_value_0 = action[1]
        effect_value_0 = effect[1]

        try:
            self.do_action_with_navigation(action_name=action[0], action_value=prev_value, time=time - 1,
                                           ax_env=ax_env, plot_env=plot_env)
            action_value_1 = self.get_action_at_t(self.robot.environment.time - 1)[1]
            self.observe_effect_with_navigation(effect_name=effect[0], ax_env=ax_env, plot_env=plot_env)
            effect_value_1 = self.get_last_observed_value_of(effect[0])

            self.do_action_with_navigation(action_name=action[0], action_value=action[1], time=time,
                                           ax_env=ax_env, plot_env=plot_env)
            action_value_2 = self.get_action_at_t(self.robot.environment.time - 1)[1]
            self.observe_effect_with_navigation(effect_name=effect[0], ax_env=ax_env, plot_env=plot_env)
            effect_value_2 = self.get_last_observed_value_of(effect[0])

        except TypeError:
            print("TypeError in off and on again")
            return 0

        # Check whether the action variable changed and changed back
        # An unobservable variable does not count as changed
        if action_is_pos:
            if np.all(action_value_0 == action_value_2) and np.all(action_value_0 != action_value_1):
                if effect_value_0 != effect_value_1 and effect_value_0 == effect_value_2:
                    confidence_change += 1
                else:
                    confidence_change -= 1
            else:
                print(f"Failed changing action {action[0]} off and on again")
        else:
            if action_value_0 == action_value_2 and action_value_0 != action_value_1:
                if effect_value_0 != effect_value_1 and effect_value_0 == effect_value_2:
                    confidence_change += 1
                else:
                    confidence_change -= 1
            else:
                print(f"Failed changing action {action[0]} off and on again")

        return confidence_change

    # Looks for new values in the incoming sensor data
    # naively assume the previous action was responsible
    # Check whether that action was responsible
    # if not, look further back in the past, up to a threshold
    def search_new_connections(self, time, intervenable_variables, ax_env, ax_ani, variable=None,
                               tested_actions=[], animate=False, plot_env=False):
        # The amount of repeats that we allow
        if self.action_history is not None:
            threshold = len(set(tuple(row) for row in self.action_history[1:, 1]))
        else:
            threshold = 1
        if len(tested_actions) < threshold:
            connection = None

            # Get the variable_names of the actions we just took
            # Make sure we step over the None actions
            action = self.get_action_at_t(time=time)
            while action is None:
                time -= 1
                action = self.get_action_at_t(time=time)

            # Check whether the action is already in the tested actions
            in_tested = False
            for tested in tested_actions:
                if action[0] == tested[0]:
                    in_tested = True
                    break

            count = 0
            while in_tested:
                time -= 1
                action = self.get_action_at_t(time=time)
                count += 1
                while action is None:
                    time -= 1
                    action = self.get_action_at_t(time=time)
                    count += 1
                    if count > 100:
                        count = 0
                        break

                # Check again if the new action is in the already tested ones
                in_tested = False
                for tested in tested_actions:
                    if action[0] == tested[0]:
                        in_tested = True
                        break
                if count > 100:
                    count = 0
                    break

            if variable is None:
                # Get the variables that just changed value
                variables = self.search_changed_values()
                variables = filter_variables(variables=variables,
                                             filter_variables=intervenable_variables)
                if len(self.fully_determined_variables) > 0:
                    variables = filter_variables(variables=variables,
                                                 filter_variables=self.fully_determined_variables)
                # Check whether the variables and actions are not empty
                if variables.size > 0:
                    if action is not None:
                        if type(action) != list:
                            action = action.tolist()
                        for variable_name in variables:
                            connection_tuple = ((action[0], action[1]), (variable_name[0], variable_name[1]))
                            # If this edge is already in the graph, assume this was the cause
                            try:
                                if connection_tuple in self.causal_graph.graph.edges:
                                    if variable_name[0] in self.non_determined_variables:
                                        self.non_determined_variables.remove(variable_name[0])
                                    break

                                if connection_tuple not in self.rejected_connections \
                                        and connection_tuple not in self.unchecked_connections:
                                    # Create a new connection for each action
                                    self.unchecked_connections.append(connection_tuple)
                                    connection = connection_tuple

                                if variable_name[0] not in self.causal_graph.supergraph.nodes and\
                                        variable_name[0] not in self.non_determined_variables:
                                    self.non_determined_variables.append(variable_name[0])
                            except TypeError:
                                print("Search new connections typeerror")

            elif variable is not None:
                connection = ((action[0], action[1]), (variable[0], variable[1]))

            if connection is not None:
                if connection in self.unchecked_connections:
                    self.unchecked_connections.remove(connection)
                result = 0
                # Break the loop if we come upon an edge that is already in the graph
                if connection in self.causal_graph.graph.edges:
                    return False
                    #if connection not in self.rejected_connections:
                result = self.review_new_connection(connection[0], connection[1], ax_env=ax_env,
                                                        ax_ani=ax_ani, plot_env=plot_env,
                                                        animate=animate)
                if result != 1:
                    # If this connection did not cause the effect, look an action further back into the past
                    #time -= 1
                    """if connection not in self.rejected_connections:
                        self.rejected_connections.append(connection)"""
                    tested_actions.append(connection[0])
                    print(f"Repeating off and on again, threshold = {threshold}, time = {time}")
                    print(f"Actions already tested = \n {tested_actions} \n")
                    return self.search_new_connections(time=time, intervenable_variables=intervenable_variables,
                                                       ax_env=ax_env,
                                                       variable=connection[1], ax_ani=ax_ani,
                                                       tested_actions=tested_actions, animate=animate, plot_env=plot_env)
                if result == 1:
                    return True
        return False

    # Repeats the last move action that was not None
    def drive_back_to_waypoint_at_time(self, time, ax_env=None, plot_env=False):
        # Repeat the last Move action
        row = self.action_history[:, 0] == time
        move = self.action_history[row, 2]
        possible_moves = self.robot.environment.possible_moves()
        if move in possible_moves:
            self.robot.environment.do_actions_move_time(move=move, ax_env=ax_env, plot_env=plot_env)

        return move

    # Returns the actions from time=t
    def get_action_at_t(self, time):
        if self.action_history is not None:
            for row in self.action_history:
                if str(row[0]) == str(time):
                    return row[1:]
        return None

    # Checks the last row of sensor_history for new values
    def search_new_values(self):
        return_values = []
        # Search in sensor history without the time column
        for row in self.sensor_history[:, 1:].transpose():
            # Only check if the value is -1 (fill value)
            # Needs to be a new value after the first seen value
            if str(row[-1]) != str(-1) and \
                    np.unique(row[str(row) != str(-1)]).size > 2:
                # If the value is new (not in history), save the variable name
                if row[-1] not in row[:-1]:
                    return_values.append(row[0])

        # if return_values:
        # print("New values found for: ", return_values)
        return np.array(return_values)

    # Checks for each variable in sensor_data
    # whether the last observed value was different
    def search_changed_values(self):
        # Check whether the variables has at least 2 different values from the sensor data
        variables = []
        # Give the data without time
        if self.sensor_history is not None:
            data = self.sensor_history[:, 1:]
            for row in data.transpose():
                if str(row[-1]) != str(-1):
                    i = 2

                    while (row.size > i and str(row[-i]) == str(-1)):
                        i += 1
                    if row[-1] != row[-i] and i != row.size:
                        variables.append([row[0], row[-1]])

            return np.array(variables)
        return None

    # Returns the column in the sensor history of a certain variable
    # Returns -1 if not found
    def get_variable_index(self, variable_name, df=None):
        if df is None:
            df = self.sensor_history
        for x, variable in enumerate(df[0]):
            if variable == variable_name:
                return x
        return -1

    # Returns the last observed value of a variable from the sensor history
    def get_last_observed_value_of(self, variable_name, df=None):
        # Filter the -1 values and return the last element
        if df is None:
            df = self.sensor_history
        col = df[:, self.get_variable_index(variable_name, df=df)]
        filter = col != str(-1)
        col = col[filter]
        if col.size > 0:
            return col[-1]
        else:
            return None

    # Called to select an action or move when there is no current task
    # Returns: action, move (one of the two is None, the other a valid action/move
    def choose_action_or_move(self, possible_actions=[], possible_moves=[]):
        action = self.choose_action(possible_actions=possible_actions)
        move = self.choose_move(possible_moves=possible_moves)

        # If one of them is None, return the pair
        if move is not None and action is None or \
                move is None and action is not None:
            return action, move
        # If both are not none, return None and one of them
        elif action is not None and move is not None:
            return None, move
            # return random.choice([(action, None), (None, move)])
        # If both are None, return a random choice of either one or None, None
        if len(possible_moves) > 0:
            random_move = random.choice(possible_moves[1:])
            random_move = tuple(random_move)
            return random.choice([(random.choice(possible_actions[1:]), None), (None, random_move)])
        return random.choice(possible_actions[1:]), None

    # Uses exploration to find next action
    def choose_action(self, possible_actions, duplicates_threshold=1):
        if self.action_history is not None:
            # Action 0 is None
            random.shuffle(possible_actions)
            for action in possible_actions:
                occurrence = 0
                for prev_action in self.action_history[-10:, 1:]:
                    if prev_action[0] == action[0]:
                        occurrence += 1
                    if occurrence > 0:
                        break
                if occurrence == 0:
                    return action
        return None

    # Uses exploration to find next move
    def choose_move(self, possible_moves, threshold=20):
        # Move 0 is None
        possible_moves = possible_moves[1:]
        random.shuffle(possible_moves)
        if self.action_history is not None:

            for move in possible_moves:
                new = True
                for prev_move in self.action_history[1:, -1]:
                    if move[1] in prev_move:
                        new = False
                        break
                if new:
                    return tuple(move)

            for move in possible_moves[1:]:
                new = True
                for prev_move in self.action_history[-threshold:, -1]:
                    if move[1] in prev_move:
                        new = False
                        break
                if new:
                    return tuple(move)

        return None

    # Adds an action to the action history, with time T
    def append_action_history(self, action, variable_name, time):
        if self.action_history is None:
            self.action_history = np.array([["time", "variable", "value"]])
        self.action_history = np.vstack((self.action_history, [time, action, variable_name]))

    # Combines a new set and the sufficiency set of "variable"
    # add the set if it is not a superset of one already in the set
    # new_set is added as a tuple
    def combine_sufficiency_sets(self, variable, new_set):
        index = self.get_variable_index(variable_name=variable)
        set_of_sets = self.sufficiency_sets[index]

        # Check for empty new set
        if new_set == ():
            pass

        # Check for empty set
        if set_of_sets == {}:
            self.sufficiency_sets[index] = {new_set}

        subset = superset_in_set_of_sets(set_of_sets, new_set)
        if subset != {}:
            # If there is already a subset
            self.sufficiency_sets[index] = set_of_sets

        superset = subset_in_set_of_sets(set_of_sets, new_set)
        if superset != {}:
            # If the new set is a subset of one in the list
            # swap the new set with that entry
            return_set = {new_set}
            for x in set_of_sets:
                return_set.update([x])
            return_set.remove(superset)
            self.sufficiency_sets[index] = return_set

        # If none of the sets are a subset (or the same) as the new set,
        # append it to the list
        return_set = {new_set}
        for x in set_of_sets:
            return_set.update([x])
        # print("added set to the list ", return_set)
        self.sufficiency_sets[index] = return_set

    # Merge dataframes of new sensor data and sensor history
    def add_new_data(self, new_sensor_data):
        """
        @param new_sensor_data: 2 rows, row 1 holds the variable names, row 2 holds the values on time=T
        """
        if self.sensor_history is None:
            self.sensor_history = new_sensor_data
            self.last_observed_sensor_history = new_sensor_data

        else:
            # Stack an empty row below the sensor history
            self.sensor_history = np.vstack((self.sensor_history, np.full([1, self.sensor_history.shape[1]], -1)))

            # Check for the new sensor data if the index already exists -> add value
            # if the index does not exist, create a new column with None and add the value at the bottom
            new_values = []
            for variable, value in zip(new_sensor_data[0, :], new_sensor_data[1, :]):
                index = np.where(self.sensor_history[0, :] == variable)
                if index[0].size > 0:
                    self.sensor_history[-1, index] = value
                elif index[0].size == 0:
                    new_values.append([variable, value])
                    # print(f"New sensor variable found with name {variable}")

            # If we have new values, add empty columns to the right and add the new variabla_name and value
            if len(new_values) > 0:
                for new_variable in new_values:
                    # Add new columns
                    self.sensor_history = np.hstack((self.sensor_history,
                                                     np.full((self.sensor_history.shape[0], 1), -1)))
                    self.last_observed_sensor_history = np.hstack((self.last_observed_sensor_history,
                                                                   np.full(
                                                                       (self.last_observed_sensor_history.shape[0], 1),
                                                                       -1)))
                    # Set index
                    self.sensor_history[0, -1] = new_variable[0]
                    self.last_observed_sensor_history[0, -1] = new_variable[0]
                    # Set value
                    self.sensor_history[-1, -1] = new_variable[1]

            # Add the last observed values to the last_observed_sensor_history
            new_row = np.array([self.get_last_observed_value_of(index, df=self.sensor_history) for index in
                                self.sensor_history[0, :]])
            self.last_observed_sensor_history = np.vstack((self.last_observed_sensor_history, new_row))

        # Make sure we add sufficiency sets and necessity sets for the new variables
        while self.sensor_history.shape[1] > len(self.sufficiency_sets):
            self.sufficiency_sets.append({})
        while self.sensor_history.shape[1] > len(self.necessity_sets):
            self.necessity_sets.append({})

class Variable:

    def __init__(self, name, flag, possible_values=None, value=0, intervenable=False, stationary=True,
                 cause_list=None, transition_operator_array=None, global_visibility=False, coords=None):
        """
        @param name: name
        @param flag: type of variable, light switch, light bulb etc..
        @param value: discrete value of the variable, 0 = off, 1 = on
        @param possible_values: list of values that the variable can take
        @param intervenable: can a Robot intervene upon this variable, 0 = no, 1 = yes
        @param stationary: does the value change over time if unaffected by the robot? 0 = no, 1 = yes
        @param cause_list: contains a list of all the causes (Variable) of this variable.
        None if this variable has none
        @param transition_operator_array: nested array of keywords, cause indices and result
        to calculate next value from causes.
        # index 0 options: ["not"], ["not", "and], ["not, "or"], ["and"], ["or"]
        Example: variable = [["and"], [1, 2], [On, Off]] -> if causes[1] and causes[2] -> On, if not ... -> Off
        If result is None, continue to the next row (used to check multiple conditions)
        None if cause_list is None
        @param global_visibility: if the variable is always visible, set to true
        """
        if possible_values is None:
            possible_values = [0, 1]
        self.name = name
        self.flag = flag
        self.value = value
        self.possible_values = possible_values
        self.intervenable = intervenable
        self.necessity_set = {}
        self.sufficiency_set = {}
        self.stationary = stationary
        self.cause_list = cause_list
        self.transition_keyword_array = transition_operator_array
        self.global_visibility = global_visibility
        self.coords = coords

    def change_value(self, new_value):
        self.value = new_value

    # Returns a list of possible actions
    def get_possible_actions(self):
        if self.intervenable:
            actions = []
            for possible_value in self.possible_values:
                if str(possible_value) != str(self.value):
                    actions.append(possible_value)
            return actions
        else:
            return None

    # Joins the causes and self.transition_keywords to get the next value
    def calculate_next_value(self, causes=None):
        """
        @param causes: the values of causes from the cause-list
        @return: the next value of this variable, calculated according to its transition_keyword_array
        """
        if self.cause_list is None:
            return self.value

        res = None
        word = None
        # Parse the expression from the transition functions one by one
        # until we get an answer that is not None
        for transition_function in self.transition_keyword_array:
            expression = ""
            no = False

            # we assume that we get one of these options:
            # ["not"], ["not", "and], ["not, "or"], ["and"], ["or"]
            if "not" in transition_function[0]:
                expression += "not ("
                no = True
            if "and" in transition_function[0]:
                word = "and"
            if "or" in transition_function[0]:
                word = "or"
            if "!=" in transition_function[0] or "xor" in transition_function[0]:
                word = "!="

            # If there is only 1 cause add it to the expression
            if len(transition_function[1]) == 1:
                expression += str(bool(int(causes[transition_function[1][0]])))
            # If we have multiple causes assume word is "and" or "or"
            elif len(transition_function[1]) > 1:
                if word is not None:
                    for cause_index in transition_function[1]:
                        expression += str(causes[cause_index]) + " " + word + " "
                    # Remove the last keyword from the expression
                    expression = expression[:-4]
            if no:
                expression += ")"

            # See what the result is after evaluating the expression
            res = eval(expression)
            res = transition_function[2][res]
            # If we get None, evaluate the next expression
            if res is not None:
                return res


class Waypoint:

    def __init__(self, name, flag, connections=[], coords=[0, 0]):
        """
        @param name: name
        @param flag: type of waypoint
        @param connections: list of waypoint names that this waypoint is connected to
        """
        self.name = name
        self.flag = flag
        self.connections = connections
        self.coords = coords

    # Adds a connection to waypoint_name
    def add_connection(self, waypoint_name):
        self.connections.append(waypoint_name)


# Class that holds the individual edges/nodes in a DAG
# Also holds the supergraph in a Bayesian Network
# this BN can be updated with the sensordata
class CausalGraph:

    def __init__(self, connection_list=[], graph=DAG(), supergraph=BayesianNetwork()):
        self.connection_list = connection_list
        self.graph = graph
        self.supergraph = supergraph
        self.fully_determined_list = []

    # Fits the sensor data to the supergraph BN
    def fit_data(self, data, actions):
        # Change the -1 values to np.NAN, for the pgmpy fit
        data[data == str(-1)] = np.NAN

        # Assume action data is the first observed value before encountering the variable
        actions_transpose = np.transpose(actions)
        for row in actions_transpose[1:]:
            if str(-1) in row:
                row[row == str(-1)] = row[row != str(-1)][1]

        actions = np.transpose(actions_transpose)
        #actions[actions == str(-1)] = np.NAN
        save_data = data

        # Data only contains the variables that can not be intervened upon (non-action variables)
        # Df can only contain columns that are nodes in the graph
        nodes = self.supergraph.nodes
        cols = data[0, :]
        df_cols = []
        action_cols = actions[0, :]
        df_action_cols = []
        for col in cols:
            df_cols.append(col in nodes)
        for action_col in action_cols:
            df_action_cols.append(action_col in nodes)
        data = data[:, df_cols]
        actions = actions[:, df_action_cols]

        # If we have the categorical "room_robot_1"
        # transform to integer values
        if "room_robot_1" in data[0, :]:
            rooms = np.unique(data[1:, data[0, :] == "room_robot_1"])
            for x, room in enumerate(rooms):
                data[data == room] = x

        # Same goes for "position_robot_1" in the actions
        if "position_robot_1" in actions[0, :]:
            for index, el in np.ndenumerate(actions):
                if "waypoint_" in el:
                    actions[index] = el[9:]

        # Merge the two numpy arrays
        merged_data = np.hstack((data, actions))

        # Create df and fit
        df = pd.DataFrame(data=merged_data[1:, :], columns=merged_data[0, :])
        #estimator = BayesianEstimator(model=self.supergraph, data=df)
        try:
            self.supergraph.fit(data=df, estimator=MaximumLikelihoodEstimator, complete_samples_only=True)
        except ValueError:
            print("ValueError in fitting BN")

    # Prints the conditional probability distributions (cpds) of the supergraph
    def print_supergraph_cpds(self):
        for cpd in self.supergraph.cpds:
            print(cpd)

    # Creates the supergraph by grouping nodes in self.graph by name
    def draw_supergraph(self, ax=None, figure=2, score=None):
        if ax is not None:
            plt.sca(ax)
        else:
            plt.figure(figure)

        title = "Generated graph"
        if score is not None:
            title = title + " \n shd score: " + str(score)
        plt.title(title)
        pos = nx.shell_layout(self.supergraph)
        nx.draw_networkx_labels(self.supergraph, pos)
        nx.draw_networkx_nodes(self.supergraph, pos, node_size=800, alpha=0.5)
        for edge in self.supergraph.edges:
            if edge[1] in self.fully_determined_list:
                weight = 5
            else:
                weight = 1
            nx.draw_networkx_edges(self.supergraph, pos, alpha=0.5, edgelist=[(edge[0], edge[1])],
                                   width=weight)

    # The graph checks the following things:
    #   the connection is new
    #   the connection is not from and to itself
    def add_connection(self, connection_from, connection_to, confidence=0.1):
        self.graph.add_edge(connection_from, connection_to, weight=confidence)
        self.supergraph.add_edge(connection_from[0], connection_to[0])

    # Updates the confidence 0 .. 1
    # Increase confidence with factor = 1
    # Decrease confidence with factor = -1
    # Increase confidence slower with higher confidence
    def update_confidence(self, connection_from, connection_to, factor, c=2.0, supergraph=False):
        if supergraph:
            self.supergraph[connection_from][connection_to]['weight'] = factor
        else:
            weight = self.graph[connection_from][connection_to]['weight']
            if factor == 1:
                self.graph[connection_from][connection_to]['weight'] = weight ** (1. / c)
            elif factor == -1:
                self.graph[connection_from][connection_to]['weight'] = weight ** c
            else:
                print("Invalid update factor")

    # prints all connections
    def print_all_connections(self):
        print(self.graph.edges)

    # Returns a numpy array with the columns edge_from, edge_to, confidence
    def numpy_array_connections(self):
        return_array = np.array(["From", "To", "Conf"])
        for edge in self.graph.edges:
            return_array = np.vstack((return_array, [edge[0], edge[1], self.graph[edge[0]][edge[1]]['weight']]))

        return np.array(return_array)

    # Returns the confidence * 10 as weights for plotting
    # gives a list of weights, 1 for each connection in self.graph
    def get_weights(self):
        weights = [self.graph[edge[0]][edge[1]]['weight'] for edge in self.graph.edges]
        # weights_int = [round(weight * 10) for weight in weights]
        # print(weights_int)
        return weights

    # Generates a DAG object
    # and plots it
    def plot_graph(self, show=False, ax=None, threshold=0.1, score=None):
        plt.sca(ax)
        title = "Found links"
        plt.title(title)
        pos = nx.shell_layout(self.graph)
        nx.draw_networkx_nodes(self.graph, pos, node_size=800, alpha=0.5)
        nx.draw_networkx_labels(self.graph, pos)

        for edge in self.graph.edges:
            nx.draw_networkx_edges(self.graph, pos, edgelist=[(edge[0], edge[1])],
                                   width=(self.graph[edge[0]][edge[1]]['weight'] * 10), alpha=0.5)
        if show:
            plt.show()


class Room:

    def __init__(self, name, waypoint_list=[], variable_list=[], origin_width_height=[0., 0., 0., 0.]):
        """
        @param name: name
        @param waypoint_list: list of waypoints contained in this room
        @param variable_list: list of variables that can be observed / intervened upon in this room
        """
        self.name = name
        self.waypoint_list = waypoint_list
        self.variable_list = variable_list
        self.origin_width_height = origin_width_height

    # Returns whether a waypoint is in this room
    def waypoint_in_room(self, waypoint_name):
        for waypoint in self.waypoint_list:
            if waypoint.name == waypoint_name:
                return True
        return False


class Environment:

    def __init__(self, name, robot, transition_function=1, room_list=[], variable_list=[],
                 waypoint_list=[], time=0):
        """
        @param room_list: list of Room objects
        @param robot: Robot object
        @param variable_list: list of Variable objects
        @param transition_function: holds a transition function for each variable in the environment.
        Use this function to calculate the values at the next time step
        Default: 1, meaning t+1 = 1 * t
        """
        self.name = name
        self.time = time
        self.waypoint_list = waypoint_list
        self.room_list = room_list
        self.robot = robot
        self.variable_list = variable_list
        self.transition_function = transition_function

    # Funtion that returns a path to a variable.
    def path_to_variable(self, variable_name):
        room = self.variable_in_which_room(variable_name=variable_name)
        position_robot = self.robot.position
        if room is not None:
            return self.navigate_from_wp_to_room(wp_from=position_robot, room_to=room)
        return None

    # Returns a list of waypoints that form a route from wp_from to wp_to
    def navigate_from_wp_to_room(self, wp_from, room_to):
        connections_from = []
        for connection in wp_from.connections:
            connections_from.append(self.get_waypoint_from_name(connection))
        connections_to = room_to.waypoint_list

        # If there is 1 point in between, return that point and the end point
        for connection_from in connections_from:
            if connection_from in connections_to:
                return [connection_from.name]
            connection_of_connection_from = []
            connection_from = self.get_waypoint_from_name(connection_from.name)
            for connection_2nd_degree in connection_from.connections:
                connection_of_connection_from.append(self.get_waypoint_from_name(connection_2nd_degree))

            for connection_of_connection in connection_of_connection_from:
                if connection_of_connection in connections_to:
                    return [connection_from.name + connection_of_connection.name]

        return None

    # Plots the environment as a figure
    def plot_environment_and_animation(self, block=False, ax_ani=None, ax_env=None, delay=0.01):

        self.plot_environment(ax_env=ax_env, delay=delay)
        self.plot_animated_graph(ax_ani=ax_ani, delay=delay)

    # Returns the variable from self.variable_list if variable_name is the same
    def get_variable_from_name(self, variable_name):
        for variable in self.variable_list:
            if variable.name == variable_name:
                return variable
        return None

    # Returns the Waypoint object from waypoint_list with the same name if it exists
    def get_waypoint_from_name(self, waypoint_name):
        for waypoint in self.waypoint_list:
            if waypoint.name == waypoint_name:
                return waypoint
        return None

    # Plots the rooms, variables and the robot
    def plot_environment(self, ax_env=None, delay=0.01):
        if ax_env is not None:
            plt.sca(ax_env)
            plt.cla()

            plt.title(
                "Time: " + str(self.time) + "\n action: " + str(self.robot.reasoner.get_action_at_t(self.time - 1)))

            for room in self.room_list:
                values = room.origin_width_height
                ax_env.add_patch(patches.Rectangle(xy=(values[0], values[1]), width=values[2], height=values[3],
                                                   fill=False))
                ax_env.text(x=values[0] + 2, y=values[1] + 2, s=room.name)
                for variable in room.variable_list:
                    ax_env.text(x=variable.coords[0], y=variable.coords[1], s=variable.name)
                    # ax_env.text(x=variable.coords[0], y=variable.coords[1] - 5, s=variable.value)

                    # Plot circles that turn on and off with the value of variables
                    if "light" in variable.name:
                        alpha = float(variable.value) * 0.5
                        ax_env.add_patch(patches.Circle(xy=variable.coords, radius=4, color='y', alpha=alpha))
                    elif "switch" in variable.name:
                        alpha = float(variable.value) * 0.5
                        ax_env.add_patch(patches.Circle(xy=variable.coords, radius=4, color='b', alpha=alpha))

                    # The arrows from switches to lights
                    if variable.cause_list:
                        for cause in variable.cause_list:
                            coord_to = self.get_variable_from_name(cause).coords
                            plt.arrow(x=coord_to[0], y=coord_to[1], dx=- coord_to[0] + 5 + variable.coords[0],
                                      dy=- coord_to[1] + variable.coords[1], color='b', head_width=3,
                                      alpha=0.2)
                for waypoint in room.waypoint_list:
                    ax_env.text(x=waypoint.coords[0] - 10, y=waypoint.coords[1], s=waypoint.name)

            # Plot the doors
            # TODO: not hardcode the door plotting
            ax_env.add_patch(patches.Rectangle(xy=(95., 15.), width=10, height=10, fill=False))
            if self.name == "4_rooms":
                ax_env.add_patch(patches.Rectangle(xy=(95., 115.), width=10, height=10, fill=False))
                # ax_env.add_patch(patches.Rectangle(xy=(15., 95.), width=10, height=10, fill=False))
                ax_env.add_patch(patches.Rectangle(xy=(175., 95.), width=10, height=10, fill=False))

            try:
                image = plt.imread('Robbie.png')
                oi = OffsetImage(image, zoom=0.3)
                box = AnnotationBbox(oi, xy=self.robot.position.coords, frameon=False)
                ax_env.add_artist(box)
            except FileNotFoundError:
                # If no file, draw rectangle instead
                ax_env.add_patch(patches.Circle(xy=self.robot.position.coords, radius=10, color='r', label='Robot'))

            plt.show(block=False)
            plt.pause(delay)

    # Plots the causal graph of the reasoner with the line thickness as confidence
    def plot_animated_graph(self, ax_ani, delay=0.01, threshold=0.1):
        if ax_ani is not None:
            plt.sca(ax_ani)
            plt.cla()
            if len(self.robot.reasoner.causal_graph.supergraph.nodes) > 0:
                if len(self.robot.reasoner.shd_score) > 0:
                    score = self.robot.reasoner.shd_score[-1][-1]
                else:
                    score = None
                self.robot.reasoner.causal_graph.draw_supergraph(ax=ax_ani, score=score)
            else:
                self.robot.reasoner.causal_graph.plot_graph(show=False, ax=ax_ani, threshold=threshold)
            plt.show(block=False)
            plt.pause(delay)

    # Returns a DAG object with the correct edges
    def correct_graph(self):
        correct_graph = DAG()
        for variable in self.variable_list:
            if variable.cause_list:
                for cause in variable.cause_list:
                    correct_graph.add_edge(cause, variable.name)
        correct_graph.add_edge("position_robot_1", "room_robot_1")
        return correct_graph

    # Set the robot position to the new waypoint
    def perform_movement(self, movement):
        if movement[1] is None:
            self.robot.reasoner.append_action_history("None", "None", self.time)
        else:
            # Check whether move is legal or not
            if movement[1] in self.robot.position.connections:
                for waypoint in self.waypoint_list:
                    if waypoint.name == movement[1]:
                        self.robot.position = waypoint
                        self.robot.room = self.waypoint_in_which_room(waypoint.name)
                        self.robot.reasoner.append_action_history("position_robot_1", movement, self.time)
                        break

    # The main method for performing actions, calculating next moves and moving time
    def do_actions_move_time(self, action=None, move=None, animate=False, ax_ani=None, delay=0.01,
                             plot_env=False, ax_env=None):
        #print("Time ", self.time, " Action, move", action, move)
        if action is not None and action in self.possible_actions():
            self.perform_action(action=action)

        elif move is not None and move in self.possible_moves():
            self.perform_movement(movement=move)

        elif action is None:
            self.perform_action(action=action)

        # Calculate next values and pass time
        self.calculate_next_values()
        self.move_time()
        observable_variables, observable_values = zip(*self.pass_observable_values(self.robot.room))
        self.robot.sensor_data = np.array([[variable.name for variable in observable_variables], observable_values])
        # Add the time
        self.robot.sensor_data = np.hstack((np.array([["Time"], [self.time]]), self.robot.sensor_data))

        # Save new values
        # print(f"New values are: \n", self.robot.sensor_data)
        self.robot.reasoner.add_new_data(self.robot.sensor_data)

        if not animate:
            ax_ani = None
        if not plot_env:
            ax_env = None
        #delay = 0.5
        self.plot_environment_and_animation(ax_ani=ax_ani, ax_env=ax_env, delay=delay)

    # Performs an action, changing the value of one variable
    def perform_action(self, action=None, value=-1):
        if action is not None:
            if action[0] is not None:
                good_action = False
                for poss_action in self.possible_actions():
                    if np.all(poss_action == action):
                        good_action = True
                if not good_action:
                    return None
                # If the action is flipping a switch, find that switch and set its value to 0 or 1
                # the opposite value than what it was before.
                for list_variable in self.variable_list:
                    if list_variable.name == action[0]:
                        list_variable.change_value(new_value=action[1])
                        self.robot.reasoner.append_action_history(action[0], action[1], self.time)
        elif action is None:
            self.robot.reasoner.append_action_history("None", "None", self.time)

    def move_time(self):
        self.time += 1

    # Returns the list of variables that exist in the given Room
    def pass_observable_variables(self, room):
        # Pass the variables that are visible globally
        global_list = []
        for variable in self.variable_list:
            if variable.global_visibility:
                global_list.append(variable)

        for own_room in self.room_list:
            if own_room.name == room.name:
                return own_room.variable_list + global_list

        return global_list

    # Returns the values of a list of variables, given their names
    # If variable_names=None return all values
    def get_values(self, variables=None):
        res = []
        if variables is None:
            for variable in self.variable_list:
                if type(variable.value) == Waypoint:
                    res.append(variable.value.name)
                else:
                    res.append(variable.value)
            return res

        for variable in variables:
            if type(variable) == Variable:
                variable = variable.name
            for own_variable in self.variable_list:
                if own_variable.name == variable:
                    res.append(own_variable.value)

        return res

    # Gives a list of both the observable variables in a room and their value
    def pass_observable_values(self, room):
        global_list = []
        for variable in self.variable_list:
            if variable.global_visibility:
                global_list.append(variable)

        variable_list = self.pass_observable_variables(room)
        if variable_list is not None:
            return_values = []
            for variable in variable_list + global_list:
                return_values.append(variable.value)

            return zip(variable_list, return_values)
        else:
            print("None in variable_list")

    # Returns a list of connected waypoint_names where the robot can move to
    def possible_moves(self):
        position = self.robot.position.name
        return_array = [(position, position)] + [(position, connection) for connection in
                                                 self.robot.position.connections]
        return np.array(return_array)

    # Returns the names of variables that can be intervened upon
    # Checks in which room the robot is and picks all the variables from that room
    # that can be intervened upon
    # TODO: change everywhere to remove argument
    def possible_actions(self, waypoint_name=None):
        robot_room = self.robot.room
        action_list = []

        for variable in robot_room.variable_list:
            possible_actions = variable.get_possible_actions()
            if possible_actions is not None:
                for possible_action in possible_actions:
                    action_list.append([variable.name, possible_action])
        return np.array(action_list)

    # Iterate over all the variables and set their next values
    # NB! effects are assumed to not be causes for other values
    def calculate_next_values(self):
        for variable in self.variable_list:

            # Set the robot position and room tracking variables
            if variable.name == "position_" + self.robot.name:
                variable.value = self.robot.position.name
            if variable.name == "room_" + self.robot.name:
                variable.value = self.robot.room.name

            # Check if the value of variables drift over time
            if variable.stationary:

                # Intervenable variables can be left alone
                if not variable.intervenable:
                    # Get the values of the causes and set the new value
                    causes = self.get_values(variable.cause_list)
                    variable.change_value(variable.calculate_next_value(causes))

    # Returns the name of the room where a specific waypoint is
    def waypoint_in_which_room(self, waypoint_name):
        for room in self.room_list:
            if room.waypoint_in_room(waypoint_name):
                return room
        return -1

    # Returns the room where a variable is
    def variable_in_which_room(self, variable_name):
        for room in self.room_list:
            for variable in room.variable_list:
                if variable.name == variable_name:
                    return room
        return None

    # Gives a list of all the variables in variable_list that can be intervened upon
    def pass_intervenable_variables(self):
        return_list = []
        for variable in self.variable_list:
            if variable.intervenable:
                return_list.append(variable.name)

        return np.array(return_list)
