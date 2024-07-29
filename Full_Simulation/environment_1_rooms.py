import copy

import classes

##### Room A
# Switches and lights
def generate_new():
    switch_1 = classes.Variable(name="switch_1", flag="switch", possible_values=[0, 1], value=0, intervenable=True,
                                stationary=True,
                                coords=[10, 40])
    switch_2 = classes.Variable(name="switch_2", flag="switch", possible_values=[0, 1], value=0, intervenable=True,
                                stationary=True,
                                coords=[70, 40])

    light_1 = classes.Variable(name="light_1", flag="light", possible_values=[0, 1], value=0, intervenable=False, stationary=True,
                               cause_list=["switch_1"], transition_operator_array=[[[], [0], [0, 1]]],
                               coords=[10, 60])
    light_2 = classes.Variable(name="light_2", flag="light", possible_values=[0, 1], value=0, intervenable=False, stationary=True,
                               cause_list=["switch_1", "switch_2"], transition_operator_array=[[["and"],
                                                                                                [0, 1], [0, 1]]],
                               coords=[40, 60])
    light_3 = classes.Variable(name="light_3", flag="light", possible_values=[0, 1], value=0, intervenable=False, stationary=True,
                               cause_list=["switch_2"], transition_operator_array=[[["not"], [0], [0, 1]]],
                               coords=[70, 60])
    # Waypoints
    waypoint_1 = classes.Waypoint(name="wall_a_1", flag="wall", coords=[20, 20],
                                  connections=["wall_a_1"])
    """waypoint_2 = classes.Waypoint(name="table_a_1", flag="table", coords=[80, 80],
                                  connections=["wall_a_1", "door_a_1", "wall_a_2"])
    waypoint_3 = classes.Waypoint(name="door_a_1", flag="door", coords=[80, 20],
                                  connections=["wall_a_1", "table_a_1", "wall_a_2"])
    waypoint_4 = classes.Waypoint(name="wall_a_2", flag="wall", coords=[20, 80],
                                  connections=["wall_a_1", "table_a_1", "door_a_1"])"""

    # Room and variable lists
    wp_list_room_a = [waypoint_1]
    var_list_room_a = [switch_1, switch_2, light_1, light_2, light_3]
    room_a = classes.Room(name="room_a", waypoint_list=wp_list_room_a, variable_list=var_list_room_a,
                          origin_width_height=[5., 5., 90., 90.])

    ##### Robot and Environment
    # Robot, environment and classes inside Robot
    causal_graph_1 = classes.CausalGraph()
    reasoner_1 = classes.Reasoner(causal_graph=causal_graph_1)
    robot_1 = classes.Robot(name="robot_1", position=waypoint_1, room=room_a, reasoner=reasoner_1)
    environment_1 = classes.Environment(name="2_rooms", robot=robot_1, room_list=[room_a],
                                        variable_list=var_list_room_a,
                                        waypoint_list=wp_list_room_a)

    # Position variable of the robot
    position_robot_1 = classes.Variable(name="position_robot_1", flag="position",
                                        possible_values=robot_1.position.connections,
                                        value=robot_1.position,
                                        intervenable=True,
                                        stationary=True, global_visibility=True)
    # Room variable of the robot
    room_robot_1 = classes.Variable(name="room_robot_1", flag="room", value=robot_1.room, intervenable=False,
                                    stationary=True, global_visibility=True)

    environment_1.variable_list = environment_1.variable_list
    robot_1.environment = environment_1
    reasoner_1.robot = robot_1
    env = copy.deepcopy(environment_1)

    return env
