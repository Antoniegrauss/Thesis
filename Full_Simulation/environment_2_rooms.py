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
                                  connections=["table_a_1", "door_a_1", "wall_a_2"])
    waypoint_2 = classes.Waypoint(name="table_a_1", flag="table", coords=[80, 80],
                                  connections=["wall_a_1", "door_a_1", "wall_a_2"])
    waypoint_3 = classes.Waypoint(name="door_a_1", flag="door", coords=[80, 20],
                                  connections=["wall_a_1", "table_a_1", "wall_a_2"])
    waypoint_4 = classes.Waypoint(name="wall_a_2", flag="wall", coords=[20, 80],
                                  connections=["wall_a_1", "table_a_1", "door_a_1"])

    # Room and variable lists
    wp_list_room_a = [waypoint_1, waypoint_2, waypoint_3, waypoint_4]
    var_list_room_a = [switch_1, switch_2, light_1, light_2, light_3]
    room_a = classes.Room(name="room_a", waypoint_list=wp_list_room_a, variable_list=var_list_room_a,
                          origin_width_height=[5., 5., 90., 90.])

    #### Room B
    # Switches and lights
    switch_3 = classes.Variable(name="switch_3", flag="switch", possible_values=[0, 1], value=0, intervenable=True, stationary=True,
                                coords=[170, 40])
    switch_4 = classes.Variable(name="switch_4", flag="switch", possible_values=[0, 1], value=0, intervenable=True, stationary=True,
                                coords=[110, 40])

    light_4 = classes.Variable(name="light_4", flag="light", possible_values=[0, 1], value=0, intervenable=False, stationary=True,
                               cause_list=["switch_3", "switch_2"], transition_operator_array=[[["or"], [0, 1], [0, 1]]],
                               coords=[110, 60])
    light_5 = classes.Variable(name="light_5", flag="light", possible_values=[0, 1], value=0, intervenable=False, stationary=True,
                               cause_list=["switch_3", "switch_4"], transition_operator_array=[[["not", "and"],
                                                                                                [0, 1], [0, 1]]],
                               coords=[140, 60])
    light_6 = classes.Variable(name="light_6", flag="light", possible_values=[0, 1], value=0, intervenable=False, stationary=True,
                               cause_list=["switch_4"], transition_operator_array=[[["not"], [0], [0, 1]]],
                               coords=[170, 60])
    # Waypoints
    waypoint_5 = classes.Waypoint(name="wall_b_1", flag="wall", coords=[180, 20],
                                  connections=["door_b_1", "table_b_1", "wall_b_2"])
    waypoint_6 = classes.Waypoint(name="door_b_1", flag="door", coords=[120, 20],
                                  connections=["wall_b_1", "table_b_1", "wall_b_2"])
    waypoint_7 = classes.Waypoint(name="table_b_1", flag="table", coords=[180, 80],
                                  connections=["wall_b_1", "door_b_1", "wall_b_2"])
    waypoint_8 = classes.Waypoint(name="wall_b_2", flag="wall", coords=[120, 80],
                                  connections=["wall_b_1", "door_b_1", "table_b_1"])

    # Add a door from waypoint_3 to waypoint_6
    waypoint_3.connections = waypoint_3.connections + ["door_b_1"]
    waypoint_6.connections = waypoint_6.connections + ["door_a_1"]

    # Room and variable lists
    wp_list_room_b = [waypoint_5, waypoint_6, waypoint_7, waypoint_8]
    var_list_room_b = [switch_3, switch_4, light_4, light_5, light_6]
    room_b = classes.Room(name="room_b", waypoint_list=wp_list_room_b, variable_list=var_list_room_b,
                          origin_width_height=[105., 5., 90., 90.])

    ##### Robot and Environment
    # Robot, environment and classes inside Robot
    causal_graph_1 = classes.CausalGraph()
    reasoner_1 = classes.Reasoner(causal_graph=causal_graph_1)
    robot_1 = classes.Robot(name="robot_1", position=waypoint_1, room=room_a, reasoner=reasoner_1)
    environment_1 = classes.Environment(name="2_rooms", robot=robot_1, room_list=[room_a, room_b],
                                        variable_list=var_list_room_a + var_list_room_b,
                                        waypoint_list=wp_list_room_a + wp_list_room_b)

    # Position variable of the robot
    position_robot_1 = classes.Variable(name="position_robot_1", flag="position",
                                        possible_values= robot_1.position.connections,
                                        value=robot_1.position,
                                        intervenable=True,
                                        stationary=True, global_visibility=True)
    # Room variable of the robot
    room_robot_1 = classes.Variable(name="room_robot_1", flag="room", value=robot_1.room, intervenable=False,
                                    stationary=True, global_visibility=True)

    environment_1.variable_list = environment_1.variable_list + [position_robot_1, room_robot_1]
    robot_1.environment = environment_1
    reasoner_1.robot = robot_1
    env = copy.deepcopy(environment_1)

    return env
