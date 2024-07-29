import copy

import classes


def generate_new():
    ##### Room A
    # Switches and lights
    switch_1 = classes.Variable(name="switch_1", flag="switch", value=0, intervenable=True, stationary=True,
                                coords=[10, 40])
    switch_2 = classes.Variable(name="switch_2", flag="switch", value=0, intervenable=True, stationary=True,
                                coords=[70, 40])

    light_1 = classes.Variable(name="light_1", flag="light", value=0, intervenable=False, stationary=True,
                               cause_list=["switch_1"], transition_operator_array=[[[""], [0], [0, 1]]],
                               coords=[10, 60])
    light_2 = classes.Variable(name="light_2", flag="light", value=0, intervenable=False, stationary=True,
                               cause_list=["switch_1", "switch_2"], transition_operator_array=[[["and"],
                                                                                                [0, 1], [0, 1]]],
                               coords=[40, 60])
    light_3 = classes.Variable(name="light_3", flag="light", value=0, intervenable=False, stationary=True,
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
    switch_3 = classes.Variable(name="switch_3", flag="switch", value=0, intervenable=True, stationary=True,
                                coords=[170, 40])
    switch_4 = classes.Variable(name="switch_4", flag="switch", value=0, intervenable=True, stationary=True,
                                coords=[110, 40])

    light_4 = classes.Variable(name="light_4", flag="light", value=0, intervenable=False, stationary=True,
                               cause_list=["switch_3", "switch_2"], transition_operator_array=[[["or"], [0, 1], [0, 1]]],
                               coords=[110, 60])
    light_5 = classes.Variable(name="light_5", flag="light", value=0, intervenable=False, stationary=True,
                               cause_list=["switch_3", "switch_6"], transition_operator_array=[[["and"],
                                                                                                [0, 1], [0, 1]]],
                               coords=[140, 60])
    light_6 = classes.Variable(name="light_6", flag="light", value=0, intervenable=False, stationary=True,
                               cause_list=["switch_4"], transition_operator_array=[[["not"], [0], [0, 1]]],
                               coords=[170, 60])
    # Waypoints
    waypoint_5 = classes.Waypoint(name="wall_b_1", flag="wall", coords=[180, 20],
                                  connections=["door_b_1", "door_b_2", "wall_b_2"])
    waypoint_6 = classes.Waypoint(name="door_b_1", flag="door", coords=[120, 20],
                                  connections=["wall_b_1", "door_b_2", "wall_b_2"])
    waypoint_7 = classes.Waypoint(name="door_b_2", flag="door", coords=[180, 80],
                                  connections=["wall_b_1", "door_b_1", "wall_b_2"])
    waypoint_8 = classes.Waypoint(name="wall_b_2", flag="wall", coords=[120, 80],
                                  connections=["wall_b_1", "door_b_1", "door_b_2"])

    # Room and variable lists
    wp_list_room_b = [waypoint_5, waypoint_6, waypoint_7, waypoint_8]
    var_list_room_b = [switch_3, switch_4, light_4, light_5, light_6]
    room_b = classes.Room(name="room_b", waypoint_list=wp_list_room_b, variable_list=var_list_room_b,
                          origin_width_height=[105., 5., 90., 90.])

    ##### Room C
    # Switches and lights
    switch_5 = classes.Variable(name="switch_5", flag="switch", value=0, intervenable=True, stationary=True,
                                coords=[10, 140])
    switch_6 = classes.Variable(name="switch_6", flag="switch", value=0, intervenable=True, stationary=True,
                                coords=[70, 140])

    light_7 = classes.Variable(name="light_7", flag="light", value=0, intervenable=False, stationary=True,
                               cause_list=["switch_5"], transition_operator_array=[[["not"], [0], [0, 1]]],
                               coords=[10, 160])
    light_8 = classes.Variable(name="light_8", flag="light", value=0, intervenable=False, stationary=True,
                               cause_list=["switch_7", "switch_8"], transition_operator_array=[[["xor"],
                                                                                                [0, 1], [0, 1]]],
                               coords=[40, 160])
    light_9 = classes.Variable(name="light_9", flag="light", value=0, intervenable=False, stationary=True,
                               cause_list=["switch_1"], transition_operator_array=[[["not"], [0], [0, 1]]],
                               coords=[70, 160])
    # Waypoints
    waypoint_9 = classes.Waypoint(name="wall_c_1", flag="wall", coords=[20, 120],
                                  connections=["table_c_1", "door_c_1", "wall_c_2"])
    waypoint_10 = classes.Waypoint(name="table_c_1", flag="table", coords=[80, 180],
                                   connections=["wall_c_1", "door_c_1", "wall_c_2"])
    waypoint_11 = classes.Waypoint(name="door_c_1", flag="door", coords=[80, 120],
                                   connections=["wall_c_1", "table_c_1", "wall_c_2"])
    waypoint_12 = classes.Waypoint(name="wall_c_2", flag="wall", coords=[20, 180],
                                   connections=["wall_c_1", "table_c_1", "door_c_1"])

    # Room and variable lists
    wp_list_room_c = [waypoint_9, waypoint_10, waypoint_11, waypoint_12]
    var_list_room_c = [switch_5, switch_6, light_7, light_8, light_9]
    room_c = classes.Room(name="room_c", waypoint_list=wp_list_room_c, variable_list=var_list_room_c,
                          origin_width_height=[5., 105., 90., 90.])

    #### Room D
    # Switches and lights
    switch_7 = classes.Variable(name="switch_7", flag="switch", value=0, intervenable=True, stationary=True,
                                coords=[170, 140])
    switch_8 = classes.Variable(name="switch_8", flag="switch", value=0, intervenable=True, stationary=True,
                                coords=[110, 140])

    light_10 = classes.Variable(name="light_10", flag="light", value=0, intervenable=False, stationary=True,
                                cause_list=["switch_5", "switch_8"],
                                transition_operator_array=[[["!="], [0, 1], [0, 1]]],
                                coords=[110, 160])
    light_11 = classes.Variable(name="light_11", flag="light", value=0, intervenable=False, stationary=True,
                                cause_list=["switch_7", "switch_2"], transition_operator_array=[[["and"],
                                                                                                 [0, 1], [0, 1]]],
                                coords=[140, 160])
    light_12 = classes.Variable(name="light_12", flag="light", value=0, intervenable=False, stationary=True,
                                cause_list=["switch_8"], transition_operator_array=[[["not"], [0], [0, 1]]],
                                coords=[170, 160])
    # Waypoints
    waypoint_13 = classes.Waypoint(name="door_d_1", flag="door", coords=[180, 120],
                                   connections=["door_d_2", "wall_d_1", "wall_d_2"])
    waypoint_14 = classes.Waypoint(name="door_d_2", flag="door", coords=[120, 120],
                                   connections=["door_d_1", "wall_d_1", "wall_d_2"])
    waypoint_15 = classes.Waypoint(name="wall_d_1", flag="wall", coords=[180, 180],
                                   connections=["door_d_1", "door_d_2", "wall_d_2"])
    waypoint_16 = classes.Waypoint(name="wall_d_2", flag="wall", coords=[120, 180],
                                   connections=["door_d_1", "door_d_2", "wall_d_1"])

    # Room and variable lists
    wp_list_room_d = [waypoint_13, waypoint_14, waypoint_15, waypoint_16]
    var_list_room_d = [switch_7, switch_8, light_10, light_11, light_12]
    room_d = classes.Room(name="room_d", waypoint_list=wp_list_room_d, variable_list=var_list_room_d,
                          origin_width_height=[105., 105., 90., 90.])

    # Add some doors
    # Room A to B
    waypoint_3.connections = waypoint_3.connections + ["door_b_1"]
    waypoint_6.connections = waypoint_6.connections + ["door_a_1"]
    # Room B to C
    waypoint_13.connections = waypoint_13.connections + ["door_b_2"]
    waypoint_7.connections = waypoint_7.connections + ["door_d_1"]

    # Room C to D
    waypoint_11.connections = waypoint_11.connections + ["door_d_2"]
    waypoint_14.connections = waypoint_14.connections + ["door_c_1"]

    ##### Robot and Environment
    # Robot, environment and classes inside Robot
    causal_graph_1 = classes.CausalGraph()
    reasoner_1 = classes.Reasoner(causal_graph=causal_graph_1)
    robot_1 = classes.Robot(name="robot_1", position=waypoint_8, room=room_b, reasoner=reasoner_1)
    environment_1 = classes.Environment(name="4_rooms", robot=robot_1, room_list=[room_a, room_b, room_c, room_d],
                                        variable_list=var_list_room_a + var_list_room_b + var_list_room_c + var_list_room_d,
                                        waypoint_list=wp_list_room_a + wp_list_room_b + wp_list_room_c + wp_list_room_d)

    # Position variable of the robot
    position_robot_1 = classes.Variable(name="position_robot_1", flag="position", value=robot_1.position,
                                        intervenable=True,
                                        stationary=True, global_visibility=True)
    # Room variable of the robot
    room_robot_1 = classes.Variable(name="room_robot_1", flag="room", value=robot_1.room, intervenable=False,
                                    stationary=True, global_visibility=True)

    environment_1.variable_list = environment_1.variable_list + [position_robot_1, room_robot_1]
    robot_1.environment = environment_1
    reasoner_1.robot = robot_1

    return copy.deepcopy(environment_1)

