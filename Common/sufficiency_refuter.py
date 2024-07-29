# Receives a set of inputs X believed to be sufficient for effect Y
# Try to refute by changing inputs while keeping the effect the same
# If an input can be changed while the output is constant it is not needed in the sufficient set
# Example ->
# A + B + C -> Y
# A + C -> Y
# Sufficient set {A, C}

# TODO: Implement Sufficiency refuter


# Tries to reduce the amount of variables in the sufficiency set
# while keeping the target the same
def reduce_sufficiency_set(set, target, light_states, switch_states, connections_array, connection_types, new_states):
    return_set = set
    # Loop through the causes
    print(f"Refuting sufficiency of {target}, with causes {set}")

    for action in set:
        # If changing the cause does not change the effect (target), it is not in the sufficiency set
        new_lights_one, new_switches_one = new_states(action, switch_states,
                                                      connections_array, connection_types)

        # Turn the switch once, check if the effect changed
        if light_states[target] == new_lights_one[target]:
            print(f"Cause {action} non-necessary in sufficiency set")
            return_set = tuple(x for x in return_set if x != action)
        # If the effect did change, check if switching back does return the effect
        # otherwise it could have been luck
        else:
            new_lights_two, new_switches_two = new_states(action, new_switches_one,
                                                          connections_array, connection_types)
            # If moving the switch back to the original position does not restore the effect
            # there is something wrong with the sufficiency set
            if light_states[target] != new_lights_two[target]:
                print(f"Cause {action} non-necessary in sufficiency set")
                return_set = tuple(x for x in return_set if x != action)

    print(f"Sufficiency set after refuting {return_set}")
    return return_set

