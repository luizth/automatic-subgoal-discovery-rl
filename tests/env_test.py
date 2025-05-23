from env import FourRooms


def test_env_wall(log):
    """
    Test the walls of the FourRooms environment.
    """
    env = FourRooms(
        start_state=30,
        goal_state=75,
        negative_states_config="default",
        max_steps=None,
        sparse_rewards=False
    )

    # Print the walls for debugging
    if log:
        print("Upper Wall:", env._upper_wall)
        print("Lower Wall:", env._lower_wall)
        print("Left Wall:", env._walls_to_left)
        print("Right Wall:", env._walls_to_right)

    # Check if the walls are correctly defined
    assert env._upper_wall == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] + [50, 51, 52, 53, 54] + [65, 66, 67, 68, 69]
    assert env._lower_wall == [90, 91, 92, 93, 94, 95, 96, 97, 98, 99] + [40, 41, 42, 43, 44] + [55, 56, 57, 58, 59]
    assert env._walls_to_left == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90] + [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    assert env._walls_to_right == [9, 19, 29, 39, 49, 59, 69, 79, 89, 99] + [4, 14, 24, 34, 44, 54, 64, 74, 84, 94]


def test_env_transition(log):
    """
    Test the transition function of the FourRooms environment.
    """
    env = FourRooms(
        start_state=30,
        goal_state=75,
        negative_states_config="default",
        max_steps=None,
        sparse_rewards=False
    )

    # Print the transition function for debugging
    if log:
        print("Transitions:", env.transitions)

    # Check if the transition function is correctly defined
    assert len(env.transitions) == env.observation_space.n
    assert len(env.transitions[0]) == 4
    assert list(env.transitions[0] ) == [0, 10, 0, 1]
    assert list(env.transitions[4] ) == [4, 14, 3, 4]
    assert list(env.transitions[5] ) == [5, 15, 5, 6]
    assert list(env.transitions[9] ) == [9, 19, 8, 9]
    assert list(env.transitions[40]) == [30, 40, 40, 41]
    assert list(env.transitions[44]) == [34, 44, 43, 44]
    assert list(env.transitions[55]) == [45, 55, 55, 56]
    assert list(env.transitions[59]) == [49, 59, 58, 59]
    assert list(env.transitions[50]) == [50, 60, 50, 51]
    assert list(env.transitions[54]) == [54, 64, 53, 54]
    assert list(env.transitions[65]) == [65, -1, 65, 66]  # to goal
    assert list(env.transitions[69]) == [69, 79, 68, 69]
    assert list(env.transitions[90]) == [80, 90, 90, 91]
    assert list(env.transitions[94]) == [84, 94, 93, 94]
    assert list(env.transitions[95]) == [85, 95, 95, 96]
    assert list(env.transitions[99]) == [89, 99, 98, 99]

    # Hallway transitions
    assert env.transitions[41, 1] == 75
    assert list(env.transitions[75]) == [41, 51, 75, 75]
    assert env.transitions[51, 0] == 75

    assert env.transitions[24, 3] == 100
    assert list(env.transitions[100]) == [100, 100, 24, 25]
    assert env.transitions[25, 2] == 100

    assert env.transitions[57, 1] == 101
    assert list(env.transitions[101]) == [57, 67, 101, 101]
    assert env.transitions[67, 0] == 101

    assert env.transitions[84, 3] == 102
    assert list(env.transitions[102]) == [102, 102, 84, 85]
    assert env.transitions[85, 2] == 102
