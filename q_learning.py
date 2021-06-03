
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def update_state(current_state, action):
    """
    Update the new state based on a previous action.
    """
    new_state = current_state + action
    # bound checking
    if new_state == -1: return 11
    elif new_state == 12: return 0
    else: return new_state


def q_learning(gamma, alpha, episodes=100):
    """
    Q learning algorithm for mouse round a clock problem.
    """

    # Initializing Q-Values
    Q = np.array(np.zeros([12,2]))

    # run through each episode
    for i in range(episodes):
        # pick a random initial state for this episode
        current_state = np.random.randint(0,12)
        if current_state == 6:
            while current_state == 6:
                current_state = np.random.randint(0,12)

        # perform episode
        while True:
            # you've made it to the exit!
            if current_state == 6: break

            # Pick an action randomly
            action = np.random.choice(actions)

            # update next state based on random choice
            next_state = update_state(current_state, actions_movement[action])

            # Compute the temporal difference
            max_q_next = max(Q[next_state, 0], Q[next_state, 1])
            change = alpha * (rewards[current_state, action] + (gamma * max_q_next) - Q[current_state, action])

            # Update the Q-Value
            Q[current_state,action] += change

            # update curr state
            current_state = next_state

        # plot current episode
        plt.title("".join(("Episode ", str(i))))
        plt.imshow(Q, cmap='hot')
        plt.pause(0.001)
        plt.clf()

    # plt.show()
    return Q


def real_time_ex():
    """
    Example of plotting graphs in real time.
    https://stackoverflow.com/questions/11874767/how-do-i-plot-in-real-time-in-a-while-loop-using-matplotlib
    """
    plt.axis([0, 10, 0, 1])

    for i in range(10):
        y = np.random.random()
        plt.scatter(i, y)
        plt.pause(0.05)

    plt.show()

if __name__ == "__main__":
    actions = [0, 1]
    actions_movement = [1, -1]

    # define rewards array (Note: [clockwise, anticlockwise])
    rewards = np.array([[-1, -1],
                        [-1, -1],
                        [-1, -1],
                        [-1, -1],
                        [-1, -1],
                        [2, -1],
                        [-1, -1],
                        [-1, 2],
                        [-1, -1],
                        [-1, -1],
                        [-1, -1],
                        [-1, -1]])

    gamma = 0.8
    learning_rate = 0.05

    np.random.seed(1)
    print(q_learning(gamma, learning_rate))
