import pandas as pd
import numpy as np
import time
import sys
import random


class QLearning:
    def __init__(self, sim_parameters, environment, perform_tests=False):
        # Q-learning parameters
        self.learning_rate = sim_parameters['alpha']
        self.discount_factor = sim_parameters['gamma']
        self.exploration_rate_min = sim_parameters['epsilon_min']
        self.exploration_rate_max = sim_parameters['epsilon_max']
        self.exploration_rate_decay = sim_parameters['epsilon_decay']

        # Simulation parameters
        self.env = environment
        self.STREAKS_TO_END = sim_parameters['streaks_to_end']
        self.MAX_EPISODES = sim_parameters['max_episodes']
        self.MAX_STEPS_PER_EPISODE = sim_parameters['max_steps']
        self.BOARD_SIZE = sim_parameters['board_size']
        self.RENDER_MAZE = sim_parameters['render_maze']
        self.N_OF_STATES = sim_parameters['n_of_states']
        # Number of discrete actions ["N", "S", "E", "W"]
        self.N_OF_ACTIONS = self.env.action_space.n

        self.q_table = self.initialize_q_table()


        self.perform_tests = perform_tests
        if self.perform_tests:
            self.results_df = pd.DataFrame(columns=[
                'episode',
                'finished_in_steps',
                'total_reward',
                'streak'
            ])

            self.sim_params_df = pd.DataFrame(
                columns=[
                    'learning_rate',
                    'discount_factor',
                    'min_exploration_rate',
                    'max_exploration_rate',
                    'exploration_rate_decay',
                    'streaks_to_end',
                    'max_episodes',
                    'max_steps_per_episode',
                    'board_size'],
                data=[[
                    self.learning_rate,
                    self.discount_factor,
                    self.exploration_rate_min,
                    self.exploration_rate_max,
                    self.exploration_rate_decay,
                    self.STREAKS_TO_END,
                    self.MAX_EPISODES,
                    self.MAX_STEPS_PER_EPISODE,
                    self.BOARD_SIZE
                ]]
            )

    def initialize_q_table(self):
        shape_of_q_table = self.N_OF_STATES + (self.N_OF_ACTIONS,)
        return np.zeros(shape_of_q_table, dtype=float)

    def simulate(self):
        # Number of repeatable streaks
        n_streaks = 0

        # Render tha maze
        self.env.render()

        # set starting exploration_rate
        exploration_rate = self.exploration_rate_max

        for episode in range(self.MAX_EPISODES):
            # Reset the environment
            initial_state = self.env.reset()

            # the initial state
            previous_state = self.convert_state_int(initial_state)
            total_reward = 0

            for t in range(1, self.MAX_STEPS_PER_EPISODE+1):

                # Search for new action
                action = self.select_action(previous_state, exploration_rate=exploration_rate)

                # Execute and evaluate new action
                observation, reward, reached_goal = self.env.step(action)

                # Generate new step
                next_state = self.convert_state_int(observation)  # coordinates of new step (tuple)

                total_reward += reward

                # Find best q_value for new state
                best_q_value = np.amax(self.q_table[next_state])

                # akcja to N S W E
                # stan to ...
                # q_table[state_0 + (action,)] - wartość Q dla każdej par stan - akcja
                # print(print(q_table))
                # print('--')

                # UPDATE q value for the given state and action
                q_value_index = previous_state + (action,)  # combination of previous state and action
                self.q_table[q_value_index] += self.learning_rate * (reward + self.discount_factor * best_q_value -
                                                                     self.q_table[previous_state + (action,)])

                # Setting up for the next iteration
                previous_state = next_state

                # Render tha maze
                if self.RENDER_MAZE:
                    if n_streaks > 70:
                        time.sleep(0.1)
                    self.env.render()

                if self.env.is_game_over():
                    sys.exit()

                if reached_goal:
                    if t <= self.BOARD_SIZE:
                        n_streaks += 1
                    else:
                        n_streaks = 0

                    print(f"Episode {episode} finished after {t} steps"
                          f" with total reward = {total_reward} (streak: {n_streaks}).")
                    break  # finish episode

                elif t == self.MAX_STEPS_PER_EPISODE:
                    print(f"Episode {episode} timed out at {t} with total reward = {total_reward}")

            if self.perform_tests:
                episode_data = {
                    'episode': int(episode),
                    'finished_in_steps': int(t),
                     'total_reward': total_reward,
                     'streak': n_streaks
                }
                self.results_df = self.results_df.append(episode_data, ignore_index=True)

            # Number of repeatable streaks where agent performed less steps than size of environment
            if n_streaks >= self.STREAKS_TO_END:
                break

            # update
            exploration_rate = self.update_exploration_rate(episode)

        if self.perform_tests:
            print(self.results_df)

    def update_exploration_rate(self, episode):
        return self.exploration_rate_min + (self.exploration_rate_max-self.exploration_rate_min) * \
               np.exp(-self.exploration_rate_decay*episode)

    def select_action(self, state, exploration_rate):
        """
        Chooses action for the current state based on two approaches.
        It has a chance to choose a random action, for the purpose of
        environment exploration.
        :param state:
        :param exploration_rate:
        :return:
        """
        if exploration_rate > np.random.uniform(0, 1):
            action = self.env.action_space.sample()  # choose random action
        else:
            # Select action with highest Q value
            action = int(np.argmax(self.q_table[state]))
        return action

    @staticmethod
    def convert_state_int(state):
        coordinates_int = tuple([int(coordinate) for coordinate in state])
        return coordinates_int
