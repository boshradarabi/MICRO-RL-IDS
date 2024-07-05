import gym
import random
import numpy as np

class IntrusionEnv(gym.Env):
    def __init__(self,x_set, y_set, len, n_action, n_state, random=True):
        """Environment
        Params
        =====
            x_set(numpy array): a matrix contains features
            y_set(numpy array): an array contains labels
            len(int): the length of dataset
            n_action(int): number of actions(classes)
            n_state(int): number of statets(features)
            random(bool): the state of selecting records in dataset
        """

        self.action_space = gym.spaces.Discrete(n_action)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(n_state,), dtype=np.float32)
        self.x, self.y = x_set, y_set
        self.random = random
        self.dataset_idx = 0
        self.len=len
        self.ep_length = self.len


    def step(self, action):
        done = False
        if (action == self.expected_action):
          reward=1
        else:
          reward=-1

        obs = self._next_obs()

        self.ep_length-=1


        if self.ep_length <= 0:
            done = True
        else:
            done = False

        return obs, reward, done, self.expected_action


    def reset(self):
        self.ep_length = self.len

        obs = self._next_obs()
        return obs


    def _next_obs(self):
        if self.random:
            next_obs_idx = random.randint(0, len(self.x) - 1)
            self.expected_action = int(self.y[next_obs_idx])
            obs = self.x[next_obs_idx]
            # print("next_obs_idx: ", next_obs_idx)

        else:
            obs = self.x[self.dataset_idx]
            self.expected_action = int(self.y[self.dataset_idx])

            self.dataset_idx += 1
            if self.dataset_idx > len(self.x):
                raise StopIteration()


        return obs