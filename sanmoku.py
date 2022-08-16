import numpy as np
import gym
from gym import spaces


class sanmoku(gym.Env):
    metadata = {'render.modes': ['human']}

    # fieldmap
    MARU = 1 # first
    BATSU = 2 # second


    WIN_PATTERN = [[[0,0],[0,1],[0,2]],
                   [[1,0],[1,1],[1,2]],
                   [[2,0],[2,1],[2,2]],
                   [[0,0],[1,0],[2,0]],
                   [[0,1],[1,1],[2,1]],
                   [[0,2],[1,2],[2,2]],
                   [[0,0],[1,1],[2,2]],
                   [[0,2],[1,1],[2,0]]]

    def is_win(self):
        win_flg = False

        # WIN_PATTERNで定義した勝ちパターンそれぞれと比較
        for l in self.WIN_PATTERN:
            wp = np.array(l)

            if np.all(self.fieldmap[tuple(wp.T)] == self.player):
                win_flg = True
                # print(f"Player: {self.player} is WIN!!")

        return win_flg

    def reset_fieldmap(self):
        self.fieldmap = np.zeros((self.grid_size, self.grid_size))

    def change_player(self):
        if self.player == self.MARU:
            self.player = self.BATSU
        else:
            self.player = self.MARU

    def set_player(self, action):
        pos_x = int(action / self.grid_size)
        pos_y = action % self.grid_size

        if self.fieldmap[pos_x, pos_y] != 0:
            # print(f"座標[{pos_x},{pos_y}]は選択済みです。")
            return False
        else:
            self.fieldmap[pos_x, pos_y] = self.player
            return True

    def __init__(self, grid_size=3):
        super(sanmoku, self).__init__()

        # fields
        self.grid_size = grid_size
        self.reset_fieldmap()
        self.action = 0
        self.reward = 0
        self.player = self.MARU

        # action num
        n_actions = 9
        self.action_space = spaces.Discrete(n_actions)

        # low=input_lower_limit, high=input_higher_limit
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(grid_size, grid_size,), dtype=np.float32)

    def reset(self):
        # initial position
        self.reset_fieldmap()
        # numpy only
        return np.array(self.fieldmap).astype(np.float32)

    def step(self, action):
        done = False
        reward = 0

        if self.set_player(action) == False:
            done = True
            reward = -100
        else:
            reward = 10

        # goal and reward
        if self.is_win():
            done = True
            reward = 100

        self.change_player()

        # fetch variables
        self.action = action
        self.reward = reward

        # infomation
        info = {}
        return np.array(self.fieldmap).astype(np.float32), reward, done, info

    # draw
    def render(self, mode='human', close=False):
        if mode != 'human':
            raise NotImplementedError()
        draw_map = ""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.fieldmap[i, j] == 0:
                    draw_map += "- "
                elif self.fieldmap[i, j] == self.MARU:
                    draw_map += "O "
                elif self.fieldmap[i, j] == self.BATSU:
                    draw_map += "X "
            draw_map += '\n'
        print("Action:", self.action, "Reward:", self.reward)
        print(draw_map)
