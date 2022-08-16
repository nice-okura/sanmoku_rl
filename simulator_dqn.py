import gym
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import *
from rl.memory import SequentialMemory
from sanmoku import sanmoku

import numpy as np

# gym
env = sanmoku(grid_size=3)
print("-Initial parameter-")
print(env.action_space) # input
print(env.observation_space) # output
print(env.reward_range) # rewards
print(env.action_space) # action
print(env.action_space.sample()) # action
print("-------------------")
# model
window_length = 1
input_shape =  (window_length,) + env.observation_space.shape
nb_actions = env.action_space.n
c = input_data = Input(input_shape)
c = Flatten()(c)
c = Dense(48, activation='relu')(c)
c = Dense(48, activation='relu')(c)
# c = Dropout(0.1)(c)
c = Dense(48, activation='relu')(c)
# c = Dense(128, activation='relu')(c)
# c = Dense(128, activation='relu')(c)
c = Dense(nb_actions, activation='linear')(c)
model = Model(input_data, c)
print(model.summary())

# rl
memory = SequentialMemory(limit=50000, window_length=window_length)
policy = EpsGreedyQPolicy() #GreedyQPolicy()# SoftmaxPolicy()
agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100, target_model_update=1e-2, policy=policy)
agent.compile(Adam())
agent.fit(env, nb_steps=10000, visualize=False, verbose=1)
agent.save_weights("weights.hdf5")

# predict
obs = env.reset()
n_steps = 20
for step in range(n_steps):
  obs = obs.reshape((1, 1, 3, 3))
  action = model.predict(obs)
  action = np.argmax(action)
  print("Step {}".format(step + 1))
  print("Action: ", action)
  obs, reward, done, info = env.step(action)
  print('obs=', obs, 'reward=', reward, 'done=', done)
  env.render(mode='human')
  if done:
    print("Goal !!", "reward=", reward)
    break
