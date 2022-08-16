from sanmoku import sanmoku

env = sanmoku(grid_size=3)
obs = env.reset()
env.render()
n_steps = 10

for step in range(n_steps):
  print("Step {}".format(step + 1))
  print(f"PLAYER:{env.player}")

  myaction = int(input("ACTION: "))

  obs, reward, done, info = env.step(myaction)
  print('obs=', obs, 'reward=', reward, 'done=', done)
  env.render()
  if done:
    print(f"reward={reward}")
    break
