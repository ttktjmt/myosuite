import myosuite
from myosuite.utils import gym

print(f"version: {myosuite.__version__}")

env = gym.make('myoChallengeBimanual-v0')
env.reset()
for _ in range(10000):
  # env.mj_render()
  env.step(env.action_space.sample()) # take a random action
env.close()