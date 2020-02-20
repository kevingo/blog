# Deep reinforcement learning 學習筆記

## DRL basic

* observation 又叫 state
* state 指的是 env 的狀態，也就是 agent 所能夠觀察到的狀態
* agent 做的事情叫做 action
* env 會根據 agent 的 action 給予 reward
* agent 的目標是讓採取 action 去 max reward
* 從開始到結束叫做一個 episode，agnet 的目標就是在一個 episode 當中去 max reward

## AC/A2C/A3C

* Policy-based learn 一個 actor
* Value-based learn 一個 critic

## Gym RL framework (environment)

* The gym library is a collection of test problems — environments — that you can use to work out your reinforcement learning algorithms. These environments have a shared interface, allowing you to write general algorithms.

* env 的 `step` function 會回傳四個物件：

1. observation (obj)
2. reward (float)
3. done (boolean)
4. info (dict)

* The process gets started by calling `reset()`, which returns an `initial observation`. (呼叫 reset() 時會回傳一個初始化的 observation)，通常用在整個 RL 開始訓練的開始步驟。

* show all env
```python
from gym import envs
all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs]
print(env_ids)
```

* 客製化 env

```python
import gym
from gym import spaces

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, arg1, arg2, ...):
    super(CustomEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
    # Example for using image as input:
    self.observation_space = spaces.Box(low=0, high=255, shape=
                    (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

  def step(self, action):
    # Execute one time step within the environment
    ...
  def reset(self):
    # Reset the state of the environment to an initial state
    ...
  def render(self, mode='human', close=False):
    # Render the environment to the screen
    ...
```

## stable_baseline - agent RL framework

* a set of implementations of Reinforcement Learning (RL) algorithms with a common interface, based on OpenAI Baselines.
* openAI stable_baseline 有什麼問題
  1. lack of comments
  2. absence of meaningful variable names and consistency (no common codestyle) and lots of duplicated code (2)
* 優點
  1. commented code and single codestyle
  2. common interface for every algorithm (Unified Interface)
  3. Support save, load model
  4. Training on Any Type of Features (current OpenAI Baselines only support continuous actions when not using images as input)



### references
1. [ML Lecture 23-1: Deep Reinforcement Learning
](https://www.youtube.com/watch?v=W8XF3ME8G2I&t=640s)