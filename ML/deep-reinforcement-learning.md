# Deep reinforcement learning 學習筆記

## DRL basic

* observation 又叫 state
* state 指的是 env 的狀態，也就是 agent 所能夠觀察到的狀態
* agent 做的事情叫做 action
* env 會根據 agent 的 action 給予 reward
* agent 的目標是讓採取 action 去 max reward
* 從開始到結束叫做一個 episode，agnet 的目標就是在一個 episode 當中去 max reward


## A3C

* Policy-based learn 一個 actor
* Value-based learn 一個 critic

## Gym RL framework

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

### references
1. [ML Lecture 23-1: Deep Reinforcement Learning
](https://www.youtube.com/watch?v=W8XF3ME8G2I&t=640s)