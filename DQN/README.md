# Deep Q-Networks
## Introduction
DQN is able to learn value functions using usch function approximators in a stable and robust way due to two innovations.
- The network is trained off-policy with samples from a replay buffer to minimize correlations between samples.
- The network is trained with a Q network to give consistent targets during temporal difference backups.
- In order to scale Q-learning, we have two changes:
1. replay buffer
	- A finite size cache, (S, A, R, St+1).
	- At each timestep, the actor and critic are updated by sampling a minibatch uniformly from the buffer.

2. separate target network
## Hyperparameters
```Python
GAMMA = 0.9 # discount facter
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # minibatch size
```
## Test envs
CartPole-v0

## Platform
OpenAI gym
http://gym.openai.com