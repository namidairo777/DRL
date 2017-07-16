# Continuous Control with deep reinforcement learning
https://arxiv.org/abs/1509.02971
## Introduction
Model-free off-policy actor-critic algorithm using deep function approximators that can learn policies in high-dimensional, contiuous action spaces. </br>

## DQN
DQN is able to learn value functions using usch function approximators in a stable and robust way due to two innovations.
- The network is trained off-policy with samples from a replay buffer to minimize correlations between samples.
- The network is trained with a Q network to give consistent targets during temporal difference backups.
- In order to scale Q-learning, we have two changes:
1. replay buffer
	- A finite size cache, (S, A, R, St+1).
	- At each timestep, the actor and critic are updated by sampling a minibatch uniformly from the buffer.

2. separate target network

## Impoosible to directly apply Q-learning to continuous action spaces
- In continuous spaces finding the greedy policy requires an optimization of a at every timestep.
- This Optimization is too slow to be practical with large, unconstrained function approximators and nontrivial action spaces.

## DPG
DPG maintains a parameterized actor function.. which specifies the current policy by deterministically mapping states to a specific action.
- The Critic Q(s, a) is learned using bellman equation as in Q-learning
- The Actor is updated by following the applying the chain rule to expected return from the start distribution J with respect to the actor parameters

## DDPG
### Challenge when using NN for RL
- Most optimization algorithms assume that the samples are independently and identically distributed.
- To make efficient use of hardware optimizations, It is essential to learn in **minibatch**, rather than online
- We create a copy of the actor and critic networks, Q and u respectively, that are used to calculating the target values. The weights of these target networks are then updated by having them slowly track the learned networks. This means that thae target values are constrained to change slowly, greatly improving the stability of learning.
Deep DPG can learn competitive policies for all of our tasks using low-dimensional observations using the same hyper-parameters and network structure.
- It requires only straightforward actor-critic architecture and learning algorithm with very few "moving parts"