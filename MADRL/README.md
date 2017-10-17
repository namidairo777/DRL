# MADRL
Multi-Agent Deep Reinforcement Learning
## First simple task - pursuit (1 target , n pursuers)
The goal of this task is similiar to my [previous multi-agent project](https://github.com/namidairo777/xiao_multiagent). However, this time we focus on AGI (artificial general intelligence).
### How to create this multi-agent oriented environment to test our algorithm for MA
- Based on OpenAI gym
	- Reward
	- Env
	- Observation

## Related Research
[Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf)
### Proposed method
1. Multi-Agent Actor Critic
Constraints: 1) the learned policies can only use local information (their own observations) at execution time; 2) no differentiable model of env dynamics; 3) no any communication method between agents.<br>
Expected gradient of expected return for agent i,
![P4](https://github.com/namidairo777/DRL/blob/master/MADRL/imgs/P4.png)

<hr>
2. Inferring Policies of other Agents
3. Agents with Policy Ensembles

## Proposed Research
- Based on MADDPG
- Hyperparameter tuning
	- [Proximal Policy Optimization](https://github.com/openai/multiagent-particle-envs)
	- Use an adaptive KL penalty to control the change of the policy at each iteration.
- Asynchronous Advantage Actor-Critic 
	- Multiple workers
	- No need for experiment replay, in other word no need for target network
	- Multi-threading programming
![Chatting](https://github.com/namidairo777/mydiary/blob/master/git_img/chatting.png)
