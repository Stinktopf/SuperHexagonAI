This section presents a PPO implementation without Stable-Baselines3, using the same hyperparameter configuration.

## Network Architecture

The PPO agent uses an **Actor-Critic network** with shared feature extraction layers. The model consists of:

- **Shared Layers:** Extract general representations from observations.
- **Policy Head:** Outputs logits for action selection.
- **Value Head:** Predicts state value for advantage estimation.

The model is implemented using the ActorCritic class:

- ::: trainer_PPO_GYM.ActorCritic.__init__
- ::: trainer_PPO_GYM.ActorCritic.forward

## Rollout Collection

During rollouts, the agent interacts with the environment and collects transitions, consisting of:

- **State:** The current observation.
- **Action:** The chosen action.
- **Reward:** The received reward.
- **Next State:** The next observation after the action.
- **Done Flag:** Indicates whether the episode has ended.

### Observation & Forward Pass
The Actor-Critic network processes the state tensor, outputting action logits and a value estimate:

```python
obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
logits, value = self.model(obs_tensor)
```

### Action Selection & Environment Step  
Actions are sampled from a categorical distribution over the logits. The selected action is executed in the environment:

```python
dist = Categorical(torch.softmax(logits, dim=-1))
action = dist.sample()
obs, reward, done, _, _ = self.env.step(action.item())
```


## Entropy Coefficient Update

To balance exploration and exploitation, the entropy coefficient (`ent_coef`) is annealed over the course of the training:

::: trainer_PPO_GYM.PPOAgent.update_ent_coef


## Generalized Advantage Estimation (GAE) [[1]](https://arxiv.org/abs/1506.02438) [[2]](https://arxiv.org/abs/1707.06347)

After collecting a batch of transitions, GAE computes advantages, reducing variance while keeping the estimator unbiased.
 
The advantage function is computed as:

$$
\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + ... + ... + (\gamma\lambda)^{T-t+1}\delta_{T-1} \text{,}
$$

$$
\text{where } \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

This is shown in the following code:

::: trainer_PPO_GYM.PPOAgent.compute_gae

## Mini-Batch Updates

Once the advantages are computed, the data is shuffled and divided into mini-batches for policy and value function updates.

### Forward Pass & Probability Ratio [[2]](https://arxiv.org/abs/1707.06347)
The PPO algorithm relies on the ratio between the new policy and the old policy to ensure that updates are constrained:

$$
r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}
$$

This ratio is computed in the code as follows:

```python
logits, values_pred = self.model(mb_obs)
ratio = torch.exp(new_logprobs - mb_old_logprobs)
```

### Clipped Policy Loss [[2]](https://arxiv.org/abs/1707.06347)
The policy loss uses the Clipped Surrogate Objective to keep updates close to the previous policy:

$$
L^{\text{CLIP}}(\theta) = -\min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right)
$$

> Unlike the original PPO paper, which defines the objective as a maximization problem, our implementation adds a negative sign to transform it into a loss function for minimization.

In code:

```python
surr1 = ratio * mb_advantages
surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * mb_advantages
policy_loss = -torch.min(surr1, surr2).mean()
```

### Value Loss [[2]](https://arxiv.org/abs/1707.06347)
The value function is optimized using a Mean Squared Error (MSE) loss:

$$
L^{\text{VF}}(\theta) = (V_\theta(s_t) - V_t^{targ})^2
$$

In code:

```python
value_loss = ((mb_returns - values_pred.squeeze()) ** 2).mean()
```

### Entropy Bonus [[2]](https://arxiv.org/abs/1707.06347)
The entropy is added to encourage exploration:

$$
L^{\text{ENT}}(\theta) = S[\pi_\theta](s_t)
$$

In code:

```python
entropy = dist.entropy().mean()
```

### Final Loss and Parameter Update [[2]](https://arxiv.org/abs/1707.06347)
The total **PPO loss function** is:

$$
L(\theta) = L^{\text{CLIP} + \text{VF} + \text{ENT}}(\theta) = \hat{\mathbb{E}}_t \left[ L^{\text{CLIP}}(\theta) + c_1 L^{\text{VF}}(\theta) - c_2 L^{\text{ENT}}(\theta) \right]
$$

> In contrast to the original PPO paper that formulates the objective for maximization, our implementation flips the signs to convert it into a minimization loss function.

In code:

```python
loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()
```

### References:

[[1]: High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

[[2]: Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
