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


## Generalized Advantage Estimation (GAE)

After collecting a batch of transitions, GAE computes advantages, reducing variance while keeping the estimator unbiased.
 
The advantage function is computed as:

$$
\delta_t = r_t + \gamma\,(1-d_{t+1})\,V(s_{t+1}) - V(s_t)
$$

$$
A_t = \delta_t + \gamma\,\lambda\,(1-d_{t+1})\,A_{t+1}
$$

$$
R_t = A_t + V(s_t)
$$

This is shown in the following code:

::: trainer_PPO_GYM.PPOAgent.compute_gae


Here is the revised version with the missing mathematical formulas for the **Mini-Batch Updates**, now in **English**:

## Mini-Batch Updates

Once the advantages are computed, the data is shuffled and divided into mini-batches for policy and value function updates.

### Forward Pass & Probability Ratio
The PPO algorithm relies on the ratio between the new policy and the old policy to ensure that updates are constrained:

$$
r_t = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}
$$

This ratio is computed in the code as follows:

```python
logits, values_pred = self.model(mb_obs)
ratio = torch.exp(new_logprobs - mb_old_logprobs)
```

### Clipped Policy Loss
The policy loss uses the Clipped Surrogate Objective to keep updates close to the previous policy:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t A_t, \text{clip}(r_t, 1 - \epsilon, 1 + \epsilon) A_t \right) \right]
$$

In code:

```python
surr1 = ratio * mb_advantages
surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * mb_advantages
policy_loss = -torch.min(surr1, surr2).mean()
```

### Value Loss
The value function is optimized using a Mean Squared Error (MSE) loss:

$$
L^{\text{VF}}(\theta) = \mathbb{E}_t \left[ (V_\theta(s_t) - R_t)^2 \right]
$$

In code:

```python
value_loss = ((mb_returns - values_pred.squeeze()) ** 2).mean()
```

### Entropy Bonus
The entropy is added to encourage exploration:

$$
L^{\text{ENT}}(\theta) = \mathbb{E}_t \left[ H(\pi_\theta) \right]
$$

In code:

```python
entropy = dist.entropy().mean()
```

### Final Loss and Parameter Update
The total **PPO loss function** is:

$$
L(\theta) = L^{\text{CLIP}}(\theta) + c_1 L^{\text{VF}}(\theta) - c_2 L^{\text{ENT}}(\theta)
$$

In code:

```python
loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()
```