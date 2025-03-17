The Gymnasium environment is presented below.

## Action Space

Despite the game's difficulty, its input system is simple, with a one-dimensional action space offering three possible actions:

0. **Stay**: Remain still. No action will be executed in the game.
1. **Left**: Rotate counterclockwise. Will call `_left(True)` at the start of the frame and `_left(False)` at the end of the frame.
2. **Right**: Rotate clockwise. Will call `_right(True)` at the start of the frame and `_right(False)` at the end of the frame.

This defines the complete set of actions required to play the game.
Notably, we introduced the "Stay" action, which refrains from pressing any input.
While this is not an action a player could typically take, it represents the absence of movement, which can be strategically beneficial in certain situations.

## Observation Space

The observation space consists of several variables that provide insights into the current state of the game. It is limited to three main variables that utilize a total of 8 values:

- **[0]: Normalized player angle:**
<div align="left">
  <img src="../images/player_angle.svg" width="70%">
</div>
::: environment.SuperHexagonGymEnv._get_norm_player_angle

- **[1]: Direction indicator to next free slot:**
The image below shows an example scenario. The method retrieves all walls and finds the edges of the closest free slot in both directions. After calculating the delta angle of both edges in relation to the player, it returns the direction with the minimal value (For discrete value buckets, see the method definition below).  
 In this example, the resulting value would be 1.0 (Right).
<div align="center">
  <img src="../images/direction_indicator.drawio.svg" width="70%">
</div>
::: environment.SuperHexagonGymEnv._get_direction_indicator

- **[2-7]: Normalized wall distances per slot:**
  ::: environment.SuperHexagonGymEnv._get_norm_wall_distances

These values are calculated at each step in the `_get_state()` method and are used as inputs for the model:

- ::: environment.SuperHexagonGymEnv._get_state

## Reward Calculation

The reward system was refined multiple times throughout the agent's development. It combines both rewards (positive values) and penalties (negative values) to provide more direct feedback. We distinguish between two cases:

- **Agent is in a free slot**

  - `1.0`: If the "Stay" action is selected.
  - `0.7`: If any action other than "Stay" is selected.
  - This will always result in a positive reward, with the value depending on the action chosen. This approach was adopted to account for the player "jittering" within a free slot, even though it may remain in a safe position. Such jittering sometimes led to a gameover when the agent hit the wall due to slight movement near the slot edges.

- **Agent is in an occupied slot**
  - `[(-1)-0]`: A continuous negative value, which depends on the distance to the closest free slot. As the agent moves further away from a free slot, the value decreases (becoming closer to -1). Conversely, the value increases as the agent approaches a free slot, reaching 0 and eventually transitioning to a reward.
  - This was implemented to give the agent a stronger incentive to reach a free slot as quickly as possible, avoiding wasting time in an occupied slot.

> "Free slot" is defined as a slot where, if the player remains in it, they will transition to the next wave of walls without resulting in a gameover.

- ::: environment.SuperHexagonGymEnv._get_reward

## Debugging Controls

Due to the sped-up nature of the game, debugging became quite challenging.
To address this, we implemented key combinations that allow us to interact with the game while it is running.
Using the `keyboard` library, we listened for key presses and created the following features:

- Freezing the entire process:
  - `Ctrl + Space`
- Slowing down time:
  - `Ctrl + Alt`
- Toggling Debug logs:
  - `Ctrl + D`

These controls enabled us to inspect specific scenarios in greater detail, gaining deeper insights into the agent's behavior and its internal values.
With these debugging tools, we were able to identify the previously discussed player rotation anomaly and also detect some challenging layouts for the agent.