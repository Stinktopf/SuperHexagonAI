## Introduction
The architecture uses memory manipulation to extract the game state directly from RAM, train an agent with Proximal Policy Optimization (PPO), and enable it to execute its own inputs in the game. This approach is based on the original memory manipulation from [SuperHexagonAI](https://github.com/polarbart/SuperHexagonAI). **TrainerPPO** handles training using **SuperHexagonGymEnv**, which processes the game state and defines possible actions. **SuperHexagonInterface** acts as a bridge to the game, leveraging **RLHook** to read from and write to **SuperHexagonExe**’s memory, allowing the agent to control the game autonomously.

## Diagram
<div align="center">
  <img src="../images/architecture.svg" width="100%">
</div>
