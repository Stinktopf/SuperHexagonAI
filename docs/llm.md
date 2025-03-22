# Introduction
The following sections highlight the various ways in which we leveraged Large Language Models (LLMs) throughout the project. 
From exploring the scope of the project, debugging, hyperparameter optimization and generating documentation, LLMs played a crucial role in enhancing efficiency and supporting development tasks.

Used LLMs:

- ChatGPT
- Grok

## Project Exploration
To gain an initial understanding of the project scope, including all its components and technologies, we used LLMs before beginning the actual implementation. 
This helped us quickly grasp the key elements and structure of the project. We specifically used LLMs to get an overview of the PPO (Proximal Policy Optimization) algorithm, exploring its components, working principles, and potential configurations.

## Exploring hyperparameters for Stable-Baseline3
We used LLMs to explore hyperparameters for Stable-Baseline3, allowing us to quickly experiment with different configurations and identify optimal settings.
This approach helped speed up the experimentation phase and ensured we explored a broader range of configurations, ultimately contributing to the improvement of the model’s performance.

## Support for PPO
We implemented the PPO algorithm with the help of the [[1]](https://arxiv.org/abs/1506.02438) [[2]](https://arxiv.org/abs/1707.06347) papers and substantial help of LLMs for the construction of the scaffolding. 
LLMs helped us clarify complex mathematical concepts and answer questions related to the implementation.

## Development of validation DQN
The validation DQN was convert from the PPO implementation via a LLM.
Its main purpose was to serve as a benchmark for comparing performance with the other models.
This is the reason why it performs so poorly.

## Debugging
For troubleshooting, we used LLMs to assist in debugging, allowing us to quickly identify issues in the code and suggesting possible solutions.
The LLMs helped us pinpoint errors more quickly such as syntax mistakes, logical inconsistencies, or missing dependencies.
However, when it came to debugging more complex ML problems, the models often fell short, as they struggled with identifying subtle issues related to model behavior, performance, or the intricacies of training data.
In those cases, manual intervention and deeper analysis were still necessary.

## Generating Mermaid Graphs
To visualize project structure, we used LLMs to generate a Mermaid graph.
These graphs illustrated the relationships between components, classes, and their methods, providing a clear overview of the system architecture.

## Generating Method Documentation
To streamline the documentation process, we used LLMs to generate method documentation, saving time and ensuring consistency.
The generated documentation was then referenced on the website, making it easily accessible for future use.

## Rephrasing and Translation
We used LLMs extensively for translating and rephrasing text from German into English, as it was often faster and provided better results.
This capability was especially valuable when dealing with large volumes of content. As we are simply faster in German. 
