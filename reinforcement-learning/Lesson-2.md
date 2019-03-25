# Lesson 2: The RL Framework: The Problem

## 2. The Setting, Revisited

The RL framework is characterized by an agent learned to interact with its environment. We will assume that time evolves
at discrete time steps.
At the initial timestep, the agent observes the environment. Then, it must select an appropriate action in response.
Then, at the next timestep in response to the agent action, the environment presents a new situation to the agent.
At the same time the environment gives the agent a reward which provides some indication of wether the agent has 
responded appropriately to the environment. Then, the process continues ...

In general, we don't need to assume that the environment shows the agent everything he needs to make well-informed
decisions. But it greatly simplifies the underlying mathematics if we do. So in this course, we'll make the assumption that the agent is able to fully observe what ever state the environment is in. And instead of referring to the agent as receiving an observation, Huntsworth say that it receives the environment state.
