# Lesson 8: RL in Continuous Spaces



## 1. Deep Reinforcement learning

We recall some basic stuff and especially that there are two groups of algorithms:

* **Model-Based Learning (Dynamic Programming)**
  * Policy Iteration
  * Value Iteration
* **Model-Free Learning**
  * Monte Carlo Methods
  * Temporal-Difference Learning

**<u>Deep Reinforcement learning:</u>**

* RL in Continuous Spaces
* Deep Q-Learning
* Policy Gradients
* Actor-Critic Methods

**<u>Resources :</u>**

- [Sutton & Barto, 2nd Ed. - Part II: Approximate Solution Methods]()



## 2. Discrete vs Continuous Spaces

**<u>Discrete :</u>**

- We have a finite number of States and Actions
- It is easy to implement them:
  - For a deterministic policy: we can a map a dictionary where the key is the state and the value the "Value Function"
  - For a stochastic policy: We define a 2D array, with, the rows are the states and the columns are the actions.
- **Limitation :** In some algorithm we had to compute a $max$ which was easy considering we only have a few values. But in the continuous case, we will end up with an optimization problem

**<u>Continuous :</u>**

- We have, States: $s \in \R^n$ and $a \in \R^m$
- use-cases:
  - A robot throwing a dart on a target
  - A cleaner Robot
- To deal with continuous spaces we will have to use:
  - Discretization
  - Function approximation



## 3. Discretization

**<u>Example 1 - Cleaning robot :</u>**

- 

We can mark off the cells where an object is present it is called **occupancy grid**. What we choosed might block the robot.



It is also possible to create a **Non-uniform Discretization**, if it is too small it may takes too much time to calculate a value function



This is the same idea as quad trees or binary space partitioning



**<u>Example 2 - Fuel composition and Speed :</u>**

- 













