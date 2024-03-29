

# Lesson 1: Introduction to RL

## 2. Prise de note Sutton chapitre 1

*Reinforceme learning is not unsupervised or supervised learning*. One of the main challenge in RL is the **trade-off between exploration and exploitation**. To obtain a lot of reward, a reinforcement learning
agent must prefer actions that it has tried in the past and found to be effective in producing reward.
But to discover such actions, it has to try actions that it has not selected before. The agent has to
exploit what it has already experienced in order to obtain reward, but it also has to explore in order to
make better action selections in the future. The dilemma is that neither exploration nor exploitation
can be pursued exclusively without failing at the task

**<u>The four elements of RL:</u>**

* **A policy:**
  * Defines the learning agent’s way of behaving at a given time.
  * In psychology would be called a set of stimulus–response rules
* **A reward signal:**
  * Defines the goal in a reinforcement learning problem
  * In a biological system, we might think of rewards as analogous to the experiences of pleasure or pain.
* **A value function:**
  * Whereas the reward signal indicates what is good in an immediate sense, a value function specifies
    what is good in the long run
* **A model for the environment (optional):**
  * This is something that mimics the behavior of the environment, or more generally, that allows inferences to be made about how the environment will behave.
  * Methods for solving reinforcement learning problems that use models and planning
    are called model-based methods, as opposed to simpler model-free methods that are explicitly trial-anderror learners—viewed as almost the opposite of planning.





## 3. The Setting

We will be concerned about learning with interactions in the field of RL we refer to the Learner or Decision Maker as the agent.
For example, the dog owner commands its dog to sit, the dog doesn’t understand so he will try one of the many actions it can do. It then waits for a feedback to maximise its reward. We will see that this situation is actually quite complexe.
Exploration-Exploitation Dilemma
Exploration: Exploring potential hypotheses for how to choose actions
Exploitation: Exploiting limited knowledge about what is already known should work well; i.e. should he settle for just one treat or should he aim higher ?
If the Puppy is a real RL agent he’s not just concerned with the reward he can get now. Instead his goal is to maximise the number of treat he can get in his entire lifetime.

## 4. OpenAI Gym
It is an open source toolkit for developing and comparing RL algorithms. We’ll use four of the environments that are available as part of this toolkit.
The first is the frozen lake environment. We will write an agent who can navigate a world without falling into pits of frozen water. Next, you’ll write an agent to play blackjack. Then, you’ll work with another small world with a large cliff where your goal is to avoid falling in. The final environment is a taxi world. We will train a taxi to pick up and drop off passengers as quickly as possible.
One of the really cool things about OpenAI Gym is that you can record your performance. 
So your agent might start off just behaving randomly but as it learns from interaction, 
you'll be able to see it choose actions in a much more intelligent way. What's also really cool is that if you're happy with how smart you've made your agents or how quickly they learn, you can upload your implementations to share your knowledge with the world. 

But I won't keep you from it any longer. For this lab, follow the tutorial links below to learn the basic syntax.
Although we won't use open agent just yet. I highly recommend following the tutorial now to build 
your excitement for the kind of implementations you'll build in this course. We'll have to spend a bit of time developing that theory before jumping in. But I guarantee you that it's more than worth it.

**Bibliography:** . 
https://openai.com/blog/openai-gym-beta/ . 
Reinforcement Learning: An Introduction by Richard S. Sutton and Andrew G. Barto . 
https://s3-us-west-1.amazonaws.com/udacity-drlnd/bookdraft2018.pdf . 