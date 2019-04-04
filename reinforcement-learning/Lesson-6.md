# Lesson 6: temporal-difference methods

## 1. Introduction

This lesson covers material in **Chapter 6** (especially 6.1-6.6) of the textbook. In this lesson we will learn " temporal-difference " or TD learning, real life is not an episodic task, we will need to come up with something else than what we previously saw.

The main idea, is if the agent plays chess, it will at every move to estimate a probability that it will be winning the game, for a car, it will have to evaluate constantly that it's going to crash. 

## 2. TD prediction: TD(0)

We will continue with the trend of solving **The prediction Problem first** given a policy $\pi$ how do we evalute its value function $v_{\pi}$. 

We recall at first the algorithm we used in the last chapter, $G_t$ had to be calculated at the end of each episode, which is not what we are interested in, so we modify slightly the definition to take the next step into account.

IMAGE

To deep into the formula:

IMAGE

What this update step accomplishes







