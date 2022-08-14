HW5, Naïve Bayesian Classification

CS 131

Neal Zhao

4/13/2022

# Assumptions

Following assumptions are made:

- The transition between classes are $0.1$
- The prior probability for both classes is $0.5$
- If a velocity is `nan`, it will be set to previous velocity value. 
- To obtain the possibility of a velocity, the velocity value will be first round to the nearest $0.5$.
- If both classes has probability $0$ after a state change, then this probability change will not be applied (value stays as the last state). 

# Extra Feature

In this implementation, no extra feature is added. 

I noticed that the velocity of birds has a lot of sudden changes, and having a higher variance, from the solution. However, this assignment asked as to build this naïve Bayesian classifier given the probability function of velocity, and I don't think my  model should *learn* from the solution, because I don't know how much the variance of velocity should influence the prediction, and tuning this hyperparameter on the influential level of the variance of velocity on this 10 data points is not reasonable. 