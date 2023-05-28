Repository contains experimental code to write Sophia and use it in your LLM model. (LLM refers to the Large Language Model). 

Sophia came out a day ago maybe? The paper does not provide any code whatsoever to write the Optimizer. But, it contains the process/Algorithm to write Sophia. Now, authors might have chosen this as writing an optimizer with given algorithmic instructions might be an easy task. I personally think, if you have the code, it becomes clearer that’s all! 

It is a quite fast-pace project/implementation which is why I have only provided the function for the optimizer and did not include any estimators yet. You can go over the code and get a fresh first hand experience of what the Optimizer looks like. 

##### What’s the repository for?

The repository is for machine learning learners and students alike and personally for myself. It is quite hard to keep up with papers and codes and most of those papers that do not provide code. Someone has to do the hard task of writing this code and reading papers and explaining them. 

Which is why, I wrote this repository. At the moment, I only wrote the hypothetical implementation of the Optimizer no fancy (Pytorch or Jax stuff yet) Kindly take the time to learn and understand. 

###### Status: Epsilon

I am thinking of writing a training script to train the GPT-2 (small) model using this optimizer using Pytorch. It will take a bit time as I have a few submissions and deadlines are near but will try to write them by 2nd week of June (no promise tho)

#### Updates
*Provide a training script with the Optimizer
* Program-out estimators and other functions
* Explain the Optimizer in a blog post (maybe fast)
* Writing a few variations of Sophia 

##### Technical requirements:

If you’re trying to write the Optimizer at its fullest and use it to perform training make sure to use either Pytorch or Jax (as mentioned in the paper) 

#### Theoretical Explanation of the Algorithm in (SPELL OUT):

1. Initialize variables for the first moment estimate (m), second moment estimate (v), and a lagged version of the second moment estimate (h).
2. Iterate from t=1 to T.
3. Compute the minibatch loss (L) for the current parameters (θt).
4. Compute the gradients (g) of the loss with respect to the parameters.
5. Update the first moment estimate (m) using a decaying average of the gradients.
6. If t is divisible by k, compute an estimate of the second moment (h_hat) using the chosen estimator.
7. Update the second moment estimate (h) using a decaying average of the estimators.
8. If t is not divisible by k, use the previous value of the second moment estimate (h).
9. Update the parameters (θt) by applying weight decay and subtracting the learning rate times the current parameters.
10. Compute the updated parameters (θt+1) by subtracting the learning rate times a clipped ratio of the first moment estimate and the maximum of the second moment estimate and a small value (ϵ).
11. Repeat the process for the next iteration.
12. Return the final updated parameters (θt+1).

##### [Paper Link](https://arxiv.org/abs/2305.14342)
