HiddenMarkovModel

Overview

The HiddenMarkovModel class implements a Hidden Markov Model (HMM) with functionalities to compute the likelihood of an observation sequence using the Forward algorithm and to decode the most likely sequence of hidden states using the Viterbi algorithm.

Initialization

HiddenMarkovModel(observation_states, hidden_states, prior_p, transition_p, emission_p)

Parameters:

observation_states (np.ndarray): List of possible observed states.

hidden_states (np.ndarray): List of possible hidden states.

prior_p (np.ndarray): Prior probabilities of the hidden states.

transition_p (np.ndarray): Transition probabilities between hidden states.

emission_p (np.ndarray): Emission probabilities from hidden states to observed states.

Methods

forward(input_observation_states: np.ndarray) -> float

Description:

Computes the probability of an observation sequence using the Forward algorithm.

Parameters:

input_observation_states (np.ndarray): Sequence of observed states.

Returns:

float: The likelihood of the given observation sequence occurring under the model.

Algorithm Steps:

Initialize the forward matrix with the prior and emission probabilities.

Iterate through time steps, updating probabilities based on previous states and transition probabilities.

Sum the final probabilities to get the likelihood of the sequence.

Edge Cases:

If the input observation sequence is empty, the function returns 0.

viterbi(decode_observation_states: np.ndarray) -> list

Description:

Finds the most probable sequence of hidden states for a given sequence of observations using the Viterbi algorithm.

Parameters:

decode_observation_states (np.ndarray): Sequence of observed states to decode.

Returns:

list: The most likely sequence of hidden states that generated the observed sequence.

Algorithm Steps:

Initialize the Viterbi table and backpointer matrix.

Iterate through time steps, computing the highest probability path to each state.

Traceback from the final state to retrieve the most likely sequence of hidden states.

Edge Cases:

If the input observation sequence is empty, the function returns an empty list.

Example Usage

import numpy as np

# Define states and probabilities
observation_states = np.array(['A', 'B', 'C'])
hidden_states = np.array(['X', 'Y'])
prior_p = np.array([0.6, 0.4])
transition_p = np.array([[0.7, 0.3], [0.4, 0.6]])
emission_p = np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])

# Create HMM instance
hmm = HiddenMarkovModel(observation_states, hidden_states, prior_p, transition_p, emission_p)

# Run forward algorithm
observations = np.array(['A', 'B', 'C'])
probability = hmm.forward(observations)
print("Forward Probability:", probability)

# Run Viterbi algorithm
best_hidden_sequence = hmm.viterbi(observations)
print("Most Likely Hidden State Sequence:", best_hidden_sequence)

Notes

This implementation assumes that observation_states and hidden_states are mapped correctly to their respective indices.

The transition_p and emission_p matrices must be properly normalized (rows should sum to 1).

License

MIT License.





# HW6-HMM

In this assignment, you'll implement the Forward and Viterbi Algorithms (dynamic programming). 


# Assignment

## Overview 

The goal of this assignment is to implement the Forward and Viterbi Algorithms for Hidden Markov Models (HMMs).

For a helpful refresher on HMMs and the Forward and Viterbi Algorithms you can check out the resources [here](https://web.stanford.edu/~jurafsky/slp3/A.pdf), 
[here](https://towardsdatascience.com/markov-and-hidden-markov-model-3eec42298d75), and [here](https://pieriantraining.com/viterbi-algorithm-implementation-in-python-a-practical-guide/). 





## Tasks and Data 
Please complete the `forward` and `viterbi` functions in the HiddenMarkovModel class. 

We have provided two HMM models (mini_weather_hmm.npz and full_weather_hmm.npz) which explore the relationships between observable weather phenomenon and the temperature outside. Start with the mini_weather_hmm model for testing and debugging. Both include the following arrays:
* `hidden_states`: list of possible hidden states 
* `observation_states`: list of possible observation states 
* `prior_p`: prior probabilities of hidden states (in order given in `hidden_states`) 
* `transition_p`: transition probabilities of hidden states (in order given in `hidden_states`)
* `emission_p`: emission probabilities (`hidden_states` --> `observation_states`)



For both datasets, we also provide input observation sequences and the solution for their best hidden state sequences. 
 * `observation_state_sequence`: observation sequence to test 
* `best_hidden_state_sequence`: correct viterbi hidden state sequence 


Create an HMM class instance for both models and test that your Forward and Viterbi implementation returns the correct probabilities and hidden state sequence for each of the observation sequences.

Within your code, consider the scope of the inputs and how the different parameters of the input data could break the bounds of your implementation.
  * Do your model probabilites add up to the correct values? Is scaling required?
  * How will your model handle zero-probability transitions? 
  * Are the inputs in compatible shapes/sizes which each other? 
  * Any other edge cases you can think of?
  * Ensure that your code accomodates at least 2 possible edge cases. 

Finally, please update your README with a brief description of your methods. 



## Task List

[TODO] Complete the HiddenMarkovModel Class methods  <br>
  [ ] complete the `forward` function in the HiddenMarkovModelClass <br>
  [ ] complete the `viterbi` function in the HiddenMarkovModelClass <br>

[TODO] Unit Testing  <br>
  [ ] Ensure functionality on mini and full weather dataset <br>
  [ ] Account for edge cases 

[TODO] Packaging <br>
  [ ] Update README with description of your methods <br>
  [ ] pip installable module (optional)<br>
  [ ] github actions (install + pytest) (optional)


## Completing the Assignment 
Push your code to GitHub with passing unit tests, and submit a link to your repository [here](https://forms.gle/xw98ZVQjaJvZaAzSA)

### Grading 

* Algorithm implementation (6 points)
    * Forward algorithm is correct (2)
    * Viterbi is correct (2)
    * Output is correct on small weather dataset (1)
    * Output is correct on full weather dataset (1)

* Unit Tests (3 points)
    * Mini model unit test (1)
    * Full model unit test (1)
    * Edge cases (1)

* Style (1 point)
    * Readable code and updated README with a description of your methods 

* Extra credit (0.5 points)
    * Pip installable and Github actions (0.5)
