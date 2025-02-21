import pytest
from hmm import HiddenMarkovModel
import numpy as np




def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    # Load HMM parameters
    observation_states = mini_hmm['observation_states']
    hidden_states = mini_hmm['hidden_states']
    prior_p = mini_hmm['prior_p']
    transition_p = mini_hmm['transition_p']
    emission_p = mini_hmm['emission_p']
    
    # Create HMM instance
    hmm = HiddenMarkovModel(observation_states, hidden_states, prior_p, transition_p, emission_p)
    
    # Run Forward and Viterbi algorithms
    observation_sequence = mini_input['observation_sequence']
    expected_forward_prob = mini_input['expected_forward_prob']
    expected_viterbi_path = mini_input['expected_viterbi_path']
    
    forward_prob = hmm.forward(observation_sequence)
    viterbi_path = hmm.viterbi(observation_sequence)
    
    # Assertions
    assert np.isclose(forward_prob, expected_forward_prob, atol=1e-5), "Forward probability mismatch"
    assert viterbi_path == expected_viterbi_path.tolist(), "Viterbi path mismatch"
    
    # Edge cases
    empty_observation_sequence = []
    assert np.isclose(hmm.forward(empty_observation_sequence), 0), "Forward probability should be 0 for empty sequence"
    assert hmm.viterbi(empty_observation_sequence) == [], "Viterbi path should be empty for empty sequence"

def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    full_hmm = np.load('./data/full_weather_hmm.npz')
    full_input = np.load('./data/full_weather_sequences.npz')
    
    # Load HMM parameters
    observation_states = full_hmm['observation_states']
    hidden_states = full_hmm['hidden_states']
    prior_p = full_hmm['prior_p']
    transition_p = full_hmm['transition_p']
    emission_p = full_hmm['emission_p']
    
    # Create HMM instance
    hmm = HiddenMarkovModel(observation_states, hidden_states, prior_p, transition_p, emission_p)
    
    # Run Forward and Viterbi algorithms
    observation_sequence = full_input['observation_sequence']
    expected_viterbi_path = full_input['expected_viterbi_path']
    
    viterbi_path = hmm.viterbi(observation_sequence)
    
    # Assertions
    assert viterbi_path == expected_viterbi_path.tolist(), "Viterbi path mismatch"
    
    # Edge case: Single observation input
    single_observation_sequence = [observation_sequence[0]]
    assert isinstance(hmm.viterbi(single_observation_sequence), list), "Viterbi should return a list for a single observation"
    assert isinstance(hmm.forward(single_observation_sequence), float), "Forward should return a probability for a single observation"