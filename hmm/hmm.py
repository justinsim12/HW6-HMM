import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        #edge case
        if len(input_observation_states) == 0:
            return 0
        #

        # Step 1. Initialize the forward matrix
        num_obs = len(input_observation_states)
        num_states = len(self.hidden_states)
        forward_matrix = np.zeros((num_obs, num_states))
        
        # Step 2. Initialize the first column
        for s in range(num_states):
            forward_matrix[0, s] = self.prior_p[s] * self.emission_p[s, self.observation_states_dict[input_observation_states[0]]]
        
        # Step 3. Recursion step
        for t in range(1, num_obs):
            for s in range(num_states):
                forward_matrix[t, s] = sum(forward_matrix[t-1, s_prev] * self.transition_p[s_prev, s] for s_prev in range(num_states)) * self.emission_p[s, self.observation_states_dict[input_observation_states[t]]]
        
        # Step 4. Return final probability
        return np.sum(forward_matrix[-1, :])


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        #edge case
        if len(decode_observation_states) == 0:
            return []
        #
        
        num_obs = len(decode_observation_states)
        num_states = len(self.hidden_states)
        
        # Step 1. Initialize matrices
        viterbi_table = np.zeros((num_obs, num_states))
        backpointer = np.zeros((num_obs, num_states), dtype=int)
        
        # Step 2. Initialize first column
        for s in range(num_states):
            viterbi_table[0, s] = self.prior_p[s] * self.emission_p[s, self.observation_states_dict[decode_observation_states[0]]]
            backpointer[0, s] = 0
        
        # Step 3. Fill in the Viterbi table
        for t in range(1, num_obs):
            for s in range(num_states):
                probabilities = [viterbi_table[t-1, s_prev] * self.transition_p[s_prev, s] for s_prev in range(num_states)]
                best_prev_state = np.argmax(probabilities)
                viterbi_table[t, s] = probabilities[best_prev_state] * self.emission_p[s, self.observation_states_dict[decode_observation_states[t]]]
                backpointer[t, s] = best_prev_state
        
        # Step 4. Traceback to find the best path
        best_path = []
        best_final_state = np.argmax(viterbi_table[-1, :])
        best_path.append(best_final_state)
        
        for t in range(num_obs - 1, 0, -1):
            best_final_state = backpointer[t, best_final_state]
            best_path.insert(0, best_final_state)
        
        # Convert state indices to state names
        best_hidden_state_sequence = [self.hidden_states_dict[state] for state in best_path]
        return best_hidden_state_sequence
