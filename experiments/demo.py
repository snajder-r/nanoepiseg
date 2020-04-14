import numpy as np

def fwd_bkw(observations, states, start_prob, trans_prob, emm_prob, end_st):
    """Forward–backward algorithm."""
    # Forward part of the algorithm
    fwd = []
    f_prev = {}
    for i, observation_i in enumerate(observations):
        f_curr = {}
        for st in states:
            if i == 0:
                # base case for the forward part
                prev_f_sum = start_prob[st]
            else:
                prev_f_list = [f_prev[k] + trans_prob[k][st] for k in states]
                prev_f_sum = prev_f_list[0]
                for vi in range(1,len(prev_f_list)):
                     prev_f_sum = np.logaddexp(prev_f_sum, prev_f_list[vi])

            f_curr[st] = emm_prob[st][observation_i] + prev_f_sum

        fwd.append(f_curr)
        f_prev = f_curr

    p_fwd_list = [f_curr[k] + trans_prob[k][end_st] for k in states]
    p_fwd = p_fwd_list[0]
    for vi in range(1,len(p_fwd_list)):
        p_fwd = np.logaddexp(p_fwd, p_fwd_list[vi])

    # Backward part of the algorithm
    bkw = []
    b_prev = {}
    for i, observation_i_plus in enumerate(reversed(observations[1:]+(None,))):
        b_curr = {}
        for st in states:
            if i == 0:
                # base case for backward part
                b_curr[st] = trans_prob[st][end_st]
            else:
                b_curr_list = [trans_prob[st][l] + emm_prob[l][observation_i_plus] + b_prev[l] for l in states]
                b_curr[st] = b_curr_list[0]
                for vi in range(len(b_curr_list)):
                    b_curr[st] = np.logaddexp(b_curr[st],b_curr_list[vi])

        bkw.insert(0,b_curr)
        b_prev = b_curr

    
    #p_bkw = sum(start_prob[l] * emm_prob[l][observations[0]] * b_curr[l] for l in states)

    # Merging the two parts
    posterior = []
    for i in range(len(observations)):
        posterior.append({st: np.exp(fwd[i][st] + bkw[i][st] - p_fwd) for st in states})

    print(p_fwd)
    return fwd, bkw, posterior


states = ('Healthy', 'Fever')
end_state = 'E'
 
observations = ('normal', 'cold', 'dizzy','cold','cold','cold','cold','cold','normal', 'dizzy')
 
start_probability = {'Healthy': np.log(0.6), 'Fever': np.log(0.4)}
 
transition_probability = {
   'Healthy' : {'Healthy': np.log(0.69), 'Fever': np.log(0.3), 'E': np.log(0.01)},
   'Fever' : {'Healthy': np.log(0.000000001), 'Fever': np.log(0.9999999), 'E': np.log(0.01)},
   }
 
emission_probability = {
   'Healthy' : {'normal': np.log(0.5), 'cold': np.log(0.4), 'dizzy': np.log(0.1)},
   'Fever' : {'normal': np.log(0.00000000001), 'cold': np.log(0.3), 'dizzy': np.log(0.6)},
   }



f,b,p = fwd_bkw(observations,states,start_probability,transition_probability, emission_probability,end_state)
print([s['Fever'] for s in f])
print([s['Fever'] for s in b])
print([s['Fever'] for s in p])
