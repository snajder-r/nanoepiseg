import numpy as np
import math
import time

def arraylogexpsum(x):
    ret = x[0]
    for i in range(1, len(x)):
        ret = logaddexp(ret,x[i])
    return ret if ret > -256 else -512

def logaddexp(a,b):
    ret = np.logaddexp(a,b)
    return ret
    if np.isscalar(a):
        if a < -256:
            ret = b
        if b < -256:
            ret = a
        else:
            ret = np.logaddexp(a,b)
        return ret if ret > -256 else -512
    else:
        ret[a<-256] = b[a<-256]
        ret[b<-256] = a[b<-256]
        ret[ret<-256] = -512
        return ret

def forward(observations, t_fn, emissions_fn, num_segments, eps=np.exp(-512)):
    M = num_segments
    N = observations.shape[0]
    F = np.zeros((N,M), dtype=np.float)+eps
    F[0,0] = 1 - F[0,:].sum() - eps
    F = np.log(F)
    start_prob = np.zeros(M)+eps
    start_prob[0] = 1
    start_prob = np.log(start_prob)

    for k in range(N):
        o = observations[k]
        for i in range(M):
            e = emissions_fn(i,o)

            if k == 0:
                F[k,i] = e + start_prob[i]
                continue 

            # Stay probability
            F[k,i] = e + F[k-1,i] + t_fn(i,i)

            
            # Move probabilty
            if i > 0:
                F[k,i] = logaddexp(F[k,i], e + F[k-1,i-1] + t_fn(i-1,i))

            # End probability
            if i == M-1:
                # if end state we could have come from anywhere to the end state:
                for j in range(M-2): # exclude last 2 because those were already handled above
                    F[k,i] = logaddexp(F[k,i], e + F[k-1,j] + t_fn(j,i))

    evidence = F[-1,-1]

    return F, evidence


def backward(observations, t_fn, emissions_fn, num_segments, eps=np.exp(-512)):
    M = num_segments
    N = observations.shape[0]
    B = np.zeros((N,M), dtype=np.float)+eps
    B[-1,-1] = 1
    B = np.log(B)


    for k in range(N-1,0,-1):
        o = observations[k]
        k = k -1
        for i in range(M):
            e_stay = emissions_fn(i,o)

            if i == M-1:
                # If i is end state, we can only stay
                B[k,i] = e_stay + B[k+1,i] + t_fn(i,i)
            else:
                e_move = emissions_fn(i+1,o)
                e_end = emissions_fn(M-1,o)
                # Move and stay probability
                B[k,i] = logaddexp(B[k+1,i] + t_fn(i,i) + e_stay, B[k+1,i+1] + t_fn(i,i+1) + e_move)
                if i < M-2:
                    # End probability only if i<M-2 because otherwise it was covered by move or stay
                    B[k,i] = logaddexp(B[k,i], B[k+1,M-1] + t_fn(i,M-1) + e_end)


    o = observations[0]
    evidence = B[0,0] + emissions_fn(0,o)
    return B, evidence

def viterbi(observations, t_fn, emissions_fn, num_segments, eps=np.exp(-512)):
    M = num_segments
    N = observations.shape[0]

    V = np.zeros((N,M), dtype=np.float)+eps
    V[0,0] = 1
    V = np.log(V)
    P = np.zeros((N,M), dtype=np.int32)

    start_prob = np.zeros(M)+eps
    start_prob[0] = 1
    start_prob = np.log(start_prob)


    for k in range(0,N-1):
        o = observations[k]
        for i in range(M):
            e = emissions_fn(i,o)
            
            if k == 0:
                V[k,i] = np.max(e + start_prob[i])
                continue

            p = np.log(np.zeros(M)+eps)

            p[i] = V[k-1,i] + t_fn(i,i)

            if i > 0:
                p[i-1] = V[k-1,i-1] + t_fn(i-1,i)

            if i==M-1:
                for j in range(M-2): # last two have been covered by stay and move
                    p[j] = V[k-1,j] + t_fn(j,i)

            p = e + p

#            print(k, o, i, np.exp(e), np.exp(p), np.argmax(p))
            V[k,i] = np.max(p)
            P[k,i] = np.argmax(p)

    V[-1,:] = -512
    V[-1,-1] = np.max(V[-2,:])
    P[-1,-1] = np.argmax(V[-2,:])

    X = np.zeros(N, dtype=np.int32)
    Z = np.zeros(N, dtype=np.float32)
    X[N-1] = M-1
    Z[N-1] = 0

    for k in range(N-2, -1, -1):
        X[k] = P[k,X[k+1]]
        Z[k] = V[k,X[k]]

    return X,Z

def main():
    eps = np.exp(-512)
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=6)

    met_comp = np.load('/home/r933r/tmp/metcomp.npy')
    comp_samples = np.load('/home/r933r/tmp/metcomp_samples.npy')

    # Methylation rate prior
    metrate = 0.2
    met_prob = 1-1/(1 + np.exp(-met_comp)/(1-metrate) - np.exp(-met_comp))

    max_segments = 5

    # Shape (M,)
    segment_p = np.zeros(max_segments) - np.log(2)
    prior_a = 0.5
    prior_lognormfactor = np.log(math.gamma(2*prior_a)/(2*math.gamma(prior_a)))


    # Shape (N,)
    test_i = 1
    signal = met_prob[test_i]
    signal = signal[met_comp[test_i]!=0]


    np.random.seed(43)
    # Test signal
    signal = np.zeros(200)
    signal[:30] = np.random.binomial(1,0.01,size=30)
    signal[30:50] = np.random.binomial(1,0.99,size=20)
    signal[50:100] = np.random.binomial(1,0.5,size=50)
    signal[100:170] = np.random.binomial(1,0.99,size=70)
    signal[170:] = np.random.binomial(1,0.01,size=30)
#    signal[:10] = 1
#    signal[10:20] = 0
#    signal[20:] = np.random.binomial(1,0.3,size=30)

    noise = np.abs(np.random.normal(loc=0,scale=0.05,size=50))
#    signal[signal==0] = signal[signal==0] + noise[signal==0]
#    signal[signal==1] = signal[signal==1] - noise[signal==1]
#    signal = np.clip(signal,0,1)

    print(signal)

    def transition_probs(i,j):

        
        # Stickyness
        t_stay = 0.5

        if i==j:
            if i==(max_segments-1):
                #End state to end state is 1
                return np.log(1)
            else:
                return np.log(t_stay)

        # Penalty for oversegmentation
        seg_penalty = 0.25
        t_end = seg_penalty + (1-t_stay-seg_penalty)*(i)/(max_segments-1)
        #print("End: ", i,j,t_end+eps)
        if j==(max_segments-1):
            # Probabiblity to go to end state is 0 if we are in start state,
            # 1-t_stay if we are in the last state, or scales in between

            return np.log(t_end + eps)

        if i==(j-1):
            # Probability to move to the next state is 1 minus probability to
            # stay minus probability to go to end state, meaning in the last
            # state move probability is 0
            #print("Move: ",i,j, 1 - t_stay - t_end + eps) 
            return np.log(1 - t_stay - t_end + eps)

        raise RuntimeError('Transition %d to %d is not a valid transition in segmentation HMM '%(i,j))


    for it in range(100):
        segment_prior = prior_lognormfactor + segment_p*(prior_a-1) + np.log(1-np.exp(segment_p)+eps)*(prior_a-1)

        def emission_probs(s, o):
            ret = segment_p[s]*o + (np.log(1-np.exp(segment_p[s])+eps))*(1-o) + segment_prior[s]
            return ret


#        print('P: ',segment_p, np.exp(segment_p))
        F,f_evidence = forward(signal, transition_probs, emission_probs, max_segments)
        B,b_evidence = backward(signal, transition_probs, emission_probs, max_segments)

        posterior = F + B - b_evidence
#        print(np.exp(posterior).sum(axis=1))

        # Maximize
        segment_sum = np.zeros(max_segments)
        segment_scale_factor = np.zeros(max_segments)
        lsignal = np.log(np.clip(signal,eps,1))
        for k in range(signal.shape[0]):
            for i in range(max_segments):
                segment_sum[i] = logaddexp(segment_sum[i], lsignal[k] + posterior[k,i])
                segment_scale_factor[i] = logaddexp(segment_scale_factor[i], posterior[k,i])
        segment_p_new = segment_sum - segment_scale_factor
        if np.max(np.abs(segment_p_new - segment_p)) < np.exp(-16):
            break
        segment_p = segment_p_new
        for s in range(max_segments):
            print(s, np.exp(segment_p[s]), np.exp(segment_scale_factor[s]))

#    print(posterior.argmax(axis=1))
    
    print("True: ",(signal[:10].sum()/10, signal[10:20].sum()/10, signal[20:].sum()/30))

    print("Predicted: ")
    X,Z = viterbi(signal, transition_probs, emission_probs, max_segments)
    for i in range(max_segments):
        if (X == i).sum()>0:
            print(i, (X==i).sum(), np.exp(segment_p[i]), signal[X==i])
    print("Predicted segments: ")
    print(X)

if __name__ == '__main__':
    main()

