import numpy as np


def arraylogexpsum(x):
    ret = x[0]
    for i in range(1, len(x)):
        ret = logaddexp(ret,x[i])
    return ret if ret > -256 else -512

def logaddexp(a,b):
    ret = np.logaddexp(a,b)
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

def forward(observations, transitions_fn, emissions_fn, num_segments, eps=np.exp(-512)):
    M = num_segments
    N = observations.shape[0]
    F = np.zeros((N,M), dtype=np.float)+eps
    F[0,0] = 1 - F[0,:].sum() - eps
    F = np.log(F)

    def fwd_it(o, F):
        for i in range(M):
            t_stay, t_move = transitions_fn(i)
            e = emissions_fn(i,o)

            if k == 0:
                F[k,i] = e + [0,-512,-512][i]
                continue 

            F[k,i] = e + F[k-1,i] 

            if i < M-1:
                F[k,i] = F[k,i] + t_stay # we only add stay transition prob if we are not in last state

            if i > 0:
                F[k,i] = logaddexp(F[k,i], e + F[k-1,i-1] + t_move)

        return F


    for k in range(0,N):
        F = fwd_it(observations[k],F)

    evidence = F[-1,-1]

    return F, evidence


def backward(observations, transitions_fn, emissions_fn, num_segments, eps=np.exp(-512)):
    M = num_segments
    N = observations.shape[0]
    B = np.zeros((N,M), dtype=np.float)+eps
    B[-1,-1] = 1
    B = np.log(B)


    for k in range(N-1,0,-1):
        o = observations[k]
        k = k -1
        for i in range(M):
            t_stay, t_move = transitions_fn(i)
            e_stay = emissions_fn(i,o)
            if i == M-1:
                B[k,i] = e_stay + B[k+1,i]
            else:
                e_move = emissions_fn(i+1,o)
                B[k,i] = logaddexp(B[k+1,i] + t_stay + e_stay, B[k+1,i+1] + t_move + e_move)

    o = observations[0]
    evidence = B[0,0] + emissions_fn(0,o)
    return B, evidence

def viterbi(observations, transitions_fn, emissions_fn, num_segments, eps=np.exp(-512)):
    M = num_segments
    N = observations.shape[0]

    V = np.zeros((N,M), dtype=np.float)+eps
    V[0,0] = 1
    V = np.log(V)
    P = np.zeros((N,M), dtype=np.int32)

    for k in range(1,N-1):
        o = observations[k]
        for i in range(M):
            t_stay, t_move = transitions_fn(i)
            e = emissions_fn(i,o)

            t_stay_arr = np.log(np.zeros(M)+eps)
            t_stay_arr[i] = t_stay

            if i == 0:
                # probability coming from start state
                p = e + V[k-1,:] + t_stay_arr
            else:
                t_move_arr = np.log(np.zeros(M)+eps)
                t_move_arr[i-1] = t_move

                # probability for any state in the middle is sum of stay and move probability
                p = e + logaddexp((V[k-1,:] + t_stay_arr), (V[k-1,:] + t_move_arr))
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
        Z[k] = V[k,X[k+1]]

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

    max_segments = 3

    # Shape (M,)
    segment_p = np.zeros(max_segments) - np.log(2)



    # Shape (N,)
    test_i = 1
    signal = met_prob[test_i]
    signal = signal[met_comp[test_i]!=0]


    # Test signal
    signal = np.zeros(50)
    signal[:10] = np.random.binomial(1,0.5,size=10)
    signal[10:20] = np.random.binomial(1,0.95,size=10)
    signal[20:] = np.random.binomial(1,0.5,size=30)
#    signal[:10] = 1
#    signal[10:20] = 0
#    signal[20:] = np.random.binomial(1,0.3,size=30)

    noise = np.abs(np.random.normal(loc=0,scale=0.05,size=50))
    signal[signal==0] = signal[signal==0] + noise[signal==0]
    signal[signal==1] = signal[signal==1] - noise[signal==1]
    signal = np.clip(signal,0,1)

    print(signal)


    print(signal.shape)

    def transition_probs(s):
        # Transition prior probabilities 
        t_stay = 0.9
        t_move = 1-t_stay
        t_stay = np.log(t_stay)
        t_move = np.log(t_move)
        return t_stay, t_move


    for it in range(100):

        def emission_probs(s, o):
            ret = segment_p[s]*o + (np.log(1-np.exp(segment_p[s])))*(1-o)
            return ret


        print('P: ',segment_p, np.exp(segment_p))
        F,f_evidence = forward(signal, transition_probs, emission_probs, max_segments)
        B,b_evidence = backward(signal, transition_probs, emission_probs, max_segments)

        posterior = F + B - b_evidence

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

#    print(posterior.argmax(axis=1))
    
    print("Estimated: ",np.exp(segment_p))
    print("True: ",(signal[:10].sum()/10, signal[10:20].sum()/10, signal[20:].sum()/30))
    X,Z = viterbi(signal, transition_probs, emission_probs, max_segments)
    print("Viterbi reflected: ",(signal[X==0].sum()/(X==0).sum(), signal[X==1].sum()/(X==1).sum(), signal[X==2].sum()/(X==2).sum()))
    print(X)

    print((X==0)[:10].sum(), (X==1)[10:20].sum(),(X==2)[20:].sum())

if __name__ == '__main__':
    main()

