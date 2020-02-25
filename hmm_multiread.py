import numpy as np
import math
import time
import matplotlib.pyplot as plt


def activation_function(x, weight=0.5):
    return (1-np.exp(-np.abs(x)*0.5))

def plot_met_profile(matrix, samples, segment):
    sample_color = {'s1':'g', 's2':'y', 's3':'r'}
    y_off = 0
    start = 0
    end = matrix.shape[1]

    plt.figure(figsize=(12,6))
    for s in ['s3','s2','s1']:
        x = np.arange(start, end)
        part_matrix = matrix[:,x][(samples==s)]
        #x = site_genomic_pos[x] # Translate to actual pos on chrom
        active_reads = np.array((part_matrix!=0).sum(axis=1)).flatten()>0
        
        part_matrix = part_matrix[active_reads]
        hasval = np.array(part_matrix != 0).flatten()
        y = np.arange(part_matrix.shape[0]) + y_off

        x, y = np.meshgrid(x,y)
        x = x.flatten()[hasval]
        y = y.flatten()[hasval]
        matrix_data = np.array(part_matrix).flatten()[hasval]
        color = [[0,1,0,activation_function(-v)] if v < 0 else [1,0,0,activation_function(v)] for v in matrix_data]

        plt.scatter(x,y, c=color, marker='|',s=35)

        x = np.ones(part_matrix.shape[0])*(x.max()+20)
        y = np.arange(part_matrix.shape[0]) +y_off
        plt.scatter(x,y, c=sample_color[s])

        y_off+=part_matrix.shape[0]

    for i in range(1,len(segment)):
        if segment[i] > segment[i-1]:
            plt.plot((i,i),(0,y_off))



def arraylogexpsum(x):
    ret = x[0]
    for i in range(1, len(x)):
        ret = logaddexp(ret,x[i])
    return ret if ret > -256 else -512

def logaddexp(a,b):
    ret = np.logaddexp(a,b)
#    return ret
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
    R = observations.shape[0]
    N = observations.shape[1]
    F = np.zeros((N,M), dtype=np.float)+eps
    F[0,0] = 1 - F[0,:].sum() - eps
    F = np.log(F)
    start_prob = np.zeros(M)+eps
    start_prob[0] = 1
    start_prob = np.log(start_prob)


    for k in range(N):
        o = observations[:,k]
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
    R = observations.shape[0]
    M = num_segments
    N = observations.shape[1]
    B = np.zeros((N,M), dtype=np.float)+eps
    B[-1,-1] = 1
    B = np.log(B)


    for k in range(N-1,0,-1):
        o = observations[:,k]
        k = k -1
        for i in range(M):
            e_stay = emissions_fn(i,o)

            if i == M-1:
                # If i is end state, we can only stay
                B[k,i] = e_stay + B[k+1,i] + t_fn(i,i)
            else:
                e_move = emissions_fn(i+1,o)
                # Move and stay probability
                B[k,i] = logaddexp(B[k+1,i] + t_fn(i,i) + e_stay, B[k+1,i+1] + t_fn(i,i+1) + e_move)
                if i < M-2:
                    # End probability only if i<M-2 because otherwise it was covered by move or stay
                    e_end = emissions_fn(M-1,o)
                    B[k,i] = logaddexp(B[k,i], B[k+1,M-1] + t_fn(i,M-1) + e_end)

    o = observations[:,0]
    evidence = B[0,0] + emissions_fn(0,o)
    return B, evidence

def viterbi(observations, t_fn, emissions_fn, num_segments, eps=np.exp(-512)):
    M = num_segments
    N = observations.shape[1]

    V = np.zeros((N,M), dtype=np.float)+eps
    V[0,0] = 1
    V = np.log(V)
    P = np.zeros((N,M), dtype=np.int32)

    start_prob = np.zeros(M)+eps
    start_prob[0] = 1
    start_prob = np.log(start_prob)


    for k in range(0,N-1):
        o = observations[:,k]
        for i in range(M):
            e = emissions_fn(i,o)
            
            if k == 0:
                V[k,i] = np.max(e + start_prob[i])
                continue

            p = np.zeros(M)-np.inf

            p[i] = V[k-1,i] + t_fn(i,i)

            if i > 0:
                p[i-1] = V[k-1,i-1] + t_fn(i-1,i)

            if i==M-1:
                for j in range(M-2): # last two have been covered by stay and move
                    p[j] = V[k-1,j] + t_fn(j,i)

            p = e + p

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
    print(met_comp.shape)

    # Methylation rate prior
    metrate = 0.2
    met_prob = 1-1/(1 + np.exp(-met_comp)/(1-metrate) - np.exp(-met_comp))
    # Binarize for now, we will figure out uncertainties later
    met_prob[met_prob>0.5] = 1
    met_prob[met_prob<=0.5] = 0
    met_prob[met_comp==0] = -1

    max_segments = 20
    num_reads = met_prob.shape[0]

    signal=met_prob

    # Shape (M,)
    segment_p = np.zeros((num_reads,max_segments)) - np.log(2)
    prior_a = 0.5
    prior_lognormfactor = np.log(math.gamma(2*prior_a)/(2*math.gamma(prior_a)))



    for it in range(100):
        segment_prior = prior_lognormfactor + segment_p*(prior_a-1) + np.log(1-np.exp(segment_p)+eps)*(prior_a-1)
        # For numerical stability:
        segment_prior = np.clip(segment_prior, -256,0)


        def transition_probs(i,j):
            # Stickyness
            t_stay = 0.8

            if i==j:
                #if i==(max_segments-1):
                #    #End state to end state is 1
                #    return np.log(1)
                #else:
                    return np.log(t_stay)

            # Penalty for oversegmentation
            seg_penalty = 0.15
            t_end = seg_penalty + (1-t_stay-seg_penalty)*(i)/(max_segments-1)
            #print("End: ", i,j,t_end+eps)

            if i==(j-1):
                segment_dissimilarity = np.mean(np.abs(np.exp(segment_p[:,i]) - np.exp(segment_p[:,j])))
                # Probability to move to the next state is 1 minus probability to
                # stay minus probability to go to end state, meaning in the last
                # state move probability is 0
                #print("Move: ",i,j, 1 - t_stay - t_end + eps) 
                return np.log(1 - t_stay - t_end + eps + segment_dissimilarity)

            if j==(max_segments-1):
                # Probabiblity to go to end state is 0 if we are in start state,
                # 1-t_stay if we are in the last state, or scales in between

                return np.log(t_end + eps)


            raise RuntimeError('Transition %d to %d is not a valid transition in segmentation HMM '%(i,j))



        def emission_probs(s,o, dbg=False):
            ret = segment_p[:,s]*o + (np.log(1-np.exp(segment_p[:,s])+eps))*(1-o)
            # Only multiply prior if likelihood is actually good, or else we might be multiplying 0 with infinity
            #ret[ret > -256] = ret[ret>-256] + segment_prior[:,s][ret>-256]
            return ret[o!=-1].mean()

#        print('P: ',segment_p, np.exp(segment_p))
        F,f_evidence = forward(signal, transition_probs, emission_probs, max_segments)
        B,b_evidence = backward(signal, transition_probs, emission_probs, max_segments)
        print(f_evidence, b_evidence)

        posterior = F + B - b_evidence

#        print(np.exp(posterior).sum(axis=1))
#        print(posterior)
#        for i in range(posterior.shape[0]):
#            if np.exp(posterior[i]).sum() > 10:
#                print("LL ", B[i])
#                print("EX ", np.exp(B[i]))

        # Maximize
        segment_sum = np.zeros(segment_p.shape)
        segment_scale_factor = np.zeros(segment_p.shape)

        for k in range(signal.shape[1]):
            for r in range(signal.shape[0]):
                o = signal[r,k]
                if o == -1:
                    continue
                lo = np.log(o+eps)
                for i in range(max_segments):
                    segment_sum[r,i] = logaddexp(segment_sum[r,i], lo + posterior[k,i])
                    segment_scale_factor[r,i] = logaddexp(segment_scale_factor[r,i], posterior[k,i])

        segment_p_new = segment_sum - segment_scale_factor
        if np.max(np.abs(segment_p_new - segment_p)) < np.exp(-8):
            break
        segment_p = segment_p_new

#    print(posterior.argmax(axis=1))
    
    print("Predicted: ")
    X,Z = viterbi(signal, transition_probs, emission_probs, max_segments)
    for i in range(max_segments):
        if (X == i).sum()>0:
            print(i, (X==i).sum())
    print("Predicted segments: ")
    print(X)
    plot_met_profile(met_comp, comp_samples, X)
    plt.savefig('/home/r933r/test.png')
    

if __name__ == '__main__':
    main()

