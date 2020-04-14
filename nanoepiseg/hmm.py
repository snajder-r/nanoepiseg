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

class SegmentationHMM:
    def __init__(self, max_segments, t_stay, t_move, seg_penalty=0, prior_a=None, eps=np.exp(-512)):
        self.eps = eps
        self.prior_a = prior_a
        if not prior_a is None:
            self.prior_lognormfactor = np.log(math.gamma(2*prior_a)/(math.gamma(prior_a)**2))
        self.num_segments = max_segments

        if t_stay + t_move + seg_penalty > 1:
            raise ValueError('t_stay + t_move + seg_penalty may not exceed 1')

        self.seg_penalty = np.array([(seg_penalty * i / max_segments) for i in range(max_segments)], dtype=np.float64)
        self.t_move = np.array([t_move -self. seg_penalty[i] for i in range(max_segments)], dtype=np.float64)
        self.t_stay = np.array([t_stay for  i in range(max_segments)], dtype=np.float64)
        self.t_end = np.array([1-self.t_move[i]-self.t_stay[i] for i in range(max_segments)], dtype=np.float64)
        self.t_move = np.log(self.t_move + eps)
        self.t_stay = np.log(self.t_stay + eps)
        self.t_end = np.log(self.t_end + eps)

    def t_fn(self, i, j):
        if i==j:
            return self.t_stay[i]

        if j==(self.num_segments-1):
            # Probability to go the last segment
            return self.t_end[i]
        if i==(j-1):
            # Probability to move to the next state 
            return self.t_move[i]

        raise RuntimeError('Transition %d to %d is not a valid transition in segmentation HMM '%(i,j))

    def forward(self, observations, e_fn):
        M = self.num_segments
        R = observations.shape[0]
        N = observations.shape[1]
        F = np.zeros((N,M), dtype=np.float)+self.eps
        F[0,0] = 1 - F[0,:].sum() - self.eps
        F = np.log(F)
        start_prob = np.zeros(M)+self.eps
        start_prob[0] = 1
        start_prob = np.log(start_prob)


        for k in range(N):
            o = observations[:,k]
            for i in range(M):
                e = e_fn(i,o)

                if k == 0:
                    F[k,i] = e + start_prob[i]
                    continue 

                # Stay probability
                F[k,i] = e + F[k-1,i] + self.t_fn(i,i)

                
                # Move probabilty
                if i > 0:
                    F[k,i] = logaddexp(F[k,i], e + F[k-1,i-1] + self.t_fn(i-1,i))

                # End probability
                if i == M-1:
                    # if end state we could have come from anywhere to the end state:
                    for j in range(M-2): # exclude last 2 because those were already handled above
                        F[k,i] = logaddexp(F[k,i], e + F[k-1,j] + self.t_fn(j,i))
        evidence = F[-1,-1]
        return F, evidence

    def backward(self, observations, e_fn):
        R = observations.shape[0]
        M = self.num_segments
        N = observations.shape[1]
        B = np.zeros((N,M), dtype=np.float)+self.eps
        B[-1,-1] = 1
        B = np.log(B)

        for k in range(N-1,0,-1):
            o = observations[:,k]
            k = k -1
            for i in range(M):
                e_stay = e_fn(i,o)

                if i == M-1:
                    # If i is end state, we can only stay
                    B[k,i] = e_stay + B[k+1,i] + self.t_fn(i,i)
                else:
                    e_move = e_fn(i+1,o)
                    # Move and stay probability
                    B[k,i] = logaddexp(B[k+1,i] + self.t_fn(i,i) + e_stay, B[k+1,i+1] + self.t_fn(i,i+1) + e_move)
                    if i < M-2:
                        # End probability only if i<M-2 because otherwise it was covered by move or stay
                        e_end = e_fn(M-1,o)
                        B[k,i] = logaddexp(B[k,i], B[k+1,M-1] + self.t_fn(i,M-1) + e_end)

        o = observations[:,0]
        evidence = B[0,0] + e_fn(0,o)
        return B, evidence

    def viterbi(self, observations, e_fn):
        M = self.num_segments
        N = observations.shape[1]

        V = np.zeros((N,M), dtype=np.float)+self.eps
        V[0,0] = 1
        V = np.log(V)
        P = np.zeros((N,M), dtype=np.int32)

        start_prob = np.zeros(M)+self.eps
        start_prob[0] = 1
        start_prob = np.log(start_prob)

        for k in range(0,N-1):
            o = observations[:,k]
            for i in range(M):
                e = e_fn(i,o)
                
                if k == 0:
                    V[k,i] = np.max(e + start_prob[i])
                    continue

                p = np.zeros(M)-np.inf

                p[i] = V[k-1,i] + self.t_fn(i,i)

                if i > 0:
                    p[i-1] = V[k-1,i-1] + self.t_fn(i-1,i)

                if i==M-1:
                    for j in range(M-2): # last two have been covered by stay and move
                        p[j] = V[k-1,j] + self.t_fn(j,i)

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

    def baum_welch(self, observations, e_fn_unsalted, tol=np.exp(-4)):
        # Initial guess of parameters
        N=observations.shape[1]
        R=observations.shape[0]
        M=self.num_segments
        segment_p = np.zeros((R,M)) - np.log(2)

        for it in range(100):
            if not self.prior_a is None:
                segment_prior = self.prior_lognormfactor + segment_p*(self.prior_a-1) + np.log(1-np.exp(segment_p)+self.eps)*(self.prior_a-1)
                # For numerical stability:
                segment_prior = np.clip(segment_prior, -10,0.5)
            else: 
                segment_prior = None

            # Salted function of emission likelihood
            e_fn = lambda i,o: e_fn_unsalted(i, o, segment_p, segment_prior)

            F,f_evidence = self.forward(observations, e_fn)
            B,b_evidence = self.backward(observations, e_fn)
            print(f_evidence, b_evidence)

            posterior = F + B - b_evidence

            # Maximize
            segment_sum = np.zeros(segment_p.shape)
            segment_scale_factor = np.zeros(segment_p.shape)

            for k in range(N):
                for r in range(R):
                    o = observations[r,k]
                    if o == -1:
                        continue
                    lo = np.log(o+self.eps)
                    certainty = np.log(0.5 + np.abs(o-0.5)+self.eps)
                    for i in range(M):
                        if posterior[k,i] > -128:
                            if o > 0.5:
                                segment_sum[r,i] = logaddexp(segment_sum[r,i],  posterior[k,i] + certainty)
                            segment_scale_factor[r,i] = logaddexp(segment_scale_factor[r,i], posterior[k,i] + certainty)

            segment_p_new = segment_sum - segment_scale_factor
            diff =  np.max(np.abs(np.exp(segment_p_new) - np.exp(segment_p)))
            print("Diff: ", diff)
            if diff < tol:
                break
            segment_p = segment_p_new
        return segment_p, segment_prior

