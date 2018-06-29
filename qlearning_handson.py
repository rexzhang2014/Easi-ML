import numpy as np
Q0 = np.zeros((36, 4))
R = np.array([[-100,-100,0,0],
[-100,0,-100,0],
[-100,0,-100,0],
[-100,0,-100,0],
[-100,0,-100,0],
[-100,0,0,-100],
[0,-100,-100,0],
[-100,-100,-100,-100],
[-100,-100,-100,-100],
[-100,-100,-100,-100],
[-100,-100,-100,-100],
[0,-100,-100,0],
[0,-100,-100,0],
[-100,-100,-100,-100],
[0,0,0,0],
[-100,100,0,0],
[-100,0,0,0],
[0,0,0,-100],
[0,-100,0,0],
[-100,0,0,-100],
[-100,-100,-100,-100],
[0,-100,0,0],
[0,0,0,0],
[0,0,0,-100],
[0,-100,0,0],
[0,0,0,-100],
[-100,-100,-100,-100],
[0,-100,0,0],
[0,0,0,0],
[0,0,0,-100],
[0,-100,-100,0],
[0,0,-100,0],
[-100,0,-100,0],
[0,0,-100,0],
[0,0,-100,0],
[0,0,-100,-100]])
A = np.array([lambda s : s-6 if s>6 else s, lambda s : s-1 if s > 1 else s, 
              lambda s : s+6 if s < 30 else s, lambda s : s+1 if s < 35 else s])
#%%
def ql(Q0, R, A, alpha, gamma, s0, max_iter, max_step=99) :
    def reward(s, a) :
        return R[s, a]
    def pi(s, a) :
        return A[a](s)
    Q = Q0
    total_reward = []
    for i in range(max_iter) :
        
        rAll = 0
        t = 0
        st = s0
        print("------INFO at iteration " + str(i) + "---------")
        while t < max_step :
            # st means state at time step t
            # st1 means state at time step t+1
            print("-----INFO at step " + str(t) + "-----")
            # Find maximal action a
            at = np.argmax(Q[st,:] + np.random.rand(Q.shape[1]))
            #print("maximal action is " + str(at))
            # Find Q(St, at)
            Q_val = Q[st, at]
            #print("Q(St,at) is " + str(Q_val))
            
            # Find current reward
            rt  = reward(st, at)
            #print("Reward at state " +str(st) + " is " + str(rt) )
                        
            # Find next status
            st1 = pi(st, at)
            #print("State transit from " + str(st) + " to " + str(st1))
            
            rAll += rt
            
            # Find max_a{Q(S_t+1, a )}
            Qt1 = np.max(Q[st1, :])
            #print("max Q(St+1, a) is " + str(Qt1))
            
            print("(at, Q_val, rt, Qt1, st, st1): " + str((at, Q_val, rt, Qt1, st, st1)))
            # Update Q-table
            Q[st, at] = (1 - alpha) * Q_val + alpha * (rt + gamma * Qt1)
            
            # all calculation done, real transit
            st = st1
            t += 1
        total_reward += [rAll]
        print( "reward in iter " + str(i) + " is " + str(rAll) )
            
    return Q, total_reward

testQ, ttl_reward = ql(Q0, R, A, 0.5, 0.5, 0, 10000)
