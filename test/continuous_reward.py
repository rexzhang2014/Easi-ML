#%%
Q0 = np.zeros((rate.shape[0], 2, 2))

A = lambda st, at : (st * at) + (1-st) * (1-at)

R = lambda p, h, c, r : (h * r if c != 0 else h) - p
#%%   
def ql_rate(principal, rate_table, Q0, R, A, alpha, gamma, s0, max_iter, max_step=99) :

    Q = Q0
    total_reward = []
    
    for i in range(max_iter) :
        
        rAll = 0
        t = 0
        st = s0
        holding = principal
        print("------INFO at iteration " + str(i) + "---------")
        while t < max_step :
            
            # st means state at time step t
            # st1 means state at time step t+1
            print("-----INFO at step " + str(t) + "-----")
            crt_rate = rate_table.loc[t, "midrate"]
            
            # Find maximal action a
            at = np.argmax(Q[t, st, :] + np.random.rand(Q.shape[2]))
            print("maximal action at time " + str(t) + " is " + str(at))
            # Find Q(St, at)
            Q_val = Q[t, at, st]
            print("Q(St,at) is " + str(Q_val))
            
            # Find current reward
            rt  = R(principal, holding, st, crt_rate)
            print("Reward at state " +str(st) + " is " + str(rt) )
                        
            # Find next status
            st1 = A(st, at)
            print("State transit from " + str(st) + " to " + str(st1))
                     
            rAll += rt
            
            
            # Find max_a{Q(S_t+1, a )}
            Qt1 = np.max(Q[t, st1, :])
            #print("max Q(St+1, a) is " + str(Qt1))
            
            # if cny is holding the holding value is dominated by the rate
            if st == 0 and st1 == 1 : 
                holding = holding * crt_rate
            elif st == 1 and st == 0 :
                holding = holding / crt_rate
                
            print("(at, Q_val, rt, Qt1, st, st1, holding): " 
                  + str((at, Q_val, rt, Qt1, st, st1, holding)))
            
            # Update Q-table
            Q[t, st, at] = (1 - alpha) * Q_val + alpha * (rt + gamma * Qt1)
            
            # all calculation done, real transit    
            st = st1
            t += 1
        last_minute_holding = holding
        total_reward += [rAll]
        print( "reward in iter " + str(i) + " is " + str(rAll) )
        print( "last_minute_holding in iter " + str(i) + " is " + str(last_minute_holding) )
            
    return Q, total_reward, last_minute_holding
policy, holding, last_minute_holding = ql_rate(100, rate, Q0, R, A, 0.5, 0.5, 0, 1, 329)
