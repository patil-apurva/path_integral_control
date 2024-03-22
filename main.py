import random
import numpy as np
import math

rng = np.random.default_rng()

#outer rectangle
xP = -0.5
yP = -0.5
A = 0.6
B = 0.6
xQ = xP + A
yQ = yP + B

# obstacles
xR1 = -0.3
yR1 = -0.4
xS1 = -0.2
yS1 = -0.25
xR2 = -0.3
yR2 = -0.15
xS2 = -0.1
yS2 = 0

#time parameters
h = 0.01 #time step
t0 = 0 #initial time
T = 10 #final time

#tuning parameters
s2 = 0.01 #s^2
a = 1 #R=a.I
b = 1 #V=b.||x(t)||^2
d = 1 #psi=d.||x(T)||^2
e = 2

G_u = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
x0 = np.array([[-0.4], [-0.4], [0], [0]]) #initial position
runs = 1000 #monte carlo runs
traj_num = 10 #number of trajectories to plot

lam = a*s2 #PDE linearization constant
k1 = -e/T 
k2 = k1
k3 = k1

s = np.sqrt(s2) #Sigma=s.I

eta = 1 #lagrange multiplier
#######################################
random.seed(1234) #reset the seed

fail_count = 0 #number of trajectories failed

for traj_itr in range(traj_num):
    print(traj_itr)
    X = x0 #to store all the positions of this trajectory
    xt = x0 #start the state from the given initial position
    
    f_xt = np.array([k1*xt[0] + xt[2]*np.cos(xt[3]), k1*xt[1] + xt[2]*np.sin(xt[3]), k2*xt[2], k3*xt[3]]) #initial f_xt
   
    safe_flag_traj = 1

    for idx in range(0, int(T/h)):
        t = idx*h

        eps_t_all = rng.normal(size=(2, runs)) #initial standard normal noise for all the MC trajectories  
        S_tau_all = np.zeros([1, runs]) #an array that stores S(tau) of each sample path starting at time t and state xt

        for i in range(runs):
            eps_t_prime = eps_t_all[:, i] #standard normal noise at t for the ith tau
            eps_t_prime = eps_t_prime.reshape((2,1))
            xt_prime = xt
            f_xt_prime = f_xt

            S_tau = 0 #the cost-to-go of the state dependent cost of tau i
            safe_flag_tau = 1

            for j in range(idx, int(T/h)): #this loop is to compute S(tau_i)
                t_prime = j*h 
                S_tau = S_tau + h*b*(np.linalg.norm(xt_prime))**2 #add the state dependent running cost

                #move tau ahead
                dummy_prime = s*eps_t_prime*math.sqrt(h)
                xt_prime = xt_prime + f_xt_prime*h + np.matmul(G_u, dummy_prime)
    
                if (((xt_prime[0]>=xR1) and (xt_prime[0]<=xS1) and (xt_prime[1]>=yR1) and (xt_prime[1]<=yS1)) or ((xt_prime[0]>=xR2) and (xt_prime[0]<=xS2) and (xt_prime[1]>=yR2) and (xt_prime[1]<=yS2)) or ((xt_prime[0]<=xP) or (xt_prime[0]>=xQ) or (xt_prime[1]<=yP) or (xt_prime[1]>=yQ))):
                    S_tau = S_tau + eta #add the boundary cost to S_tau
                    safe_flag_tau = 0
                    break #end this tau

                eps_t_prime = rng.normal(size=(2, 1)) #standard normal noise at new t_prime. Will be used in the next iteration 
                f_xt_prime = np.array([k1*xt_prime[0] + xt_prime[2]*np.cos(xt_prime[3]), k1*xt_prime[1] + xt_prime[2]*np.sin(xt_prime[3]), k2*xt_prime[2], k3*xt_prime[3]]) #f_xt_prime at new t_prime. Will be used in the next iteration 

            if safe_flag_tau==1: #if tau has not collided 
                S_tau = S_tau + d*(np.linalg.norm(xt_prime))**2 #add the terminal cost to S_tau

            S_tau_all[0,i] = S_tau #save the cost of tau i

        denom_i = np.exp(-S_tau_all/lam) #(size: (1 X runs))
        numer = np.matmul(eps_t_all, denom_i.transpose()) #(size: (2 X 1))
        denom = np.sum(denom_i) #scalar

        ut = (s/math.sqrt(h))*numer/denom #the agent control input

        # move the trajectory forward. Update the position with the control inputs ut=> x(t+h) = x(t) + f.h + G_u.u(t).h + Sigma*dw
        eps = rng.normal(size=(2, 1))
        dummy = ut*h + s*eps*math.sqrt(h)
        xt = xt + f_xt*h + np.matmul(G_u, dummy)
      
        X = np.concatenate((X, xt), axis=1) #stack the new position
       
        if (((xt[0]>=xR1) and (xt[0]<=xS1) and (xt[1]>=yR1) and (xt[1]<=yS1)) or ((xt[0]>=xR2) and (xt[0]<=xS2) and (xt[1]>=yR2) and (xt[1]<=yS2)) or ((xt[0]<=xP) or (xt[0]>=xQ) or (xt[1]<=yP) or (xt[1]>=yQ))):
            fail_count = fail_count + 1 
            safe_flag_traj = 0
            break #end this traj   

        f_xt = np.array([k1*xt[0] + xt[2]*math.cos(xt[3]), k1*xt[1] + xt[2]*math.sin(xt[3]), k2*xt[2], k3*xt[3]]) #update f(x(t)) for the next t => t=t+h. Will be used in the next iteration 
       
fail_prob = fail_count/traj_num
print (fail_prob)
