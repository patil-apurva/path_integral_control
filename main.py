import random
import numpy as np
import math

rng = np.random.default_rng()

xP = -0.5
yP = -0.5
A = 0.6
B = 0.6

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
h = 0.01
t0 = 0
T = 10

#tuning parameters
s2 = 0.01
a = 1
b = 1
d = 1
e = 2

G_u = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
x0 = np.array([[-0.4], [-0.4], [0], [0]])
runs = 100
traj_num = 10

lam = a*s2
k1 = -e/T
k2 = k1
k3 = k1

s = np.sqrt(s2)

xQ = xP + A
yQ = yP + B

eta = 0.6
#######################################
random.seed(1234)

fail_count = 0

for traj_itr in range(traj_num):
    print(traj_itr)
    X = x0 #to store all the states
    xt = x0
    
    f_xt = np.array([k1*xt[0] + xt[2]*np.cos(xt[3]), k1*xt[1] + xt[2]*np.sin(xt[3]), k2*xt[2], k3*xt[3]])
   
    safe_flag_traj = 1

    for idx in range(0, int(T/h)):
        t = idx*h
        eps_t_all = rng.normal(size=(2, runs))
        S_tau_all = np.zeros([1, runs]) 

        for i in range(runs):
            eps_t_prime = eps_t_all[:, i]
            eps_t_prime = eps_t_prime.reshape((2,1))
            xt_prime = xt
            f_xt_prime = f_xt

            S_tau = 0
            safe_flag_tau = 1

            for j in range(idx, int(T/h)):
                t_prime = j*h 
                S_tau = S_tau + h*b*(np.linalg.norm(xt_prime))**2

                dummy_prime = s*eps_t_prime*math.sqrt(h)
                
                xt_prime = xt_prime + f_xt_prime*h + np.matmul(G_u, dummy_prime)
    
                if (((xt_prime[0]>=xR1) and (xt_prime[0]<=xS1) and (xt_prime[1]>=yR1) and (xt_prime[1]<=yS1)) or ((xt_prime[0]>=xR2) and (xt_prime[0]<=xS2) and (xt_prime[1]>=yR2) and (xt_prime[1]<=yS2)) or ((xt_prime[0]<=xP) or (xt_prime[0]>=xQ) or (xt_prime[1]<=yP) or (xt_prime[1]>=yQ))):
                    S_tau = S_tau + eta
                    safe_flag_tau = 0
                    break

                eps_t_prime = rng.normal(size=(2, 1))
                f_xt_prime = np.array([k1*xt_prime[0] + xt_prime[2]*np.cos(xt_prime[3]), k1*xt_prime[1] + xt_prime[2]*np.sin(xt_prime[3]), k2*xt_prime[2], k3*xt_prime[3]])

            if safe_flag_tau==1:
                S_tau = S_tau + d*(np.linalg.norm(xt_prime))**2

            S_tau_all[0,i] = S_tau


        
        denom_i = np.exp(-S_tau_all/lam)
        numer = np.matmul(eps_t_all, denom_i.transpose())
        denom = np.sum(denom_i)

        ut = (s/math.sqrt(h))*numer/denom

        eps = rng.normal(size=(2, 1))

        dummy = ut*h + s*eps*math.sqrt(h)
        xt = xt + f_xt*h + np.matmul(G_u, dummy)
      
        X = np.concatenate((X, xt), axis=1)
       
        if (((xt[0]>=xR1) and (xt[0]<=xS1) and (xt[1]>=yR1) and (xt[1]<=yS1)) or ((xt[0]>=xR2) and (xt[0]<=xS2) and (xt[1]>=yR2) and (xt[1]<=yS2)) or ((xt[0]<=xP) or (xt[0]>=xQ) or (xt[1]<=yP) or (xt[1]>=yQ))):
            fail_count = fail_count + 1
            safe_flag_traj = 0
            break

        f_xt = np.array([k1*xt[0] + xt[2]*math.cos(xt[3]), k1*xt[1] + xt[2]*math.sin(xt[3]), k2*xt[2], k3*xt[3]])
       
fail_prob = fail_count/traj_num
print (fail_prob)
print(fail_count)

