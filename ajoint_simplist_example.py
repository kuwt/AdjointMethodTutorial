import numpy as np
import matplotlib.pyplot as plt
import pdb

pdb.set_trace()
##################################
# User define parameters
#################################
u0 = 1
deltat = 0.5

def f_func(u,theta):
    return u*theta
    
def g_func(u,theta):
    return theta*theta + u*u

def dfdtheta_func(u,theta):
    return u

def dfdu_func(u,theta):
    return theta

def dgdtheta_func(u,theta):
    return 2*theta

def dgdu_func(u,theta):
    return 2*u

def lambda_func(lam,u,theta):
    return -dfdu_func(u,theta) * lam - dgdu_func(u,theta) 
##################################
# Utilities
#################################

def explicit_euler_forward_step(u_i, theta_i, delta_t, f_func):
    u_ip1 = u_i + delta_t * f_func(u_i,theta_i)
    return u_ip1

def explicit_euler_backward_step(u_ip1, theta_ip1, lambda_ip1, delta_t, f_func):
    u_i = u_ip1 - delta_t * lambda_func(lambda_ip1,u_ip1,theta_ip1)
    return u_i

def compute_totalcost(u, theta, g_func):
    return np.sum(g_func(u,theta))

#############################
# testing functions
################################
# test forward simulation
def test1():
    timespan = 3
    theta_list = np.asarray([1,2,3])
    u_list = np.asarray([])

    # time loop
    u_last = u0
    u_list = np.append(u_list, u_last)
    for i in range(timespan):
        u = explicit_euler_forward_step(u_last, theta_list[i], deltat, f_func)
        u_last = u
        u_list = np.append(u_list, u_last)
    plt.plot(u_list)
    plt.show()

    theta_list = np.append(theta_list, 0)
    print("cost = {}".format(compute_totalcost(u_list,theta_list, g_func)))

# test derivative computation
def test2():
    timespan = 3
    theta_list = np.asarray([1,2,3])
    u_list = np.asarray([])

    # time loop
    u_last = u0
    u_list = np.append(u_list, u_last)
    for i in range(timespan):
        u = explicit_euler_forward_step(u_last, theta_list[i], deltat, f_func)
        u_last = u
        u_list = np.append(u_list, u_last)

    lambda_list = np.asarray([])
    lambda_final = 0
    lambda_last = lambda_final
    lambda_list = np.append(lambda_list, lambda_last)

    for i in range(timespan):
        timeStamp = timespan - i - 1
        lambda_current = explicit_euler_backward_step(u_list[timeStamp], theta_list[timeStamp], lambda_last, deltat, f_func)
        lambda_last = lambda_current
        lambda_list = np.append(lambda_list, lambda_last)
    lambda_list = lambda_list[::-1]

    theta_list = np.append(theta_list, 0)
    derivative = dgdtheta_func(u_list,theta_list) + lambda_list * dfdtheta_func(u_list,theta_list)
    print("derivative = {}".format(derivative))

if __name__ == '__main__':
    test1()
    test2()
    
