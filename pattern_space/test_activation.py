import numpy as np
import matplotlib.pyplot as plt
from parameters import beta, U, S, a

r_s = np.linspace(-10, 10, 200)

def activation_active(r, theta_0):
    return np.exp(beta*r*(1-a/S))/((S-1)*np.exp(-beta*r*a/S) + np.exp(beta*r*(1-a/S)) + np.exp(beta*(theta_0+U)))


def activation_inactive(r, theta_0):
    return np.exp(-beta*r*a/S)/(S*np.exp(-beta*r*a/S) + np.exp(beta*(theta_0+U)))


def activation_active_approx(r, theta_0):
    return np.exp(beta*r*(1-a/S))/(np.exp(beta*r*(1-a/S)) + np.exp(beta*(theta_0+U)))

def activation_inactive_approx(r, theta_0):
    return np.exp(-beta*r*a/S)/(S*np.exp(-beta*r*a/S)+np.exp(beta*(theta_0+U)))

plt.plot(r_s, activation_active(r_s, 0))
plt.plot(r_s, activation_inactive(r_s, 0))
plt.plot(r_s, activation_active_approx(r_s, 0))
# plt.plot(r_s, activation_inactive_approx(r_s, 0))
plt.vlines(U, 0, 1)
plt.show()
