from numba import njit
from numba import jit
import numpy as np
import time
import timeit
import matplotlib.pyplot as plt

"""
Example 1: Simple performance test

Obtained directly from https://numba.pydata.org/numba-examples/

"""
print('===Example 1====\nSimple compilation test\n')
x = np.arange(100).reshape(10, 10)

@njit
def go_fast(a): # Function is compiled and runs in machine code
    trace = 0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
go_fast(x)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
go_fast(x)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))

"""
Example 2: Corona Virus Simulation 

This example models the time it takes for the corona virus to affect 100% of a population of 
1 Million people based on various behavioral rules. 
Data obtained from: https://ourworldindata.org/coronavirus#our-data-sources

"""
print('\n\n===Example 2====\nCorona-19 Simulation\n')

pop = 1e6

def infect_time_control(infected, day2double):
    day = 1
    while infected < pop:
        infected = np.multiply(infected, np.power(2, np.divide(day, day2double)))
        day += 1
    return day

@njit
def infect_time_numba(infected, day2double):
    day = 1
    while infected < pop:
        infected = np.multiply(infected, np.power(2, np.divide(day, day2double)))
        day += 1
    return day

# Some Real-World Examples
people_infected = 1  # initial number of people infected
world_rate = 7  # rate for doubling in the world
us_rate = 5  # rate for doubling in the us
southKorea_rate = 28 # rate for doubling in South Korea

days_world = infect_time_numba(people_infected, world_rate)
days_us = infect_time_numba(people_infected, us_rate)
days_southKorea = infect_time_numba(people_infected, southKorea_rate)

print("It will take %d days to infect %d people in the World" % (days_world, pop))
print("It will take %d days to infect %d people in the U.S." % (days_us, pop))
print("It will take %d days to infect %d people in South Korea" % (days_southKorea, pop))

time_control = []
time_numba = []
rates = np.arange(1, 1001)
days_control = []
days_numba = []

for rate in rates:
    # Control
    start = time.time()
    days_control.append(infect_time_control(1, rate))
    end = time.time()
    time_control.append(end - start)

    # Numba
    start = time.time()
    days_numba.append(infect_time_numba(1, rate))
    end = time.time()
    time_numba.append(end - start)

fig, ax = plt.subplots()
fig.set_tight_layout(True)

ax.scatter(rates[1:],time_control[1:], linewidths=0.01, marker='.')
ax.scatter(rates[1:],time_numba[1:], linewidths=0.01, marker='.')
ax.set_xlabel('Rate [days to double]')
ax.set_ylabel('Time to execute [s]')
ax.legend(['Control', 'Numba'])
ax.set_title('Performance Comparison')

fig.show()
fig.savefig('Performance Comparison.png')

fig2, ax2 = plt.subplots()
fig2.set_tight_layout(True)

ax2.scatter(rates,days_numba, linewidths=0.01, marker='.')
ax2.set_xlabel('Rate [days to double]')
ax2.set_ylabel('Days to 1 million infected')
ax2.set_title('Time to Infect 1m People vs Doubling Rate')

fig2.show()
fig2.savefig('Infected Time.png')
