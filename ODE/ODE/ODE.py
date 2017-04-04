import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# total population of interest: N.
N = 1000
# initial number of infected and recovered individuals
# at t=0: I0 and R0.
I0, R0 = 1, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = 1
# Contact rate, a, and mean removal rate, b
# the contact rate is divided by the population count
# the removal rate is given in 1/t units
# so, if we assume days as our time unit, then 1/10 would
# be a removal rate of 10 days, in other words, people 
# stay infectious for 10 days!
a, b = 0.2/N, 1./10
# A grid of time points (in days)
t = np.linspace(0, 160, 160)
# Let's define the ODE system here for the SIR equations:
def sir(y, t, a, b):
    S, I, R = y #remember that odeint deals with a vector!
    dS_dt = -a * I * S
    dI_dt = a * I * S - b * I
    dR_dt = b * I
    return dS_dt, dI_dt, dR_dt

# Here, we need to define our initial state:
y0 = S0, I0, R0
# Now, use odeint to integrate this over the time interval
# to get the populations of interest
p = odeint(sir, y0, t, args=(a,b))
S, I, R = p.T
# Here, we plot the data
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S, 'b', lw=2, label='Susceptible')
ax.plot(t, I, 'r', lw=2, label='Infected')
ax.plot(t, R, 'g', lw=2, label='Removed (immune or dead)')
ax.set_xlabel('Time (in days)')
ax.set_ylabel('Number')
ax.set_ylim(0,1200)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid()
ax.legend()
plt.show()