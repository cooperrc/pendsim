from numpy.lib.function_base import kaiser
from numpy.lib.twodim_base import triu_indices_from
from pendulum import controller, pendulum, sim, utils, viz
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

dt, t_final = 0.01, 10
c1, c2, c3, c4, fshift = 60, 2.0, 4, .9, 3
def force_fn(t):
  return c1 * np.sin(c2*t) * c3/(c4*np.sqrt(np.pi)) * np.exp(-((t-fshift)/c4)**2)
noise_scale = [.3,.3,.3,.3]
simu = sim.Simulation(dt, t_final, force_fn, noise_scale=noise_scale)
pend = pendulum.Pendulum(4,2,2.0, cfric=.3, pfric=0.1, initial_state=np.array([0,0,0.1,0]))
Q = [0,0,1,0]
R = 5e-6

cont = controller.UKF_GP(pend, dt, window=50, rbflength=25, s=10, testplot=False)
resgpr = simu.simulate(pend, cont)
resgpr.to_parquet('./sim_data.gzip', compression='gzip')

fig, ax1 = plt.subplots(nrows=2, ncols=2, sharex=True)
ax1i = {0:(0,0),1:(0,1),2:(1,0),3:(1,1)}
labels = ['x', 'xd', 't', 'td']
# static charts

# ensemble chart
for i, l in enumerate(labels):
  ax1[ax1i[i]].scatter(resgpr.index, resgpr[('measured state', l)].values, c='#333', marker='.', label='Measured ' + l)
  ax1[ax1i[i]].plot(resgpr[('state', l)], 'k', label='True ' + l)
  ax1[ax1i[i]].plot(resgpr[('est_control', l)], 'r--', label='Filtered (Standard) ' + l)
  ax1[ax1i[i]].plot(resgpr[('est_gp', l)], 'b', label='Filtered (GP) ' + l)
  ax1[ax1i[i]].legend()

# separate charts
for i, l in enumerate(labels):
  fig, ax = plt.subplots()
  ax.scatter(resgpr.index, resgpr[('measured state', l)].values, c='#333', marker='.', label='Measured ' + l)
  ax.plot(resgpr[('state', l)], 'k', label='True ' + l)
  ax.plot(resgpr[('est_control', l)], 'r--', label='Filtered (Standard) ' + l)
  ax.plot(resgpr[('est_gp', l)], 'b', label='Filtered (GP) ' + l)
  ax.legend()
 
# make errors
for i, l in enumerate(labels):
  resgpr[('est_error', l)] = (resgpr[('state', l)] - resgpr[('est_control', l)]).abs()
  resgpr[('est_error_gp', l)] = (resgpr[('state', l)] - resgpr[('est_gp', l)]).abs()

# error charts
for i, l in enumerate(labels):
  fig, ax = plt.subplots()
  ax.plot(resgpr[('est_error', l)], 'r--', label='Error - UKF Only')
  ax.plot(resgpr[('est_error', l)], 'b-', label='Error - UKF + GP')


# animation
anim_data = {
  ('state', 't') : {
    'type' : 'line',
    'label' : 'True State',
    'plotpoints' : 50,
    'linestyle' : '-',
    'color' : 'black',
  },
  ('measured state', 't') : {
    'type' : 'scatter',
    'label' : 'Measured State',
    'plotpoints' : 50,
    'color' : 'grey',
  },
  ('est_control','t') : {
    'type' : 'line',
    'label' : 'Standard UKF',
    'plotpoints' : 50,
    'color' : 'red',
    'linestyle' : '--',
  },
  ('est_gp' , 't') : {
    'type' : 'line',
    'label' : 'UKF with GP',
    'plotpoints' : 50,
    'color' : 'purple',
    'linestyle' : '-',
  },
}

visu = viz.Visualizer(resgpr, pend, speed=1)
anim = visu.animate(pltdata=anim_data, interval=30)
anim.save('./resgpr.mp4')

plt.show()