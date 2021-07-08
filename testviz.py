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
noise_scale = [.1,.1,.1,.1]
simu = sim.Simulation(dt, t_final, force_fn, noise_scale=noise_scale)
pend = pendulum.Pendulum(4,2,2.0, cfric=.3, pfric=0.1, initial_state=np.array([0,0,0.1,0]))
Q = [0,0,1,0]
R = 5e-6
cont2 = controller.LQR_UKF(pend, dt, 24, Q, R, 6)
cont3 = controller.LQR_GPR(pend, dt, 10, 10, Q, R)

results_kf = simu.simulate(pend, cont2)
pts = 50
kfdata = {
   ('state', 't') : {
      'type' : 'line',
      'label' : 'true state',
      'plotpoints' : pts,
      'linestyle' : '-',
      'color' : 'green'
   },
   ('est','t') : {
      'type' : 'line',
      'label' : 'estimated state',
      'plotpoints' : pts,
      'linestyle' : '-',
      'color' : 'blue'
   },
   ('measured state','t') : {
      'type' : 'scatter',
      'label' : 'measured state',
      'plotpoints' : pts,
      'color' : '#333'
   }
}
resgpr = simu.simulate(pend, cont3)
# shift k+1 prediction from k -> k+1
resgpr[('lpred1','t')] = resgpr[('lpred','t')].shift(1, fill_value=0.0)
resgpr[('nlpred1','t')] = resgpr[('nlpred','t')].shift(1, fill_value=0.0)
resgpr[('nlpred1_upp','t')] = (resgpr[('nlpred','t')] + resgpr[('sigma', 't')]).shift(1, fill_value=0.0)
resgpr[('nlpred1_low','t')] = (resgpr[('nlpred','t')] - resgpr[('sigma', 't')]).shift(1, fill_value=0.0)
resgpr[('lpred err','t')] = (resgpr[('lpred1','t')] - resgpr[('est','t')]).abs()
resgpr[('nlpred err','t')] = (resgpr[('nlpred1','t')] - resgpr[('est','t')]).abs()
visu2 = viz.Visualizer(resgpr, pend, speed=2)
gpdata = {
   ('est', 't') : {
      'label' : 'estimated state',
      'plotpoints' : pts,
      'type' : 'line',
      'linestyle' : '-',
      'color' : 'black'
   },
   ('lpred1', 't') : {
      'type' : 'line',
      'label' : 'linear prediction',
      'plotpoints' : pts,
      'linestyle' : '--',
      'color' : 'red'
   },
   ('nlpred1', 't') : {
      'type' : 'line',
      'label' : 'nonlinear prediction',
      'plotpoints' : pts,
      'linestyle' : '-',
      'color' : 'blue'
   },
   ('nlpred1_upp', 't') : {
      'type' : 'line',
      'label' : '+1 std',
      'plotpoints' : pts,
      'linestyle' : ':',
      'color' : 'lightblue',
   },
   ('nlpred1_low','t') : {
      'type' : 'line',
      'label' : '-1 std',
      'plotpoints' : pts,
      'linestyle' : ':',
      'color' : 'lightblue'
   },
}
gpdata2 = {
   ('nlpred err' ,'t') : {
      'type' : 'scatter',
      'label' : 'Nonlinear Error',
      'plotpoints' : pts,
      'color' : 'blue'
   },
   ('lpred err' ,'t') : {
      'type' : 'scatter',
      'label' : 'Linear Error',
      'plotpoints' : pts,
      'color' : 'red'
   }
}


kfvisu = viz.Visualizer(results_kf, pend, speed=1)
gpvisu = viz.Visualizer(resgpr, pend, speed=1)
anim1 = gpvisu.animate(pltdata=gpdata, interval=40)
anim2 = gpvisu.animate(pltdata=gpdata2, interval=40)
anim3 = kfvisu.animate(pltdata=kfdata, interval=40)

anim1.save('./gp_state_data.mp4')
anim2.save('./gp_error.mp4')
anim3.save('./kf.mp4')
plt.show()