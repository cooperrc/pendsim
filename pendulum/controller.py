from operator import length_hint
import numpy as np
from scipy.signal import cont2discrete
from pendulum.utils import array_to_kv, wrap_pi, sign
import copy
# necessary for GPR
from sklearn import preprocessing, gaussian_process
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool


np.set_printoptions(precision=5,suppress=True)

class Controller(object):
    '''
    Class template for pendulum controller
    '''
    def __init__(self, init_state):
        self.init_state = init_state
    
    def policy(self, state):
        '''
        A controller must have a policy action.
        
        Parameters
        ----------
        state: (:obj:`float`, :obj:`float`, :obj:`float` :obj:`float`)
            The current system state
        
        Returns
        -------
        :obj:`float`
            The controller action, in force applied to the cart.
        '''
        raise NotImplementedError

class PID(Controller):
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integrator = 0
        self.prev = 0
    
    def policy(self, state, t, dt):
        err = - (state[2]  + np.pi) % (2*np.pi) - np.pi
        errd = (err - self.prev) / dt
        self.integrator += err
        action = self.kp * err + self.ki * self.integrator + self.kd * errd
        self.prev = err

        data = {}
        labels = ['x', 'xd', 't', 'td']
        data.update(array_to_kv('zeros', labels, np.zeros(len(labels)) ))
        return action, data

class LQR(Controller):
    def __init__(self, pend, dt, window, Q, R):
        self.window = window
        A = pend.jacA
        B = pend.jacB
        C, D = np.zeros((1, A.shape[0])), np.zeros((1, 1))
        sys_disc = cont2discrete((A,B,C,D), dt, method='zoh')
        self.A, self.B = sys_disc[0], np.atleast_2d(sys_disc[1])
        self.Q = np.diag(Q)
        self.R = np.atleast_2d(R)

    def policy(self, state, t, dt):
        action = do_lqr(self.window, self.A, self.B, self.Q, self.R, state)
        data = {}
        labels = ['x', 'xd', 't', 'td']
        return action, {}

class NoController(Controller):
    def __init__(self):
        pass
    
    def policy(self, state, t, dt):
        return 0, {}

class BangBang(Controller):
    def __init__(self, setpoint, magnitude, threshold):
        '''Simple "BangBang" style controller:
        if it's on turn it off
        if it's off turn it on

        Parameters
        ----------
        setpoint : :obj:`float`
            angle, radians
        magnitude : :obj:`float`
            system gain
        threshold : :obj:`float`
            max angle
        '''
        self.setpoint = setpoint
        self.magnitude = magnitude
        self.threshold = np.pi/4
    
    def policy(self, state, t, dt):
        error = state[2] - self.setpoint
        action = 0
        if error > 0.1 and state[2] < self.threshold:
            action = -self.magnitude
        elif error < -0.1 and state[2] > -self.threshold:
            action = self.magnitude
        else:
            action = 0
        return action, {}

class LQR_GPR(Controller):
    '''A controller that controls with an LQR controller (with a swing-up strategy)
    but which estimates true state using Gaussian Process Regression
    
    Parameters
    ----------
    pend : Pendulum
        The pendulum to be controlled
    dt : float
        simulation timestep
    window : int
        the number of timesteps for the LQR (forward) window
    bwindow : int
        the number of timesteps for collecting the GP train set
    Q : np.array or array-like
        4x1 array for the `Q` control matrix 
    R : float
        scalar cost of control
    
    '''
    def __init__(self, pend, dt, window, bwindow, Q, R, s=10):
        # LQR params
        self.window = window
        self.Q = np.diag(Q)
        self.R = np.atleast_2d(R)
        self.pend = pend
        self.dt = dt

        # model params
        A = pend.jacA
        B = pend.jacB
        C, D = np.zeros((1, A.shape[0])), np.zeros((1, 1))
        sys_disc = cont2discrete((A,B,C,D), dt, method='zoh')
        self.A, self.B = sys_disc[0], np.atleast_2d(sys_disc[1])

        # GPR Params
        self.M = bwindow
        self.tick = 0
        # priors for supervised learning problem (gpr)
        self.priors_gp = []
        # priors for calculating variance (kf)
        self.priors = []

        # kf params
        self.s = s
        self.vw = 20
        self.kf = self.create_ukf()

    @ignore_warnings(category=ConvergenceWarning)
    def policy(self, state, t, dt):
        self.priors.append(state)        
        l = max(self.tick - self.vw, 1)
        u = self.tick
        # do kalman filtering
        if self.tick >= 2:
            var = np.std(np.vstack(self.priors[l:u]), axis=0)
        else:
            var = np.asarray([0.4] * 4)
        self.kf.Q = np.diag(var)
        # Kalman filter predict
        self.kf.predict()
        # Kalman filter update
        self.kf.update(np.array(state))

        state = self.kf.x

        ### Wrap 
        x = copy.deepcopy(wrap_pi(state))
        ### Solve LQR
        if np.abs(x[2]) < np.pi/4:
            action = do_lqr(self.window, self.A, self.B, self.Q, self.R, state)
        else:
            action = self.swingup(state, 50)

        action = 0

        if self.tick > 2:
            loweri = max(self.tick-self.M, 1)
            upperi = self.tick
            xk1 = np.atleast_2d(self.priors_gp)[loweri:upperi] # k-1 state

            xk = np.vstack( (np.atleast_2d(self.priors_gp)[loweri+1:upperi,:4], state) )
            linearpred = np.dot(xk1[:,:4], self.A) + np.dot(np.atleast_2d(xk1[:,4]).T, self.B.T)

            y = np.atleast_2d(xk - linearpred)[:,2] # M x n_d
            z = np.atleast_2d(xk1) # M x n_z
            SC = preprocessing.StandardScaler()
            SC = SC.fit(z)
    
            z_trans = SC.transform(z)
            rq = gaussian_process.kernels.RBF(4.0, length_scale_bounds=(0.75,5))
            ck = gaussian_process.kernels.ConstantKernel(constant_value=1.0)
            no = gaussian_process.kernels.WhiteKernel(noise_level=1, noise_level_bounds=(0.0001,1))
            gp = gaussian_process.GaussianProcessRegressor(
                kernel=rq*ck+ no,
                n_restarts_optimizer=6,
                alpha=1e-6
            )
            gp.fit(z_trans, y)
            indata = np.atleast_2d(list(state) + [action])
            indata_trans = SC.transform(indata)
            mu, sigma = gp.predict(indata_trans, return_std=True)
        else:
            mu, sigma = 0.0,0.0000001
        

        lpred = np.dot(np.atleast_2d(state), self.A) + np.dot(self.B, action).T
        nlpred = np.squeeze(lpred[0,2]) - mu

        mu = np.float64(np.squeeze(mu))
        sigma = np.float64(np.squeeze(sigma))
        lpred = np.float64(np.squeeze(lpred[0,2]))
        nlpred = np.float64(np.squeeze(nlpred))

        data = {
            ('mu','t') : mu,
            ('sigma','t'): sigma,
            ('lpred','t') : lpred,
            ('nlpred','t') : nlpred,
        }
        labels = ['x', 'xd', 't', 'td']
        data.update(array_to_kv('est', labels, state))
        # keep track of history
        self.tick += 1
        self.priors_gp.append(list(state) + [action])
        return action, data

    def create_ukf(self):
        def fx(x, dt): return self.A.dot(x)
        def hx(x): return x
        points2 = sigma_points.SimplexSigmaPoints(4)
        kf = UnscentedKalmanFilter(4, 4, self.dt, hx, fx, points2)
        # initialize noise
        kf.Q = np.diag([.2] * 4)
        # initialize smoothing
        kf.R = np.diag([self.s] * 4)
        return kf

    def swingup(self, x,k):
        m, g, l = self.pend.m, self.pend.g, self.pend.l
        E_norm = 2*m*g*l
        E = m * g * l * (np.cos(x[2]) - 1) # 0 = upright
        beta = E/E_norm
        u = k* beta * sign(x[3] * np.cos(x[2]))
        return - u

from filterpy.kalman.UKF import UnscentedKalmanFilter
from filterpy.kalman import sigma_points

class UKF_GP(Controller):
    def __init__(self, pend, dt, window = 10, s=10, rbflength=20, testplot=False):
        self.pend = pend
        self.dt = dt
        self.A, self.B = self._get_linear_sys(pend, dt)
        self.s = s
        self.window = window + 1
        self.n = 0
        self.testplot = testplot
        self.rbflength = rbflength

        self.priors, self.ts = [], []
        def fx(x, dt, mu=0): 
            xk1 = np.dot(self.A, x) - mu
            return xk1
        # standard UKF with no mu arg passed
        self.ukf_control = self.create_ukf(fx)
        self.ukf_gp = self.create_ukf(fx)

    def get_gp_train(self, ts, priors):
        l = max(self.n - self.window, 1)
        u = self.n
        # x_k
        xk = np.array(priors[l+1:u])
        # x_{k-1}
        xk1 = np.array(priors[l:u-1])
        xk1_pred = np.dot(self.A, xk.T)
        y = xk1_pred.T - xk1
        z = np.array(ts[l+1:u])
        return z, y

    def policy(self, state, t, dt):
        self.priors.append(state)
        self.ts.append(t)
        if self.n >= 10:
            # get training values
            z, y = self.get_gp_train(self.ts, self.priors)
            mu, sigma = np.empty((y.shape[1],)), np.empty((y.shape[1],))

            #### PLOTTING
            if self.testplot:
                fig, ax = plt.subplots(ncols=2, nrows=2)
                axi = {0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)}
                lab = {0:'x',1:'xd',2:'t',3:'td'}

            for i, yi in enumerate(y.T):
                rbf = gaussian_process.kernels.RBF(dt*15, length_scale_bounds='fixed')
                wk = gaussian_process.kernels.WhiteKernel(.2)
                gp = gaussian_process.GaussianProcessRegressor(
                    kernel = rbf + wk, 
                    n_restarts_optimizer=8,
                )
                gp.fit(z.reshape(-1,1), yi.reshape(-1,1))
                mu[i], sigma[i] = gp.predict(np.atleast_2d(t), return_std=True)

                #### PLOTTING
                if self.testplot:
                    testz = np.linspace(z.min(), z.max(), 100)
                    testmu, testsigma = gp.predict(testz.reshape(-1,1), return_std=True)
                    testmu = testmu.squeeze()
                    ax[axi[i]].scatter(z, yi, marker='.', c='#222222')
                    ax[axi[i]].plot(testz, testmu, 'b-', label='GP Prediction')
                    ax[axi[i]].fill_between(testz, testmu-testsigma, testmu+testsigma, alpha=0.1, color='b')
            if self.testplot: plt.show()



        else:
            mu, sigma = np.zeros((4,)), np.zeros((4,))

        self.ukf_control.predict(**{'mu': 0})
        self.ukf_control.update(state)
        est_control = self.ukf_control.x

        self.ukf_gp.predict(**{'mu': mu})
        self.ukf_gp.update(state, np.diag(sigma))
        est_gp = self.ukf_gp.x

        self.n += 1
        # write data
        data = {}
        labels = ['x','xd','t','td']
        data.update(array_to_kv('est_control', labels, est_control))
        data.update(array_to_kv('est_gp',labels, est_gp))
        return 0, data

    def create_ukf(self, fx):
        def hx(x): return x
        points2 = sigma_points.SimplexSigmaPoints(4)
        kf = UnscentedKalmanFilter(4, 4, self.dt, hx, fx, points2)
        kf.Q = np.diag([.2] * 4)
        kf.R = np.diag([self.s] * 4)
        return kf


    @staticmethod
    def _get_linear_sys(pend, dt):
        A, B = pend.jacA, pend.jacB
        C, D = np.zeros((1, A.shape[0])), np.zeros((1, 1))
        sys_disc = cont2discrete((A, B, C, D), dt, method='zoh')
        return sys_disc[0], np.atleast_2d(sys_disc[1])




class LQR_UKF(Controller):
    def __init__(self, pend, dt, window, Q, R, s=10, var_window=10):
        self.pend = pend
        self.dt = dt
        self.window = window
        self.Q = np.diag(Q)
        self.R = np.atleast_2d(R)
        self.s = s
        self.vw = var_window
        self.A, self.B = self._get_linear_sys(pend, dt)
        self.tick = 0
        self.priors = []
        self.kf = self.create_ukf()

    @staticmethod
    def _get_linear_sys(pend, dt):
        A, B = pend.jacA, pend.jacB
        C, D = np.zeros((1, A.shape[0])), np.zeros((1, 1))
        sys_disc = cont2discrete((A, B, C, D), dt, method='zoh')
        return sys_disc[0], np.atleast_2d(sys_disc[1])

    def create_ukf(self):
        def fx(x, dt): return self.A.dot(x)
        def hx(x): return x
        points2 = sigma_points.SimplexSigmaPoints(4)
        kf = UnscentedKalmanFilter(4, 4, self.dt, hx, fx, points2)
        # initialize noise
        kf.Q = np.diag([.2] * 4)
        # initialize smoothing
        kf.R = np.diag([self.s] * 4)
        return kf

    def policy(self, state, t, dt):
        self.priors.append(state)
        # Update variance
        l = max(self.tick - self.vw, 1)
        u = self.tick
        if self.tick >= 2:
            var = np.std(np.vstack(self.priors[l:u]), axis=0)
        else:
            var = np.asarray([0.4] * 4)
        self.kf.Q = np.diag(var)
        # Kalman filter predict
        self.kf.predict()
        # Kalman filter update
        self.kf.update(np.array(state))
        # Get control action using estimated state
        action = do_lqr(self.window, self.A, self.B, self.Q, self.R, self.kf.x)
        self.tick += 1
        # store data
        data = {}
        labels = ['x', 'xd', 't', 'td']
        data.update(array_to_kv('est', labels, self.kf.x ))
        data.update(array_to_kv('var', labels, var))
        return action, data

def quadform(x, H):
    x = np.atleast_2d(x)
    if not ( (x.shape[1] == 1) or (x.shape[1] == H.shape[1]) ):
        raise ValueError('axis 1 doesn\'t match or x not a vector! x: {}, H: {}'.format(x.shape, H.shape))
    return x.T @ H @ x

def do_lqr(w, A, B, Q, R, x):
    P = [None] * (w+1)
    P[w] = Q
    for k in range(w, 0, -1):
        ApkB = A.T @ P[k] @ B
        BpkA = B.T @ P[k] @ A
        c3 = np.linalg.pinv(R + quadform(B, P[k]))
        P[k-1] = quadform(A, P[k]) - ApkB @ c3 @ BpkA + Q
    u = [None] * w
    for k in range(w):
        c1 = np.linalg.inv(R + quadform(B, P[k]))
        c2 = B.T @ P[k] @ A
        u[k] = c1 @ c2 @ x
    return float(np.squeeze(u[0]))
