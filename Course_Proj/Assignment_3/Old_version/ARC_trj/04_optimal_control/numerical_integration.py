# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 08:07:36 2020

Test different integration schemes and their derivatives.

@author: student
"""

import numpy as np
import pinocchio as pin


class Integrator:
    ''' A class implementing different numerical integrator schemes '''

    def __init__(self, name):
        self.name = name

    def integrate(self, ode, x_init, U, t_init, dt, ndt, N, scheme):
        ''' Integrate the given ODE and returns the resulting trajectory:
            - ode: the ordinary differential equation to integrate
            - x_init: the initial state
            - U: trajectory of control inputs, one constant value for each time step dt
            - t_init: the initial time
            - dt: the time step of the trajectory
            - ndt: the number of inner time steps for each time step
            - N: the number of time steps
            - scheme: the name of the integration scheme to use
        '''
        n = x_init.shape[0]
        t = np.zeros((N*ndt+1))*np.nan
        x = np.zeros((N*ndt+1, n))*np.nan
        dx = np.zeros((N*ndt, n))*np.nan
        h = dt/ndt  # inner time step
        x[0, :] = x_init
        t[0] = t_init

        if(scheme == 'RK-1'):
            for i in range(x.shape[0]-1):
                ii = int(np.floor(i/ndt))
                f = ode.f(x[i, :], U[ii, :], t[i])
                dx[i, :] = f
                x[i+1, :] = x[i, :] + h*f
                t[i+1] = t[i] + h
        elif(scheme == 'RK-2'):   # explicit midpoint method
            for i in range(x.shape[0]-1):
                ii = int(np.floor(i/ndt))
                x0 = x[i, :]
                k1 = ode.f(x0,            U[ii, :], t[i])
                k2 = ode.f(x0 + 0.5*h*k1, U[ii, :], t[i]+0.5*h)
                dx[i, :] = k2
                x[i+1, :] = x0 + h*k2
                t[i+1] = t[i] + h
        elif(scheme == 'RK-2-Heun'):
            for i in range(x.shape[0]-1):
                ii = int(np.floor(i/ndt))
                x0 = x[i, :]
                k1 = ode.f(x0,        U[ii, :], t[i])
                k2 = ode.f(x0 + h*k1, U[ii, :], t[i]+h)
                dx[i, :] = 0.5*(k1+k2)
                x[i+1, :] = x0 + h*dx[i, :]
                t[i+1] = t[i] + h
        elif(scheme == 'RK-3'):  # Kutta's third-order method
            for i in range(x.shape[0]-1):
                ii = int(np.floor(i/ndt))
                x0 = x[i, :]
                k1 = ode.f(x0,                  U[ii, :], t[i])
                k2 = ode.f(x0 + h*0.5*k1,       U[ii, :], t[i]+0.5*h)
                k3 = ode.f(x0 + h*(-k1 + 2*k2), U[ii, :], t[i]+h)
                dx[i, :] = (k1 + 4*k2 + k3)/6.0
                x[i+1, :] = x0 + h*dx[i, :]
                t[i+1] = t[i] + h
        elif(scheme == 'RK-4'):
            for i in range(x.shape[0]-1):
                ii = int(np.floor(i/ndt))
                x0 = x[i, :]
                k1 = ode.f(x0,            U[ii, :], t[i])
                k2 = ode.f(x0 + 0.5*h*k1, U[ii, :], t[i]+0.5*h)
                k3 = ode.f(x0 + 0.5*h*k2, U[ii, :], t[i]+0.5*h)
                k4 = ode.f(x0 + h * k3,   U[ii, :], t[i]+h)
                dx[i, :] = (k1 + 2*k2 + 2*k3 + k4)/6.0
                x[i+1, :] = x0 + h*dx[i, :]
                t[i+1] = t[i] + h
        self.dx = dx
        self.t = t
        self.x = x
        return x[::ndt, :]

    def integrate_w_sensitivities_u(self, ode, x_init, U, t_init, dt, N, scheme):
        ''' Integrate the given ODE and returns the resulting trajectory.
            Compute also the derivative of the x trajectory w.r.t. U.
            - ode: the ordinary differential equation to integrate
            - x_init: the initial state
            - U: trajectory of control inputs, one constant value for each time step dt
            - t_init: the initial time
            - dt: the time step of the trajectory
            - N: the number of time steps
            - scheme: the name of the integration scheme to use
        '''
        nx = x_init.shape[0]
        nu = ode.nu
        t = np.zeros((N+1))*np.nan
        x = np.zeros((N+1, nx))*np.nan
        dx = np.zeros((N+1, nx))*np.nan
        dXdU = np.zeros(((N+1)*nx, N*nu))
        h = dt
        x[0, :] = x_init
        t[0] = t_init

        I = np.identity(nx)
        if(scheme == 'RK-1'):
            for i in range(N):
                (f, f_x, f_u) = ode.f(x[i, :], U[i, :], t[i], jacobian=True)
                dx[i, :] = f
                x[i+1, :] = x[i, :] + h*f
                t[i+1] = t[i] + h

                phi_x = I + h*f_x
                phi_u = h * f_u
                ix, ix1, ix2 = i*nx, (i+1)*nx, (i+2)*nx
                iu, iu1 = i*nu, (i+1)*nu
                dXdU[ix1:ix2, :] = phi_x.dot(dXdU[ix:ix1, :])
                dXdU[ix1:ix2, iu:iu1] += phi_u
        elif(scheme == 'RK-4'):
            for i in range(x.shape[0]-1):
                x1 = x[i, :]
                t1 = t[i]
                (k1, f1_x, f1_u) = ode.f(x1, U[i, :], t1, jacobian=True)
                k1_x = f1_x
                k1_u = f1_u

                x2 = x1 + 0.5*h*k1
                t2 = t[i]+0.5*h
                (k2, f2_x, f2_u) = ode.f(x2, U[i, :], t2, jacobian=True)
                k2_x = f2_x.dot(I + 0.5*h*k1_x)
                k2_u = f2_u + 0.5*h*f2_x @ k1_u

                x3 = x1 + 0.5*h*k2
                t3 = t[i]+0.5*h
                (k3, f3_x, f3_u) = ode.f(x3, U[i, :], t3, jacobian=True)
                k3_x = f3_x.dot(I + 0.5*h*k2_x)
                k3_u = f3_u + 0.5*h*f3_x @ k2_u

                x4 = x1 + h * k3
                t4 = t[i]+h
                (k4, f4_x, f4_u) = ode.f(x4, U[i, :], t4, jacobian=True)
                k4_x = f4_x.dot(I + h*k3_x)
                k4_u = f4_u + h*f4_x @ k3_u

                dx[i, :] = (k1 + 2*k2 + 2*k3 + k4)/6.0
                x[i+1, :] = x1 + h*dx[i, :]
                t[i+1] = t[i] + h

                phi_x = I + h*(k1_x + 2*k2_x + 2*k3_x + k4_x)/6.0
                phi_u = h*(k1_u + 2*k2_u + 2*k3_u + k4_u)/6.0
                ix, ix1, ix2 = i*nx, (i+1)*nx, (i+2)*nx
                iu, iu1 = i*nu, (i+1)*nu
                dXdU[ix1:ix2, :] = phi_x.dot(dXdU[ix:ix1, :])
                dXdU[ix1:ix2, iu:iu1] += phi_u
        else:
            return None
        self.dx = dx
        self.t = t
        self.x = x
        return (x, dXdU)

    def check_sensitivities_u(self, ode, x_init, t_init, dt, N, scheme, N_TESTS=10):
        eps = 1e-8
        nx = x_init.shape[0]
        nu = ode.nu
        for iii in range(N_TESTS):
            U = np.random.rand(N, nu)
            (X, dXdU) = self.integrate_w_sensitivities_u(
                ode, x_init, U, t_init, dt, N, scheme)
            X = X.reshape(X.shape[0]*X.shape[1])
            dXdU_fd = np.zeros(((N+1)*nx, N*nu))
            for i in range(N):
                for j in range(nu):
                    U_bar = np.copy(U)
                    U_bar[i, j] += eps
                    X_bar = self.integrate(
                        ode, x_init, U_bar, t_init, dt, 1, N, scheme)
                    X_bar = X_bar.reshape(X_bar.shape[0]*X_bar.shape[1])
                    dXdU_fd[:, i*nu+j] = (X_bar-X)/eps
            dXdU_err = dXdU - dXdU_fd

            print("Error in sensitivities", np.max(np.abs(dXdU_err)))
            if(np.max(np.abs(dXdU_err)) > np.sqrt(eps)):
                print("dXdU", dXdU)
                print("dXdU_fd", dXdU_fd)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import utils.plot_utils as plut
    import time
    from ode import ODERobot_wc, ODERobot, ODELinear, ODESin, ODEStiffDiehl
    from utils.robot_loaders import loadUR_urdf, loadUR
    from utils.robot_wrapper import RobotWrapper
    from utils.robot_simulator import RobotSimulator, ContactPoint, ContactSurface
    import numerical_integration_conf as conf_int
    import single_shooting_conf as conf
    import math
    from numpy.linalg import norm

    np.set_printoptions(precision=3, suppress=True)

    ''' Test DDP with a manipulator
    '''
    print("".center(conf.LINE_WIDTH, '#'))
    print(" Numerical integration ".center(conf.LINE_WIDTH, '#'))
    print("".center(conf.LINE_WIDTH, '#'), '\n')

    N = int(conf.T/conf.dt)                 # horizon size
    dt = conf.dt               # time step
    DEBUG = False
    PLOT_STUFF = 1
    linestyles = ['-', '--', ':', '-.']

    # system = 'ur'
    system = 'ur_wc'
    # system = 'linear'
    # system = 'sin'
    # system = 'stiff-diehl'

    # choose the number of inner steps so that the number of function evaluations
    # is the same for every method
    integrators = []
    # used as ground truth
    # integrators += [{'scheme': 'RK-4',      'ndt': 1000}]
    # integrators += [{'scheme': 'RK-1',      'ndt': 12}]
    # integrators += [{'scheme': 'RK-2',      'ndt': 6}]
    # integrators += [{'scheme': 'RK-2-Heun', 'ndt': 6}]
    # integrators += [{'scheme': 'RK-3',      'ndt': 4}]
    integrators += [{'scheme': 'RK-4',      'ndt': 3}]

    # integrators += [{'scheme': 'RK-1',      'ndt': 10}]
    # integrators += [{'scheme': 'RK-2',      'ndt': 10}]
    # integrators += [{'scheme': 'RK-2-Heun', 'ndt': 10}]
    # integrators += [{'scheme': 'RK-3',      'ndt': 10}]
    # integrators += [{'scheme': 'RK-4',      'ndt': 10}]


    if(system == 'ur'):

        r = loadUR_urdf()
        robot = RobotWrapper(r.model, r.collision_model, r.visual_model)

        simu = RobotSimulator(conf, robot)
        
        nq, nv = robot.nq, robot.nv     # join position and velocity size
        n = nq+nv                       # state size
        m = robot.na                    # control size
        U = np.zeros((N, m))            # initial guess for control inputs
        x0 = np.concatenate((conf.q0, np.zeros(robot.nv)))  # initial state

        # ODE problem definition
        ode = ODERobot('ode', robot)

    elif(system == 'ur_wc'):

        r = loadUR_urdf()
        robot = RobotWrapper(r.model, r.collision_model, r.visual_model)

        # Initialize the candidate contact points and surfaces vectors
        contact_points = []
        contact_surfaces = []

        # Fill the candidate contact points and surfaces vectors
        for name in conf.contact_frames:
            contact_points += [ContactPoint(robot.model, robot.data, name)]

        contact_surfaces += [ContactSurface(conf.contact_surface_name, conf.contact_surface_pos,
                                            conf.contact_normal, conf.K, conf.B, conf.mu)]
        
        simu = RobotSimulator(conf, robot)
        # visualize surface in the viewer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        if(simu.gui):
            simu.gui.addFloor('world/'+name)
            simu.gui.setLightingMode('world/'+name, 'OFF')
            z = np.array([0.,0.,1.])
            axis = np.cross(conf.contact_normal, z)
            if(norm(axis)>1e-6):
                angle = math.atan2(np.linalg.norm(axis), conf.contact_normal.dot(z))
                aa = pin.AngleAxis(angle, axis)
                H = pin.SE3(aa.matrix(), conf.contact_surface_pos)
                simu.gui.applyConfiguration('world/'+name, pin.se3ToXYZQUATtuple(H))
            else:
                simu.gui.applyConfiguration('world/'+name, conf.contact_surface_pos.tolist()+[0.,0.,0.,1.])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        
        nq, nv = robot.nq, robot.nv     # join position and velocity size
        n = nq+nv                       # state size
        m = robot.na                    # control size
        U = np.zeros((N, m))            # initial guess for control inputs
        x0 = np.concatenate((conf.q0, np.zeros(robot.nv)))  # initial state

        # ODE problem definition
        ode = ODERobot_wc('ode', robot, contact_points, contact_surfaces)

    elif(system == 'linear'):
        A = np.array([[-100.0]])
        b = np.array([0.0])
        x0 = np.array([100.0])
        ode = ODELinear('linear', A, b)
    elif(system == 'linear2'):
        A = np.array([[-10.0, 1.0],
                      [0.0, -100.0]])
        b = np.array([0.0, 0.0])
        x0 = np.array([10.0, 7.0])
        ode = ODELinear('linear2', A, b)
    elif(system == 'sin'):
        x0 = np.array([0.0])
        ode = ODESin('sin', A=np.array([1.0]), f=np.array(
            [20.0]), phi=np.array([0.0]))
    elif(system == 'stiff-diehl'):
        x0 = np.array([0.0])
        ode = ODEStiffDiehl()

#    # create simulator to compute ground  truth motion
#    simu = RobotSimulator(conf, robot)
#    print("Compute ground truth")
#    x = np.zeros((N,n))
#    dx = np.zeros((N,n))
#    for i in range(0, N):
#        time_start = time.time()
#        q, v, f = simu.simulate(tau, dt, ndt)
#        x[i,:nq] = q
#        x[i,nq:] = v
#        dx[i,:nq] = v
#        if(i>0):
#            dx[i,nq:] = (v-x[i-1,nq:])/dt
#        time_spent = time.time() - time_start
#        if(conf.simulate_real_time and time_spent < dt):
#            time.sleep(dt-time_spent)
#    sol = {'ground-truth': np.copy(x)}
#    dx = {'ground-truth': np.copy(dx)}

    x_coarse = {}
    x_fine = {}
    t_fine = {}
    dx = {}
    integrator = Integrator('integrator')

    for params in integrators:
        scheme = params['scheme']
        name = scheme+'_ndt_'+str(params['ndt'])
        print("Integrate with ", scheme, 'ndt=', params['ndt'])
        t = 0.0
#        integrator.check_sensitivities_u(ode, x0, t, dt, N, scheme, N_TESTS=1)
        x_coarse[name] = integrator.integrate(ode, x0, U, t, dt, 
                                              params['ndt'], N, scheme)
        x_fine[name] = np.copy(integrator.x)
        t_fine[name] = np.copy(integrator.t)
        dx[name] = np.copy(integrator.dx)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # simu = RobotSimulator(conf, robot)
    for i in range(x_coarse['RK-4_ndt_3'].shape[0]):
        time_start = time.time()
        q = x_coarse['RK-4_ndt_3'][i, :nq]
        robot.display(q)
        time_spent = time.time() - time_start
        if(time_spent < dt):
            time.sleep(dt-time_spent)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    # PLOT STUFF
    time = np.arange(0.0, N*dt, dt)

    if(PLOT_STUFF):
        max_plots = 6
        if(x0.shape[0] == 1):
            nplot = 1
            (f, ax) = plut.create_empty_figure()
            ax = [ax]
        else:
            nplot = int(min(max_plots, x0.shape[0])/2)
            (f, ax) = plut.create_empty_figure(nplot, 2)
            ax = ax.reshape(nplot*2)
        i_ls = 0
        for name, x in sorted(x_fine.items()):
            for i in range(len(ax)):
                ls = linestyles[i_ls]
                ax[i].plot(t_fine[name], x[:, i], ls, label=name, alpha=0.7)
                ax[i].set_xlabel('Time [s]')
                ax[i].set_ylabel(r'$x_'+str(i)+'$')
            i_ls = (i_ls+1) % len(linestyles)
        leg = ax[0].legend()
        leg.get_frame().set_alpha(0.5)

#        if(x0.shape[0]==1):
#            nplot = 1
#            (f, ax) = plut.create_empty_figure()
#            ax = [ax]
#        else:
#            nplot = int(min(max_plots, x0.shape[0])/2)
#            (f, ax) = plut.create_empty_figure(nplot,2)
#            ax = ax.reshape(nplot*2)
#        i_ls = 0
#        for name, dxi in sorted(dx.items()):
#            for i in range(len(ax)):
#                ls = linestyles[i_ls]
#                ax[i].plot(t_fine[name], dxi[:,i], ls, label=name, alpha=0.7)
#                ax[i].set_xlabel('Time [s]')
#                ax[i].set_ylabel(r'$\dot{x}_'+str(i)+'$')
#            i_ls = (i_ls+1)%len(linestyles)
#        leg = ax[0].legend()
#        leg.get_frame().set_alpha(0.5)

    print("Simulation finished")
    plt.show()
