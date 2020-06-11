# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 18:12:04 2020

@author: student
"""

import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize
from ode import ODERobot
from numerical_integration import Integrator
import time

class Empty:
    def __init__(self):
        pass
    

class SingleShootingProblem:
    ''' A simple solver for a single shooting OCP.
        In the current version, the solver considers only a cost function (no path constraints).
    '''
    
    def __init__(self, name, ode, x0, dt, N, integration_scheme, simu):
        self.name = name
        self.ode = ode
        self.integrator = Integrator('integrator')
        self.x0 = x0
        self.dt = dt
        self.N = N
        self.integration_scheme = integration_scheme
        self.simu = simu

        self.frame_id = self.simu.robot.model.getFrameId('tool0')
        
        self.nq = int(x0.shape[0]/2)
        self.nx = x0.shape[0]
        self.nu = self.ode.nu
        self.X = np.zeros((N, self.x0.shape[0]))
        self.U = np.zeros((N, self.nu))
        self.last_cost = 0.0
        self.running_costs = []
        self.final_costs = []

        self.visu = False
        
    # Add the running cost to the problem with its weight
    def add_running_cost(self, c, weight=1):
        self.running_costs += [(weight,c)]
    
    # Add the final cost to the problem with its weight
    def add_final_cost(self, c):
        self.final_costs += [c]
        
    # Compute the running cost of one step (with compute) and integrate it over the horizon
    def running_cost(self, X, U):
        ''' Compute the running cost integral '''
        cost = 0.0
        t = 0.0
        for i in range(U.shape[0]):         # Integration over the horizon
            for (w,c) in self.running_costs:
                cost += w * dt * c.compute(X[i,:], U[i,:], t, recompute=True)   # Computation
                t += self.dt
        return cost
    
    # Compute the running cost and its gradient of one step (with compute) and integrate it over the horizon
    def running_cost_w_gradient(self, X, U, dXdU):
        ''' Compute the running cost integral and its gradient w.r.t. U'''
        cost = 0.0
        grad = np.zeros(self.N*self.nu)
        t = 0.0
        nx, nu = self.nx, self.nu
        for i in range(U.shape[0]):         # Integration over the horizon
            for (w,c) in self.running_costs:
                
                ci, ci_x, ci_u = c.compute_w_gradient(X[i,:], U[i,:], t, recompute=True)   # Computation
                dci = ci_x.dot(dXdU[i*nx:(i+1)*nx,:]) 
                dci[i*nu:(i+1)*nu] += ci_u
                
                cost += w * self.dt * ci
                grad += w * self.dt * dci
                t += self.dt

        return (cost, grad)
        
    # Compute the final cost that depends only by the final states x_N  
    def final_cost(self, x_N):
        ''' Compute the final cost '''
        cost = 0.0

        for c in self.final_costs:
            cost += c.compute(x_N, recompute=True)
        return cost
        
    # Compute the final cost and its gradient that depends only by the final states x_N
    def final_cost_w_gradient(self, x_N, dxN_dU):
        ''' Compute the final cost and its gradient w.r.t. U'''
        cost = 0.0

        grad = np.zeros(self.N*self.nu)
        for c in self.final_costs:
            ci, ci_x = c.compute_w_gradient(x_N, recompute=True)
            dci = ci_x.dot(dxN_dU)
            cost += ci
            grad += dci
        return (cost, grad)
        
    # Compute the overall cost for a given input sequence y
    def compute_cost(self, y):
        ''' Compute cost function '''
        # compute state trajectory X from control y
        U = y.reshape((self.N, self.nu))
        t0, ndt = 0.0, 1

        # Integrate the dynamics finding the sequence of X = [x_1 x_2 x_3 ... x_N]
        X = self.integrator.integrate(self.ode, self.x0, U, t0, self.dt, ndt, 
                                      self.N, self.integration_scheme)
        
        # compute cost
        run_cost = self.running_cost(X, U)
        fin_cost = self.final_cost(X[-1,:])
        cost = run_cost + fin_cost
        
        # store X, U and cost
        self.X, self.U = X, U
        self.last_cost = cost        
        return cost

    # Compute the overall cost and its gradient for a given input sequence y
            # The gradient is computed using finite diffrence
            #       grad = ( cost(y + eps) - cost(y) )/eps
    def compute_cost_w_gradient_fd(self, y):
        ''' Compute both the cost function and its gradient using finite differences '''
        eps = 1e-8
        y_eps = np.copy(y)
        grad = np.zeros_like(y)
        cost = self.compute_cost(y)     # Use the previous copute_cost funfion
        for i in range(y.shape[0]):
            y_eps[i] += eps
            cost_eps = self.compute_cost(y_eps)
            y_eps[i] = y[i]
            grad[i] = (cost_eps - cost) / eps
        return (cost, grad)
        
    # Compute the overall cost and its gradient for a given input sequence y
    def compute_cost_w_gradient(self, y):
        ''' Compute cost function and its gradient '''
        # compute state trajectory X from control y
        U = y.reshape((self.N, self.nu))
        t0 = 0.0

        # Integrate the dynamics finding the sequence of X = [x_1 x_2 x_3 ... x_N]
        # In the integration, it computes also the sensitivities
        X, dXdU = self.integrator.integrate_w_sensitivities_u(self.ode, self.x0, U, t0, 
                                                              self.dt, self.N, 
                                                              self.integration_scheme)
        
        # compute cost
        (run_cost, grad_run) = self.running_cost_w_gradient(X, U, dXdU)
        (fin_cost, grad_fin) = self.final_cost_w_gradient(X[-1,:], dXdU[-self.nx:,:])
        cost = run_cost + fin_cost
        grad = grad_run + grad_fin
        
        # store X, U and cost
        self.X, self.U = X, U
        self.last_cost = cost        
        return (cost, grad)
        
    # Solve the problem:
    def solve(self, y0=None, method='BFGS', use_finite_difference=False, max_iter_grad = 100, max_iter_fd = 100):
        ''' Solve the optimal control problem '''
        # Given an initial guess for the input...
        if(y0 is None):
            y0 = np.zeros(self.N*self.nu)
            
        self.iter = 0
        print('Start optimizing')
        # ... start to iterate using minimize function:
        #     minimize("compute cost function", 
        #              "initial guess for the input", 
        #              jac (?), 
        #              method (?),
        #              "callback fnc to show the motion at each iteration", 
        #              "options")
        if(use_finite_difference):
            r = minimize(self.compute_cost_w_gradient_fd, y0, jac=True, method=method, 
                     callback=self.clbk, options={'maxiter': max_iter_fd, 'disp': True})
        else:
            r = minimize(self.compute_cost_w_gradient, y0, jac=True, method=method, 
                     callback=self.clbk, options={'maxiter': max_iter_grad, 'disp': True})
        return r
        

    def sanity_check_cost_gradient(self, N_TESTS=10):
        ''' Compare the gradient computed with finite differences with the one
            computed by deriving the integrator
        '''
        for i in range(N_TESTS):
            y = np.random.rand(self.N*self.nu)
            (cost, grad_fd) = self.compute_cost_w_gradient_fd(y)
            (cost, grad) = self.compute_cost_w_gradient(y)
            grad_err = grad-grad_fd
            if(np.max(np.abs(grad_err))>1):
                print('Grad:   ', grad)
                print('Grad FD:', grad_fd)
            else:
                print('Everything is fine', np.max(np.abs(grad_err)))
        
    # Callback function to show the motion during the iteration
    def clbk(self, xk):
        print('Iter %3d, cost %5f'%(self.iter, self.last_cost))
        self.iter += 1
        if (self.iter%20 == 0 and self.visu):
            self.display_motion()
        return False
        
    # Function that display the motion
    def display_motion(self, slow_down_factor=1):
        for i in range(0, self.N):
            time_start = time.time()
            q = self.X[i,:self.nq]
            self.simu.display(q)        
            time_spent = time.time() - time_start
            if(time_spent < slow_down_factor*self.dt):
                time.sleep(slow_down_factor*self.dt-time_spent)
        

if __name__=='__main__':
    import utils.plot_utils as plut
    import matplotlib.pyplot as plt
    from random import uniform
    from utils.robot_loaders import loadUR_urdf, loadPendulum
    from utils.robot_wrapper import RobotWrapper
    from utils.robot_simulator import RobotSimulator
    from cost_functions import OCPFinalCostFrame, OCPFinalCostState, OCPRunningCostQuadraticControl
    from cost_functions import OCPRunningCostFixedPoint, OCPFinalCostFixedPoint, OCPFinalCostZeroVelocity
    from cost_functions import OCPRunningCostTrajectory, OCPRunningCostPosturalTask, OCPRunningCostTrajectoryJS 
    from cost_functions import OCPRunningLockEERot, OCPRunningCostMinJointVel, OCPRunningOrthTool, OCPRunningCostQuadraticControl
    from utils.robot_simulator import RobotSimulator, ContactPoint, ContactSurface
    from trajectory import TrajectoryJS, TrajectoryLin, TrajectoryCirc
    from utils.inverse_kinematic import InverseKinematic
    from ode import ODERobot, ODERobot_wc
    import single_shooting_conf as conf
    np.set_printoptions(precision=3, linewidth=200, suppress=True)
        
    dt = conf.dt                 # time step
    T = conf.T
    N = int(T/dt);         # horizon size
    PLOT_STUFF = 1
    linestyles = ['-*', '--*', ':*', '-.*']
    system=conf.system
    
    if(system=='ur'):
        r = loadUR_urdf()
    elif(system=='double-pendulum'):
        r = loadDoublePendulum()
    elif(system=='pendulum'):
        r = loadPendulum()

    robot = RobotWrapper(r.model, r.collision_model, r.visual_model)
    # Initialize the candidate contact points and surfaces vectors
    contact_points = []
    contact_surfaces = []

    # Fill the candidate contact points and surfaces vectors
    for name in conf.contact_frames:
        contact_points += [ContactPoint(robot.model, robot.data, name)]
    for name in conf.contact_surface_name:
        contact_surfaces += [ContactSurface(conf.contact_surface_name, conf.contact_surface_pos,
                                        conf.contact_normal, conf.K, conf.B, conf.mu)]

    nq, nv = robot.nq, robot.nv    
    n = nq+nv                       # state size
    m = robot.na                    # control sizecost_scale
    U = np.zeros((N,m))             # initial guess for control inputs
    
    ode = ODERobot('ode', robot)
        
    # create simulator 
    simu = RobotSimulator(conf, robot)

    # Initialize the trajectory that we would like to track
    traj = TrajectoryCirc(conf.T)
    # traj = TrajectoryLin(conf.p0, conf.p_des, conf.T)
    traj.compute(0)
    p_0 = np.zeros(3)
    np.copyto(p_0, traj.pos)

    # evaluate inverse kinematics for q0
    IK = InverseKinematic(robot, eps=1e-3, MAX_ITER=10000)
    M_0 = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    # M_0 = np.array([[0,1,0],[1,0,0],[0,0,-1]])
    q_0 = IK.compute(p_0, M_0, conf.q0)
    x0 = np.concatenate((q_0, np.zeros(6)))
    
    # check position of ee
    H = robot.framePlacement(x0[:robot.nq], robot.model.getFrameId('tool0'), True)
    pos = H.translation
    print('Initial position from trajectory : ', p_0.T)
    print('Initial position from inverse kinematics: ', pos.T)
    print('q_0 = ',q_0.T) 

# first problem
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    problem0 = SingleShootingProblem('ssp', ode, x0, dt, N, conf.integration_scheme, simu)
    problem0.visu = False

    # Initial guess for U
    h = robot.nle(q_0, np.zeros(6), True)           # Gravity compensation term
    h_ = h
    # for i in range(h.shape[0]):
    #     h_[i] = h_[i] + uniform(-0.1, 0.1)                  # Randomize a bit the initial guess

    y0 = np.zeros(N*m)
    for i in range(0, N-1):
        U[i,:] = h #*0.99                                # Set the gravity compensation term as initial guess
        y0[m*i:m*(i+1)] = h #*0.99 
    print('Initial guess - h:\n', h)
    
    print("\nShowing initial motion in viewer")
    # simulate motion with initial guess - only to show the motion - (display_motion function)
    nq = robot.model.nq                                                                     #
    integrator = Integrator('tmp')                                                          #
    X = integrator.integrate(ode, x0, U, 0.0, dt, 1, N, conf.integration_scheme)            #
                                                                                            #
    problem0.X = X                                                                           #
    problem0.display_motion()    # Use the problem0 display fnc to show the motion            #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    

    # Compute the initial condition in task space given an initial condition in joint space
    robot.computeAllTerms(q_0, np.zeros(6))
    H = robot.framePlacement(q_0, problem0.frame_id, False)
    p_0 = H.translation

    traj.compute(conf.T)
    p_des = np.zeros(3)
    np.copyto(p_des, traj.pos)

    # Initialize the trajectory that we would like to track in joint space
    # traj_js = TrajectoryJS(conf.q0, conf.q_des, conf.T)
    # traj_js.compute(conf.T)
    # q_fin = traj_js.q

    # Add the costs to the problem - the costs are defined in the cost_functions file
    # Final costs :

    zero_vel = OCPFinalCostZeroVelocity(robot, 1e-1)                     # Final cost penalizing the final velocity in joint space
    problem0.add_final_cost(zero_vel)

    traj_cost = OCPRunningCostTrajectory(robot, traj, dt)                # Cost to follow a desired trajectory - class Trajectory - in task space
    problem0.add_running_cost(traj_cost, 8)

    # Solve OCP
    problem0.solve(y0=y0,use_finite_difference=conf.use_finite_difference, max_iter_grad = 100, max_iter_fd = 50)
    # problem.solve(method='Nelder-Mead') # solve with non-gradient based solver
    # problem0.display_motion(slow_down_factor=3)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# second problem with contact
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # create OCP
    ode_wc = ODERobot_wc('ode', robot, contact_points, contact_surfaces)
    
    problem = SingleShootingProblem('ssp', ode_wc, x0, dt, N, conf.integration_scheme, simu)
    problem.visu = True

    # Initial guess for U
    h = robot.nle(q_0, np.zeros(6), True)           # Gravity compensation term
    h_ = h
    # for i in range(h.shape[0]):
    #     h_[i] = h_[i] + uniform(-0.1, 0.1)                  # Randomize a bit the initial guess

    y0 = np.zeros(N*m)
    for i in range(0, N-1):
        U[i,:] = h #*0.99                                # Set the gravity compensation term as initial guess
        y0[m*i:m*(i+1)] = h #*0.99 
    print('Initial guess - h:\n', h)
    
    print("\nShowing initial motion in viewer")
    # simulate motion with initial guess - only to show the motion - (display_motion function)
    nq = robot.model.nq                                                                     #
    integrator = Integrator('tmp')                                                          #
    X = integrator.integrate(ode, x0, U, 0.0, dt, 1, N, conf.integration_scheme)            #
                                                                                            #
    problem.X = X                                                                           #
    problem.display_motion()    # Use the problem display fnc to show the motion            #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    

    # Compute the initial condition in task space given an initial condition in joint space
    robot.computeAllTerms(q_0, np.zeros(6))
    H = robot.framePlacement(q_0, problem.frame_id, False)
    p_0 = H.translation

    traj.compute(conf.T)
    p_des = np.zeros(3)
    np.copyto(p_des, traj.pos)

    # Initialize the trajectory that we would like to track in joint space
    # traj_js = TrajectoryJS(conf.q0, conf.q_des, conf.T)
    # traj_js.compute(conf.T)
    # q_fin = traj_js.q

    # Add the costs to the problem - the costs are defined in the cost_functions file
    # Final costs :
    if (conf.FIN_FIXED_POINT):
        final_fixP = OCPFinalCostFixedPoint(robot, p_des, conf.weight_pos)      # Final cost penalizing the final position in task space
        problem.add_final_cost(final_fixP)
    if (conf.FIN_VEL):
        zero_vel = OCPFinalCostZeroVelocity(robot, conf.weight_vel)             # Final cost penalizing the final velocity in joint space
        problem.add_final_cost(zero_vel)
    # Cost penalizing final position and velocity in joint space (Prof)
    # fin_js_pos = OCPFinalCostState(robot, q_fin, np.zeros(6), conf.weight_pos, conf.weight_vel)
    # problem.add_final_cost(fin_js_pos)

    # Running costs :
    if (conf.RUN_FIXED_POINT):
        p_des_fp = np.array([ 0. , 0.4 , 0. ])
        fix_point = OCPRunningCostFixedPoint(robot, p_des_fp, dt)               # Cost to stay in a fixed point in task space
        problem.add_running_cost(fix_point, conf.weight_traj)
    if (conf.RUN_QUAD):
        control_cost = OCPRunningCostQuadraticControl(robot, dt)                # Cost to minimize the effort (Prof)
        problem.add_running_cost(control_cost, conf.weight_u)
    if (conf.RUN_TRAJ_TS):
        traj_cost = OCPRunningCostTrajectory(robot, traj, dt)                   # Cost to follow a desired trajectory - class Trajectory - in task space
        problem.add_running_cost(traj_cost, conf.weight_traj)
    if (conf.RUN_TRAJ_JS):
        traj_cost = OCPRunningCostTrajectoryJS(robot, problem0.X, dt)           # Cost to follow a desired trajectory - class Trajectory - in joint space
        problem.add_running_cost(traj_cost, conf.weight_post_JS)
    if (conf.RUN_POSTURAL):
        post_js = OCPRunningCostPosturalTask(robot, q_0, dt)                    # Cost to keep a desired configuration during the task
        problem.add_running_cost(post_js, conf.weight_post)
    # lock_ee_rot = OCPRunningLockEERot(robot, dt)                              # Cost to limit the freedom of the last three joint
    # problem.add_running_cost(lock_ee_rot, 1e-2)
    if (conf.RUN_JOINT_VEL):
        min_j_vel = OCPRunningCostMinJointVel(robot, dt)                        # Cost to minimize the joint velocity during the trajectory
        problem.add_running_cost(min_j_vel, conf.weight_vel_r)
    if (conf.RUN_ORTH_EE):
        orth_ee = OCPRunningOrthTool(robot, problem.frame_id, dt)               # Cost function to keep the end effctor orthogonal to the ground
        problem.add_running_cost(orth_ee, conf.weight_orth)

    # Eventually check with the sanity test
    if (conf.sanity_check):
        problem.sanity_check_cost_gradient()
    
    # Solve OCP
    U_2 = np.copy(problem0.U)
    problem.solve(y0=U_2,use_finite_difference=conf.use_finite_difference, max_iter_grad = 150, max_iter_fd = 50)
    # problem.solve(method='Nelder-Mead') # solve with non-gradient based solver


    # OUTPUTS
    print('U norm:', norm(problem.U))       # Norm of the controls
    print('X_N\n', problem.X[-1,:].T)       # final configuration vector - X_N = [ q_N  dq_N ] -
    # print('X_N_des\n', np.concatenate((q_fin, np.zeros(6))).T)       # final configuration vector - X_N = [ q_N  dq_N ] -

    # Compute the initial and final position in task space
    H = robot.framePlacement(problem.X[0,:robot.nq], robot.model.getFrameId('tool0'), True)
    pos = H.translation
    print('\nInitial position:\n', pos.T)
    H = robot.framePlacement(problem.X[-1,:robot.nq], robot.model.getFrameId('tool0'), True)
    pos = H.translation
    print('Final position:\n', pos.T)

    # Compute the trajectory final position
    traj.compute(0.0)
    p_0 = traj.pos
    print('\nDesired initial position:\n', p_0.T)
    traj.compute(conf.T)
    p_fin = traj.pos
    print('Desired final position:\n', p_fin.T)
    
    # create simulator 
    print('Showing final motion in viewer')
    problem.display_motion(slow_down_factor=3)


    # Plots
    x_real = np.zeros((N+1, 3))
    x_traj = np.zeros((N+1, 3))
    time   = np.zeros(N+1)
    err = 0.
    for i in range(N+1):
        t = i*conf.dt
        time[i] = t
        traj.compute(t)
        H = robot.framePlacement(problem.X[i,:robot.nq],problem.frame_id,True)
        x_real[i,:] = H.translation
        x_traj[i,:] = traj.pos
        err += norm(traj.pos - x_real[i,:])
    err_mean = err/(N+1)
    print('Mean tracking error : ', err_mean)

    max_plots = 3
    if(x_real.shape[0] == 1):
        nplot = 1
        (f, ax) = plut.create_empty_figure()
        ax = [ax]
    else:
        nplot = min(max_plots, x_real.shape[0])
        (f, ax) = plut.create_empty_figure(nplot, 1)
    i_ls = 0
    for i in range(len(ax)):
        ls = linestyles[0]
        ax[i].plot(time, x_real[:, i], ls, label='real', alpha=0.7)
        ls = linestyles[1]
        ax[i].plot(time, x_traj[:, i], ls, label='traj', alpha=0.7)
        ax[i].set_xlabel('Time [s]')
        if i==0:
            ax[i].set_ylabel(r'$x$')
        if i==1:
            ax[i].set_ylabel(r'$y$')
        if i==2:
            ax[i].set_ylabel(r'$z$')
    leg = ax[0].legend()

    plt.show()

