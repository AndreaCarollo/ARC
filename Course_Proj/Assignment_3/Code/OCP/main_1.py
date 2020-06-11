# -*- coding: utf-8 -*-
"""
Created on Jun 2020

@author: Carollo Andrea - Tomasi Matteo

This file compute the OCP of a given trajectory
and then use the output of the OCP as reference
for a Reactive Control that has to provide a
target force at the end effector
"""

if __name__=='__main__':
    import numpy as np
    import time as tim
    import utils.plot_utils as plut
    import matplotlib.pyplot as plt
    import main_conf as conf
    from numpy.linalg import norm
    from scipy.optimize import minimize
    from random import uniform
    from utils.robot_loaders import loadUR_urdf, loadPendulum
    from utils.robot_wrapper import RobotWrapper
    from utils.cost_functions import OCPFinalCostFrame, OCPFinalCostState, OCPRunningCostQuadraticControl
    from utils.cost_functions import OCPRunningCostFixedPoint, OCPFinalCostFixedPoint, OCPFinalCostZeroVelocity
    from utils.cost_functions import OCPRunningCostTrajectory, OCPRunningCostPosturalTask, OCPRunningCostTrajectoryJS 
    from utils.cost_functions import OCPRunningLockEERot, OCPRunningCostMinJointVel, OCPRunningOrthTool
    from utils.robot_simulator import RobotSimulator, ContactPoint, ContactSurface
    from utils.trajectory import TrajectoryJS, TrajectoryLin, TrajectoryCirc
    from utils.inverse_kinematic import InverseKinematic
    from ode import ODERobot, ODERobot_wc, ContactSurface_ODE, ContactPoint_ODE
    from single_shooting import SingleShootingProblem
    from numerical_integration import Integrator
    np.set_printoptions(precision=3, linewidth=200, suppress=True)


   # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INITIALIZATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        
    dt = conf.dt           # time step
    T = conf.T
    N = int(T/dt);         # horizon size
    PLOT_STUFF = 1
    linestyles = ['-*', '--*', ':*', '-.*']
    system=conf.system

    r = loadUR_urdf()
    robot = RobotWrapper(r.model, r.collision_model, r.visual_model)

    # Initialize parameters
    nq, nv = robot.nq, robot.nv     # joint position and velocity size    
    n = nq+nv                       # state size
    m = robot.na                    # control sizecost_scale
    U = np.zeros((N,m))             # initial guess for control inputs
       
    # create simulator 
    simu = RobotSimulator(conf, robot)

    # Initialize the trajectory that we would like to track
    traj = TrajectoryCirc(conf.T)
    # traj = TrajectoryLin(conf.p0, conf.p_des, conf.T)
    traj.compute(0)
    p_0 = np.zeros(3)
    np.copyto(p_0, traj.pos)                       # desired position

    # evaluate inverse kinematics for q0
    IK = InverseKinematic(robot, eps=1e-3, MAX_ITER=10000)
    M_0 = np.array([[1,0,0],[0,-1,0],[0,0,-1]])  # desired orientation

    q_0 = IK.compute(p_0, M_0, conf.q0)
    x0 = np.concatenate((q_0, np.zeros(6)))        # initial states
    
    # check position of ee
    H = robot.framePlacement(x0[:robot.nq], robot.model.getFrameId('tool0'), True)
    pos = H.translation
    print('Initial position from trajectory : ', p_0.T)
    print('Initial position from inverse kinematics: ', pos.T)
    print('q_0 = ',q_0.T) 

    simu.init(q0=q_0, v0=np.zeros(robot.nv))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ OCP - WITHOUT CONTACTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    
    # Load the ODE problem: without contact
    ode = ODERobot('ode', robot)

    # Create the OCP
    problem0 = SingleShootingProblem('ssp', ode, x0, dt, N, conf.integration_scheme, simu)
    problem0.visu = False

    # Initial guess for U - gravity compensation
    h = robot.nle(q_0, np.zeros(6), True)           # Gravity compensation term
    
    y0 = np.zeros(N*m)
    for i in range(0, N-1):
        U[i,:] = h
        y0[m*i:m*(i+1)] = h 
    print('Initial guess - h:\n', h)
    
    # simulate motion with initial guess - only to show the motion - (display_motion function)
    if(conf.SHOW_THE_FIRST_MOVEMENT):
        nq = robot.model.nq                                                                     
        integrator = Integrator('tmp')                                                          
        X = integrator.integrate(ode, x0, U, 0.0, dt, 1, N, conf.integration_scheme)                                                                          
        problem0.X = X 
        
        print("\nShowing initial motion in viewer")                                                                          
        problem0.display_motion()    # Use the problem0 display fnc to show the motion          
    

    # Compute the initial condition in task space given an initial condition in joint space
    robot.computeAllTerms(q_0, np.zeros(6))
    H = robot.framePlacement(q_0, problem0.frame_id, False)
    p_0 = H.translation

    # Trajectory final point
    traj.compute(conf.T)
    p_des = np.zeros(3)
    np.copyto(p_des, traj.pos)


    # Add the costs to the problem - the costs are defined in the cost_functions file
    # Final costs:
    zero_vel = OCPFinalCostZeroVelocity(robot, 1e-1)                     # Final cost penalizing the final velocity in joint space
    problem0.add_final_cost(zero_vel)
    # Running costs:
    traj_cost = OCPRunningCostTrajectory(robot, traj, dt)                # Cost to follow a desired trajectory - class Trajectory - in task space
    problem0.add_running_cost(traj_cost, 8)

    # Solve OCP
    problem0.solve(y0=y0,use_finite_difference=conf.use_finite_difference, max_iter_grad = 100, max_iter_fd = 50)
    # problem.solve(method='Nelder-Mead') # solve with non-gradient based solver
    problem0.display_motion(slow_down_factor=3)

    # Error evaluation
    x_real = np.zeros((N+1, 3))
    x_traj = np.zeros((N+1, 3))
    time1  = np.zeros(N+1)
    err = 0.
    for i in range(N+1):
        t = i*conf.dt
        time1[i] = t
        traj.compute(t)
        H = robot.framePlacement(problem0.X[i,:robot.nq],problem0.frame_id,True)
        x_real[i,:] = H.translation
        x_traj[i,:] = traj.pos
        err += norm(traj.pos - x_real[i,:])
    err_mean = err/(N+1)

    print('Mean tracking error after the first OCP : ', err_mean)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ REACTIVE  CONTROL ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    
    F_des = np.array([ 0. , 0. , 50. ])

    # Create a simulator for the reactive control part (the contacts are different)
    simu_rc = RobotSimulator(conf, robot)
    simu_rc.init(q0=q_0, v0=np.zeros(robot.nv))

    # Add the contact surf for the reactive control
    for name in conf.contact_frames:
        simu_rc.add_candidate_contact_point(name)
    simu_rc.add_contact_surface(conf.contact_surface_name, conf.contact_surface_pos, 
                                conf.contact_normal, conf.K_RC, conf.B_RC, conf.mu)

    ndt    = conf.ndt_rc

    tau    = np.empty((robot.na,N*ndt))*np.nan     # joint torque
    tau_c  = np.empty((robot.na,N*ndt))*np.nan     # coulomb friction
    f      = np.zeros((3       ,N*ndt))            # contact force
    q      = np.empty((robot.nq,N*ndt))*np.nan     # joint position    
    v      = np.empty((robot.nv,N*ndt))*np.nan     # joint velocity
    q_ref  = np.empty((robot.nq,N*ndt))*np.nan     # reference joint position    
    v_ref  = np.empty((robot.nv,N*ndt))*np.nan     # reference joint velocity
    x      = np.empty((3       ,N*ndt))*np.nan     # end-effector position
    x_ref  = np.empty((3       ,N*ndt))*np.nan     # reference end-effector position

    # Linear interpolation between two OCP time step
    ii = 0
    for i in range(0, N*ndt, 1):
        if i%ndt == 0:
            q_ref[:,i] = problem0.X[ii,:robot.nq].T
            v_ref[:,i] = problem0.X[ii,robot.nq:].T
            ii += 1
        else:
            k = (i-(ii-1)*ndt)/ndt
            dX = problem0.X[ii,:] - problem0.X[ii-1,:]
            q_ref[:,i] = (problem0.X[ii-1,:robot.nq] + dX[:robot.nq]*k).T
            v_ref[:,i] = (problem0.X[ii-1,robot.nq:] + dX[robot.nq:]*k).T


    dt_RC   = conf.T/(N*ndt)
    t       = 0
    for i in range(0, N*ndt):
        time_start = tim.time()

        # read current state from simulator
        v[:,i] = simu_rc.v
        q[:,i] = simu_rc.q
        if(simu_rc.f.shape[0]==3):
            f[:,i] = simu_rc.f

        robot.computeAllTerms(q[:,i], v[:,i])
        M = robot.mass(q[:,i], False)
        h = robot.nle(q[:,i], v[:,i], False)

        J6 = robot.frameJacobian(q[:,i], problem0.frame_id, False)
        J  = J6[:3,:]            # take first 3 rows of J6

        # Force tracking
        f_star   = F_des + conf.kp*(F_des - f[:,i])
        tau[:,i] = - J.T.dot(f_star)

        # Motion tracking
        ddq_0    = conf.kp_j*(q_ref[:,i]-q[:,i]) + conf.kp_j*(v_ref[:,i]-v[:,i])
        tau[:,i] += M.dot(ddq_0)

        # End-Effector position
        H        = robot.framePlacement(q[:,i], problem0.frame_id, False)
        x[:,i]   = H.translation

        # Trajectory position
        traj.compute(t)
        x_ref[:,i] = traj.pos

        # Simulate the time step
        simu_rc.simulate(tau[:,i].T, conf.dt_RC, 1)
        tau_c[:,i] = simu_rc.tau_c

        if i%int(1/conf.dt_RC) == 0:
            print("Time %.3f\n"%(t))
        t += conf.dt_RC

        time_spent = tim.time() - time_start
        if(conf.simulate_real_time and time_spent < conf.dt_RC): 
            tim.sleep(3*conf.dt_RC-time_spent)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #    


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PLOT STUFF ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    

    if (conf.PLOT_TRACK_OCP):
        max_plots = 3
        if(x_real.shape[0] == 1):
            nplot = 1
            (ff, ax) = plut.create_empty_figure()
            ax = [ax]
        else:
            nplot = min(max_plots, x_real.shape[0])
            (ff, ax) = plut.create_empty_figure(nplot, 1)
        i_ls = 0
        ax[0].set_title("Trajectories")
        for i in range(len(ax)):
            ls = linestyles[0]
            ax[i].plot(time1, x_real[:, i], ls, label='traj OCP', alpha=0.7)
            ls = linestyles[1]
            ax[i].plot(time1, x_traj[:, i], ls, label='traj ref', alpha=0.7)
            ax[i].set_xlabel('Time [s]')
            if i==0:
                ax[i].set_ylabel(r'$x$ [m]')
            if i==1:
                ax[i].set_ylabel(r'$y$ [m]')
            if i==2:
                ax[i].set_ylabel(r'$z$ [m]')
        leg = ax[0].legend()

    time    = np.arange(0.0, conf.T, conf.dt_RC)
    f_ref   = np.zeros((3, N*ndt))
    f_ref[2,:] = F_des[2]*np.ones(N*ndt).T

    if(conf.PLOT_CONTACT_FORCE):    
        (ff, ax) = plut.create_empty_figure(3)
        ax = ax.reshape(3)
        ax[0].set_title("Contact Forces")
        for i in range(3):
            ax[i].plot(time, f[i,:], label='f')
            ax[i].plot(time, f_ref[i,:], '--', label='f ref')
            ax[i].set_xlabel('Time [s]')
            # ax[i].set_ylabel(r'f_'+str(i)+' [N]')
            if i==0:
                ax[i].set_ylabel(r'$F_x$ [N]')
            if i==1:
                ax[i].set_ylabel(r'$F_y$ [N]')
            if i==2:
                ax[i].set_ylabel(r'$F_z$ [N]')
        leg = ax[0].legend()
        leg.get_frame().set_alpha(0.5)

        f_N = np.array([f[:,i].dot(conf.contact_normal) for i in range(f.shape[1])])
        f_T = np.array([norm(f[:,i] - f_N[i]*conf.contact_normal) for i in range(f.shape[1])])

        (ff, ax) = plut.create_empty_figure(1)
        ax.set_title("Contact surface Friction and Tangential forces")
        ax.plot(time, conf.mu*f_N, label='mu*Normal')
        ax.plot(time, f_T, '--', label='Tangential')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(r'Force [N]')
        leg = ax.legend()
        leg.get_frame().set_alpha(0.5)

    if(conf.PLOT_EE_POS):    
        (ff, ax) = plut.create_empty_figure(3)
        ax = ax.reshape(3)
        ax[0].set_title("End-Effector position")
        for i in range(3):
            ax[i].plot(time1, x_real[:, i], ls, label='traj OCP', alpha=0.7)
            ax[i].plot(time, x[i,:], label='traj RC')
            ax[i].plot(time, x_ref[i,:], '--', label='traj ref')
            ax[i].set_xlabel('Time [s]')
            if i==0:
                ax[i].set_ylabel(r'$x$ [m]')
            if i==1:
                ax[i].set_ylabel(r'$y$ [m]')
            if i==2:
                ax[i].set_ylabel(r'$z$ [m]')
        leg = ax[0].legend()
        leg.get_frame().set_alpha(0.5)

    if(conf.PLOT_JOINT_POS):    
        (ff, ax) = plut.create_empty_figure(int(robot.nv/2),2)
        ax = ax.reshape(robot.nv)
        ax[0].set_title("Joints position")
        for i in range(robot.nv):
            ax[i].plot(time, q[i,:], label='q')
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel(r'$q_'+str(i)+'$ [rad]')
        leg = ax[0].legend()
        leg.get_frame().set_alpha(0.5)

    if(conf.PLOT_JOINT_VEL):    
        (ff, ax) = plut.create_empty_figure(int(robot.nv/2),2)
        ax = ax.reshape(robot.nv)
        ax[0].set_title("Joint velocities")
        for i in range(robot.nv):
            ax[i].plot(time, v[i,:], label='v')
    #        ax[i].plot(time, v_ref[i,:], '--', label='v ref')
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel(r'v_'+str(i)+' [rad/s]')
        leg = ax[0].legend()
        leg.get_frame().set_alpha(0.5)
    
    if(conf.PLOT_TORQUES):    
        (ff, ax) = plut.create_empty_figure(int(robot.nv/2),2)
        ax = ax.reshape(robot.nv)
        ax[0].set_title("Joints torques")
        for i in range(robot.nv):
            ax[i].plot(time, tau[i,:], label=r'$\tau$ '+str(i))
            ax[i].plot(time, tau_c[i,:], label=r'$\tau_c$ '+str(i))
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel('Torque [Nm]')
        leg = ax[0].legend()
        leg.get_frame().set_alpha(0.5)

    plt.show()


