# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 18:12:04 2020

@author: student
"""
if __name__=='__main__':
    import numpy as np
    from numpy.linalg import norm
    from scipy.optimize import minimize
    from ode import ODERobot
    from numerical_integration import Integrator
    import utils.plot_utils as plut
    import matplotlib.pyplot as plt
    from random import uniform
    from utils.robot_loaders import loadUR_urdf, loadPendulum
    from example_robot_data.robots_loader import loadDoublePendulum
    from utils.robot_wrapper import RobotWrapper
    from utils.robot_simulator import RobotSimulator
    import time
    from single_shooting import SingleShootingProblem
    from cost_functions import OCPFinalCostFrame, OCPFinalCostState, OCPRunningCostQuadraticControl
    from cost_functions import OCPRunningCostFixedPoint, OCPFinalCostFixedPoint, OCPFinalCostZeroVelocity
    from cost_functions import OCPRunningCostTrajectory, OCPRunningCostPosturalTask, OCPRunningCostTrajectoryJS, OCPRunningLockEERot
    import main_conf as conf
    np.set_printoptions(precision=3, linewidth=200, suppress=True)
        
    dt = conf.dt                 # time step
    T = conf.T
    N = int(T/dt);         # horizon size
    PLOT_STUFF = 1
    linestyles = ['-*', '--*', ':*', '-.*']
    system=conf.system
    
    
    r = loadUR_urdf()
    robot = RobotWrapper(r.model, r.collision_model, r.visual_model)

    nq, nv = robot.nq, robot.nv     # joint position and velocity size
    n = nq+nv                       # state size
    m = robot.na                    # control sizecost_scale
    U = np.zeros((N,m))             # initial guess for control inputs
    ode = ODERobot('ode', robot)
    
    # create simulator 
    simu = RobotSimulator(conf, robot)
    
    # create OCP
    problem = SingleShootingProblem('ssp', ode, conf.x0, dt, N, conf.integration_scheme, simu)

    # Initial guess for U
    h = robot.nle(conf.q0, np.zeros(6), True)           # Gravity compensation term
    h_ = h
    for i in range(h.shape[0]):
        h_[i] = h_[i] + uniform(-0.1, 0.1)                  # Randomize a bit the initial guess

    y0 = np.zeros(N*m)
    for i in range(0, N-1):
        U[i,:] = h_#*0.99                                # Set the gravity compensation term as initial guess
        y0[m*i:m*(i+1)] = h_#*0.99 
    print('Initial guess - h:\n', h)
    
    print("\nShowing initial motion in viewer")
    # simulate motion with initial guess - only to show the motion - (display_motion function)
    nq = robot.model.nq                                                                     #
    integrator = Integrator('tmp')                                                          #
    X = integrator.integrate(ode, conf.x0, U, 0.0, dt, 1, N, conf.integration_scheme)       #
                                                                                            #
    problem.X = X                                                                           #
    problem.display_motion()    # Use the problem display fnc to show the motion            #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    

    # Compute the initial condition in task space given an initial condition in joint space
    robot.computeAllTerms(conf.q0, np.zeros(6))
    H = robot.framePlacement(conf.q0, problem.frame_id, False)
    p_0 = H.translation

    # Initialize the trajectory that we would like to track
    traj = conf.Trajectory(p_0, conf.p_des, conf.T)
    traj.compute(conf.T)
    p_des = np.zeros(3)
    np.copyto(p_des, traj.pos)

    # Initialize the trajectory that we would like to track in joint space
    # traj_js = conf.TrajectoryJS(conf.q0, conf.q_des, conf.T)
    # traj_js.compute(conf.T)
    # q_fin = traj_js.q

    # Add the costs to the problem - the costs are defined in the cost_functions file
    # Final costs :
    final_fixP = OCPFinalCostFixedPoint(robot, conf.p_des, conf.weight_pos)         # Final cost penalizing the final position in task space
    problem.add_final_cost(final_fixP)
    zero_vel = OCPFinalCostZeroVelocity(robot, conf.weight_vel)                     # Final cost penalizing the final velocity in joint space
    problem.add_final_cost(zero_vel)
    # Cost penalizing final position and velocity in joint space (Prof)
    # fin_js_pos = OCPFinalCostState(robot, q_fin, np.zeros(6), conf.weight_pos, conf.weight_vel)
    # problem.add_final_cost(fin_js_pos)

    # Running costs :
    # fix_point = OCPRunningCostFixedPoint(robot, p_des, dt)                          # Cost to stay in a fixed point in task space
    # problem.add_running_cost(fix_point, conf.weight_traj)
    control_cost = OCPRunningCostQuadraticControl(robot, dt)                        # Cost to minimize the effort (Prof)
    problem.add_running_cost(control_cost, conf.weight_u)
    traj_cost = OCPRunningCostTrajectory(robot, traj, dt)                           # Cost to follow a desired trajectory - class Trajectory - in task space
    problem.add_running_cost(traj_cost, conf.weight_traj)
    # traj_cost = OCPRunningCostTrajectoryJS(robot, traj_js, dt)                      # Cost to follow a desired trajectory - class Trajectory - in joint space
    # problem.add_running_cost(traj_cost, conf.weight_post)
    post_js = OCPRunningCostPosturalTask(robot, conf.q_post, dt)                    # Cost to keep a desired configuration during the task
    problem.add_running_cost(post_js, conf.weight_post)
    lock_ee_rot = OCPRunningLockEERot(robot, dt)
    problem.add_running_cost(lock_ee_rot, 1e-2)

    # Eventually check with the sanity test
    # problem.sanity_check_cost_gradient()
    
    # Solve OCP
    problem.solve(y0=y0,use_finite_difference=conf.use_finite_difference)
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
    print('Desired final position:\n', p_des.T)
    
    # create simulator 
    print('Showing final motion in viewer')
    problem.display_motion(slow_down_factor=3)
