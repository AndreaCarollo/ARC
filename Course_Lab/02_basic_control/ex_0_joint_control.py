import numpy as np
from numpy import nan
import matplotlib.pyplot as plt
import utils.plot_utils as plut
import time
from utils.robot_loaders import loadUR
from utils.robot_wrapper import RobotWrapper
from utils.robot_simulator import RobotSimulator
import ex_0_conf as conf

print("".center(conf.LINE_WIDTH,'#'))
print(" Joint Space Control - Manipulator ".center(conf.LINE_WIDTH, '#'))
print("".center(conf.LINE_WIDTH,'#'), '\n')

PLOT_JOINT_POS = 1
PLOT_JOINT_VEL = 1
PLOT_JOINT_ACC = 0
PLOT_TORQUES = 1

r = loadUR()
robot = RobotWrapper(r.model, r.collision_model, r.visual_model)
simu = RobotSimulator(conf, robot)

N = int(conf.T_SIMULATION/conf.dt)      # number of time steps
tau    = np.empty((robot.na, N))*nan    # joint torques
tau_c  = np.empty((robot.na, N))*nan    # joint Coulomb torques
q      = np.empty((robot.nq, N+1))*nan  # joint angles
v      = np.empty((robot.nv, N+1))*nan  # joint velocities
dv     = np.empty((robot.nv, N+1))*nan  # joint accelerations
q_ref  = np.empty((robot.nq, N))*nan
v_ref  = np.empty((robot.nv, N))*nan
dv_ref = np.empty((robot.nv, N))*nan

two_pi_f             = 2*np.pi*conf.freq   # frequency (time 2 PI)
two_pi_f_amp         = np.multiply(two_pi_f, conf.amp)
two_pi_f_squared_amp = np.multiply(two_pi_f, two_pi_f_amp)

t = 0.0
dt = conf.dt
q[:,0], v[:,0] = simu.q, simu.v
kp, kd = conf.kp, conf.kd
PRINT_N = int(conf.PRINT_T/conf.dt)

for i in range(0, N):
    time_start = time.time()
    
    # set reference trajectory
    q_ref[:,i]  = conf.q0 +  conf.amp*np.sin(two_pi_f*t + conf.phi)
    v_ref[:,i]  = two_pi_f_amp * np.cos(two_pi_f*t + conf.phi)
    dv_ref[:,i] = - two_pi_f_squared_amp * np.sin(two_pi_f*t + conf.phi)
    
    # read current state from simulator
    v[:,i] = simu.v
    q[:,i] = simu.q
    
    # compute mass matrix M, bias terms h, gravity terms g
    M = robot.mass(q[:,i])
    h = robot.nle(q[:,i], v[:,i])
    g = robot.gravity(q[:,i])
    
    # implement your control law here
    tau[:,i] = np.zeros(robot.na)
    flag = 'PD + gravity' # 'PD' or 'PID' or 'PD + gravity' or 'PD + non linear control'
    if( flag == 'PD'):
        tau[:,i]  = kp*( q_ref[:,i] - q[:,i] ) + kd*( v_ref[:,i] - v[:,i] )
    elif( flag == 'PID'):
        tau[:,i]  = kp*( q_ref[:,i] - q[:,i] ) + kd*( v_ref[:,i] - v[:,i] )
    elif( flag == 'PD + gravity'):
        tau[:,i]  = kp*( q_ref[:,i] - q[:,i] ) + kd*( v_ref[:,i] - v[:,i] ) + M.dot(g)

    # send joint torques to simulator
    simu.simulate(tau[:,i], dt, conf.ndt)
    tau_c[:,i] = simu.tau_c
        
    if i%PRINT_N == 0:
        print("Time %.3f"%(t))
    t += conf.dt
        
    time_spent = time.time() - time_start
    if(conf.simulate_real_time and time_spent < conf.dt): 
        time.sleep(conf.dt-time_spent)

# PLOT STUFF
time = np.arange(0.0, N*conf.dt, conf.dt)

if(PLOT_JOINT_POS):    
    (f, ax) = plut.create_empty_figure(int(robot.nv/2),2)
    ax = ax.reshape(robot.nv)
    for i in range(robot.nv):
        ax[i].plot(time, q[i,:-1], label='q')
        ax[i].plot(time, q_ref[i,:], '--', label='q ref')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(r'$q_'+str(i)+'$ [rad]')
        leg = ax[i].legend()
        leg.get_frame().set_alpha(0.5)
        
if(PLOT_JOINT_VEL):    
    (f, ax) = plut.create_empty_figure(int(robot.nv/2),2)
    ax = ax.reshape(robot.nv)
    for i in range(robot.nv):
        ax[i].plot(time, v[i,:-1], label='v')
        ax[i].plot(time, v_ref[i,:], '--', label='v ref')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(r'v_'+str(i)+' [rad/s]')
        leg = ax[i].legend()
        leg.get_frame().set_alpha(0.5)
        
if(PLOT_JOINT_ACC):    
    (f, ax) = plut.create_empty_figure(int(robot.nv/2),2)
    ax = ax.reshape(robot.nv)
    for i in range(robot.nv):
        ax[i].plot(time, dv[i,:-1], label=r'$\dot{v}$')
        ax[i].plot(time, dv_ref[i,:], '--', label=r'$\dot{v}$ ref')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(r'$\dot{v}_'+str(i)+'$ [rad/s^2]')
        leg = ax[i].legend()
        leg.get_frame().set_alpha(0.5)
   
if(PLOT_TORQUES):    
    (f, ax) = plut.create_empty_figure(int(robot.nv/2),2)
    ax = ax.reshape(robot.nv)
    for i in range(robot.nv):
        ax[i].plot(time, tau[i,:], label=r'$\tau$ '+str(i))
        ax[i].plot(time, tau_c[i,:], label=r'$\tau_c$ '+str(i))
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel('Torque [Nm]')
        leg = ax[i].legend()
        leg.get_frame().set_alpha(0.5)
        
plt.show()
