import numpy as np
from numpy import nan
from numpy.linalg import inv, norm, pinv
import matplotlib.pyplot as plt
import utils.plot_utils as plut
import time, sys
from utils.robot_loaders import loadUR
from utils.robot_wrapper import RobotWrapper
from utils.robot_simulator import RobotSimulator
import ex_4_conf as conf

print("".center(conf.LINE_WIDTH,'#'))
print(" IC Control - Manipulator ".center(conf.LINE_WIDTH, '#'))
print("".center(conf.LINE_WIDTH,'#'), '\n')

PLOT_TORQUES = 0
PLOT_EE_POS = 0
PLOT_EE_VEL = 0
PLOT_EE_ACC = 0

r = loadUR()
robot = RobotWrapper(r.model, r.collision_model, r.visual_model)

tests = []
tests += [{'controller': 'OSC', 'kp': 50,  'friction': 0}]
tests += [{'controller': 'IC',  'kp': 50,  'friction': 0}]
tests += [{'controller': 'OSC', 'kp': 100, 'friction': 0}]
tests += [{'controller': 'IC',  'kp': 100, 'friction': 0}]

tests += [{'controller': 'OSC', 'kp': 100, 'friction': 2}]
tests += [{'controller': 'IC',  'kp': 100, 'friction': 2}]
tests += [{'controller': 'OSC', 'kp': 200, 'friction': 2}]
tests += [{'controller': 'IC',  'kp': 200, 'friction': 2}]

tests += [{'controller': 'OSC', 'kp': 200, 'friction': 4}]
tests += [{'controller': 'IC',  'kp': 200, 'friction': 4}]
tests += [{'controller': 'OSC', 'kp': 400, 'friction': 4}]
tests += [{'controller': 'IC',  'kp': 400, 'friction': 4}]

# tests += [{'controller': 'OSC', 'kp': 1100, 'friction': 20}]
# tests += [{'controller': 'IC',  'kp': 1100, 'friction': 20}]
# tests += [{'controller': 'OSC', 'kp': 1400, 'friction': 20}]
# tests += [{'controller': 'IC',  'kp': 1400, 'friction': 20}]

# get the ID corresponding to the frame we want to control
assert(robot.model.existFrame(conf.frame_name))
frame_id = robot.model.getFrameId(conf.frame_name)

simu = RobotSimulator(conf, robot)
tracking_err_osc = []   # list to contain the tracking error of OSC
tracking_err_ic  = []   # list to contain the tracking error of IC

kp_j, kd_j = conf.kp_j, conf.kd_j
lamb = 8

for (test_id, test) in  enumerate(tests):
    description = str(test_id)+' Controller '+test['controller']+' kp='+\
                  str(test['kp'])+' friction='+str(test['friction'])
    print(description)
    kp = test['kp']
    kd = 2*np.sqrt(kp)
    tau_coulomb_max = test['friction']*np.ones(6)   # expressed as percentage of torque max    
    simu.set_coulomb_friction(tau_coulomb_max)      # change Coulomb friction in simulator
    simu.init(conf.q0)                              # initialize simulation state
    
    nx, ndx = 3, 3                          # size of x and its time derivative
    N = int(conf.T_SIMULATION/conf.dt)      # number of time steps
    tau     = np.empty((robot.na, N))*nan    # joint torques
    tau_c   = np.empty((robot.na, N))*nan    # joint Coulomb torques
    q       = np.empty((robot.nq, N+1))*nan  # joint angles
    v       = np.empty((robot.nv, N+1))*nan  # joint velocities
    dv      = np.empty((robot.nv, N+1))*nan  # joint accelerations
    x       = np.empty((nx,  N))*nan        # end-effector position
    dx      = np.empty((ndx, N))*nan        # end-effector velocity
    ddx     = np.empty((ndx, N))*nan        # end effector acceleration
    x_ref   = np.empty((nx,  N))*nan        # end-effector reference position
    dx_ref  = np.empty((ndx, N))*nan        # end-effector reference velocity
    ddx_ref = np.empty((ndx, N))*nan        # end-effector reference acceleration
    ddx_des = np.empty((ndx, N))*nan        # end-effector desired acceleration
    
    two_pi_f             = 2*np.pi*conf.freq   # frequency (time 2 PI)
    two_pi_f_amp         = np.multiply(two_pi_f, conf.amp)
    two_pi_f_squared_amp = np.multiply(two_pi_f, two_pi_f_amp)
    
    t = 0.0
    PRINT_N = int(conf.PRINT_T/conf.dt)
    
    for i in range(0, N):
        time_start = time.time()
        
        # set reference trajectory
        x_ref[:,i]  = conf.x0 +  conf.amp*np.sin(two_pi_f*t + conf.phi)
        dx_ref[:,i]  = two_pi_f_amp * np.cos(two_pi_f*t + conf.phi)
        ddx_ref[:,i] = - two_pi_f_squared_amp * np.sin(two_pi_f*t + conf.phi)
        
        # read current state from simulator
        v[:,i] = simu.v
        q[:,i] = simu.q
        
        # compute mass matrix M, bias terms h, gravity terms g
        robot.computeAllTerms(q[:,i], v[:,i])
        M = robot.mass(q[:,i], False)
        h = robot.nle(q[:,i], v[:,i], False)
        g = robot.gravity(q[:,i])
        
        J6 = robot.frameJacobian(q[:,i], frame_id, False)
        J = J6[:3,:]            # take first 3 rows of J6
        H = robot.framePlacement(q[:,i], frame_id, False)
        x[:,i] = H.translation # take the 3d position of the end-effector
        v_frame = robot.frameVelocity(q[:,i], v[:,i], frame_id, False)
        dx[:,i] = v_frame.linear # take linear part of 6d velocity
    #    dx[:,i] = J.dot(v[:,i])
        dJdq = robot.frameAcceleration(q[:,i], v[:,i], None, frame_id, False).linear

        # Friction compensation term
        # tau_comp_f = tau_coulomb_max * np.tanh( 20* v[:,i] )

        if(test['controller']=='OSC'):

            Lambda = inv( J.dot( inv(M) ).dot(J.T) )
            mu = Lambda.dot( J.dot( inv(M) ).dot(h) - dJdq )
            ddx_des[:,i] = ddx_ref[:,i] + kp * (x_ref[:,i] - x[:,i]) + kd * (dx_ref[:,i] - dx[:,i])
            f_d = Lambda.dot(ddx_des[:,i]) + mu

            ddq_d = kp_j * ( conf.q0 - q[:,i] ) - kd_j * v[:,i]
            tau_0 = M.dot(ddq_d) + h
                                                                                             # Friction compensation
            tau[:,i] = J.T.dot(f_d) + ( np.identity(6) - J.T.dot( pinv(J.T) ) ).dot( tau_0 ) #  + tau_comp_f

        elif(test['controller']=='IC'):

            ddq_d = kp_j * ( conf.q0 - q[:,i] ) - kd_j * v[:,i]
            tau_0 = M.dot(ddq_d)

            ddx_des[:,i] = kp * (x_ref[:,i] - x[:,i]) + kd * (dx_ref[:,i] - dx[:,i])
                                                                                                                   # This should be the additional term    # Friction compensation
            tau[:,i] = h + J.T.dot( lamb * ddx_des[:,i] ) + ( np.identity(6) - J.T.dot( pinv(J.T) ) ).dot( tau_0 ) # - pinv( J.dot( inv(M) ) ).dot(dJdq)   # + tau_comp_f

        else: 
            print('ERROR: Unknown controller', test['controller'])
            sys.exit(0)
        
        # send joint torques to simulator
        simu.simulate(tau[:,i], conf.dt, conf.ndt)
        tau_c[:,i] = simu.tau_c
        ddx[:,i] = J.dot(simu.dv) + dJdq
        t += conf.dt
            
        time_spent = time.time() - time_start
        if(conf.simulate_real_time and time_spent < conf.dt): 
            time.sleep(conf.dt-time_spent)
    
    tracking_err = np.sum(norm(x_ref-x, axis=0))/N
    desc = test['controller']+' kp='+str(test['kp'])+' fri='+str(test['friction'])
    if(test['controller']=='OSC'):        
        tracking_err_osc += [{'value': tracking_err, 'description': desc}]
    elif(test['controller']=='IC'):
        tracking_err_ic += [{'value': tracking_err, 'description': desc}]
    else:
        print('ERROR: Unknown controller', test['controller'])
    
    print('Average tracking error %.3f m\n'%(tracking_err))
    
    # PLOT STUFF
    tt = np.arange(0.0, N*conf.dt, conf.dt)
    
    if(PLOT_EE_POS):    
        (f, ax) = plut.create_empty_figure(nx)
        ax = ax.reshape(nx)
        for i in range(nx):
            ax[i].plot(tt, x[i,:], label='x')
            ax[i].plot(tt, x_ref[i,:], '--', label='x ref')
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel(r'x_'+str(i)+' [m]')
        leg = ax[0].legend()
        leg.get_frame().set_alpha(0.5)
        ax[0].set_title(description)
        
    if(PLOT_EE_VEL):    
        (f, ax) = plut.create_empty_figure(nx)
        ax = ax.reshape(nx)
        for i in range(nx):
            ax[i].plot(tt, dx[i,:], label='dx')
            ax[i].plot(tt, dx_ref[i,:], '--', label='dx ref')
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel(r'dx_'+str(i)+' [m/s]')
        leg = ax[0].legend()
        leg.get_frame().set_alpha(0.5)
        ax[0].set_title(description)
       
    if(PLOT_EE_ACC):    
        (f, ax) = plut.create_empty_figure(nx)
        ax = ax.reshape(nx)
        for i in range(nx):
            ax[i].plot(tt, ddx[i,:], label='ddx')
            ax[i].plot(tt, ddx_ref[i,:], '--', label='ddx ref')
            ax[i].plot(tt, ddx_des[i,:], '-.', label='ddx des')
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel(r'ddx_'+str(i)+' [m/s^2]')
        leg = ax[0].legend()
        leg.get_frame().set_alpha(0.5)
        ax[0].set_title(description)
         
    if(PLOT_TORQUES):    
        (f, ax) = plut.create_empty_figure(int(robot.nv/2),2)
        ax = ax.reshape(robot.nv)
        for i in range(robot.nv):
            ax[i].plot(tt, tau[i,:], label=r'$\tau$ '+str(i))
            ax[i].plot(tt, tau_c[i,:], label=r'$\tau_c$ '+str(i))
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel('Torque [Nm]')
        leg = ax[0].legend()
        leg.get_frame().set_alpha(0.5)
        ax[0].set_title(description)

(f, ax) = plut.create_empty_figure()
for (i,err) in enumerate(tracking_err_osc):
    ax.plot(i, err['value'], 's', markersize=20, label=err['description'])
for (i,err) in enumerate(tracking_err_ic):
    ax.plot(i, err['value'], 'o', markersize=20, label=err['description'])
ax.set_xlabel('Test')
ax.set_ylabel('Mean tracking error [m]')
leg = ax.legend()
leg.get_frame().set_alpha(0.5)

plt.show()
