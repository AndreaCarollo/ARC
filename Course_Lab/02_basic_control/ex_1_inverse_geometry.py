import numpy as np
from numpy import nan
from numpy.linalg import norm
import matplotlib.pyplot as plt
import Course_Lab.utils.plot_utils as plut
import time
from Course_Lab.utils.robot_loaders import loadUR
from Course_Lab.utils.robot_wrapper import RobotWrapper
from Course_Lab.utils.robot_simulator import RobotSimulator
import ex_1_conf as conf

print("".center(conf.LINE_WIDTH, '#'))
print(" Inverse Geometry- Manipulator ".center(conf.LINE_WIDTH, '#'))
print("".center(conf.LINE_WIDTH, '#'), '\n')

PLOT_JOINT_POS = 0
PLOT_FRAME_POS = 1
PLOT_COST = 1
PLOT_GRADIENT = 1

r = loadUR()
robot = RobotWrapper(r.model, r.collision_model, r.visual_model)
simu = RobotSimulator(conf, robot)

# display target position with red sphere in viewer
simu.gui.addSphere('world/target', conf.REF_SPHERE_RADIUS,
                   conf.REF_SPHERE_COLOR)
simu.gui.applyConfiguration(
    'world/target', conf.x_des.tolist()+[0., 0., 0., 1.])

# get the ID corresponding to the frame we want to work with
assert(robot.model.existFrame(conf.frame_name))
frame_id = robot.model.getFrameId(conf.frame_name)

N = conf.MAX_ITER
q = np.empty((robot.nq, N+1))*nan  # joint angles
x = np.empty((3, N))*nan           # frame position
cost = np.empty(N)*nan
grad_norm = np.empty(N)*nan        # gradient norm
q[:, 0] = conf.q0
x_des = conf.x_des
iter_line_search = 0
alpha = 1e-1
beta  = conf.beta
gamma = conf.gamma
regu  = conf.hessian_regu

for i in range(N):
    # compute frame placement H and Jacobian J
    robot.computeJointJacobians(q[:, i])
    robot.framesForwardKinematics(q[:, i])
    J6 = robot.frameJacobian(q[:, i], frame_id)
    H = robot.framePlacement(q[:, i], frame_id)

    x[:, i] = H.translation  # take the 3d position of the end-effector
    J = J6[:3, :]            # take first 3 rows of J6
    e = x_des - x[:, i]
    cost[i] = norm(e)

    ''' The problem to solve is:
            minimize g(q) 
        where:
            g(q) = 0.5 || x(q) - x_{des} ||^2
            g(q) = 0.5 || e(q) ||^2
            g(q) = 0.5 e^T e
        The gradient of g(q) is:
            b = dg/dq = de/dq e = J^T e
        The exact Hessian of g(q) is:
            H = J^T J + e^T dJ/dq
        We use an approximate Hessian to save computation time and ensure
        Hessian is always positive semidefinite (which is needed to ensure that
        Newton direction is a descent direction):
            H = J^T J
        This is known as Gauss-Newton Hessian approximation.
        Newton's step direction is:
            \Delta q = -H^{-1} b = -(J^T J)^{-1} J^T e = -J^{+} e
        where the overscript + indicates the Moore-Penrose pseudo-inverse.
    '''
    # implement your inverse geometry here
    gradient = J.T.dot(e)
    delta_q = -np.linalg.inv(J.T.dot(J) + regu*np.identity(6)).dot(gradient)

    # if gradient is null you are done
    grad_norm[i] = norm(gradient)
    if(grad_norm[i] < conf.gradient_threshold):
        print("Terminate because gradient is (almost) zero:", grad_norm[i])
        print("Problem solved after %d iterations with error %f" % (i, norm(e)))
        break

    if(not conf.line_search):
        q[:, i+1] = q[:, i] + delta_q
    else:
        # back-tracking line search
        alpha = 1.0
        gamma = 0.01
        beta = 0.1
        iter_line_search = 0
        x_try = x[:, i] + alpha*J.dot(delta_q)
        e_try = x_try - x_des
        reduction = cost[i] - norm(e_try)

        while True:
            q_try = q[:, i] + alpha*delta_q
            x_try = x[:, i] + alpha*J.dot(delta_q)
            e_try = x_try - x_des
            reduction = cost[i] - norm(e_try)
            if reduction < gamma*alpha*cost[i]:
                q[:, i+1] = q_try
                break
            alpha = beta*alpha
            iter_line_search += 1

    # display current configuration in viewer
    if i % conf.DISPLAY_N == 0:
        simu.display(q[:, i])
        time.sleep(0.01)

    if i % conf.PRINT_N == 0:
        print("Iteration %d, ||x_des-x||=%f, norm(gradient)=%f" %
              (i, norm(e), grad_norm[i]))

    if(iter_line_search == N):
        break

iters = i

# PLOT STUFF
if(PLOT_COST):
    (f, ax) = plut.create_empty_figure(1, 2)
    ax[0].plot(cost)
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Cost')
    if(not np.any(np.isnan(cost))):
        ax[0].set_yscale('log')
    ax[1].plot(grad_norm)
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Gradient norm')
    if(not np.any(np.isnan(grad_norm))):
        ax[1].set_yscale('log')

if(PLOT_FRAME_POS):
    (f, ax) = plut.create_empty_figure(3, 1)
    ax = ax.reshape(3)
    for i in range(3):
        ax[i].plot(x[i, :], label='x')
        ax[i].plot([0, iters], [x_des[i], x_des[i]], 'r:', label='x')
        ax[i].set_xlabel('Iteration')
        ax[i].set_ylabel(r'$x_'+str(i)+'$ [m]')
        leg = ax[i].legend()
        leg.get_frame().set_alpha(0.5)

if(PLOT_JOINT_POS):
    (f, ax) = plut.create_empty_figure(int(robot.nv/2), 2)
    ax = ax.reshape(robot.nv)
    for i in range(robot.nv):
        ax[i].plot(q[i, :-1], label='q')
        ax[i].set_xlabel('Iteration')
        ax[i].set_ylabel(r'$q_'+str(i)+'$ [rad]')
        leg = ax[i].legend()
        leg.get_frame().set_alpha(0.5)

plt.show()
