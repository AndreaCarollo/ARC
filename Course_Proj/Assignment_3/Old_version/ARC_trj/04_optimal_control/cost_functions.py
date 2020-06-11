# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 18:12:04 2020

@author: student
"""

import numpy as np
from numpy.linalg import norm
import pinocchio as pin


class Empty:
    def __init__(self):
        pass
        
        
class OCPFinalCostState:
    ''' Cost function for reaching a desired state of the robot
    '''
    def __init__(self, robot, q_des, v_des, weight_pos, weight_vel):
        self.robot = robot
        self.nq = robot.model.nq
        self.q_des = q_des   # desired joint angles
        self.v_des = v_des  # desired joint velocities
        self.weight_pos = weight_pos
        self.weight_vel = weight_vel
        
    def compute(self, x, recompute=True):
        ''' Compute the cost given the final state x '''
        q = x[:self.nq]
        v = x[self.nq:]
        e = q-self.q_des
        de = v - self.v_des
        cost = 0.5*self.weight_pos*e.dot(e) + 0.5*self.weight_vel*de.dot(de)
        return cost
        
    def compute_w_gradient(self, x, recompute=True):
        ''' Compute the cost and its gradient given the final state x '''
        q = x[:self.nq]
        v = x[self.nq:]
        e = q-self.q_des
        de = v - self.v_des
        cost = 0.5*self.weight_pos*e.dot(e) + 0.5*self.weight_vel*de.dot(de)
        grad =  np.concatenate((self.weight_pos*e, self.weight_vel*de))
        return (cost, grad)
        
        
class OCPRunningCostQuadraticControl:
    ''' Quadratic cost function for penalizing control inputs 
    '''
    def __init__(self, robot, dt):
        self.robot = robot
        self.dt = dt
        
    def compute(self, x, u, t, recompute=True):
        ''' Compute the cost for a single time instant'''
        cost = 0.5*u.dot(u)
        return cost
        
    def compute_w_gradient(self, x, u, t, recompute=True):
        ''' Compute the cost for a single time instant and its gradient w.r.t. x and u '''
        cost = 0.5*u.dot(u)
        grad_x = np.zeros(x.shape[0])
        grad_u = u
        return (cost, grad_x, grad_u)


class OCPFinalCostFrame:
    ''' Cost function for reaching a desired position-velocity with a frame of the robot
        (typically the end-effector).
    '''
    def __init__(self, robot, frame_name, p_des, dp_des, weight_vel):
        self.robot = robot
        self.nq = robot.model.nq
        self.frame_id = robot.model.getFrameId(frame_name)
        assert(robot.model.existFrame(frame_name))
        self.p_des  = p_des   # desired 3d position of the frame
        self.dp_des = dp_des  # desired 3d velocity of the frame
        self.weight_vel = weight_vel
        
    def compute(self, x, recompute=True):
        ''' Compute the cost given the final state x '''
        q = x[:self.nq]
        v = x[self.nq:]

        H = self.robot.framePlacement(q, self.frame_id, recompute)
        p = H.translation # take the 3d position of the end-effector
        v_frame = self.robot.frameVelocity(q, v, self.frame_id, recompute)
        dp = v_frame.linear # take linear part of 6d velocity
        
        cost = norm(p-self.p_des) + self.weight_vel*norm(dp - self.dp_des)
        
        return cost
        
    # gradient not implemented yet




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ OUR FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FINAL  COST ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class OCPFinalCostFixedPoint:
    ''' Cost function for reaching a desired point with the End-Effector
    '''
    def __init__(self, robot, p_des, weight_fpos):
        self.robot = robot
        self.nq = robot.model.nq
        self.p_des = p_des  # deired position
        self.weight_fpos = weight_fpos
        self.frame_id = self.robot.model.getFrameId('tool0')
        
    def compute(self, x, recompute=True):
        ''' Compute the cost given the final state x '''

        H = self.robot.framePlacement(x[:self.nq], self.frame_id, recompute)
        # v_frame = robot.frameVelocity(x[:self.nq], x[self.nq:], self.frame_id, recompute)
        # J6 = self.robot.frameJacobian(x[:self.nq], self.frame_id, recompute)

        pos = H.translation    # take the 3d position of the end-effector
        # vel = v_frame.linear   # take linear part of 6d velocity
        # J = J6[:3,:]           # take first 3 rows of J6
        
        tmp = pos-self.p_des
        cost = 0.5*(tmp).dot(tmp)

        return cost
        
    def compute_w_gradient(self, x, recompute=True):
        ''' Compute the cost and its gradient given the final state x '''

        H = self.robot.framePlacement(x[:self.nq], self.frame_id, recompute)
        # v_frame = robot.frameVelocity(x[:self.nq], x[self.nq:], self.frame_id, recompute)
        J6 = self.robot.frameJacobian(x[:self.nq], self.frame_id, recompute)

        pos = H.translation    # take the 3d position of the end-effector
        # vel = v_frame.linear   # take linear part of 6d velocity
        J = J6[:3,:]           # take first 3 rows of J6
        
        tmp = pos-self.p_des
        cost = 0.5*(tmp).dot(tmp)

        grad = np.zeros(x.shape[0])
        grad[:self.nq] =  tmp.dot(J)

        return (cost, grad)


class OCPFinalCostZeroVelocity:
    ''' Cost function for reaching the final point at zero velocity
    '''
    def __init__(self, robot, weight_vel):
        self.nq = robot.nq
        self.weight_vel = weight_vel
        
    def compute(self, x, recompute=True):
        ''' Compute the cost given the final state x '''

        v = x[self.nq:]
        cost = 0.5*self.weight_vel*v.dot(v)

        return cost
        
    def compute_w_gradient(self, x, recompute=True):
        ''' Compute the cost and its gradient given the final state x '''

        v = x[self.nq:]
        cost = 0.5*self.weight_vel*v.dot(v)

        grad = np.zeros(x.shape[0])
        grad[self.nq:] = self.weight_vel*v

        return (cost, grad)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RUNNING  COST ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class OCPRunningCostFixedPoint:
    ''' Cost function that penalize all the points - of the End-Effector - except the desired position 
    '''
    def __init__(self, robot, p_des, dt):
        self.robot = robot
        self.nq = self.robot.nq
        self.p_des = p_des
        self.dt = dt
        self.frame_id = self.robot.model.getFrameId('tool0')
        
    def compute(self, x, u, t, recompute=True):
        ''' Compute the cost for a single time instant'''

        H = self.robot.framePlacement(x[:self.nq], self.frame_id, recompute)
        # v_frame = robot.frameVelocity(x[:self.nq], x[self.nq:], self.frame_id, recompute)
        # J6 = self.robot.frameJacobian(x[:self.nq], self.frame_id, recompute)

        pos = H.translation    # take the 3d position of the end-effector
        # vel = v_frame.linear   # take linear part of 6d velocity
        # J = J6[:3,:]           # take first 3 rows of J6

        tmp = pos-self.p_des
        cost = 0.5*(tmp).dot(tmp)

        return cost
        
    def compute_w_gradient(self, x, u, t, recompute=True):
        ''' Compute the cost for a single time instant and its gradient w.r.t. x and u '''

        H = self.robot.framePlacement(x[:self.nq], self.frame_id, recompute)
        # v_frame = robot.frameVelocity(x[:self.nq], x[self.nq:], self.frame_id, recompute)
        J6 = self.robot.frameJacobian(x[:self.nq], self.frame_id, recompute)

        pos = H.translation    # take the 3d position of the end-effector
        # vel = v_frame.linear   # take linear part of 6d velocity
        J = J6[:3,:]           # take first 3 rows of J6

        tmp = pos-self.p_des
        cost = 0.5*(tmp).dot(tmp)

        grad_x = np.zeros(x.shape[0])
        grad_x[0:self.robot.nq] = tmp.dot(J)

        grad_u = np.zeros(u.shape[0])

        return (cost, grad_x, grad_u)


class OCPRunningCostTrajectory:
    ''' Cost function for a given refernce trajectory
    '''
    def __init__(self, robot, trajectory, dt):
        self.robot = robot
        self.nq = self.robot.nq
        self.dt = dt
        self.frame_id = self.robot.model.getFrameId('tool0')
        self.trajectory = trajectory
        
    def compute(self, x, u, t, recompute=True):
        ''' Compute the cost for a single time instant'''

        self.trajectory.compute(t)

        H = self.robot.framePlacement(x[:self.nq], self.frame_id, recompute)
        # v_frame = robot.frameVelocity(x[:self.nq], x[self.nq:], self.frame_id, recompute)
        # J6 = self.robot.frameJacobian(x[:self.nq], self.frame_id, recompute)

        pos = H.translation    # take the 3d position of the end-effector
        # vel = v_frame.linear   # take linear part of 6d velocity
        # J = J6[:3,:]           # take first 3 rows of J6

        tmp = pos-self.trajectory.pos
        # tmp[2] = 2*tmp[2]
        cost = 0.5*(tmp).dot(tmp)

        return cost
        
    def compute_w_gradient(self, x, u, t, recompute=True):
        ''' Compute the cost for a single time instant and its gradient w.r.t. x and u '''

        self.trajectory.compute(t)

        if (recompute):
            self.robot.computeAllTerms(x[:self.nq], x[self.nq:])
        H = self.robot.framePlacement(x[:self.nq], self.frame_id, False)
        # v_frame = robot.frameVelocity(x[:self.nq], x[self.nq:], self.frame_id, False)
        J6 = self.robot.frameJacobian(x[:self.nq], self.frame_id, False)

        pos = H.translation    # take the 3d position of the end-effector
        # vel = v_frame.linear   # take linear part of 6d velocity
        J = J6[:3,:]           # take first 3 rows of J6

        tmp = pos-self.trajectory.pos
        tmp[2] = 2*tmp[2]
        cost = 0.5*(tmp).dot(tmp)

        grad_x = np.zeros(x.shape[0])
        tmp[2] = 2*tmp[2]
        grad_x[0:self.robot.nq] = tmp.dot(J)

        grad_u = np.zeros(u.shape[0])

        return (cost, grad_x, grad_u)


class OCPRunningCostTrajectoryJS:
    ''' Cost function for a given refernce trajectory
    '''
    def __init__(self, robot, q_ref, dt):
        self.robot = robot
        self.nq = self.robot.nq
        self.dt = dt
        self.frame_id = self.robot.model.getFrameId('tool0')
        # self.trajectory = trajectory
        self.q_ref = q_ref
        
    def compute(self, x, u, t, recompute=True):
        ''' Compute the cost for a single time instant'''

        # self.trajectory.compute(t)
        tmp_time = int(t // self.dt)
        tmp = x[:self.nq]-self.q_ref[tmp_time,:self.nq]
        cost = 0.5*(tmp).dot(tmp)

        return cost
        
    def compute_w_gradient(self, x, u, t, recompute=True):
        ''' Compute the cost for a single time instant and its gradient w.r.t. x and u '''

        # self.trajectory.compute(t)
        tmp_time = int(t // self.dt)
        tmp = x[:self.nq]-self.q_ref[tmp_time,:self.nq]
        cost = 0.5*(tmp).dot(tmp)

        grad_x = np.zeros(x.shape[0])
        grad_x[0:self.robot.nq] = tmp

        grad_u = np.zeros(u.shape[0])

        return (cost, grad_x, grad_u)
        

class OCPRunningCostPosturalTask:
    ''' Cost function to keep a given posture during the task 
    '''
    def __init__(self, robot, q_post, dt):
        self.q_post = q_post
        self.dt = dt
        self.robot = robot
        self.nq = self.robot.nq

    def compute(self, x, u, t, recompute=True):
        ''' Compute the cost for a single time instant'''

        tmp = x[:self.nq] - self.q_post
        tmp[0] = 0.0
        cost = 0.5*(tmp).dot(tmp)

        return cost

    def compute_w_gradient(self, x, u, t, recompute=True):
        ''' Compute the cost for a single time instant and its gradient w.r.t. x and u '''

        tmp = x[:self.nq] - self.q_post
        tmp[0] = 0.0
        cost = 0.5*(tmp).dot(tmp)

        grad_x = np.zeros(x.shape[0])
        grad_x[0:self.robot.nq] = tmp
        grad_x[0] = 0.0

        grad_u = np.zeros(u.shape[0])

        return (cost, grad_x, grad_u)


class OCPRunningCostMinJointVel:
    ''' Cost function to minimize the joint velocity
    '''
    def __init__(self, robot, dt):
        self.dt = dt
        self.robot = robot
        self.nq = self.robot.nq

    def compute(self, x, u, t, recompute=True):
        ''' Compute the cost for a single time instant'''

        v = x[self.nq:]
        # v[0] = 0.0
        cost = 0.5*(v).dot(v)

        return cost

    def compute_w_gradient(self, x, u, t, recompute=True):
        ''' Compute the cost for a single time instant and its gradient w.r.t. x and u '''

        v = x[self.nq:]
        # v[0] = 0.0
        cost = 0.5*(v).dot(v)

        grad_x = np.zeros(x.shape[0])
        grad_x[self.robot.nq:] = v

        grad_u = np.zeros(u.shape[0])

        return (cost, grad_x, grad_u)


class OCPRunningLockEERot:
    ''' Cost function to keep a given posture during the task 
    '''
    def __init__(self, robot, dt):
        self.dt = dt
        self.nq = robot.nq

    def compute(self, x, u, t, recompute=True):
        ''' Compute the cost for a single time instant'''

        tmp = x[3:self.nq]
        cost = 0.5*(tmp).dot(tmp)

        return cost

    def compute_w_gradient(self, x, u, t, recompute=True):
        ''' Compute the cost for a single time instant and its gradient w.r.t. x and u '''

        tmp = x[3:self.nq]
        cost = 0.5*(tmp).dot(tmp)

        grad_x = np.zeros(x.shape[0])
        grad_x[3:self.nq] = tmp

        grad_u = np.zeros(u.shape[0])

        return (cost, grad_x, grad_u)


# class OCPRunningOrthTool:
#     ''' Cost function to keep the end effector orthogonal to the plane
#     '''
#     def __init__(self, robot, frame_id, dt):
#         self.dt = dt
#         self.robot = robot
#         self.frame_id = frame_id
#         self.nq = robot.nq
#         # self.R_ref = np.array( [[-1.,  0.,  0.],
#         #                         [ 0.,  0., -1.],
#         #                         [ 0., -1.,  0.]])
#         self.R_ref = np.array( [[ 1.,  0.,  0.],
#                                 [ 0.,  1.,  0.],
#                                 [ 0.,  0.,  1.]])
        

#     def compute(self, x, u, t, recompute=True):
#         ''' Compute the cost for a single time instant'''

#         H = self.robot.framePlacement(x[:self.nq], self.frame_id, True)

#         error = pin.log3(self.R_ref.T.dot(H.rotation))
#         error[:2] = - error[:2]     # To compensate the transformation
#         error[2] = 0
#         cost = 0.5*(error).dot(error)

#         return cost

#     def compute_w_gradient(self, x, u, t, recompute=True):
#         ''' Compute the cost for a single time instant and its gradient w.r.t. x and u '''

#         if (recompute):
#             self.robot.computeAllTerms(x[:self.nq], x[self.nq:])
#         H = self.robot.framePlacement(x[:self.nq], self.frame_id, False)
#         # v_frame = robot.frameVelocity(x[:self.nq], x[self.nq:], self.frame_id, False)
#         J6 = self.robot.frameJacobian(x[:self.nq], self.frame_id, False)

#         pos = H.translation    # take the 3d position of the end-effector
#         # vel = v_frame.linear   # take linear part of 6d velocity
#         J = J6[3:,:]           # take first 3 rows of J6

#         error = pin.log3(self.R_ref.T.dot(H.rotation))
#         error[2] = 0
#         cost = 0.5*(error).dot(error)

#         grad_x = np.zeros(x.shape[0])
#         grad_x[self.robot.nq:] = error.dot(J)

#         grad_u = np.zeros(u.shape[0])

#         return (cost, grad_x, grad_u)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ ALTERNATIVE IMPLEMENTATION ~~~~~~~~~~~~~~~~~~~~~~~~~~ #
class OCPRunningOrthTool:
    ''' Cost function to keep the end effector orthogonal to the plane
        How does this work?
        The rotation matrix have teh following shape

                [ R11 R12 R13 ]   [   |   |   ]
        M_rot = [ R21 R22 R23 ] = [ i | j | k ]
                [ R31 R32 R33 ]   [   |   |   ]

        Where i, j and k are the norm vector corresponding to the transformed x, y and z axis.
        Since, we want that the end-effector is orthogonal to the plane, we can impose that the
        y-axis (the one exing from the end-effector) is alligned with the normal to the plane.
        If the plane is the ground, the normal will be n = [ 0 0 -1 ].T and, the cost function will be
        (the minus in the last element of n indicates that the transformed y-axis must point downward)

        L = 0.5*((j-n).T*(j-n)) = 0.5*(j[1]^2 + j[2]^2 + (j[3]+1)^2)

        The gradient is obtained computing the matrix analytically and extracting the derivatives
        wrt the joint position.
    '''
    def __init__(self, robot, frame_id, dt):
        self.dt = dt
        self.robot = robot
        self.frame_id = frame_id
        self.nq = robot.nq
        self.M_ee = np.array([[-1 , 0 , 0 ],[ 0 , 0 , 1 ],[ 0 , 1 , 0 ]]) 
    

    def compute_gradients_from_MTX(self, q):
        ''' Compute the value of the gradients of the second column of the rotation matrix, founded analytically.
            This column corresponds to the transformed y axis.
        '''

        # y norm vector derivatives
        dM12_dq = np.zeros(6)
        dM12_dq[0] = 0
        dM12_dq[1] = -np.sin(q[4]) * ((np.cos(q[1]) * np.sin(q[2]) + np.sin(q[1]) * np.cos(q[2])) * np.sin(q[3]) - (np.cos(q[1]) * np.cos(q[2]) - np.sin(q[1]) * np.sin(q[2])) * np.cos(q[3]))
        dM12_dq[2] = -np.sin(q[4]) * ((np.cos(q[1]) * np.sin(q[2]) + np.sin(q[1]) * np.cos(q[2])) * np.sin(q[3]) - (np.cos(q[1]) * np.cos(q[2]) - np.sin(q[1]) * np.sin(q[2])) * np.cos(q[3]))
        dM12_dq[3] = -np.sin(q[4]) * ((np.cos(q[1]) * np.sin(q[2]) + np.sin(q[1]) * np.cos(q[2])) * np.sin(q[3]) - (np.cos(q[1]) * np.cos(q[2]) - np.sin(q[1]) * np.sin(q[2])) * np.cos(q[3]))
        dM12_dq[4] = -(np.sin(q[3]) * (np.sin(q[1]) * np.sin(q[2]) - np.cos(q[1]) * np.cos(q[2])) - (np.cos(q[1]) * np.sin(q[2]) + np.sin(q[1]) * np.cos(q[2])) * np.cos(q[3])) * np.cos(q[4])
        dM12_dq[5] = 0

        dM22_dq = np.zeros(6)
        dM22_dq[0] = ((-np.cos(q[2]) * np.cos(q[3]) + np.sin(q[2]) * np.sin(q[3])) * np.cos(q[1]) + np.sin(q[1]) * (np.sin(q[2]) * np.cos(q[3]) + np.cos(q[2]) * np.sin(q[3]))) * np.sin(q[4]) * np.cos(q[0]) - np.sin(q[0]) * np.cos(q[4])
        dM22_dq[1] = -np.sin(q[4]) * np.sin(q[0]) * (np.sin(q[3]) * (np.sin(q[1]) * np.sin(q[2]) - np.cos(q[1]) * np.cos(q[2])) - (np.cos(q[1]) * np.sin(q[2]) + np.sin(q[1]) * np.cos(q[2])) * np.cos(q[3]))
        dM22_dq[2] = -np.sin(q[4]) * np.sin(q[0]) * (np.sin(q[3]) * (np.sin(q[1]) * np.sin(q[2]) - np.cos(q[1]) * np.cos(q[2])) - (np.cos(q[1]) * np.sin(q[2]) + np.sin(q[1]) * np.cos(q[2])) * np.cos(q[3]))
        dM22_dq[3] = -np.sin(q[4]) * np.sin(q[0]) * (np.sin(q[3]) * (np.sin(q[1]) * np.sin(q[2]) - np.cos(q[1]) * np.cos(q[2])) - (np.cos(q[1]) * np.sin(q[2]) + np.sin(q[1]) * np.cos(q[2])) * np.cos(q[3]))
        dM22_dq[4] = ((-np.cos(q[2]) * np.cos(q[3]) + np.sin(q[2]) * np.sin(q[3])) * np.cos(q[1]) + np.sin(q[1]) * (np.sin(q[2]) * np.cos(q[3]) + np.cos(q[2]) * np.sin(q[3]))) * np.cos(q[4]) * np.sin(q[0]) - np.cos(q[0]) * np.sin(q[4])
        dM22_dq[5] = 0

        dM32_dq = np.zeros(6)
        dM32_dq[0] = ((-np.cos(q[2]) * np.cos(q[3]) + np.sin(q[2]) * np.sin(q[3])) * np.cos(q[1]) + np.sin(q[1]) * (np.sin(q[2]) * np.cos(q[3]) + np.cos(q[2]) * np.sin(q[3]))) * np.sin(q[4]) * np.sin(q[0]) + np.cos(q[0]) * np.cos(q[4])
        dM32_dq[1] = ((-np.sin(q[2]) * np.cos(q[3]) - np.cos(q[2]) * np.sin(q[3])) * np.cos(q[1]) + (-np.cos(q[2]) * np.cos(q[3]) + np.sin(q[2]) * np.sin(q[3])) * np.sin(q[1])) * np.sin(q[4]) * np.cos(q[0])
        dM32_dq[2] = ((-np.sin(q[2]) * np.cos(q[3]) - np.cos(q[2]) * np.sin(q[3])) * np.cos(q[1]) + (-np.cos(q[2]) * np.cos(q[3]) + np.sin(q[2]) * np.sin(q[3])) * np.sin(q[1])) * np.sin(q[4]) * np.cos(q[0])
        dM32_dq[3] = ((-np.sin(q[2]) * np.cos(q[3]) - np.cos(q[2]) * np.sin(q[3])) * np.cos(q[1]) + (-np.cos(q[2]) * np.cos(q[3]) + np.sin(q[2]) * np.sin(q[3])) * np.sin(q[1])) * np.sin(q[4]) * np.cos(q[0])
        dM32_dq[4] = -((-np.cos(q[2]) * np.cos(q[3]) + np.sin(q[2]) * np.sin(q[3])) * np.cos(q[1]) + np.sin(q[1]) * (np.sin(q[2]) * np.cos(q[3]) + np.cos(q[2]) * np.sin(q[3]))) * np.cos(q[4]) * np.cos(q[0]) - np.sin(q[0]) * np.sin(q[4])
        dM32_dq[5] = 0

        return (dM12_dq,dM22_dq,dM32_dq)


    def compute(self, x, u, t, recompute=True):
        ''' Compute the cost for a single time instant '''

        H = self.robot.framePlacement(x[:self.nq], self.frame_id, recompute)
        self.M = self.M_ee.dot(H.rotation)
        
        cost = 0.5*(self.M[0,1]**2 + self.M[1,1]**2 + (self.M[2,1] + 1)**2)

        return cost


    def compute_w_gradient(self, x, u, t, recompute=True):
        ''' Compute the cost for a single time instant and its gradient w.r.t. x and u '''

        H = self.robot.framePlacement(x[:self.nq], self.frame_id, recompute)
        self.M = self.M_ee.dot(H.rotation)
        
        cost = 0.5*(self.M[0,1]**2 + self.M[1,1]**2 + (self.M[1,2]+1)**2)

        (dM12_dq, dM22_dq, dM32_dq) = self.compute_gradients_from_MTX(x[:self.nq])

        grad_x = np.zeros(x.shape[0])
        grad_x[:self.robot.nq] = self.M[0,1]*dM12_dq + self.M[1,1]*dM22_dq + (self.M[2,1]+1)*dM32_dq

        grad_u = np.zeros(u.shape[0])

        return (cost, grad_x, grad_u)

