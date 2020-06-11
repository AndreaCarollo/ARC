# -*- coding: utf-8 -*-
"""
Created on Jun 2020

@author: Carollo Andrea - Tomasi Matteo
"""

import numpy as np
import pinocchio as pin
from numpy.linalg import norm, solve

class InverseKinematic:
    ''' Compute the inverse kinematics
    '''

    def __init__(self, robot, eps = 1e-4, MAX_ITER = 1000, DT = 1e-1):
        self.robot = robot
        self.model = robot.model
        self.data = robot.data
        self.eps = eps
        self.n = MAX_ITER
        self.success = False
        self.DT = DT
        self.damp = 1e-6

    def compute(self, x_des, M_des, q0):

        self.frame_id = self.model.getFrameId('tool0')
        q_i = q0

        H_des = pin.SE3(M_des, x_des)

        for i in range(self.n):
            
            self.robot.computeAllTerms(q_i, np.zeros(6))
            H = self.robot.framePlacement(q_i, self.frame_id, False)

            dH = H_des.actInv(H)
            error = pin.log(dH).vector

            if (norm(error) < self.eps):
                self.success = True
                break

            # J = self.robot.frameJacobian(q_i, self.frame_id, False)
            J = pin.computeJointJacobian(self.model, self.data, q_i, 6)
            v = - J.T.dot(solve(J.dot(J.T) + self.damp * np.eye(6), error))
            q_i = pin.integrate(self.model, q_i, v*self.DT)

        return q_i



# if __name__=='__main__':





