import numpy as np
import numpy.linalg as la
import utils.transformations as tr
import math


class ImuParameters:
    def __init__(self):
        self.sigma_a_n = 0.0365432018302     # acc noise.   m/(s*sqrt(s)), continuous noise sigma
        self.sigma_w_n = 0.00367396706572     # gyro noise.  rad/sqrt(s), continuous noise sigma
        self.sigma_a_b = 0.000433     # acc bias     m/sqrt(s^5), continuous bias sigma
        self.sigma_w_b = 2.66e-05     # gyro bias    rad/sqrt(s^3), continuous bias sigma


class ESEKF(object):
    def __init__(self, init_nominal_state: np.array, imu_parameters: ImuParameters):
        """
        :param init_nominal_state: [ p, v, q, a_b, w_b, g ], a 19x1 or 1x19 vector
        :param imu_parameters: imu parameters
        """
        self.nominal_state = init_nominal_state
        if self.nominal_state[6] < 0:
            self.nominal_state[6:10] *= -1    # force the quaternion has a positive real part.
        self.imu_parameters = imu_parameters

        # initialize noise covariance matrix
        self.noise_covar = np.zeros((12, 12))
        # assume the noises (especially sigma_a_n) are isotropic so that we can precompute self.noise_covar and save it.
        self.noise_covar[0:3, 0:3] = (imu_parameters.sigma_a_n**2) * np.eye(3)
        self.noise_covar[3:6, 3:6] = (imu_parameters.sigma_w_n**2) * np.eye(3)
        self.noise_covar[6:9, 6:9] = (imu_parameters.sigma_a_b**2) * np.eye(3)
        self.noise_covar[9:12, 9:12] = (imu_parameters.sigma_w_b**2) * np.eye(3)
        

        Fi = np.zeros((18,12))
        Fi[3:15, :] = np.eye(12)
        Vi = self.noise_covar[:3,:3]*0.01
        Thetai = self.noise_covar[3:6,3:6]*0.01
        Ai = self.noise_covar[6:9,6:9]*0.01
        Omegai = self.noise_covar[9:12,9:12]*0.01
        Qi=np.zeros((12,12))
        Qi[:3,:3] = Vi
        Qi[3:6,3:6] = Thetai
        Qi[6:9,6:9] = Ai
        Qi[9:12,9:12] = Omegai
        self.error_covar = Fi @ Qi @ Fi.T

        self.last_predict_time = 0.0
        
        # for convergence judgement
        self.delta_T = [1e5,1e5,1e5,1e5,1e5,1e5]

    def predict(self, imu_measurement: np.array):
        """
        :param imu_measurement: [t, w_m, a_m]
        :return: 
        """
        if self.last_predict_time == imu_measurement[0]:
            return
        # we predict error_covar first, because __predict_nominal_state will change the nominal state.
        self.__predict_error_covar(imu_measurement)
        self.__predict_nominal_state(imu_measurement)
        self.last_predict_time = imu_measurement[0]  # update timestamp

    def update_with_pose(self, gt_measurement: np.array, measurement_covar: np.array):
        """
        :param gt_measurement: [p, q], a 7x1 or 1x7 vector
        :param measurement_covar: a 6x6 symmetrical matrix = diag{sigma_p^2, sigma_theta^2}
        :return: 
        """
        """
        we simulate a system that measure the errors between the nominal state and ground-truth state directly,
        so that we can avoid the direct subtracting of quaternions.
        
        we define q1 - q2 = conjugate(q2) x q1, so that q2 x (q1 - q2) = q1.
        
        ground_truth - nominal_state = delta = H @ error_state + noise
        """
        H = np.zeros((6, 18)) #这里并不是7*18,因为选择轴角对四元数作了参数化,相当于对四元数对应的参数作error state的更新，则Eq. 279中对角度的偏导数结果同p,v一样，变成了单位阵
        H[0:3, 0:3] = np.eye(3) #\partial{p} ni 
        H[3:6, 6:9] = np.eye(3) #\partial{q} pixni
        PHt = self.error_covar @ H.T  # 18x6
        # compute Kalman gain. HPH^T, project the error covariance to the measurement space.
        K = PHt @ la.inv(H @ PHt + measurement_covar)  # 18x6

        # update error covariance matrix
        self.error_covar = (np.eye(18) - K @ H) @ self.error_covar
        # force the error_covar to be a symmetrical matrix
        self.error_covar = 0.5 * (self.error_covar + self.error_covar.T)

        # compute the measurements according to the nominal state and ground-truth state.
        if gt_measurement[3] < 0:
            gt_measurement[3:7] *= -1
        gt_p = gt_measurement[0:3]
        gt_q = gt_measurement[3:7]
        q = self.nominal_state[6:10]

        delta = np.zeros((6, 1))
        delta[0:3, 0] = gt_p - self.nominal_state[0:3]
        delta_q = tr.quaternion_multiply(tr.quaternion_conjugate(q), gt_q)
        if delta_q[0] < 0:
            delta_q *= -1
        angle = math.asin(la.norm(delta_q[1:4]))
        if math.isclose(angle, 0):
            axis = np.zeros(3,)
        else:
            axis = delta_q[1:4] / la.norm(delta_q[1:4])
        delta[3:6, 0] = angle * axis

        # compute state errors.
        errors = K @ delta #18x1

        # inject errors to the nominal state
        self.nominal_state[0:3] += errors[0:3, 0]  # update position
        dq = tr.quaternion_about_axis(la.norm(errors[6:9, 0]), errors[6:9, 0])
        # print(dq)
        self.nominal_state[6:10] = tr.quaternion_multiply(q, dq)  # update rotation
        self.nominal_state[6:10] /= la.norm(self.nominal_state[6:10])
        if self.nominal_state[6] < 0:
            self.nominal_state[6:10] *= -1
        self.nominal_state[3:6] += errors[3:6, 0]
        self.nominal_state[10:] += errors[9:, 0]  # update the rest.

        """
        reset errors to zero and modify the error covariance matrix.
        we do nothing to the errors since we do not save them.
        but we need to modify the error_covar according to P = GPG^T
        """
        G = np.eye(18)
        G[6:9, 6:9] = np.eye(3) - tr.skew_matrix(0.5 * errors[6:9, 0])
        self.error_covar = G @ self.error_covar @ G.T
    
    def update_with_sdf(self, points: np.array, sdf_normals: np.array, sdf_residual: np.array, point_normal_cross: np.array, measurement_covar: np.array):
        """
        :param points: nx3 array
        :param point_normal_cross: nx3 array get by 'points x sdf_normals'
        :param measurement_covar: nxn covariance matrix of the measurements (lidar points)
        :return: 
        """
        n = points.shape[0]
        # print(points.shape)
        # print(sdf_normals.shape)
        # print(sdf_residual.shape)
        # print(point_normal_cross.shape)
        # print(measurement_covar.shape)
        H = np.zeros((n, 18)) #这里并不是7*18,因为选择轴角对四元数作了参数化,相当于对四元数对应的参数作error state的更新，则Eq. 279中对角度的偏导数结果同p,v一样，变成了单位阵
        H[:,0:3] = sdf_normals #\partial{p} ni
        H[:, 6:9] = point_normal_cross #\partial{q} pixni

        PHt = self.error_covar @ H.T  # 18x6
        # compute Kalman gain. HPH^T, project the error covariance to the measurement space.
        K = PHt @ la.inv(H @ PHt + measurement_covar)  # 18x6
        
        # compute Kalman gain. HPH^T, project the error covariance to the measurement space.
        # using Eq.20 of fast-lio
        # Rinv = np.linalg.inv(measurement_covar)
        # LASER_POINT_COV = 0.00015
        # K = la.inv(H.T @ H + la.inv(self.error_covar/LASER_POINT_COV)) @ H.T #18xn

        # update error covariance matrix
        self.error_covar = (np.eye(18) - K @ H) @ self.error_covar
        # force the error_covar to be a symmetrical matrix
        self.error_covar = 0.5 * (self.error_covar + self.error_covar.T)

        # compute the measurements according to the nominal state and ground-truth state.
        # if gt_measurement[3] < 0:
        #     gt_measurement[3:7] *= -1
        # gt_p = gt_measurement[0:3]
        # gt_q = gt_measurement[3:7]
        q = self.nominal_state[6:10]

        # compute state errors.
        errors = K @ sdf_residual #18x18x18x1->18x1

        # print(errors.shape)

        # inject errors to the nominal state
        self.nominal_state[0:3] += errors[0:3]  # update position
        dq = tr.quaternion_about_axis(la.norm(errors[6:9]), errors[6:9])
        # print(dq)
        self.nominal_state[6:10] = tr.quaternion_multiply(q, dq)  # update rotation
        self.nominal_state[6:10] /= la.norm(self.nominal_state[6:10])
        if self.nominal_state[6] < 0:
            self.nominal_state[6:10] *= -1
        self.nominal_state[3:6] += errors[3:6]
        self.nominal_state[10:] += errors[9:]  # update the rest.
        
        # for pin-slam usage
        self.delta_T = np.eye(4)
        self.delta_T[:3,:3] = tr.quaternion_matrix(dq)[:3,:3]
        self.delta_T[:3, 3] = errors[0:3]
        """
        reset errors to zero and modify the error covariance matrix.
        we do nothing to the errors since we do not save them.
        but we need to modify the error_covar according to P = GPG^T
        """
        G = np.eye(18)
        G[6:9, 6:9] = np.eye(3) - tr.skew_matrix(0.5 * errors[6:9])
        self.error_covar = G @ self.error_covar @ G.T

    def __predict_nominal_state(self, imu_measurement: np.array):
        p = self.nominal_state[:3].reshape(-1, 1)
        v = self.nominal_state[3:6].reshape(-1, 1)
        q = self.nominal_state[6:10]
        a_b = self.nominal_state[10:13].reshape(-1, 1)
        w_b = self.nominal_state[13:16]
        g = self.nominal_state[16:19].reshape(-1, 1)

        w_m = imu_measurement[1:4].copy()
        a_m = imu_measurement[4:7].reshape(-1, 1).copy()

        dt = imu_measurement[0] - self.last_predict_time
        assert dt > 0.
        if dt > 0.1: dt = 0.001 # for the first reading

        
        # eq 260a~260f
        R = tr.quaternion_matrix(q)[:3, :3]
        p_next = p + v*dt+0.5*(R@(a_m-a_b)+g)*dt*dt
        v_next = v + (R@(a_m-a_b)+g)*dt
        angle = la.norm(w_m-w_b)
        axis = w_m / angle
        dR = tr.rotation_matrix(dt * angle, axis)
        dq = tr.quaternion_from_matrix(dR, True)
        q_next = tr.quaternion_multiply(q, dq)

        self.nominal_state[:3] = p_next.reshape(3,)
        self.nominal_state[3:6] = v_next.reshape(3,)
        self.nominal_state[6:10] = q_next


    def __predict_error_covar(self, imu_measurement: np.array):
        
        w_m = imu_measurement[1:4]
        a_m = imu_measurement[4:7]
        dt = imu_measurement[0] - self.last_predict_time
        assert dt > 0.
        if dt > 0.1: dt = 0.001 # for the first reading

        a_b = self.nominal_state[9:12]
        w_b = self.nominal_state[12:15]
        q = self.nominal_state[6:10]
        R = tr.quaternion_matrix(q)[:3, :3]

        Fx = np.zeros((18, 18))
        Fx[0:3, 0:3] = np.eye(3)
        Fx[0:3, 3:6] = np.eye(3) * dt
        Fx[3:6, 3:6] = np.eye(3)
        Fx[3:6, 6:9] = -R @ tr.skew_matrix(a_m - a_b) * dt
        Fx[3:6, 9:12] = -R * dt
        Fx[3:6, 15:18] = np.eye(3) * dt
        
        phi = (w_m-w_b) * dt
        angle = la.norm(phi)
        axis = phi / angle
        dR = tr.rotation_matrix(angle, axis)[:3, :3]
        Fx[6:9, 6:9] = dR.T
        Fx[6:9, 12:15] = -np.eye(3)*dt
        Fx[9:18, 9:18] = np.eye(9)

        Fi = np.zeros((18,12))
        Fi[3:15, :] = np.eye(12)

        Vi = self.noise_covar[:3,:3]*dt*dt
        Thetai = self.noise_covar[3:6,3:6]*dt*dt
        Ai = self.noise_covar[6:9,6:9]*dt
        Omegai = self.noise_covar[9:12,9:12]*dt
        Qi=np.zeros((12,12))
        Qi[:3,:3] = Vi
        Qi[3:6,3:6] = Thetai
        Qi[6:9,6:9] = Ai
        Qi[9:12,9:12] = Omegai
        
        # propagate时，error_state为零向量，Eq.268
        self.error_covar = Fx @ self.error_covar @ Fx.T + Fi @ Qi @ Fi.T 

