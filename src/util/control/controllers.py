import numpy as np
from src.util.state.state import HighLevelState
from src.util.trajectory.trajectory import RawTrajectory
from src.util.state.state import LowLevelState

# An interface that defines a "ControlAction"
class ControlAction:
    def __init__(self):
        raise NotImplementedError

# Control action connected to the LowLevelState class
# This class holds data that affects the "body state" of the system
class LowLevelControlAction(ControlAction):
    def __init__(self, accel=0.0, delta=0.0):
        self.accel = accel
        self.delta = delta

# Control action connected with the HighLevelState class
# This class is essentially going to be passed into the LowLevelState class
# This control action is derived from the global state and the desired trajectory
# that is to be followed (sometimes this is known as the "Twist" message in ROS systems)
class HighLevelControlAction(ControlAction):
    def __init__(self, vel_linear, vel_rotational):
        self.vel_linear = vel_linear
        self.vel_rotational = vel_rotational

# A controller interface to be inherited by Concrete controller classes, for example the PIDController class
class Controller:
    def __init__(_):
        raise NotImplementedError
    
    def calculate_control(_):
        raise NotImplementedError

    def saturate(_):
        raise NotImplementedError
    
    def update_gains(_):
        raise NotImplementedError
    
    def setpoint_reset(_):
        raise NotImplementedError
    
    def update_time_parameters(_):
        raise NotImplementedError

# An implemented PID Feedback controller from a textbook by Randal Beard
# It implements the Band limited derivative (dirty derivative) and a simple anti-windup
# scheme for the integrator potion (using simple saturation)
class PIDController(Controller):
    def __init__(self, kp=0.0, ki=0.0, kd=0.0, Ts=0.005, llim=-100.0, ulim=100.0, sigma=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.Ts = Ts
        self.llim = llim
        self.ulim = ulim
        self.sigma = sigma
        self.beta = (2.0*self.sigma - self.Ts) / (2.0*self.sigma + self.Ts)
        self.y_d1 = 0
        self.error_d1 = 0
        self.y_dot = 0
        self.error_dot = 0
        self.integrator = 0

    def calculate_control(_, y_r, y):
        # Compute current error
        error = y_r - y
        # Integrate erro rusing trapazoidal rule
        _.integrator = _.integrator + ((_.Ts/2) * (error + _.error_d1))

        # Prevent integrator unsaturation
        if _.ki != 0.0:
            integrator_unsat = _.ki*_.integrator
            _.integrator=_.saturate(integrator_unsat)/_.ki
        
        # Differentiate error
        _.error_dot = _.beta*_.error_dot + (((1-_.beta)/_.Ts) * (error - _.error_d1))

        # PID Control
        u_unsat = (_.kp * error) + (_.ki * _.integrator) + (_.kd * _.error_dot)

        # Saturate control input
        u_sat = _.saturate(u_unsat)

        _.error_d1 = error
        _.y_d1 = y
        return u_sat

    def saturate(self, u):
        return max(min(self.ulim, u), self.llim)
    
    def update_time_parameters(self, Ts, sigma):
        self.Ts = Ts
        self.sigma = sigma
        self.beta = (2.0*self.sigma - self.Ts) / (2.0*self.sigma + self.Ts)
    
    def setpoint_reset(self, y_r, y):
        self.integrator = 0
        self.error_d1 = y_r - y
        self.error_dot = 0
    
    def update_gains(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

# A class that consists of two controllers for the 2D car system.
# The controllers are separate in this implementation, the velocity control is its own independent PID controller
# and the steering control is also its own independe PID controller
class LowLevelController:
    def __init__(self, vel_controller: PIDController, steer_controller: PIDController):
        self.vel_controller = vel_controller
        self.steer_controller = steer_controller
    
    def calculate_control(self, desired: LowLevelState, actual: LowLevelState):
        # Get acceleration from body velocity error
        accel = self.vel_controller.calculate_control(desired.velocity.x, actual.velocity.x)
        # Get steering angle from body angular velocity error
        # TODO: Change this so that this controller outputs a "torque" which then changes
        # delta in the update method of the car (representing the steering system as a first order velocity model)
        delta = self.steer_controller.calculate_control(desired.ang_vel, actual.ang_vel)
        action = LowLevelControlAction(accel = accel, delta=delta)
        return action
        
class Stanley:
    def __init__(self):
        pass

    def calculate_controls(self, traj:RawTrajectory, state:HighLevelState, length:float):
        # controller gains
        klong = 1.0
        ksteer = 0.1

        return_state = HighLevelState()
        t = state.time
        # Check if time exceeds traj time
        if(t>traj.t[-2]):
            return return_state

        # Get index position to use to get current desired state
        index = 0
        for i in range(0,len(traj.t)):
            if(traj.t[i] < t):
                index = i
            elif(traj.t[i] > t):
                break

        # Get interpolation %
        if t != 0:
            interp_ratio =  (t - traj.t[index]) / (traj.t[index+1] - traj.t[index])
        else:
            interp_ratio = 0

        # Get interpolated desired state
        state0 = traj.x[:, index]
        state1 = traj.x[:, index+1]
        desired_state = state0 + ((state1-state0)*interp_ratio)

        # Calculate return_state

        # return desired vel form trajectory
        return_state.vel_mag = desired_state[2]

        # Pull desired heading

        # Calculate Lat and long and heading errors
        pos_body = state.position
        pos_traj = np.array([desired_state[0], desired_state[1]])
        lat_error = np.dot((pos_traj - pos_body), (0,1))
        long_error = np.dot((pos_traj - pos_body),(1,0))
        head_error = desired_state[3] - state.theta

        # Get cross track error steering angle
        delta = head_error + np.arctan2(ksteer*lat_error, (0.001 + state.vel_mag))

        # update desired omega from delta
        omega = state.vel_mag*np.tan(delta)/length
        return_state.omega = omega
        return return_state
