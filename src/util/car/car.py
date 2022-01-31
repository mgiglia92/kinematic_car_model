from pygame.math import Vector2
import numpy as np
from src.util.control.controllers import Stanley
from src.util.trajectory.trajectory import RawTrajectory
from util.control.controllers import ControlAction, LowLevelController, PIDController, LowLevelControlAction
from util.state.state import LowLevelState, HighLevelState

# This car class holds all of the data related to the global state, body state, controllers, and other limiting parameters for the systems
# This class also holds the update() method, which is just the discrete time integration math to update the position and angle of the car
# to be drawn on the screen in pygame. This update() method also integrates all other internal state variables by calling the
# controller's calculate_control() method.
class FullCar:
    def __init__(self, global_state: HighLevelState, body_state: LowLevelState, controller: LowLevelController, max_steer = np.pi/4, length = 4):
        self.global_state = global_state
        self.max_steer = max_steer
        self.max_velocity = 100.0
        self.controller=controller
        self.body_state = body_state
        self.control_action = LowLevelControlAction()
        self.length = length
    
    # Integration method
    def low_level_update(self, desired: LowLevelState):
        dt = self.controller.vel_controller.Ts
        # Get control action
        self.control_action = self.controller.calculate_control(desired, self.body_state)

        # Integrate velocity
        self.body_state.velocity += Vector2(self.control_action.accel*dt, 0.0)
        # Integrate theta
        self.body_state.ang_vel = self.body_state.velocity.x * np.tan(self.control_action.delta) / self.length
        # self.body_state.ang_vel = 1.0
        self.global_state.theta += (180/np.pi)*self.body_state.ang_vel*dt 
        # Integrate position (.rotate uses degrees!)(
        self.global_state.position += self.body_state.velocity.rotate(-1*self.global_state.theta)*dt
        self.global_state.vel_mag = self.body_state.velocity.x
    
    def update(self, trajectory: RawTrajectory, controller: Stanley):
        control_state = controller.calculate_controls(trajectory, self.global_state, self.length)
        des = LowLevelState()
        des.velocity.x=control_state.vel_mag
        des.velocity.y=0.0
        des.ang_vel = control_state.omega
        self.low_level_update(desired = des)

# An example car class found online
class Car:
    def __init__(self, x, y, angle=0.0, length=4, max_steering=30, max_acceleration=5.0):
        self.position = Vector2(x, y)
        self.velocity = Vector2(0.0, 0.0)
        self.angle = angle
        self.length = length
        self.max_acceleration = max_acceleration
        self.max_steering = max_steering
        self.max_velocity = 20
        self.brake_deceleration = 10
        self.free_deceleration = 2

        self.acceleration = 0.0
        self.steering = 0.0

    def update(self, dt):
        self.velocity += (self.acceleration * dt, 0)
        self.velocity.x = max(-self.max_velocity, min(self.velocity.x, self.max_velocity))

        if self.steering:
            turning_radius = self.length / np.sin(self.steering)
            angular_velocity = self.velocity.x / turning_radius
        else:
            angular_velocity = 0

        self.position += self.velocity.rotate(-self.angle) * dt
        self.angle += angular_velocity * dt