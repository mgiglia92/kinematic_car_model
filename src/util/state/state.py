from pygame.math import Vector2
import numpy as np

class State:
    def __init__(self):
        raise NotImplementedError
    
    
# The "body state" of the system, in this case it is the body linear velocities and the body angular velocity
# Velocity is in m/s
# ang_vel is in rad/sec
class LowLevelState(State):
    def __init__(self, velocity=Vector2(0.0, 0.0), ang_vel=0.0):
        self.velocity = velocity
        self.ang_vel = ang_vel

# The global state of the system, the position and angle of the system in the global reference frame
# Note that theta here is in degrees NOT radians. This is because pygame uses degrees not radians 
# to rotate objects before drawing them.
# Position is in meters
# theta is in degrees
class HighLevelState(State):
    def __init__(self, time = 0.0, position = Vector2(30,30), vel_mag = 0.0, theta=0.0, omega=0.0):
        self.time = time
        self.position = position
        self.vel_mag = vel_mag
        self.theta = theta
        self.omega=omega

    def get_casadi_form(self):
        return self.time, np.array([ self.position.x,
                        self.position.y,                    
                        self.vel_mag,
                        self.theta,
                        self.omega
                        ], dtype=np.float32)