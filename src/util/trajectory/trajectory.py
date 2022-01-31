
import src.util.control.controllers as controllers
from src.util.state.state import HighLevelState
import scipy.interpolate as interpolate
import numpy as np
from pygame.math import Vector2
from casadi import *
# plotting
from pylab import plot, step, figure, legend, show, spy, quiver, scatter

class TrajectoryGenerator:
    def __init__(self):
        pass

    def generate_trajectory_from_2_state(self, initState: HighLevelState, finalState: HighLevelState):
        ti, xi = initState.get_casadi_form()
        tf, xf = finalState.get_casadi_form()
        N = 10
        opti = Opti()
        X = opti.variable(5, N+1) # State trajectory
        U = opti.variable(2, N) # Control Trajectory (acceleration, angular_velocity)
        T = np.linspace(ti, tf, N+1)
        # Dynamic constraints
        f = lambda x,u: vertcat(cos(x[3])*x[2], sin(x[3])*x[2], u[0], x[4], u[1])

        dt = (tf-ti)/N # length of a control interval
        for k in range(N): # loop over control intervals
            # Runge-Kutta 4 integration
            k1 = f(X[:,k],         U[:,k])
            k2 = f(X[:,k]+dt/2*k1, U[:,k])
            k3 = f(X[:,k]+dt/2*k2, U[:,k])
            k4 = f(X[:,k]+dt*k3,   U[:,k])
            x_next = X[:,k] + dt/6*(k1+2*k2+2*k3+k4) 
            opti.subject_to(X[:,k+1]== x_next) # close the gaps

        # sum 
        g = lambda x, y: (((x[0]-y[0])**2) + (((x[1]-y[1])**2)))

        # Objectives
        for i in range(N):
            opti.minimize(1e4*(X[0, i+1] - X[0, i])**2)
            opti.minimize(1e4*(X[1, i+1] - X[1, i])**2)


        # set other constraints
        opti.subject_to(opti.bounded(-10, X[2], 10))
        opti.subject_to(opti.bounded(-20, U[0, :], 20))
        opti.subject_to(opti.bounded(-2, U[1, :], 2))
        opti.subject_to(X[:,0] == xi)
        opti.subject_to(X[:,-1] == xf)

        # set initial guess
        for i in range(N+1):

            opti.set_initial(X[0,i], (xf[0] - xi[0])/(N+1-i))
            opti.set_initial(X[1,i], (xf[1] - xi[1])/(N+1-i))
        opti.set_initial(U, np.repeat(vertcat(1,0), N, axis=1))

        # solver settings
        opti.solver("ipopt")
        sol = opti.solve()



        scatter(sol.value(X[0,:]), sol.value(X[1,:]), label='pos traj')
        u = sol.value(cos(X[3,:])*X[2,:])*10
        v = sol.value(sin(X[3,:])*X[2,:])*10
        quiver(sol.value(X[0,:]), sol.value(X[1,:]), u, v)

        show(block=False)
        print(sol)
        return RawTrajectory(t=T, state=sol.value(X), control=sol.value(U))

    def convert_state_to_casadi_form(state):
        pass

class RawTrajectory:
    def __init__(self, t:np.array, state:np.array, control:np.array):
        self.t = t
        self.x = state
        self.u = control



class Trajectory:
    def __init__(self):
        self.states = []

    def append_state(self, state):
        self.states.append(state)

    def update_spline(self):
        if not all(type(x) is HighLevelState for x in self.states):
            raise TypeError
        self.update_spline_from_state_array(self.states)

    def update_spline_from_state_array(self, stateList: list):
        # Check that each element in state list is a HighLevelState
        if not all(type(x) is HighLevelState for x in stateList):
            raise TypeError

        time=np.empty(0,)
        x=np.empty(0,)
        y=np.empty(0,)
        vel_mag = np.empty(0,)
        theta = np.empty(0,)

        for s in stateList:
            time = np.append(time, s.time)
            x = np.append(x, s.position.x)
            y = np.append(y, s.position.y)
            vel_mag = np.append(vel_mag, s.vel_mag)
            theta = np.append(theta, s.theta)
        x = np.expand_dims(x, axis=1)
        y = np.expand_dims(y, axis=1)
        pos = np.concatenate((x,y), axis=1)
        velx = np.expand_dims((np.cos(theta)*vel_mag), axis=1)
        vely = np.expand_dims((np.sin(theta)*vel_mag), axis=1)
        vel = np.concatenate((velx, vely), axis=1)
        omega = np.gradient(theta, time[1]-time[0])


        self.spline_trajectory_x = interpolate.BPoly.from_derivatives(time, np.stack((x, velx), axis=1))
        self.spline_trajectory_y = interpolate.BPoly.from_derivatives(time, np.stack((y, vely), axis=1))

def plot_splined_traj_from_trajectory(traj: Trajectory):
    import matplotlib.pyplot as plt
    plt.figure(2)

    # Generate t_fine
    t_fine = np.linspace(traj.states[0].time, traj.states[-1].time, 501)
    x = traj.spline_trajectory_x(t_fine)
    velx = traj.spline_trajectory_x.derivative(1)(t_fine)
    y = traj.spline_trajectory_y(t_fine)
    vely = traj.spline_trajectory_y.derivative(1)(t_fine)

    plt.scatter(x,y)
    plt.show()

def plot_splined_traj_from_state_list(stateList):
    import matplotlib.pyplot as plt
    plt.figure(1)
    time=np.empty(0,)
    x=np.empty(0,)
    y=np.empty(0,)
    vel_mag = np.empty(0,)
    theta = np.empty(0,)

    for s in stateList:
        time = np.append(time, s.time)
        x = np.append(x, s.position.x)
        y = np.append(y, s.position.y)
        vel_mag = np.append(vel_mag, s.vel_mag)
        theta = np.append(theta, s.theta)
    u = np.cos(theta)
    v = np.sin(theta)
    plt.scatter(x,y)
    plt.quiver(x,y,u,v)
    plt.show()

if __name__ == "__main__":
    x0 = HighLevelState(time=0, position=Vector2(30,30), vel_mag=0.0, theta=0.0, omega=0.0)
    xf = HighLevelState(time=10, position=Vector2(100, 100), vel_mag=0.0, theta=0.0, omega=0.0)
    trajGenerator = TrajectoryGenerator()
    traj = trajGenerator.generate_trajectory_from_2_state(x0, xf)
    
    # Pure pursuit controller test
    test_state = HighLevelState(time=5.25, position = Vector2(45, 40), vel_mag = 2.0, theta = np.pi/4, omega=0.0)
    traj_cont = controllers.PurePursuit()
    control_state = traj_cont.calculate_controls(traj, test_state)

    x = test_state.position.x
    y = test_state.position.y
    theta = test_state.theta
    u = cos(theta)
    v = sin(theta)
    scatter(x,y)
    quiver(x,y,u,v)
    ud = cos(control_state.theta)
    vd = sin(control_state.theta)
    quiver(x,y,ud,vd)
    print("SD")


    # stateList = []
    # f = 0.1 # Frequency of circle
    # time = np.linspace(0, 20, 11)
    # trajectory = Trajectory()
    # for t in time:
    #     theta = 2*np.pi*f*t
    #     x = 100*np.cos(theta + (np.pi/2)) + 30
    #     y = 100*np.sin(theta + (np.pi/2)) + 30
    #     vel_mag = 10 # velocity magnitude
    #     state = HighLevelState(time = t, position = Vector2(x,y), vel_mag = vel_mag, theta=theta)
    #     stateList.append(state)
    #     trajectory.append_state(state)
        
        

    # trajectory.update_spline_from_state_array(stateList)
    # plot_splined_traj_from_trajectory(trajectory)
    # plot_splined_traj_from_state_list(stateList)
