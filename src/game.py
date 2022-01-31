import os
from math import sin, radians, degrees, copysign
from src.util.control.controllers import Stanley
from util.car.car import Car, FullCar
import pygame
from pygame.math import Vector2
import numpy as np
import matplotlib.pyplot as plt

from util.control.controllers import LowLevelController, PIDController
from util.state.state import HighLevelState, LowLevelState
from util.trajectory.trajectory import TrajectoryGenerator


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("2D Car Kinematics (Bicycle Model)")
        width = 1280
        height = 720
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.exit = False
        self.vel_data = []
        self.time = []
        self.Ts = 0.01

    def run(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "car_image.png")
        car_image = pygame.image.load(image_path)

        image_scale = 1/50
        image_dims = Vector2(2401, 1192)*image_scale
        car_image = pygame.transform.scale(car_image, image_dims)
        # car = Car(100,100, angle=0)
        vel_controller = PIDController(kp=100.0, ki=0.0, kd=00.0, Ts=self.Ts, sigma=10, llim=-30, ulim=30)
        steer_controller = PIDController(kp=100.0, ki=0.0, kd=0.0, Ts=self.Ts, sigma=10, llim=-1*np.pi, ulim=np.pi)
        controller = LowLevelController(vel_controller=vel_controller, steer_controller=steer_controller)
        global_state = HighLevelState(time=0, position=Vector2(30,30), vel_mag=0, theta=0)
        body_state = LowLevelState(Vector2(0,0), ang_vel=0.0)
        car = FullCar(global_state=global_state, body_state=body_state, controller=controller)
        ppu = int(2401*image_scale/car.length)
        # ppu = 5

        desired = LowLevelState(Vector2(10.0, 0), ang_vel=1.5)

        t_n = 0 # Number of steps taken (for plotting vs. time purposes)

        # Generate trajectory
        x0 = HighLevelState(time=0, position=Vector2(30,30), vel_mag=0.0, theta=0.0, omega=0.0)
        xf = HighLevelState(time=5, position=Vector2(40, 40), vel_mag=0.0, theta=0.0, omega=0.0)
        trajGenerator = TrajectoryGenerator()
        traj = trajGenerator.generate_trajectory_from_2_state(x0, xf)
        ppcontroller = Stanley()

        while not self.exit:
            dt = self.clock.get_time() / 1000.0

            # Event queue
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True

            # User input
            pressed = pygame.key.get_pressed()
            
            # Reset car position if out of screen
            if(car.global_state.position.x*ppu > 1200):
                car.global_state.position.x = 0

            #Store data inside list testing
            t_n += 1

            # Update car global state
            car.global_state.time = t_n*self.Ts
            # Car logic goes here
            car.update(trajectory=traj, controller=ppcontroller)
            
            # Drawing
            # self.screen.fill((0, 0, 0))

            # Draw trajectory
            for i in range(len(traj.t)):
                pos = Vector2(x=traj.x[0,i], y=traj.x[1,i])
                pygame.draw.circle(self.screen, center=pos*ppu, radius=10, color = (255,0,0))
            
            rotated = pygame.transform.rotate(car_image, car.global_state.theta)
            rect = rotated.get_rect()
            augmented_pos = car.global_state.position*ppu - (rect.width / 2, rect.height / 2)
            self.screen.blit(rotated, augmented_pos)
            self.clock.tick(self.ticks)
            pygame.display.flip()

            
            print("millis: " + str(t_n*self.Ts) + " theta: " + str(car.global_state.theta) + \
                " angvel_a:" + str(car.body_state.ang_vel) + " action:" + str(car.control_action.delta))
            # print("millis: " + str(millis) + " vd: " + str(desired.velocity) + " va: " + str(car.velocity.x) + " accel: " + str(car.control_action.accel))
        pygame.quit()


if __name__ == '__main__':
    game = Game()
    game.run()