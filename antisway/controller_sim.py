import os
import numpy as np
import matplotlib.pyplot as plt
import pygame
import sys
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import controller_fetch as FE
import scipy.io

from return_handler import return_handler

class ctrl_sim:
    def __init__(self, param_file):
        self.param_file = param_file
        self.load_parameters()
        self.run_simulation()

    def load_parameters(self):
        with open(self.param_file, 'r') as file:
            lines = file.readlines()
            params = {}
            for line in lines:
                key, value = line.strip().split(': ')
                params[key.strip()] = value.strip()

        # Variables
        self.controller_enabled = bool(params.get('controller_enabled', False))
        self.run_sim = bool(params.get('run_sim', False))
        self.kp = float(params.get('kp', 0.0))
        self.ki = float(params.get('ki', 0.0))

        self.antisway_enabled = bool(params.get('antisway_enabled', False))
        self.antisway_gain = float(params.get('antisway_gain', 0.0))

        self.m0 = float(params.get('m0', 0.0))
        self.m1 = float(params.get('m1', 0.0))
        self.b0 = float(params.get('b0', 0.0))
        self.b1 = float(params.get('b1', 0.0))
        self.g = float(params.get('g', 0.0))
        self.l = float(params.get('l', 0.0))
        self.xi = float(params.get('xi', 0.0))
        self.ti = float(params.get('ti', 0.0))
        self.dThi = float(params.get('dThi', 0.0))
        self.dt = float(params.get('dt', 0.0))
        self.sim_speed = float(params.get('sim_speed', 0.0))
        self.sim_length = float(params.get('sim_length', 0.0))
        self.motor_constant = float(params.get('motor_constant', 0.0))
        self.amp_constant = float(params.get('amp_constant', 0.0))
        self.gear_radius = float(params.get('gear_radius', 0.0))

        self.vel_type = params.get('vel_type', 'Step')
        self.force_type = params.get('force_type', 'Step')
        self.ramp = float(params.get('ramp', 0.0))
        self.force_ramp = float(params.get('force_ramp', 0.0))

        self.u_type = params.get('u_type', 'Step')
        self.u_ramp = float(params.get('u_ramp', 0.0))

        self.model_type = params.get('model_type', 'Anti-Sway')

        self.plot_F = bool(params.get('plot_F', False))
        self.plot_u = bool(params.get('plot_u', False))
        self.plot_u_Vel = bool(params.get('plot_u_Vel', False))
        self.plot_X = bool(params.get('plot_X', False))
        self.plot_Vset = bool(params.get('plot_Vset', False))
        self.plot_Vact = bool(params.get('plot_Vact', False))
        self.plot_Theta = bool(params.get('plot_Theta', False))
        self.plot_Vsend = bool(params.get('plot_Vsend', False))
        self.grid_on = bool(params.get('grid_on', False))
        
        

    def get_vset(self, v_type, time, dt, ramp):
        Vset = np.ones_like(time)
        v_type = str(v_type)
        if v_type == 'Step':
            length = len(Vset)
            divide = int(length/4)
            for i in range(len(Vset)):
                if i < divide:
                    Vset[i] = 0
                else:
                    Vset[i] = ramp
        else:
            Vset = np.zeros_like(time)
        return Vset


    def plot_matplotlib(self, time, X, dX, Vset, Theta, F, M1, vsend, u=None, dTheta=None):
        
        u_Vel = []
        l = self.l
        for v, d0 in zip(dX, dTheta):
            u_Vel.append(v+d0*l)
            
        torque = []; amps = []; volts = []; rpm = []
        for fnow, v in zip(F, dX):
            torque.append(fnow*self.gear_radius)
            amps.append(torque[-1]/self.motor_constant)
            volts.append(amps[-1]/self.amp_constant)
            rev_per_s = v/(2*np.pi*self.gear_radius)
            curr_rpm = rev_per_s*60
            rpm.append(curr_rpm)
            
        outputs = f'Outputs: ThetaMax={int(100*max(np.abs(Theta))*180/np.pi)/100}deg, Vp Max={int(max(u_Vel)*100)/100} m/s, Vt Max={int(max(dX)*100)/100} m/s'
        inputs = f'Inputs: Ki={self.ki} (Ns/m), Kp={self.kp} (Ns/m), Gain={self.antisway_gain}, Model={self.model_type}'
        
        figure = plt.figure(figsize=(12, 6))

        if self.plot_X:
            plt.plot(time, X, label='Trolley X (m)', color="green")
        if self.plot_Vset:
            plt.plot(time, Vset, label='V Desired Trolley (m/s)', color="blue")
        if self.plot_Vact:
            plt.plot(time, dX, label='V Actual Trolley (m/s)', color="red")
        if self.plot_u_Vel:
            plt.plot(time, np.array(u_Vel), label= 'V Actual Mass (m/s)', color="orange")
        if self.plot_F:
            plt.plot(time, np.array(F), label='F Motor (N)', color="pink")
        if u is not None and self.plot_u:
            plt.plot(time, np.array(u), label= 'F on Mass (N)', color="purple")
        if self.plot_Theta:
            plt.plot(time, 3*np.array(Theta), label = '3*Theta (rad)', color="cyan")
        if self.plot_Vsend:
            plt.plot(time, np.array(vsend), label = "Motor V_desired", color="grey")  
                
        if self.grid_on: plt.grid()
        plt.title(f'{inputs}\n{outputs}', fontsize=16)
        plt.xlabel('Time (s)', fontsize=16)
        plt.ylabel('Variables', fontsize = 16)
        plt.legend()
        figure.savefig('plot.pdf')

    def run_simulation(self):
            # Write your simulation logic here
            dt = self.dt
            controller = self.controller_enabled
            Kp = self.kp
            Ki = self.ki
            anti_sway = self.antisway_enabled
            as_gain = self.antisway_gain
            M0 = self.m0
            M1 = self.m1
            B0 = self.b0
            B1 = self.b1
            g = self.g
            l = self.l
            Xi = self.xi
            Ti = self.ti
            simulation_speed = self.sim_speed
            simulation_length = self.sim_length
            v_type = self.vel_type
            ramp = self.ramp
            percentage = 0.4
            dThi = self.dThi
            
            #initialize
            self.time = np.arange(0,simulation_length,dt)
            time = self.time
            Vset = self.get_vset(v_type, time, dt, ramp)

            #CURRENTLY UNUSED OLD PARAMS
            #force parameters:
            DISTANCE = False #specify the distance and get there with 0 sway. else velocity mode
            F_app = 0
            F = self.get_vset(self.force_type, time, dt, self.force_ramp)
            
            u = None

            theta_max = 40 #degrees, maximum sway angle
            track_destination = 10 #meters, for distance mode
            velocity_set = 3 #for velocity mode

            stored = [F, time, dt, M0, M1, B0, B1, g, l, Xi, Ti, F_app, percentage, controller, velocity_set, Kp, Ki, Vset, anti_sway, as_gain, u, dThi]
            
            # Run simulation
            Theta, X, dX, F, dTheta, vsend = FE.fetch_pendulum_mode(stored)
            return_handler(velocity_ct = dX, angle_ct = Theta, action_ct = F, target = Vset)

            #self.plot_matplotlib(time, X, dX, Vset, Theta, F, M1, vsend, u, dTheta)
            


if __name__ == "__main__":
    param_file = 'controller_sim_params.txt'
    sim_interface = ctrl_sim(param_file)
    # Add code to run the simulation using the parameters loaded