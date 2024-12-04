import matplotlib.pyplot as plt
import numpy as np
import random

class PIController:
    def __init__(self, kp, ki, M, dt):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.dt = dt  # Time step
        self.M = M
        self.integral = 0

    def compute(self, error):
        # Proportional term
        proportional_term = self.kp * error

        # Integral term
        self.integral += error * self.dt
        integral_term = self.ki * self.integral

        # Calculate the control signal
        control_signal = (proportional_term + integral_term)*self.M

        return control_signal

#SIMULATE LOOP
def fetch_pendulum_mode(stored):
    F, time, dt, M0, M1, B0, B1, g, l, Xi, Ti, F_app, percentage, controller, velocity_set, Kp, Ki, Vset, anti_sway, as_gain, u, dThi = stored
    if controller:
        PI_control = PIController(Kp,Ki, M1, dt)
        F = [0]
    
    ddX = [0]; dX = [0]; X = [Xi]
    ddTheta = [0]; dTheta = [dThi]; Theta = [Ti]
    vsend = [0]
    
    def step_X(i, F, dt):
        ddX_n = (F-(M0*l*ddTheta[i])-(B1*dX[i]))/(M0+M1)
        dX_n = ddX_n*dt + dX[i]
        X_n = dX_n*dt + X[i]
        ddX.append(ddX_n); dX.append(dX_n); X.append(X_n)

    def step_Theta(i, dt):
        ddTheta_n = (-ddX[i]-g*Theta[i]-B0*dTheta[i])/l
        dTheta_n = ddTheta_n*dt + dTheta[i]
        Theta_n = dTheta_n*dt + Theta[i]
        ddTheta.append(ddTheta_n); dTheta.append(dTheta_n); Theta.append(Theta_n)
        
    
    for i in range(len(time)):
        if not i == len(time) - 1:
            if controller:
                if anti_sway:
                    K = 2*np.sqrt(l/g)*as_gain
                    F.append(PI_control.compute((Vset[i]+(K*g*Theta[i]))-dX[i]))
                    boi = (Vset[i]+(K*g*Theta[i]))
                else:
                    F.append(PI_control.compute(Vset[i]-dX[i]))
                    boi = Vset[i]
            step_X(i, F[i], dt)
            step_Theta(i, dt)
            vsend.append(boi)
    return Theta, X, dX, F, dTheta, vsend


def fetch_F_actual(stored):
    F, time, dt, M0, M1, B0, B1, g, l, Xi, Ti, F_app, percentage, controller, velocity_set, Kp, Ki, Vset, anti_sway, as_gain, u, dThi, act_theta, act_vel = stored
    PI_control = PIController(Kp,Ki, M1, dt)
    F = [0]
    v_desired = [0]
        
    for i in range(len(time)):
        if not i == len(time) - 1:
            K = 2*np.sqrt(l/g)*as_gain
            boi = (Vset[i]+(K*g*act_theta[i]))
            F.append(PI_control.compute(Vset[i]+(K*g*act_theta[i]))-act_vel[i])
            v_desired.append(boi)
    return F, v_desired


#SIMULATE LOOP
def fetch_tracking_mode(stored):
    F, time, dt, M0, M1, B0, B1, g, l, Xi, Ti, F_app, percentage, controller, velocity_set, Kp, Ki, Vset, anti_sway, as_gain, u, dThi = stored
    
    if controller:
        PI_control = PIController(Kp,Ki, M1, dt)
        F = [0]
        
    def saturate(Val):
        if np.abs(Val) > 12.9:
            return np.abs(Val)*12.9/Val
        else:
            return Val
    
    
    ddX = [0]; dX = [0]; X = [Xi]
    ddTheta = [0]; dTheta = [dThi]; Theta = [Ti]
    
    def step_X(i, F, u, dt):
        #+(u*Theta[i])
        ddX_n = (F+(u*np.abs(Theta[i]))-(M0*l*(ddTheta[i])+(B1*dX[i])))/(M0+M1)
        dX_n = ddX_n*dt + dX[i]
        X_n = dX_n*dt + X[i]
        ddX.append(ddX_n); dX.append(dX_n); X.append(X_n)

    def step_Theta(i, F, u, dt):
        #ddTheta_n = (g*Theta[i] - (l/M0)*(u+F))/(l-((l**2)/(M0*(M1+M0))))
        ddTheta_n = (-1*g*Theta[i]/l) + u/(l*(M0+M1)) - ddX[i]/l
        dTheta_n = ddTheta_n*dt + dTheta[i]
        Theta_n = dTheta_n*dt + Theta[i]
        ddTheta.append(ddTheta_n); dTheta.append(dTheta_n); Theta.append(Theta_n)
        
    Vset = 0
    vsend = [0]
    for i in range(len(time)):
        if not i == len(time) - 1:
            #if i > 0:
               # f_pass += g*M0*Theta[i]
            if controller:
                K = 2*np.sqrt(l/g)*as_gain
                F.append(saturate(PI_control.compute(((Vset + K*g*Theta[i]))-dX[i])))
                
            step_X(i, F[i], u[i], dt)
            step_Theta(i, F[i], u[i], dt)
            vsend.append((Vset + K*g*Theta[i]))
            
    return Theta, X, dX, F, dTheta, vsend