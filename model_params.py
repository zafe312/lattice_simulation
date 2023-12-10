import numpy as np

def param():
    Nx = 16;Ny = 16;Nz = 16;dt = 1e-1;ti = 1 ; epsilon = 0.0
    Lx = 5; Ly = 5; Lz = 5;
    prefix = 30 #number of domain pairs (consecutive true and false) within the length of the box
    T = 15
    Nt = int(round((T-ti)/float(dt)))
    return prefix, Lx, Ly, Lz, Nx, Ny, Nz, dt, ti, epsilon, T, Nt

def plot_times():
    prefix, Lx, Ly, Lz, Nx, Ny, Nz, dt, ti, epsilon, T, Nt = param()
    plt_pts = []
    for i in [11,21,31,41,51,61,71,81,91,101,111,121,131,141,151,161,171,181,191]:
        if i <= T:
            plt_pts.append(int((i-ti)/dt))
    
    return plt_pts
