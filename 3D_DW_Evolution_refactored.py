"""
3D cosmological evolution equation for MLRSM potential solved by finite differences.
"""
from model_params import param
import time, sys
import numpy as np
import os, psutil # for checking memory uses
import h5py
import glob

def mem():
  print(f' Memory in use is {int(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)} mb \n') # gives mb of memory uses

def solver(prefix, I, Vu, Vv, lam, rho3, eta, a0, epsilon, Lx, Ly, Lz, Nx, Ny, Nz, dt, ti, T,
           user_action=None):
    print(f'\nLattice size is {(Lx,Ly,Lz)} with {(Nx,Ny,Nz)} = {(Nx)*(Ny)*(Nz)} lattice points.')
    print(f'ti = {ti}, tf = {T}, dt = {dt}, Nt = {int(round((T-ti)/float(dt)))}\n')
    print('Initializing and performing 1st step...')
    

    x = np.linspace(0, Lx, Nx, endpoint=False)  # mesh points in x dir
    y = np.linspace(0, Ly, Ny, endpoint=False)  # mesh points in y dir
    z = np.linspace(0, Lz, Nz, endpoint=False)  # mesh points in y dir
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    
    Bx = dt**2/(a0*dx)**2
    By = dt**2/(a0*dy)**2
    Bz = dt**2/(a0*dz)**2

    xv = x[:,np.newaxis,np.newaxis]          # for vectorized function evaluations
    yv = y[np.newaxis,:,np.newaxis]
    zv = z[np.newaxis,np.newaxis,:]

    stability_limit = (float(a0))*(1/np.sqrt(1/dx**2 + 1/dy**2))
    if dt <= 0:                # max time step?
        safety_factor = -dt    # use negative dt as safety factor
        dt = safety_factor*stability_limit
    elif dt > stability_limit:
        raise ValueError('error: dt=%g exceeds the stability limit %g' % \
              (dt, stability_limit))
        
    Nt = int(round((T-ti)/float(dt)))
    t = np.linspace(ti, ti+Nt*dt, Nt)    # mesh points in time
    dt2 = dt**2

    # Allow f and V to be None or 0
    if Vu is None or Vu == 0:
        Vu = lambda x, y, z: np.zeros((x.shape[0], y.shape[1], z.shape[2]))
            
    if Vv is None or Vv == 0:
        Vv = lambda x, y, z: np.zeros((x.shape[0], y.shape[1], z.shape[2]))


    order = 'C'
    u   = np.zeros((Nx,Ny,Nz), order=order)   # solution array
    u[:,:,:]   = -1
    u_1 = np.zeros((Nx,Ny,Nz), order=order)   # solution at t-dt
    u_1[:,:,:] = -1
    u_2 = np.zeros((Nx,Ny,Nz), order=order)   # solution at t-2*dt
    u_2[:,:,:] = -1
    
    v   = np.zeros((Nx,Ny,Nz), order=order)   # solution array
    v[:,:,:]   = -1
    v_1 = np.zeros((Nx,Ny,Nz), order=order)   # solution at t-dt
    v_1[:,:,:] = -1
    v_2 = np.zeros((Nx,Ny,Nz), order=order)   # solution at t-2*dt
    v_2[:,:,:] = -1
    v_2[:,:,:] = -1
    

    Ix = range(0, u.shape[0])
    Iy = range(0, u.shape[1])
    Iz = range(0, u.shape[2])
    It = range(0, t.shape[0])
    
    Jx = range(0, v.shape[0])
    Jy = range(0, v.shape[1])
    Jz = range(0, v.shape[2])

    import time; t0 = time.process_time()          # for measuring CPU time

    # Load initial condition into u_1
    u_1[:,:,:], v_1[:,:,:] = I(xv, yv, zv)
    u[:,:,:], v[:,:,:] = I(xv, yv, zv)
    u_2[:,:,:], v_2[:,:,:] = I(xv, yv, zv)

    if user_action is not None:
        user_action(u_1, v_1, x, xv, y, yv, z, zv, t, 0)

    # Special formula for first time step
    n = 0
    # First step requires a special formula
   
    V_u = Vu(xv, yv, zv)
    V_v = Vv(xv, yv, zv)
    u, v = advance(u, u_1, u_2, v, v_1, v_2, lam, rho3, n, eta, epsilon, Bx, By, Bz, ti, dt2, V_u=V_u, V_v=V_v, step1=True)
    vev = [(u.mean(),v.mean())]
    print(f'n = {n}, Av l = {round(u.mean(),5)}, Av r = {round(v.mean(),5)}')

    if user_action is not None:
        user_action(u, v, x, xv, y, yv, z, zv, t, 1)

    # Update data structures for next step
    #u_2[:] = u_1;  u_1[:] = u  # safe, but slower
    u_2, u_1, u = u_1, u, u_2
    v_2, v_1, v = v_1, v, v_2
    
    print('Entering loop...')
    for n in It[1:-1]:
        iter_t0 = time.process_time() # check iteration time
        
        u, v = advance(u, u_1, u_2, v, v_1, v_2, lam, rho3, n, eta, epsilon, Bx, By, Bz, ti, dt2)
        vev.append((u.mean(),v.mean()))
        print(f'n = {n}, Av l = {round(u.mean(),5)}, Av r = {round(v.mean(),5)}')


        if user_action is not None:
            if user_action(u, v, x, xv, y, yv, z, zv, t, n+1):
                break

        # Update data structures for next step
        #u_2[:] = u_1;  u_1[:] = u  # safe, but slower
        u_2, u_1, u = u_1, u, u_2
        v_2, v_1, v = v_1, v, v_2
        
        iter_t1 = time.process_time()
        iter_t = iter_t1 - iter_t0
#         print(f'n = {n} iteration done in {round(iter_t,3)} sec.')
        if n % 50 == 0:
          print(f'n = {n} iteration done.')
    
    # save vev as txt file
    filename = f'{prefix}_vev_{round(epsilon,3)}.csv'
    if glob.glob(filename):
        os.remove(filename)
    np.savetxt(filename, vev)
    print('vev exported :)')
    
    # Important to set u = u_1 if u is to be returned!
    t1 = time.process_time()
    # dt might be computed in this function so return the value
    return dt, t1 - t0


def advance(u, u_1, u_2, v, v_1, v_2, lam, rho3, n, eta, epsilon, Bx, By, Bz, ti, dt2,
                       V_u=None, V_v=None, step1=False):
    dt = np.sqrt(dt2)  # save
    Bx = Bx/(n*dt+ti); By = By/(n*dt+ti); Bz = Bz/(n*dt+ti)
    A = dt/(2*(ti+n*dt))

    u_xx = u_1[:-2,1:-1,1:-1] - 2*u_1[1:-1,1:-1,1:-1] + u_1[2:,1:-1,1:-1]
    u_yy = u_1[1:-1,:-2,1:-1] - 2*u_1[1:-1,1:-1,1:-1] + u_1[1:-1,2:,1:-1]
    u_zz = u_1[1:-1,1:-1,:-2] - 2*u_1[1:-1,1:-1,1:-1] + u_1[1:-1,1:-1,2:]
    
    v_xx = v_1[:-2,1:-1,1:-1] - 2*v_1[1:-1,1:-1,1:-1] + v_1[2:,1:-1,1:-1]
    v_yy = v_1[1:-1,:-2,1:-1] - 2*v_1[1:-1,1:-1,1:-1] + v_1[1:-1,2:,1:-1]
    v_zz = v_1[1:-1,1:-1,:-2] - 2*v_1[1:-1,1:-1,1:-1] + v_1[1:-1,1:-1,2:]
    
    if step1:
        u[1:-1,1:-1,1:-1] = (1 - ((lam*dt2)/2)*(u_1[1:-1,1:-1,1:-1]*u_1[1:-1,1:-1,1:-1] - \
                       eta**2))*u_1[1:-1,1:-1,1:-1] - (rho3/2)*(dt2/2)*(u_1[1:-1,1:-1,1:-1]*v_1[1:-1,1:-1,1:-1]*v_1[1:-1,1:-1,1:-1]) - \
                       ((3/2)*A-1)*dt*V_u[1:-1,1:-1,1:-1] + \
                       Bx*u_xx/2 + By*u_yy/2 + Bz*u_zz/2
                       
        v[1:-1,1:-1,1:-1] = (1 - ((lam*dt2)/2)*(v_1[1:-1,1:-1,1:-1]*v_1[1:-1,1:-1,1:-1] - \
                       eta**2))*v_1[1:-1,1:-1,1:-1] - (rho3/2)*(dt2/2)*(v_1[1:-1,1:-1,1:-1]*u_1[1:-1,1:-1,1:-1]*u_1[1:-1,1:-1,1:-1]) - \
                       ((3/2)*A-1)*dt*V_v[1:-1,1:-1,1:-1] + \
                       Bx*v_xx/2 + By*v_yy/2 + Bz*v_zz/2
    else:
        u[1:-1,1:-1,1:-1] = ((2 - lam*dt2*(u_1[1:-1,1:-1,1:-1]*u_1[1:-1,1:-1,1:-1] - \
                       eta**2))*u_1[1:-1,1:-1,1:-1] - (rho3/2)*dt2*(u_1[1:-1,1:-1,1:-1]*v_1[1:-1,1:-1,1:-1]*v_1[1:-1,1:-1,1:-1]) + \
                       ((3/2)*A-1)*u_2[1:-1,1:-1,1:-1] + \
                       Bx*u_xx + By*u_yy + Bz*u_zz)/(1+(3/2)*A)
        
        v[1:-1,1:-1,1:-1] = ((2 - lam*dt2*(v_1[1:-1,1:-1,1:-1]*v_1[1:-1,1:-1,1:-1] - \
                       eta**2))*v_1[1:-1,1:-1,1:-1] - (rho3/2)*dt2*(v_1[1:-1,1:-1,1:-1]*u_1[1:-1,1:-1,1:-1]*u_1[1:-1,1:-1,1:-1]) + \
                       ((3/2)*A-1)*v_2[1:-1,1:-1,1:-1] + \
                       Bx*v_xx + By*v_yy + Bz*v_zz)/(1+(3/2)*A)

    # Boundary condition u=0
    Bxx = np.sqrt(Bx/3)
    Byy = np.sqrt(By/3)
    Bzz = np.sqrt(Bz/3)
    tol_bc = 1e-4

    k = 0
    # u[:,:,k] = -1
    if abs(u[:,:,u.shape[2]-1].max()+1) > tol_bc:
        u[:,:,k] = u[:,:,u.shape[2]-1]
    else:
        u[:,:,k] = u_1[:,:,k] #-1.0
    k = u.shape[2] - 1
    # u[:,:,k] = -1
    u[0,0,k] = u_1[0,0,k] - Bxx*(u_1[0,0,k] - u_1[1,0,k]) - Byy*(u_1[0,0,k] - u_1[0,1,k]) - Bzz*(u_1[0,0,k] - u_1[0,0,k-1])
    u[0,1:,k] = u_1[0,1:,k] - Bxx*(u_1[0,1:,k] - u_1[1,1:,k]) - Byy*(u_1[0,1:,k] - u_1[0,:-1,k]) - Bzz*(u_1[0,1:,k] - u_1[0,1:,k-1])
    u[1:,0,k] = u_1[1:,0,k] - Bxx*(u_1[1:,0,k] - u_1[:-1,0,k]) - Byy*(u_1[1:,0,k] - u_1[1:,1,k]) - Bzz*(u_1[1:,0,k] - u_1[1:,0,k-1])
    u[1:,1:,k] = u_1[1:,1:,k] - Bxx*(u_1[1:,1:,k] - u_1[:-1,1:,k]) - Byy*(u_1[1:,1:,k] - u_1[1:,:-1,k]) - Bzz*(u_1[1:,1:,k] - u_1[1:,1:,k-1])
    
    k = 0
    # v[:,:,k] = -1
    if abs(v[:,:,v.shape[2]-1].max()+1) > tol_bc:
        v[:,:,k] = v[:,:,v.shape[2]-1]
    else:
        v[:,:,k] = v_1[:,:,k] #-1.0
    k = v.shape[2] - 1
    # v[:,:,k] = -1
    v[0,0,k] = v_1[0,0,k] - Bxx*(v_1[0,0,k] - v_1[1,0,k]) - Byy*(v_1[0,0,k] - v_1[0,1,k]) - Bzz*(v_1[0,0,k] - v_1[0,0,k-1])
    v[0,1:,k] = v_1[0,1:,k] - Bxx*(v_1[0,1:,k] - v_1[1,1:,k]) - Byy*(v_1[0,1:,k] - v_1[0,:-1,k]) - Bzz*(v_1[0,1:,k] - v_1[0,1:,k-1])
    v[1:,0,k] = v_1[1:,0,k] - Bxx*(v_1[1:,0,k] - v_1[:-1,0,k]) - Byy*(v_1[1:,0,k] - v_1[1:,1,k]) - Bzz*(v_1[1:,0,k] - v_1[1:,0,k-1])
    v[1:,1:,k] = v_1[1:,1:,k] - Bxx*(v_1[1:,1:,k] - v_1[:-1,1:,k]) - Byy*(v_1[1:,1:,k] - v_1[1:,:-1,k]) - Bzz*(v_1[1:,1:,k] - v_1[1:,1:,k-1])

    j = 0
    # u[:,j,:] = -1
    if abs(u[:,u.shape[1]-1,:].max()+1) > tol_bc:
        u[:,j,:] = u[:,u.shape[1]-1,:]
    else:
        u[:,j,:] = u_1[:,j,:] #-1.0
    j = u.shape[1]-1
    # u[:,j,:] = -1
    u[0,j,0] = u_1[0,j,0] - Bxx*(u_1[0,j,0] - u_1[1,j,0]) - Byy*(u_1[0,j,0] - u_1[0,j-1,0]) - Bzz*(u_1[0,j,0] - u_1[0,j,1])
    u[0,j,1:] = u_1[0,j,1:] - Bxx*(u_1[0,j,1:] - u_1[1,j,1:]) - Byy*(u_1[0,j,1:] - u_1[0,j-1,1:]) - Bzz*(u_1[0,j,1:] - u_1[0,j,:-1])
    u[1:,j,0] = u_1[1:,j,0] - Bxx*(u_1[1:,j,0] - u_1[:-1,j,0]) - Byy*(u_1[1:,j,0] - u_1[1:,j-1,0]) - Bzz*(u_1[1:,j,0] - u_1[1:,j,1])
    u[1:,j,1:] = u_1[1:,j,1:] - Bxx*(u_1[1:,j,1:] - u_1[:-1,j,1:]) - Byy*(u_1[1:,j,1:] - u_1[1:,j-1,1:]) - Bzz*(u_1[1:,j,1:] - u_1[1:,j,:-1])
    
    j = 0
    # v[:,j,:] = -1
    if abs(v[:,v.shape[1]-1,:].max()+1) > tol_bc:
        v[:,j,:] = v[:,v.shape[1]-1,:]
    else:
        v[:,j,:] = v_1[:,j,:] #-1.0
    j = v.shape[1]-1
    # v[:,j,:] = -1
    v[0,j,0] = v_1[0,j,0] - Bxx*(v_1[0,j,0] - v_1[1,j,0]) - Byy*(v_1[0,j,0] - v_1[0,j-1,0]) - Bzz*(v_1[0,j,0] - v_1[0,j,1])
    v[0,j,1:] = v_1[0,j,1:] - Bxx*(v_1[0,j,1:] - v_1[1,j,1:]) - Byy*(v_1[0,j,1:] - v_1[0,j-1,1:]) - Bzz*(v_1[0,j,1:] - v_1[0,j,:-1])
    v[1:,j,0] = v_1[1:,j,0] - Bxx*(v_1[1:,j,0] - v_1[:-1,j,0]) - Byy*(v_1[1:,j,0] - v_1[1:,j-1,0]) - Bzz*(v_1[1:,j,0] - v_1[1:,j,1])
    v[1:,j,1:] = v_1[1:,j,1:] - Bxx*(v_1[1:,j,1:] - v_1[:-1,j,1:]) - Byy*(v_1[1:,j,1:] - v_1[1:,j-1,1:]) - Bzz*(v_1[1:,j,1:] - v_1[1:,j,:-1])
    
    i = 0
    # u[i,:,:] = -1
    if abs(u[u.shape[0]-1,:,:].max()+1) > tol_bc:
        u[i,:,:] = u[u.shape[0]-1,:,:]
    else:
        u[i,:,:] = u_1[i,:,:] #-1.0
    i = u.shape[0] - 1
    # u[i,:,:] = -1
    u[i,0,0] = u_1[i,0,0] - Bxx*(u_1[i,0,0] - u_1[i-1,0,0]) - Byy*(u_1[i,0,0] - u_1[i,1,0]) - Bzz*(u_1[i,0,0] - u_1[i,0,1])
    u[i,0,1:] = u_1[i,0,1:] - Bxx*(u_1[i,0,1:] - u_1[i-1,0,1:]) - Byy*(u_1[i,0,1:] - u_1[i,1,1:]) - Bzz*(u_1[i,0,1:] - u_1[i,0,:-1])
    u[i,1:,0] = u_1[i,1:,0] - Bxx*(u_1[i,1:,0] - u_1[i-1,1:,0]) - Byy*(u_1[i,1:,0] - u_1[i,:-1,0]) - Bzz*(u_1[i,1:,0] - u_1[i,1:,1])
    u[i,1:,1:] = u_1[i,1:,1:] - Bxx*(u_1[i,1:,1:] - u_1[i-1,1:,1:]) - Byy*(u_1[i,1:,1:] - u_1[i,:-1,1:]) - Bzz*(u_1[i,1:,1:] - u_1[i,1:,:-1])
    
    i = 0
    # v[i,:,:] = -1
    if abs(v[v.shape[0]-1,:,:].max()+1) > tol_bc:
        v[i,:,:] = v[v.shape[0]-1,:,:]
    else:
        v[i,:,:] = v_1[i,:,:] #-1.0
    i = v.shape[0] - 1
    # v[i,:,:] = -1
    v[i,0,0] = v_1[i,0,0] - Bxx*(v_1[i,0,0] - v_1[i-1,0,0]) - Byy*(v_1[i,0,0] - v_1[i,1,0]) - Bzz*(v_1[i,0,0] - v_1[i,0,1])
    v[i,0,1:] = v_1[i,0,1:] - Bxx*(v_1[i,0,1:] - v_1[i-1,0,1:]) - Byy*(v_1[i,0,1:] - v_1[i,1,1:]) - Bzz*(v_1[i,0,1:] - v_1[i,0,:-1])
    v[i,1:,0] = v_1[i,1:,0] - Bxx*(v_1[i,1:,0] - v_1[i-1,1:,0]) - Byy*(v_1[i,1:,0] - v_1[i,:-1,0]) - Bzz*(v_1[i,1:,0] - v_1[i,1:,1])
    v[i,1:,1:] = v_1[i,1:,1:] - Bxx*(v_1[i,1:,1:] - v_1[i-1,1:,1:]) - Byy*(v_1[i,1:,1:] - v_1[i,:-1,1:]) - Bzz*(v_1[i,1:,1:] - v_1[i,1:,:-1])
    
    return u, v



def model(save_plot=True):
    """
    Defines initial field config, model parameters, grid parameters and calls solver function.
    plot_method=1 applies mesh function, =2 means surf, =0 means no plot.
    """
    # Grid and model parameters
    prefix, Lx, Ly, Lz, Nx, Ny, Nz, dt, ti, epsilon, T, _ = param()

    lam = 0.1; rho3 = 1.7; eta = 1; a0 = 1; m=eta*np.sqrt((rho3-(2*lam))/2)
    Bx = 1; By = 1; Bz = 1
    
    # Clean up plot files
    import glob, os
    for name in glob.glob('tmp_*.png'):
        os.remove(name)
    if glob.glob('field_l.h5'):
        os.remove('field_l.h5')
    if glob.glob('field_r.h5'):
        os.remove('field_r.h5')
    name = f'{prefix}_movie_{round(epsilon,3)}.mp4'
    if glob.glob(name):
        os.remove(name)

    def I(x, y, z):
        """Initial field configuration of u."""
        
        # random sized domains of correlation size, small randomness
        from random import randint
        freq = 2*np.pi*int(prefix)/Lx;
        freq_list = [2*np.pi*i/Lx for i in np.linspace(1e-3,prefix,1000)]
        phi = np.sin(freq*x+np.pi/2*randint(-10,10)/10)*np.sin(freq*y+np.pi/2*randint(-10,10)/10)*np.sin(freq*z+np.pi/2*randint(-10,10)/10)
        for fr in freq_list[:-1]:
            phi += (randint(-400,400)/500)*np.sin(fr*x+np.pi/2*randint(-10,10)/10)*np.sin(fr*y+np.pi/2*randint(-10,10)/10)*np.sin(fr*z+np.pi/2*randint(-10,10)/10)
        return eta*np.sin(np.arctan(np.exp(m*phi/(abs(phi).max()*freq)))), eta*np.cos(np.arctan(np.exp(m*phi/(abs(phi).max()*freq))))
        

        # correlation sized domains with 5 high frequency modes added
        # from random import randint
        # freq = 2*np.pi*int(prefix)/Lx;
        # freq_list = [2*np.pi*i*int(prefix)/Lx for i in np.linspace(2,5,4)]
        # phi = np.sin(freq*x+np.pi/2*randint(-10,10)/10)*np.sin(freq*y+np.pi/2*randint(-10,10)/10)*np.sin(freq*z+np.pi/2*randint(-10,10)/10)
        # for fr in freq_list[:-1]:
        #     phi += (randint(-5,5)/50)*np.sin(fr*x+np.pi/2*randint(-10,10)/10)*np.sin(fr*y+np.pi/2*randint(-10,10)/10)*np.sin(fr*z+np.pi/2*randint(-10,10)/10)
        # return eta*np.sin(np.arctan(np.exp(m*phi/(abs(phi).max()*freq)))), eta*np.cos(np.arctan(np.exp(m*phi/(abs(phi).max()*freq))))
        
        
        # random sized domains of correlation size, randomized highly
        # from random import randint
        # freq = 2*np.pi*int(prefix)/Lx;
        # freq_list = [2*np.pi*i/Lx for i in np.linspace(1e-3,prefix,1000)]
        # phi = np.sin(freq*x+np.pi/2*randint(-10,10)/10)*np.sin(freq*y+np.pi/2*randint(-10,10)/10)*np.sin(freq*z+np.pi/2*randint(-10,10)/10)
        # for fr in freq_list[:-1]:
        #     phi += (randint(-4,4)/50)*np.sin(fr*x+np.pi/2*randint(-10,10)/10)*np.sin(fr*y+np.pi/2*randint(-10,10)/10)*np.sin(fr*z+np.pi/2*randint(-10,10)/10)
        # return eta*np.sin(np.arctan(np.exp(m*phi/(abs(phi).max()*freq)))), eta*np.cos(np.arctan(np.exp(m*phi/(abs(phi).max()*freq))))
        
    
    
    def Vu(x, y, z):
        # return (y-x-z)*0
        
        # mesh of cuboids of fixed width
        # freq = 2*np.pi*int(prefix)/Lx; width = 1/np.sqrt(lam)
        # return 1e-5*np.tanh((0.1*np.sin(freq*x+np.pi/2)*np.sin(freq*y+np.pi/2)*np.sin(freq*z+np.pi/2)/(width*freq)+1e-5)**-1)
        
        # velocities according to random domains
        from random import randint
        freq = 2*np.pi*int(prefix)/Lx; width = 1/np.sqrt(lam)
        freq_list = [2*np.pi*i/Lx for i in range(prefix,prefix+6)]
        phi = np.sin(freq*x+np.pi/2*randint(-10,10)/10)*np.sin(freq*y+np.pi/2*randint(-10,10)/10)*np.sin(freq*z+np.pi/2*randint(-10,10)/10)
        for fr in freq_list[:-1]:
            phi += (randint(-100,100)/1200)*np.sin(fr*x+np.pi/2*randint(-10,10)/10)*np.sin(fr*y+np.pi/2*randint(-10,10)/10)*np.sin(fr*z+np.pi/2*randint(-10,10)/10) + 1e-10
        return 1e-15*np.tanh(phi/abs(phi).max()) # closer to zero faster the velocity
        
    def Vv(x, y, z):
        # return (y-x-z)*0
        
        # mesh of cuboids of fixed width
        # freq = 2*np.pi*int(prefix)/Lx; width = 1/np.sqrt(lam)
        # return 1e-5*np.tanh((0.1*np.sin(freq*x+np.pi/2)*np.sin(freq*y+np.pi/2)*np.sin(freq*z+np.pi/2)/(width*freq)+1e-5)**-1)
        
        # velocities according to random domains
        from random import randint
        freq = 2*np.pi*int(prefix)/Lx; width = 1/np.sqrt(lam)
        freq_list = [2*np.pi*i/Lx for i in range(prefix,prefix+6)]
        phi = np.sin(freq*x+np.pi/2*randint(-10,10)/10)*np.sin(freq*y+np.pi/2*randint(-10,10)/10)*np.sin(freq*z+np.pi/2*randint(-10,10)/10)
        for fr in freq_list[:-1]:
            phi += (randint(-100,100)/1200)*np.sin(fr*x+np.pi/2*randint(-10,10)/10)*np.sin(fr*y+np.pi/2*randint(-10,10)/10)*np.sin(fr*z+np.pi/2*randint(-10,10)/10) + 1e-10
        return 1e-15*np.tanh(phi/abs(phi).max()) # closer to zero faster the velocity


    def plot_u(u, v, x, xv, y, yv, z, zv, t, n):
            
        from mpl_toolkits import mplot3d
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12,6))
        #ax = plt.axes(projection='3d')
        ax1 = fig.add_subplot(1,2,1,projection='3d')

        u_surf = ax1.plot_surface(xv[:,:,0], yv[:,:,0], u[:,:,int(Nz/2)],cmap='viridis', edgecolor='none')
            
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel(f'$l(x,y,{round(Nz/2*(zv[0,0,1]-zv[0,0,0]),1)})$')
        ax1.set_zlim(-0.5, 1.5)
            
        ax2 = fig.add_subplot(1,2,2,projection='3d')
        v_surf = ax2.plot_surface(xv[:,:,0], yv[:,:,0], v[:,:,int(Nz/2)],cmap='viridis', edgecolor='none')
            
        title = f'n = {n}, t = {round(ti+n*dt,3)}, $\\tau$ = {round(2*np.sqrt(ti+n*dt),3)}, max = {round(u.max(),3),round(v.max(),3)}, min = {round(u.min(),3),round(v.min(),3)}, av = {round(u.mean(),3),round(v.mean(),3)}'
        fig.suptitle(title)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel(f'$r(x,y,{round(Nz/2*(zv[0,0,1]-zv[0,0,0]),1)})$')
        ax2.set_zlim(-0.5, 1.5)
            
        
        time.sleep(0) # pause between frames

        # uncomment following lines to save array into gz file
        #filename = 'field_%04d.gz' % n
        #u_reshaped = u.reshape(u.shape[0], -1)
        #np.savetxt(filename, u_reshaped) # most time consuming step
            
        # uncomment following lines to save h5 files
#         filename = 'field_%04d.h5' % n
#         def hdf5_write(data,name):
#             f = h5py.File(name, "w")
#             f.create_dataset('data', data=data)
#         hdf5_write(u,filename)



        # function to append data to h5py file

        def hdf5_write(data,name):
            f = h5py.File(name, "w")
            f.create_dataset(f'field_{n}', data=data)
            f.close()


        filename = 'field_l.h5'
        if glob.glob(filename):
            f = h5py.File(filename, "a")
            f[f'field_{n}'] = u
            f.close()
        else:
            hdf5_write(u,filename) 
            
        filename = 'field_r.h5'
        if glob.glob(filename):
            f = h5py.File(filename, "a")
            f[f'field_{n}'] = v
            f.close()
        else:
            hdf5_write(v,filename)           


        if save_plot:
            filename = 'tmp_%04d.png' % n
            plt.savefig(filename)  # time consuming!
       # ax1.collections.remove(u_surf)
       # ax2.collections.remove(v_surf)
        ax1.cla()
        ax2.cla()
        plt.close('all')
        plt.draw()
        time.sleep(1)

    dt, cpu = solver(prefix, I, Vu, Vv, lam, rho3, eta, a0, epsilon, Lx, Ly, Lz, Nx, Ny, Nz, dt, ti, T,
                     user_action=plot_u)
    print(f'Total time taken is {cpu/3600} hr')
    
    # Make video files
    fps = 29  # frames per second
    codec2ext = dict(libx264='mp4')
    filespec = 'tmp_%04d.png'
    movie_program = 'ffmpeg'  # or 'avconv'
    for codec in codec2ext:
        ext = codec2ext[codec]
        cmd = f'{movie_program} -r {fps} -i {filespec} -vcodec {codec} {prefix}_movie_{round(epsilon,3)}.{ext}'
        os.system(cmd)
        
    for name in glob.glob('tmp_*.png'):
        os.remove(name)


if __name__ == '__main__':
    model()
