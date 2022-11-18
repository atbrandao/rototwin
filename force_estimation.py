# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 17:10:18 2021

@author: HR7O
"""

import numpy as np
import scipy.linalg as la
import ross as rs
from numpy.random import randn , rand , seed
import copy
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense

def prep(M,K,C,G,w):
    '''
    Prepares the variables for further calculations by 
    calculating the System' state space matrices
    
    Returns:
        A , H , evecs , evals , Minv , M_2N
    '''
    
    N = len(M)

    Z = np.zeros((N,N))
    I = np.eye(N)
    
    Minv = np.vstack([np.hstack([Z,Z]),
                       np.hstack([Z,la.inv(M)])])
    M_2N = np.vstack([np.hstack([Z,Z]),
                       np.hstack([Z,M])])
    
    A = np.vstack( [np.hstack([Z,I]),
                    np.hstack([la.solve(-M,K),la.solve(-M,w*G+C)])] )
    
    
    evals , evecs = la.eig(A)
    i_sort = np.argsort(np.abs(np.imag(evals)))
    evals = evals[i_sort]
    evecs = evecs[:,i_sort]
    
    
    H = la.inv(1.j*w*np.eye(2*N) - A)
    
    return A , H , evecs , evals , Minv , M_2N




def forced_displacement(rotor,d,w,prep1=None):
    '''
    Estimates the force distribution and deflected shape of a rotor
    from the output measurements in d.
    
    Hypothesis:
    Considers that forces happen only at the measurement stations.
    
    Returns:
        x_full , F_est
    '''
    
    if prep1 == None:
        A,H,evecs,evals,Minv,M_2N = prep(rotor.M(),rotor.K(w),rotor.C(w),rotor.G(),w)
    else:
        A,H,evecs,evals,Minv,M_2N = prep1
    N = rotor.ndof
    
    dof = list(d.keys()) # list of nodes where the response is known
    dof2 = [a+N for a in dof]
    
    # Forced displacement
    x = np.array([d[k] for k in dof]).reshape((len(dof),1))
    H_inv = la.inv((H@Minv)[np.ix_(dof,dof2)])
    F = H_inv@x
    B = np.zeros((2*N,1)).astype(complex)
    B[np.ix_(dof2,[0])] = F
    B = Minv @ B
    
    x_full = (H @ B)[:N] # Vetor de resposta ao deslocamento imposto d
    
    F_est = rotor.M() @ B[N:]
    
    return x_full , F_est




# d = {i:x_full[i] for i in range(N)}
# dof = list(d.keys())
# dof2 = [a+N for a in dof]

# d = {0:10,
#       2:-1.j*10}
# dof = list(d.keys())
# dof2 = [a+N for a in dof]


# Modal reduction for unbalancance distribution
def modal_reduction(rotor,d,w,modes=None,prep1=None):
    
    '''
    Estimates the force distribution and deflected shape of a rotor
    from the output measurements in d.
    
    Hypothesis:
    Considers that only the modes given in 'modes' are excited.
    
    Returns:
        x_full , F_est
    '''
    
    if prep1 == None:
        A,H,evecs,evals,Minv,M_2N = prep(rotor.M(),rotor.K(w),rotor.C(w),rotor.G(),w)
    else:
        A,H,evecs,evals,Minv,M_2N = prep1
    N = rotor.ndof
    
    dof = list(d.keys())
    dof2 = [a+N for a in dof]

    A2 = la.inv(evecs) @ A @ evecs
    H2 = la.inv(1.j*w*np.eye(2*N) - A2)
    
    if type(modes) != list and type(modes) != np.ndarray:
        aux = [i for i in range(len(evals)) if np.abs(np.imag(evals[i])) > 0 and \
                  np.abs(np.real(evals[i])/np.imag(evals[i])) < 1]
            
        if type(modes) == int:
            modes = aux[:2*modes]
        else:           
            modes = aux[:2*len(dof)]
    
    
    x = np.vstack([np.array([d[k] for k in dof]).reshape((len(dof),1)),
                   np.array([d[k]*1.j*w for k in dof]).reshape((len(dof),1))])
    
    H_inv2 = la.pinv((evecs @ H2)[np.ix_(dof+dof2,modes)])
    B_red = H_inv2 @ x
    
    
    B2 = np.zeros((2*N,1)).astype(complex)
    B2[np.ix_(modes,[0])] = B_red
    
    B_est = np.zeros((2*N,1)).astype(complex)
    B_est[N:] = (evecs @ B2)[N:]
    
    F_est = rotor.M() @ B_est[N:]
    
    x_full = (H @ B_est)[:N]
    
    return x_full , F_est




def unbalance_distribution(rotor,d,w,unb_nodes,prep1=None):
    
    '''
    Estimates the force distribution and deflected shape of a rotor
    from the output measurements in d.
    
    Hypothesis:
    Considers that all forces are pure unbalance at
    stations given in 'unb_nodes'.
    
    Returns:
        x_full , F_est
    '''
    
    if prep1 == None:
        A,H,evecs,evals,Minv,M_2N = prep(rotor.M(),rotor.K(w),rotor.C(w),rotor.G(),w)
    else:
        A,H,evecs,evals,Minv,M_2N = prep1
    N = rotor.ndof
    
    dof = list(d.keys())
    
    dof2x = [4*a+N for a in unb_nodes]
    dof2y = [4*a+N+1 for a in unb_nodes]
    
    # Forced displacement
    x = np.array([d[k] for k in dof]).reshape((len(dof),1))
    H_inv = la.pinv((H@Minv)[np.ix_(dof,dof2x)]-1.j*(H@Minv)[np.ix_(dof,dof2y)])
    F = H_inv@x
    B = np.zeros((2*N,1)).astype(complex)
    B[np.ix_(dof2x,[0])] = F
    B[np.ix_(dof2y,[0])] = -F*1.j
    B = Minv @ B
    
    x_full = (H @ B)[:N] # Vetor de resposta ao deslocamento imposto d
    
    F_est = rotor.M() @ B[N:]
    
    return x_full , F_est




def xy_forces(rotor,d,w,nodes,prep1=None):
    
    '''
    Estimates the force distribution and deflected shape of a rotor
    from the output measurements in d.
    
    Hypothesis:
    Considers that only tranverse X and Y forces at stations
    given in 'nodes' are present.
    
    Returns:
        x_full , F_est
    '''
    
    if prep1 == None:
        A,H,evecs,evals,Minv,M_2N = prep(rotor.M(),rotor.K(w),rotor.C(w),rotor.G(),w)
    else:
        A,H,evecs,evals,Minv,M_2N = prep1
    N = rotor.ndof
    
    dof = list(d.keys()) # list of nodes where the response is known
    dof2x = [4*a+N for a in nodes]
    dof2y = [4*a+N+1 for a in nodes]
    
    # Forced displacement
    x = np.array([d[k] for k in dof]).reshape((len(dof),1)) 
    H_inv = la.pinv((H@Minv)[np.ix_(dof,dof2x+dof2y)])
    F = H_inv@x
    B = np.zeros((2*N,1)).astype(complex)
    B[np.ix_(dof2x+dof2y,[0])] = F
    B = Minv @ B
    
    x_full = (H @ B)[:N] # Vetor de resposta ao deslocamento imposto d
    
    F_est = rotor.M() @ B[N:]
    
    return x_full , F_est


    
def create_rotor(r,plot_rotor=False,seed_state=None):
    
    '''
    Creates a ross.Rotor object with reasonable characteritics of a typical
    multistage, between bearings machine.
    
    Returns:
        rotor , w
    '''
    
    if seed_state:
        seed(seed_state)
    
    rpm = (3600 + rand()*6000)
    w = rpm/60*2*np.pi
    
    L = rand()*2 + 1
    b_span = (0.65 + rand()*0.1)*L
    d_bearing = b_span/(10+rand()*8)
    D_shaft = d_bearing*1.5
    b1_pos = (L-b_span)*(0.4+rand()*0.2)
    
    D = np.linspace(d_bearing*0.8,d_bearing,int(b1_pos/d_bearing*0.7))
    D = np.append(D,np.array([d_bearing]*int(np.ceil(b1_pos/d_bearing*0.3))))
    i_b1 = len(D)
    D = np.append(D,np.linspace(d_bearing,D_shaft,int(b_span/d_bearing/3)))
    i_imp1 = len(D)
    D = np.append(D,np.array([D_shaft]*int(b_span/d_bearing/3)))
    D = np.append(D,np.linspace(D_shaft,d_bearing,int(b_span/d_bearing/3)))
    i_b2 = len(D)
    D = np.append(D,np.array([d_bearing]*int((L-b_span-b1_pos)/d_bearing*0.3)))
    D = np.append(D,np.linspace(d_bearing,d_bearing*0.8,int(L/d_bearing)-len(D)))
    
    shaft = [
        rs.ShaftElement(
            n=i,
            L=d_bearing,
            odl=D[i],
            idl=0,
            odr=D[i],
            idr=0,
            material=rs.materials.steel,
            rotary_inertia=True,
            gyroscopic=True,
        )
        for i in range(len(D))
    ]
    rotor = rs.Rotor(shaft)
    
    n_imp = int(2+rand()*(int(b_span/d_bearing/3)))
    
    d_imp = D_shaft*2
    L_imp = d_bearing/4
    m_imp = rotor.m*(0.05+0.05*rand())
    
    Ip = 1/8*m_imp*(d_imp**2+D_shaft**2)
    Id = 1/12*m_imp*(3/4*(d_imp**2+D_shaft**2)+L_imp**2)
    
    disks = [rs.DiskElement(i,m=m_imp,Ip=Ip,Id=Id) for i in range(i_imp1,i_imp1+n_imp)]
    disk_coupling = [rs.DiskElement(rotor.nodes[-1],m=m_imp/2,Ip=Ip/2,Id=Id/2)]
    disks = disks + disk_coupling
    
    ## Bearing and Seal coefficients for speed = 3566rpm
    K = rotor.K(0)
    K[0,0] += 1e20
    K[-4,-4] += 1e20
    try:
        kxx = 1/la.inv(K)[i_b1*4,i_b1*4]
    except:
        kxx = 1e6
        
    if rotor.m < 200:
        kyy = kxx
    else:
        kyy = kxx * np.sqrt(rotor.m/200)
    kxy = kxx*0.1
    kyx = -kxy
    cxx = kxx*1e-3
    cyy = kyy*1e-3
    
    brg1 = rs.BearingElement(kxx=kxx,kxy=kxy,kyx=kyx,kyy=kyy,
                                 cxx=cxx,cxy=0,cyx=0,cyy=cyy,frequency=[w],n=i_b1)    
    brg2 = rs.BearingElement(kxx=kxx,kxy=kxy,kyx=kyx,kyy=kyy,
                                 cxx=cxx,cxy=0,cyx=0,cyy=cyy,frequency=[w],n=i_b2)

    rotor = rs.Rotor(shaft,disks,[brg1,brg2])
    
    
    
    modal = rotor.run_modal(w)
    
    j = 0
    while any(np.abs(modal.wd/w-1) < 0.15) and j < 50:
        rpm = (3600 + rand()*6000)
        w = rpm/60*2*np.pi
        modal = rotor.run_modal(w)
        j += 1
        
    if j == 20:
        print(f'Rotor number {r} has low Separation Margin. SM = {min(np.abs(modal.wd/w-1))*100:.1f} %.')
    
    
    return rotor , w

def create_real_rotor(rotor,uncertainty=0.2,seed_state=None):
    
    '''
    Creates a rotor with an given uncertainty added at the
    bearing elements coefficients.
    
    Returns:
        real_rotor
    '''
    
    if seed_state:
        seed(seed_state)
    
    real_rotor = copy.deepcopy(rotor)    
    
    for i, b in enumerate(rotor.bearing_elements):
        if b.frequency is None:
            b.frequency = [0]
        
        b_real = rs.BearingElement(kxx=(1+randn()*uncertainty)*np.array([a for a in b.kxx]),
                                   kxy=(1+randn()*uncertainty)*np.array([a for a in b.kxy]),
                                   kyx=(1+randn()*uncertainty)*np.array([a for a in b.kyx]),
                                   kyy=(1+randn()*uncertainty)*np.array([a for a in b.kyy]),
                                   cxx=(1+randn()*uncertainty)*np.array([a for a in b.cxx]),
                                   cxy=(1+randn()*uncertainty)*np.array([a for a in b.cxy]),
                                   cyx=(1+randn()*uncertainty)*np.array([a for a in b.cyx]),
                                   cyy=(1+randn()*uncertainty)*np.array([a for a in b.cyy]),
                                   frequency=(1+randn()*uncertainty)*np.array([a for a in b.frequency]),
                                   n=b.n)
        real_rotor.bearing_elements[i] = b_real
        
    
    return real_rotor


def calc_rotor(rotor,real_rotor,w,r=1,plot_rotor=False,
               full_output=False,prep1=None,seed_state=None):
    
    '''
    Applies a random distributed load to the 'real_rotor' model and tries
    to estimate the deflected shape and force distribution using the 'rotor'
    model and all four estimators.
    
    It also calculates the normalized correlation matrix U.
    
    The variables inp and out gather some selected elements from the
    matrix U and may be used to train ML models.
    
    Returns:
        inp , out , inp2 , out2
        
    if full_output:
        Returns:
            inp , out , inp2 , out2 , U , U2 , x_or , F , x1 , F1 , x2 , F2 , x3 , F3 , x4 , F4
    '''

    if seed_state:
        seed(seed_state)  
        
    M = rotor.M()
    
    N = rotor.ndof
    
    
    TM = la.inv(1.j*w*np.eye(real_rotor.ndof*2) - real_rotor.A(w)) # real_rotor.transfer_matrix(w)
    Z = np.zeros((real_rotor.ndof,real_rotor.ndof))
    Minv_real = np.vstack([np.hstack([Z,Z]),
                       np.hstack([Z,la.inv(real_rotor.M())])])
    
    
    i_imp1 = int(len(rotor.nodes)/2)
    
    F = 1e-15*np.ones((N,1)).astype(complex)
    # distributed randomized unbalance
    for i in rotor.nodes:
        F[i*4] = w**2*M[4*i,4*i]*(1e-3*(2*rand()+2.5)/w) * np.exp(1.j*rand()*2*np.pi)
        F[i*4+1] = -F[i*4]*1.j
    
    # Right end concentrated unbalance
    if rand() < 0.5:
        F[rotor.nodes[-1]*4] = max(F)*4 * np.exp(1.j*2*np.pi*rand())
        F[rotor.nodes[-1]*4+1] = -F[rotor.nodes[-1]*4]*1.j        
    
    # Left end concentrated unbalance
    if rand() < 0.5:
        F[0] = max(F)*4 * np.exp(1.j*2*np.pi*rand())
        F[0] = -F[0]*1.j           
    
    # Unbalance and unidirection forces at disks
    for dk in rotor.disk_elements:
        if rand() < 0.2:
            F[dk.n*4] = max(F)*4 * np.exp(1.j*2*np.pi*rand())
        if rand() < 0.2:
            F[dk.n*4+1] = max(F)*4 * np.exp(1.j*2*np.pi*rand())
        if rand() < 0.2:
            F[dk.n*4] = w**2*M[4*dk.n,4*dk.n]*(1e-3*(6/w)) * np.exp(1.j*rand()*2*np.pi)
            F[dk.n*4+1] = -F[dk.n*4]*1.j
        
        
                
    F2 = np.vstack([np.zeros((N,1)).astype(complex),
                   F])
    x_or = TM @ (Minv_real @ F2)
    x_or = x_or[:real_rotor.ndof]

    i_b1 = rotor.bearing_elements[0].n
    i_b2 = [be.n for be in rotor.bearing_elements if be.n < rotor.nodes[-1]][-1]
    probe_lna = (i_b1-1,0)
    probe_la = (i_b2+1,0)
    
    d = {probe_lna[0]*4+0:x_or[probe_lna[0]*4+0],
          probe_lna[0]*4+1:x_or[probe_lna[0]*4+1],
          probe_la[0]*4+0:x_or[probe_la[0]*4+0],
          probe_la[0]*4+1:x_or[probe_la[0]*4+1],}
   
    if prep1 == None:
        prep1 = prep(rotor.M(),rotor.K(w),rotor.C(w),rotor.G(),w)
    
    x1, F1 = forced_displacement(rotor,d,w,prep1)
    
    x2, F2 = modal_reduction(rotor,d,w,16,prep1)    
    
    x3, F3 = unbalance_distribution(rotor,d,w,rotor.nodes,prep1)  
    
    x4, F4 = xy_forces(rotor,d,w,[0]+[a.n for a in rotor.disk_elements]+[rotor.nodes[-1]],prep1)
    

    X = np.hstack([x_or,x1,x2,x3,x4])
    X = X.astype(np.clongdouble)
    
    U = X.transpose() @ X
    U = la.inv(np.diag(U)*np.eye(len(U))) @ U
    
    if plot_rotor: 
        rotor.plot_rotor().write_image(f'plot_rotor/rotor_{r}.png')
        
    inp = np.array( [U[1,2],U[1,3],U[1,4],U[2,3],U[2,4],U[3,4]] )
    out = np.array( [U[0,1],U[0,2],U[0,3],U[0,4]] )
    
    X2 = np.hstack([F,F1,F2,F3,F4])
    X2 = X2.astype(np.clongdouble)
    
    U2 = X2.transpose() @ X2
    U2[U2 == 0] = 1e15  # Evitar o erro de criar um zewro na diagonal de U2 por um número muito grande
    try:
        U2 = la.inv(np.diag(U2)*np.eye(len(U2))) @ U2
    except:
        print('Singular matrix.')
        # print(U2)
    
    if plot_rotor: 
        rotor.plot_rotor().write_image(f'plot_rotor/rotor_{r}.png')
        
    inp2 = np.array( [U2[1,2],U2[1,3],U2[1,4],U2[2,3],U2[2,4],U2[3,4]] )
    out2 = np.array( [U2[0,1],U2[0,2],U2[0,3],U2[0,4]] )
    
   
    if full_output:
        return inp , out , inp2 , out2 , U , U2 , x_or , F , x1 , F1 , x2 , F2 , x3 , F3 , x4 , F4
    else:
        return inp , out , inp2 , out2
    
    


def calc_d(probe_vib,probe_runout=None):
    
    '''
    Used to calculate the X and Y complex vibration components from arbitrarily
    oriented vibration probe measurements given by 'probe_vib'. The output d
    can be directly used by the estimators.
    
    probe_vib must be a dict object in the following form:
    
    probe_vib = {(station , orientation [deg]) : (vibration [um pk-pk] , phase [deg]),
                 (station , orientation [deg]) : (vibration [um pk-pk] , phase [deg]),}
    
    Returns:
        d
    '''

    d = {}
    for p in probe_vib.keys():
        
        v = probe_vib[p]
        if p[0]*4 not in d:
            d[p[0]*4] = 0
            d[p[0]*4+1] = 0
            
        d[p[0]*4] += v[0]/2e6 * np.exp(1.j*v[1]*np.pi/180) * np.cos(p[1]*np.pi/180)
        d[p[0]*4+1] += v[0]/2e6 * np.exp(1.j*v[1]*np.pi/180) * np.sin(p[1]*np.pi/180)
    
    if probe_runout!= None:
        for p in probe_runout.keys():
            
            ro = probe_runout[p]
                
            d[p[0]*4] -= ro[0]/2e6 * np.exp(1.j*ro[1]*np.pi/180) * np.cos(p[1]*np.pi/180)
            d[p[0]*4+1] -= ro[0]/2e6 * np.exp(1.j*ro[1]*np.pi/180) * np.sin(p[1]*np.pi/180)
    
    return d
    


def plot_deflected_shape(rotor,rpm,probes,vibration,runout=None,
                         model_best=None,model_value=None,scaler_x=None,scaler_y=None,nodes_xy=[]):
    """Plot estimated deflected shape based on measured vibrations

    Parameters
    ----------
    rotor: ross.Rotor
        Rotor model.
        
    rpm: float
        Rotor speed in rpm.
        
    probes: list of tuples
        list containing all measured point in the form of (node,angle)
        
    vibration: list of tuples
        list containing all measures from the above mentioend probes.
        tuples must be like (um pk-pk amplitude,phase angle)
        
    model_best: keras model 
        trained keras model that defines the best deflected shape estimator
        
    model_value: keras model 
        trained keras model that defines the error estimation for each estimator
        
    scaler: StandardScaler
        StardardScaler fitted to the input data

    Returns
    -------
    plotly Figure object
        
    """
    
    w = rpm*2*np.pi/60
    
    probe_vib = {}
    
    for i,p in enumerate(probes):
        probe_vib[p] = vibration[i]
        
    if runout != None:
        probe_runout = {}
        for i,p in enumerate(probes):
            probe_runout[p] = runout[i]
    else: 
        probe_runout = None
    
    d = calc_d(probe_vib,probe_runout)
    
    prep1 = prep(rotor.M(),rotor.K(w),rotor.C(w),rotor.G(),w)
    
    x1, F1 = forced_displacement(rotor,d,w,prep1)

    x2, F2 = modal_reduction(rotor,d,w,16,prep1)
    
    x3, F3 = unbalance_distribution(rotor,d,w,rotor.nodes,prep1)  
    
    x4, F4 = xy_forces(rotor,d,w,[0]+[a.n for a in rotor.disk_elements]+[rotor.nodes[-1]]+nodes_xy,prep1)
    
    N = len(x1)
    l = rotor.nodes_pos
    t = np.arange(0,2*np.pi*0.95,np.pi/15)
    data_orbits = []
    for i, p in enumerate(d.keys()):
        if i == 0:
            sl = True
        else:
            sl = False
        if p%4 == 0:
           data_orbits.append(go.Scatter3d(x=[rotor.nodes_pos[p//4]]*len(t),
                                           z=np.abs(d[p+1])*np.cos(t+np.angle(d[p+1])),
                                           y=np.abs(d[p])*np.cos(t+np.angle(d[p])),
                                           mode = 'lines',
                                           showlegend=sl,
                                           line={'color':'black',
                                                 'width':2,},
                                           name='Probe Orbits',
                                           legendgroup='Orbits'))
           data_orbits.append(go.Scatter3d(x=[rotor.nodes_pos[p//4]],
                                           z=[np.abs(d[p+1])*np.cos(t+np.angle(d[p+1]))[0]],
                                           y=[np.abs(d[p])*np.cos(t+np.angle(d[p]))[0]],
                                           mode = 'markers',
                                           showlegend=False,
                                           marker={'color':'black',
                                                 'size':2,},
                                           name='Probe Orbits',
                                           legendgroup='Orbits'))
            
    
    
    fig = go.Figure(data=[go.Scatter3d(x=l,z=np.real(x1[1::4]).reshape((N//4)),y=np.real(x1[::4]).reshape((N//4)),mode='lines',line={'width':5},name='Forced Displacement'),
                          go.Scatter3d(x=l,z=np.real(x2[1::4]).reshape((N//4)),y=np.real(x2[::4]).reshape((N//4)),mode='lines',line={'width':5},name='Modal Reduction'),
                          go.Scatter3d(x=l,z=np.real(x3[1::4]).reshape((N//4)),y=np.real(x3[::4]).reshape((N//4)),mode='lines',line={'width':5},name='Unbalance distribution'),
                          go.Scatter3d(x=l,z=np.real(x4[1::4]).reshape((N//4)),y=np.real(x4[::4]).reshape((N//4)),mode='lines',line={'width':5},name='X-Y Forces'),
                          go.Scatter3d(x=l,z=[0]*len(l),y=[0]*len(l),name='Linha neutra',showlegend=False,mode='lines',line={'width':1,'color':'black','dash':'dash'}),
                          ]+data_orbits)
    max_x = max((np.max(np.abs(x1)),np.max(np.abs(x2)),np.max(np.abs(x3)),np.max(np.abs(x4))))
    fig.update_layout(scene=dict(yaxis={'range':[2*-max_x,2*max_x],
                                        'title':'X [m]'},
                                 zaxis={'range':[2*-max_x,2*max_x],
                                        'title':'Y [m]'},
                                 xaxis={'title':'Axial position [m]'},))
    

    if scaler_x != None:
        X = np.hstack([x1,x2,x3,x4])
        
        U = X.transpose() @ X
        U = la.inv(np.diag(U)*np.eye(len(U))) @ U
        
        inp = np.array( [U[0,1],U[0,2],U[0,3],U[1,2],U[1,3],U[2,3]] )
        
        i00 = np.abs(inp-1)
        i01 = np.angle(inp-1)
        X = np.append(i00,np.append(i01,[w])).reshape((1,13))
        
        best_est = (np.argmax(model_best.predict(scaler_x.transform(X))))
        if scaler_y == None:
            erro_est = np.round(100*(model_value.predict(scaler_x.transform(X))))
        else:
            erro_est = np.round(100*scaler_y.inverse_transform(model_value.predict(scaler_x.transform(X))))
            
        erro = []
        for e in erro_est[0]:
            if e < 0:
                erro.append('< 5%')
            elif e > 1000:
                erro.append('> 1000%')
            else:
                erro.append(f'{e:.1f}%')
        max_x0 = ((np.max(np.abs(x1[::4])),np.max(np.abs(x2[::4])),np.max(np.abs(x3[::4])),np.max(np.abs(x4[::4]))))[best_est]
        max_x1 = ((np.max(np.abs(x1[1::4])),np.max(np.abs(x2[1::4])),np.max(np.abs(x3[1::4])),np.max(np.abs(x4[1::4]))))[best_est]
        max_x = max([max_x0,max_x1])
        
        fig.update_layout(title_text=f'Best estimator: {fig.data[best_est].name} - {erro}',
                          scene=dict(yaxis={'range':[2*-max_x,2*max_x],
                                            'title':'X [m]'},
                                     zaxis={'range':[2*-max_x,2*max_x],
                                            'title':'Y [m]'},
                                     xaxis={'title':'Axial position [m]'},))
        
    return fig

def plot_forces(rotor,rpm,probes,vibration,runout=None,
                model_best=None,scaler=None,nodes_xy=[]):
    """Plot estimated deflected shape based on measured vibrations

    Parameters
    ----------
    rotor: ross.Rotor
        Rotor model.
        
    rpm: float
        Rotor speed in rpm.
        
    probes: list of tuples
        list containing all measured point in the form of (node,angle)
        
    vibration: list of tuples
        list containing all measures from the above mentioend probes.
        tuples must be like (um pk-pk amplitude,phase angle)
        
    model_best: keras model 
        trained keras model that defines the best force distribution estimator
        
    scaler: StandardScaler
        StardardScaler fitted to the input data

    Returns
    -------
    plotly Figure object
        
    """
    
    w = rpm*2*np.pi/60
    
    probe_vib = {}
    
    for i,p in enumerate(probes):
        probe_vib[p] = vibration[i]
        
    if runout != None:
        probe_runout = {}
        for i,p in enumerate(probes):
            probe_runout[p] = runout[i]
    else: 
        probe_runout = None
    
    d = calc_d(probe_vib,probe_runout)
    
    prep1 = prep(rotor.M(),rotor.K(w),rotor.C(w),rotor.G(),w)
    
    x1, F1 = forced_displacement(rotor,d,w,prep1)

    x2, F2 = modal_reduction(rotor,d,w,16,prep1)
    
    x3, F3 = unbalance_distribution(rotor,d,w,rotor.nodes,prep1)  
    
    x4, F4 = xy_forces(rotor,d,w,[0]+[a.n for a in rotor.disk_elements]+[rotor.nodes[-1]]+nodes_xy,prep1)
    
    N = len(x1)
    l = rotor.nodes_pos
        
    t = np.arange(0,2*np.pi*0.95,np.pi/15)
    fig = go.Figure()
    max_amp = max((np.max(np.abs(F1)),np.max(np.abs(F2)),np.max(np.abs(F3)),np.max(np.abs(F4))))
    
    
    sl1 = True
    sl2 = True
    sl3 = True
    sl4 = True
    
    for i, a in enumerate(l):        
            
        if np.abs(F1[4*i+1]) > 0.05*np.max(np.abs(F1)) or np.abs(F1[4*i]) > 0.05*np.max(np.abs(F1)):
            
            fig.add_trace(go.Scatter3d(x=[a]*len(t),
                                       z=np.abs(F1[4*i+1])*np.cos(t+np.angle(F1[4*i+1])),
                                       y=np.abs(F1[4*i])*np.cos(t+np.angle(F1[4*i])),
                                       mode = 'lines',
                                       showlegend=sl1,
                                       line={'color':'blue'},
                                       name='Forced Displacement',
                                       legendgroup='Forced Displacement'),)
            fig.add_trace(go.Scatter3d(x=[a],
                                       z=np.abs(F1[4*i+1])*np.cos(np.angle(F1[4*i+1])),
                                       y=np.abs(F1[4*i])*np.cos(np.angle(F1[4*i])),
                                       mode = 'markers',
                                       showlegend=False,
                                       marker={'color':'blue','size':2},
                                       name='Forced Displacement',
                                       legendgroup='Forced Displacement'),)
            sl1 = False
            
        if np.abs(F2[4*i+1]) > 0.05*np.max(np.abs(F2)) or np.abs(F2[4*i]) > 0.05*np.max(np.abs(F2)):
            fig.add_trace(go.Scatter3d(x=[a]*len(t),
                                       z=np.abs(F2[4*i+1])*np.cos(t+np.angle(F2[4*i+1])),
                                       y=np.abs(F2[4*i])*np.cos(t+np.angle(F2[4*i])),
                                       mode = 'lines',
                                       showlegend=sl2,
                                       line={'color':'red'},
                                       name='Modal Reduction',
                                       legendgroup='Modal Reduction'),)
            fig.add_trace(go.Scatter3d(x=[a],
                                       z=np.abs(F2[4*i+1])*np.cos(np.angle(F2[4*i+1])),
                                       y=np.abs(F2[4*i])*np.cos(np.angle(F2[4*i])),
                                       mode = 'markers',
                                       showlegend=False,
                                       marker={'color':'red','size':2},
                                       name='Modal Reduction',
                                       legendgroup='Modal Reduction'),)
            sl2 = False
            
        if np.abs(F3[4*i+1]) > 0.05*np.max(np.abs(F3)) or np.abs(F3[4*i]) > 0.05*np.max(np.abs(F3)):
            fig.add_trace(go.Scatter3d(x=[a]*len(t),
                                       z=np.abs(F3[4*i+1])*np.cos(t+np.angle(F3[4*i+1])),
                                       y=np.abs(F3[4*i])*np.cos(t+np.angle(F3[4*i])),
                                       mode = 'lines',
                                       showlegend=sl3,
                                       line={'color':'green'},
                                       name='Unbalance Distribution',
                                       legendgroup='Unbalance Distribution'),)
            fig.add_trace(go.Scatter3d(x=[a],
                                       z=np.abs(F3[4*i+1])*np.cos(np.angle(F3[4*i+1])),
                                       y=np.abs(F3[4*i])*np.cos(np.angle(F3[4*i])),
                                       mode = 'markers',
                                       showlegend=False,
                                       marker={'color':'green','size':2},
                                       name='Unbalance Distribution',
                                       legendgroup='Unbalance Distribution'),)
            sl3 = False
            
        if np.abs(F4[4*i+1]) > 0.05*np.max(np.abs(F4)) or np.abs(F4[4*i]) > 0.05*np.max(np.abs(F4)):
            fig.add_trace(go.Scatter3d(x=[a]*len(t),
                                       z=np.abs(F4[4*i+1])*np.cos(t+np.angle(F4[4*i+1])),
                                       y=np.abs(F4[4*i])*np.cos(t+np.angle(F4[4*i])),
                                       mode = 'lines',
                                       showlegend=sl4,
                                       line={'color':'orange'},
                                       name='X-Y Forces',
                                       legendgroup='X-Y Forces'),)
            fig.add_trace(go.Scatter3d(x=[a],
                                       z=np.abs(F4[4*i+1])*np.cos(np.angle(F4[4*i+1])),
                                       y=np.abs(F4[4*i])*np.cos(np.angle(F4[4*i])),
                                       mode = 'markers',
                                       showlegend=False,
                                       marker={'color':'orange','size':2},
                                       name='X-Y Forces',
                                       legendgroup='X-Y Forces'),)
            sl4 = False
        
        
                     
    
    fig.update_layout(scene=dict(yaxis={'range':[2*-max_amp,2*max_amp],
                                        'title':'X [N]'},
                                 zaxis={'range':[2*-max_amp,2*max_amp],
                                        'title':'Y [N]'},
                                 xaxis={'title':'Axial position [m]'},))
    

    if scaler != None:
        X = np.hstack([F1,F2,F3,F4])
        
        U = X.transpose() @ X
        U = la.inv(np.diag(U)*np.eye(len(U))) @ U
        
        inp = np.array( [U[0,1],U[0,2],U[0,3],U[1,2],U[1,3],U[2,3]] )
        
        i00 = np.abs(inp-1)
        i01 = np.angle(inp-1)
        X = np.append(i00,np.append(i01,[w])).reshape((1,13))
        
        best_est = (np.argmax(model_best.predict(scaler.transform(X))))
        
        max_F = ((np.max(np.abs(F1)),np.max(np.abs(F2)),np.max(np.abs(F3)),np.max(np.abs(F4))))[best_est]
        
        fig.update_layout(title_text=f'Best estimator: {fig.data[best_est].name}',
                          scene=dict(yaxis={'range':[2*-max_F,2*max_F],
                                            'title':'X [N]'},
                                     zaxis={'range':[2*-max_F,2*max_F],
                                            'title':'Y [N]'},
                                     xaxis={'title':'Axial position [m]'},))
        
        # fig.update_layout(title_text=f'Best estimator: {fig.data[best_est].name}',)
        
    return fig
    
    
    
    
def run_rotor_batch(rotor,N_runs,rpm_min,rpm_max=None):    
    '''
    Runs N_runs of estimation. Each run considers a randomly created real_rotor
    created usin the 'create_real_rotor' function and randomly distributed 
    loads using the 'calc_rotor' function.
    
    The variables inp and out are returned to be used in ML models.
    
    Returns:
        inp , out , inp2 , out2 , w_arr
    
    '''
    
    # N_runs = 1000
    inp = []
    out = []
    inp2 = []
    out2 = []
    w_list = []
    
    if rpm_max == None:
        w = 2*np.pi/60 * rpm_min
        prep1 = prep(rotor.M(),rotor.K(w),rotor.C(w),rotor.G(),w)
        
    
    
    for r in range(N_runs):
        if int(r/N_runs*100) > int((r-1)/N_runs*100):
            
            print(f'{int(r/N_runs*100)} %')
        
        if rpm_max != None:
            w = 2*np.pi/60 * (rpm_min + rand()*(rpm_max-rpm_min))
            prep1 = prep(rotor.M(),rotor.K(w),rotor.C(w),rotor.G(),w)
        
        real_rotor = create_real_rotor(rotor)
                
        try:
            a, b, c, d = calc_rotor(rotor,real_rotor,w,r,prep1=prep1)
            w_list.append(w)
            inp.append(a)
            out.append(b)
            inp2.append(c)
            out2.append(d)
        except:
            aux = None
        
       
    
    inp = np.array(inp)
    inp2 = np.array(inp2)
    out = np.array(out)
    out2 = np.array(out2)
    w_arr = np.array(w_list)
    
    return inp , out , inp2 , out2 , w_arr
    
    
def train_model_best(inp,out,w_arr,
                    model_name=None,scale=True,test_size=0.2,ep=150):
    '''
    Trains a ANN model to find the minimum value of 'out' given
    the variables 'inp' and 'w_arr'.
    
    Returns:
        NN_model, scaler_X
    
    '''
    
    encoder = LabelEncoder()
    encoder.fit(np.argmin(np.abs(out-1),1))
    encoded_y = encoder.transform(np.argmin(np.abs(out-1),1))
    
    y = np_utils.to_categorical(encoded_y)    
    
    X = np.hstack([np.abs(inp-1),np.angle(inp-1),w_arr.reshape((len(w_arr),1))])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    
    scaler_X = preprocessing.StandardScaler().fit(X_train)
    
    if scale:
        X_scaled = scaler_X.transform(X_train)
        X_test = scaler_X.transform(X_test)        
        
    else:
        X_scaled = X_train
        X_test = X_test
    
    NN_model = Sequential()
    
    NN_model.add(Dense(256,input_dim = len(X[0,:]),activation='relu'))
    NN_model.add(Dense(128,activation='relu'))
    NN_model.add(Dense(64,activation='relu'))
    NN_model.add(Dense(32,activation='relu'))
    NN_model.add(Dense(16,activation='relu'))
    NN_model.add(Dense(8,activation='relu'))
    NN_model.add(Dense(len(y[0,:]),activation='softmax'))
                 
    NN_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy') # Categorical
    NN_model.summary()
    
    log = NN_model.fit(X_scaled, y_train, epochs=ep, batch_size=16)
    
    score = NN_model.evaluate(X_test, y_test, verbose = 0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    
    if model_name != None:
        NN_model.save(model_name)
    
    return NN_model, scaler_X

def train_model_value(inp,out,w_arr,
                    model_name=None,scale=True,test_size=0.2,ep=150):
    '''
    Trains a ANN model to estimate the absolute values of 'out' given
    the variables 'inp' and 'w_arr'.
    
    Returns:
        NN_model, scaler_X, scaler_y
    
    '''
    
    y = np.abs(out-1)
    
    X = np.hstack([np.abs(inp-1),np.angle(inp-1),w_arr.reshape((len(w_arr),1))])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    
    scaler_X = preprocessing.StandardScaler().fit(X_train)
    scaler_y = preprocessing.StandardScaler().fit(y_train)
    
    scale = True
    if scale:
        X_scaled = scaler_X.transform(X_train)
        X_test = scaler_X.transform(X_test)
        
        y_train = scaler_y.transform(y_train)
        y_test = scaler_y.transform(y_test)
        
    else:
        X_scaled = X_train
        X_test = X_test
    
    NN_model = Sequential()
    
    NN_model.add(Dense(256,input_dim = len(X[0,:]),activation='relu'))
    NN_model.add(Dense(128,activation='relu'))
    NN_model.add(Dense(64,activation='relu'))
    NN_model.add(Dense(32,activation='relu'))
    NN_model.add(Dense(16,activation='relu'))
    NN_model.add(Dense(8,activation='relu'))
    NN_model.add(Dense(len(y[0,:])))
                 
    NN_model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics='accuracy') # Regression
    NN_model.summary()
    
    log = NN_model.fit(X_scaled, y_train, epochs=ep, batch_size=16)
    
    score = NN_model.evaluate(X_test, y_test, verbose = 0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    
    if model_name != None:
        NN_model.save(model_name)
    
    return NN_model, scaler_X, scaler_y
        

def test_estimators(rotor,w,seed_state=None,uncertainty=0.2):
    '''
    Calculates the deflected shape of the rotor using the four different 
    estimators and returns the plotly.Figure object with deflected 
    shapes and force distribution.
    
    Returns:
        fig_d , fig_F
    
    '''
    
    real_rotor = create_real_rotor(rotor,uncertainty=uncertainty,seed_state=seed_state)
    aux = calc_rotor(rotor,real_rotor,w,full_output=True,seed_state=seed_state)
    inp , out , inp2 , out2 , U , U2 , x_or , F , x1 , F1 , x2 , F2 , x3 , F3 , x4 , F4 = aux
        
    N = len(x_or) 
    l = np.array(rotor.nodes_pos)
    
    i_b1 = rotor.bearing_elements[0].n
    i_b2 = rotor.bearing_elements[-1].n
    probes = [i_b1-1,i_b2+1]
    
    t = np.arange(0,2*np.pi*0.95,np.pi/15)
    data_orbits = []
    for i, p in enumerate(probes):
        if i == 0:
            sl = True
        else:
            sl = False
       
        data_orbits.append(go.Scatter3d(x=[rotor.nodes_pos[p]]*len(t),
                                        z=np.abs(x_or[4*p+1])*np.cos(t+np.angle(x_or[4*p+1])),
                                        y=np.abs(x_or[4*p])*np.cos(t+np.angle(x_or[4*p])),
                                        mode = 'lines',
                                        showlegend=sl,
                                        line={'color':'black',
                                              'width':2,},
                                        name='Probe Orbits',
                                        legendgroup='Orbits'))
    
    fig_d = go.Figure(data=[go.Scatter3d(x=l,z=np.real(x_or[1::4]).reshape((N//4)),y=np.real(x_or[::4]).reshape((N//4)),mode='lines',line={'width':5},name='Original'),
                          go.Scatter3d(x=l,z=np.real(x1[1::4]).reshape((N//4)),y=np.real(x1[::4]).reshape((N//4)),mode='lines',line={'width':5},name='Forced Displacement'),
                          go.Scatter3d(x=l,z=np.real(x2[1::4]).reshape((N//4)),y=np.real(x2[::4]).reshape((N//4)),mode='lines',line={'width':5},name='Modal Reduction'),
                           go.Scatter3d(x=l,z=np.real(x3[1::4]).reshape((N//4)),y=np.real(x3[::4]).reshape((N//4)),mode='lines',line={'width':5},name='Unbalance distribution'),
                          go.Scatter3d(x=l,z=np.real(x4[1::4]).reshape((N//4)),y=np.real(x4[::4]).reshape((N//4)),mode='lines',line={'width':5},name='X-Y Forces'),
                          go.Scatter3d(x=l,z=0*l,y=0*l,name='Linha neutra',showlegend=False,mode='lines',line={'width':1,'color':'black','dash':'dash'}),
                          ]+data_orbits)
    max_x = (np.max(np.abs(x1)),np.max(np.abs(x2)),np.max(np.abs(x3)),np.max(np.abs(x4)))[np.argmin(np.abs(out))]
    fig_d.update_layout(title_text=f'Erros: {[str(a)+"%" for a in np.round(np.abs(out-1)*100)]}',
                      scene=dict(yaxis={'range':[2*-max_x,2*max_x],
                                            'title':'X [m]'},
                                     zaxis={'range':[2*-max_x,2*max_x],
                                            'title':'Y [m]'},
                                     xaxis={'title':'Axial position [m]'},))
    
    
    fig_F = go.Figure()
    max_amp = ((np.max(np.abs(F1)),np.max(np.abs(F2)),np.max(np.abs(F3)),np.max(np.abs(F4))))[np.argmin(np.abs(out2))]
    
    sl0 = True
    sl1 = True
    sl2 = True
    sl3 = True
    sl4 = True
    
    for i, a in enumerate(l):   
        
        if np.abs(F[4*i+1]) > 0.05*np.max(np.abs(F)) or np.abs(F[4*i]) > 0.05*np.max(np.abs(F)):
            
            fig_F.add_trace(go.Scatter3d(x=[a]*len(t),
                                       z=np.abs(F[4*i+1])*np.cos(t+np.angle(F[4*i+1])),
                                       y=np.abs(F[4*i])*np.cos(t+np.angle(F[4*i])),
                                       mode = 'lines',
                                       showlegend=sl0,
                                       line={'color':'black'},
                                       name='Original',
                                       legendgroup='Original'),)
            fig_F.add_trace(go.Scatter3d(x=[a],
                                       z=np.abs(F[4*i+1])*np.cos(np.angle(F[4*i+1])),
                                       y=np.abs(F[4*i])*np.cos(np.angle(F[4*i])),
                                       mode = 'markers',
                                       showlegend=False,
                                       marker={'color':'black','size':2},
                                       name='Original',
                                       legendgroup='Original'),)
            sl0 = False
            
        if np.abs(F1[4*i+1]) > 0.05*np.max(np.abs(F1)) or np.abs(F1[4*i]) > 0.05*np.max(np.abs(F1)):
            
            fig_F.add_trace(go.Scatter3d(x=[a]*len(t),
                                       z=np.abs(F1[4*i+1])*np.cos(t+np.angle(F1[4*i+1])),
                                       y=np.abs(F1[4*i])*np.cos(t+np.angle(F1[4*i])),
                                       mode = 'lines',
                                       showlegend=sl1,
                                       line={'color':'blue'},
                                       name='Forced Displacement',
                                       legendgroup='Forced Displacement'),)
            fig_F.add_trace(go.Scatter3d(x=[a],
                                       z=np.abs(F1[4*i+1])*np.cos(np.angle(F1[4*i+1])),
                                       y=np.abs(F1[4*i])*np.cos(np.angle(F1[4*i])),
                                       mode = 'markers',
                                       showlegend=False,
                                       marker={'color':'blue','size':2},
                                       name='Forced Displacement',
                                       legendgroup='Forced Displacement'),)
            sl1 = False
            
        if np.abs(F2[4*i+1]) > 0.05*np.max(np.abs(F2)) or np.abs(F2[4*i]) > 0.05*np.max(np.abs(F2)):
            fig_F.add_trace(go.Scatter3d(x=[a]*len(t),
                                       z=np.abs(F2[4*i+1])*np.cos(t+np.angle(F2[4*i+1])),
                                       y=np.abs(F2[4*i])*np.cos(t+np.angle(F2[4*i])),
                                       mode = 'lines',
                                       showlegend=sl2,
                                       line={'color':'red'},
                                       name='Modal Reduction',
                                       legendgroup='Modal Reduction'),)
            fig_F.add_trace(go.Scatter3d(x=[a],
                                       z=np.abs(F2[4*i+1])*np.cos(np.angle(F2[4*i+1])),
                                       y=np.abs(F2[4*i])*np.cos(np.angle(F2[4*i])),
                                       mode = 'markers',
                                       showlegend=False,
                                       marker={'color':'red','size':2},
                                       name='Modal Reduction',
                                       legendgroup='Modal Reduction'),)
            sl2 = False
            
        if np.abs(F3[4*i+1]) > 0.05*np.max(np.abs(F3)) or np.abs(F3[4*i]) > 0.05*np.max(np.abs(F3)):
            fig_F.add_trace(go.Scatter3d(x=[a]*len(t),
                                       z=np.abs(F3[4*i+1])*np.cos(t+np.angle(F3[4*i+1])),
                                       y=np.abs(F3[4*i])*np.cos(t+np.angle(F3[4*i])),
                                       mode = 'lines',
                                       showlegend=sl3,
                                       line={'color':'green'},
                                       name='Unbalance Distribution',
                                       legendgroup='Unbalance Distribution'),)
            fig_F.add_trace(go.Scatter3d(x=[a],
                                       z=np.abs(F3[4*i+1])*np.cos(np.angle(F3[4*i+1])),
                                       y=np.abs(F3[4*i])*np.cos(np.angle(F3[4*i])),
                                       mode = 'markers',
                                       showlegend=False,
                                       marker={'color':'green','size':2},
                                       name='Unbalance Distribution',
                                       legendgroup='Unbalance Distribution'),)
            sl3 = False
            
        if np.abs(F4[4*i+1]) > 0.05*np.max(np.abs(F4)) or np.abs(F4[4*i]) > 0.05*np.max(np.abs(F4)):
            fig_F.add_trace(go.Scatter3d(x=[a]*len(t),
                                       z=np.abs(F4[4*i+1])*np.cos(t+np.angle(F4[4*i+1])),
                                       y=np.abs(F4[4*i])*np.cos(t+np.angle(F4[4*i])),
                                       mode = 'lines',
                                       showlegend=sl4,
                                       line={'color':'orange'},
                                       name='X-Y Forces',
                                       legendgroup='X-Y Forces'),)
            fig_F.add_trace(go.Scatter3d(x=[a],
                                       z=np.abs(F4[4*i+1])*np.cos(np.angle(F4[4*i+1])),
                                       y=np.abs(F4[4*i])*np.cos(np.angle(F4[4*i])),
                                       mode = 'markers',
                                       showlegend=False,
                                       marker={'color':'orange','size':2},
                                       name='X-Y Forces',
                                       legendgroup='X-Y Forces'),)
            sl4 = False
        
        
                     
    
    fig_F.update_layout(title_text=f'Erros: {[str(a)+"%" for a in np.round(np.abs(out2-1)*100)]}',
                      scene=dict(yaxis={'range':[2*-max_amp,2*max_amp],
                                        'title':'X [N]'},
                                 zaxis={'range':[2*-max_amp,2*max_amp],
                                        'title':'Y [N]'},
                                 xaxis={'title':'Axial position [m]'},))
    
    return fig_d , fig_F
        
    
        
        
        
        
        
        
        
        
        
        
        