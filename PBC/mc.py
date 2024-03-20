import random,os
from numba import njit
import torch
from torch.autograd import Variable
import numpy as np
current_cwd=os.getcwd()

# lattice is 10x10
L1,L2=10,10
K=3#padding size
k=int((K-1)/2)
site_number=int(L1*L2)

#PBC padding
@njit
def make_PBC_spin_lattice(spin_lattice):
    PBC=np.zeros(shape=(len(spin_lattice),L1+K-1,L2+K-1))
    k=int((K-1)/2)
    PBC[:,k:L1+k,k:L2+k]=spin_lattice[:,:,:]
    PBC[:,:k,:k]=spin_lattice[:,-k:,-k:]
    PBC[:,:k, -k:]=spin_lattice[:,-k:, :k]
    PBC[:,-k:, :k]=spin_lattice[:,:k, -k:]
    PBC[:,-k:, -k:]=spin_lattice[:,:k, :k]
    PBC[:,:k, k:L2+k]=spin_lattice[:,-k:, :]
    PBC[:,-k:, k:L2+k]=spin_lattice[:,:k, :]
    PBC[:,k:L1+k, :k]=spin_lattice[:,:, -k:]
    PBC[:,k:L1+k, -k:]=spin_lattice[:,:, :k]
    return PBC #[batchsize,L+K-1,L+K-1]


def rot90(s):
    s=s.reshape((len(s),1,L1+K-1,L2+K-1))
    s1=np.array(s)
    s2=np.rot90(s1,1,axes=(2,3)).copy()
    s3=np.rot90(s1,2,axes=(2,3)).copy()
    s4=np.rot90(s1,3,axes=(2,3)).copy()
    return s1,s2,s3,s4


def WS_CALCULATE(spin_lattice, net, gradient_option, rank):
    if gradient_option==0:
        with torch.no_grad():
            s1=make_PBC_spin_lattice(spin_lattice.reshape(1,L1,L2))
            spin_input_1=torch.DoubleTensor(s1.reshape(1,1,L1+K-1,L2+K-1))
            ws = net(spin_input_1).squeeze()
            return ws.data.numpy()
    else:
        s1 = make_PBC_spin_lattice(spin_lattice.reshape(1,L1,L2))
        spin_input_1 = torch.DoubleTensor(s1.reshape(1,1,L1+K-1,L2+K-1))
        ws = net(spin_input_1).squeeze()
        params = list(net.parameters())
        for i in params:
            i.grad = None
        ws.backward()
        Os = [(i.grad/ws).data.numpy() for i in params]
        Os_flattern=np.concatenate([ele.reshape(-1) for ele in Os[:]])
        return ws.data.numpy(), Os_flattern

def calculate_sprime(spin_lattice, Rb, R0, Omega, Delta):#PBC
    E_s=0
    propose_batch=[]
    for i in range(site_number):
        iy=int(i%L2)
        ix=int((i-iy)/L1)
        site_1=[ix,iy]
        spin_1=spin_lattice[ix,iy]
        propose_batch.append([site_1,-1*spin_1,Omega/2])
        E_s-=Delta*(1+spin_1)/2
        for j in range(i+1,site_number):
            jy=int(j%L2)
            jx=int((j-jy)/L1)
            dx,dy=np.abs(jx-ix),np.abs(jy-iy)
            dx=min([dx,L1-dx])
            dy=min([dy,L2-dy])
            distance=np.sqrt(dx**2+dy**2)
            if distance<=R0:
                spin_2=spin_lattice[jx,jy]
                E_s+=Omega*(Rb/distance)**6*(1+spin_1)*(1+spin_2)/4
    return E_s,propose_batch


def make_s_prime(spin_lattice,propose_batch):
    s_prime_batch=np.zeros(shape=(len(propose_batch),L1,L2)) #[batchsize,L,L]
    s_prime_batch[:,:,:]=spin_lattice[:,:]
    for i,propose in enumerate(propose_batch):
        site_1, spin_2, sign=propose[0],propose[1],propose[2]
        s_prime_batch[i,site_1[0],site_1[1]]=spin_2
    return s_prime_batch #[batchsize,L,L]


def Energy_on_spin(spin_lattice, net, rank):
    Rb, R0, Omega, Delta=1.6,6,1,3
    ws, O_s = WS_CALCULATE(spin_lattice, net, 1, rank)
    E_s,propose_batch = calculate_sprime(spin_lattice, Rb, R0, Omega, Delta)
    # computing the total non-diagonal Es elements
    batchsize = len(propose_batch)
    s_prime = make_s_prime(spin_lattice, propose_batch)  # [batchsize,L,L]
    s_prime_PBC = make_PBC_spin_lattice(s_prime)  # [batchsize,L+K-1,L+K-1]
    s1 = s_prime_PBC.reshape((batchsize, 1, L1 + K - 1, L2 + K - 1))
    with torch.no_grad():
        ws_1_batch = net(torch.DoubleTensor(s1)).squeeze()
        sign_batch = [e[-1] for e in propose_batch]
        ws_1_batch = ws_1_batch.data.numpy()/ws
        ws_1_batch = ws_1_batch * np.array(sign_batch)
        E_s += np.sum(ws_1_batch)
    return E_s, O_s

def choose_site():
    x1=np.random.randint(0,L1)
    y1=np.random.randint(0,L2)
    return [x1,y1]

def MC_fliponesite(Nsweep,net,spin_lattice,rank):
    Es_list=[]
    Os_list=[]
    sweep_count=0
    collected_samples=0
    ws=WS_CALCULATE(spin_lattice,net,0,rank)
    while collected_samples<Nsweep:
        sweep_count += 1
        site=choose_site()
        spin=spin_lattice[site[0],site[1]]
        #flip a spin
        spin_lattice[site[0], site[1]]=-1*spin
        ws_1=WS_CALCULATE(spin_lattice,net,0,rank)
        P=(ws_1/ws)**2
        r=np.random.uniform(0,1)
        if P>r:
            ws=ws_1
        else:
            #flip back
            spin_lattice[site[0], site[1]]=spin
        if sweep_count%site_number==0:
            ele1=Energy_on_spin(spin_lattice,net,rank)
            Es = ele1[0]
            if np.abs(Es/site_number)<1e3:
                collected_samples+=1
                Es_list.append(ele1[0])
                Os_list.append(ele1[1])
                if collected_samples%100==0:
                    with open('tmp','a') as f:
                        f.write("{},{},{}\n".format(collected_samples,Es/site_number,ws))
            else:
                pass
                with open("fly_away","a") as f:
                    f.write("{} {}\n".format(Es,rank))
    return Es_list,Os_list,spin_lattice

def MC_eq(Nsweep,net,spin_lattice,rank):
    sweep_count=0
    flip_count=0
    ws=WS_CALCULATE(spin_lattice,net,0,rank)
    while flip_count<Nsweep:
        sweep_count += 1
        site=choose_site()
        spin=spin_lattice[site[0],site[1]]
        #flip a spin
        spin_lattice[site[0], site[1]]=-1*spin
        ws_1=WS_CALCULATE(spin_lattice,net,0,rank)
        P=(ws_1/ws)**2
        r=np.random.uniform(0,1)
        if P>r:
            ws=ws_1
            flip_count+=1
        else:
            #flip back
            spin_lattice[site[0], site[1]]=spin
    return spin_lattice

def MC_sequence(Nsweep,net,spin_lattice,rank):
    Es_list,Os_list=[],[]
    collected_samples=0
    ws=WS_CALCULATE(spin_lattice,net,0,rank)
    while collected_samples<Nsweep:
        for i in range(L1):
            for j in range(L2):
                spin=spin_lattice[i,j]
                #flip one site
                spin_lattice[i,j]=-spin
                ws_1=WS_CALCULATE(spin_lattice,net,0,rank)
                P=(ws_1/ws)**2
                r=np.random.uniform(0,1)
                if P>r:
                    ws=ws_1
                else:#flip back
                    spin=spin_lattice[i,j]
                    spin_lattice[i,j]=-spin
        ele1=Energy_on_spin(spin_lattice,net,rank)
        Es = ele1[0]
        if -1e4<Es/site_number<1e4:
            collected_samples+=1
            Es_list.append(ele1[0])
            Os_list.append(ele1[1])
            if collected_samples%(256)==0:
                print("collected samples: {}".format(collected_samples))
                with open('tmp','a') as f:
                    f.write("{} {},{},{}\n".format(rank,collected_samples,Es/site_number,ws))
        else:
            with open("fly_away","a") as f:
                f.write("{} {}\n".format(Es,rank))
    return Es_list,Os_list,spin_lattice
