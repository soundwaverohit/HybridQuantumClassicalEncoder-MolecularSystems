import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random,os,pickle,math
import numpy as np
from numba import njit
from scipy.linalg import cho_factor, cho_solve
from mpi4py import MPI
from mc import MC_fliponesite,MC_eq,MC_sequence
from calculate_movements import calculate_parameters
#torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled = False

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
current_dir=os.getcwd()
'''if rank%24==0:
    os.system('rm -rf /dev/shm/*')
    os.system('cp -r /home/quaninfo/dsj2010/home-ssd/tJModel_U1 /dev/shm')
    os.chdir('/dev/shm/tJModel_U1')
    os.system('make clean')
    os.system('make')
    os.chdir(current_dir)'''


L1,L2=10,10
K=3
k=int((K-1)/2)
site_number=int(L1*L2)

# build neural network
class single_CNN(nn.Module):
    def __init__(self,K):
        super(single_CNN, self).__init__()
        self.K=K
        self.conv1 = nn.Conv2d(1, 128, K, padding=0)
        self.maxpool = nn.MaxPool2d(2,stride=2)
        self.deconv = nn.ConvTranspose2d(128, 1, 2, stride=2, padding=0)
    def nn(self,x):
        #x shape is [B,1,L+K-1,L+K-1]
        bs=len(x)
        x=self.conv1(x) #[B,C,L,L]
        x=self.maxpool(x) #[B,C,L/2,L/2]
        x=self.deconv(x) #[B,1,L,L]
        x=x.view(bs,-1)*5 #[B,L^2]
        x=torch.prod(x,1)
        return x
    def forward(self, x):
        x=self.nn(x)
        return x

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


if rank == 0:
    net=single_CNN(K).double()
    for i in net.parameters():
        if i.dim() > 1:
            nn.init.xavier_uniform_(i)
        else:
            nn.init.uniform_(i,-5e-3,5e-3)
    print("Total parameter number is {}".format(get_n_params(net)))
    params_amp_nn=list(net.parameters())[:]
    param_number_list=[len(params_amp_nn[i].reshape(-1)) for i in range(len(params_amp_nn))]
    # assign pre-trained weights
    '''with open("model_chk_760","rb") as f:
        weights=pickle.load(f)
    for i in range(len(weights))[:]:
        params_amp_nn[i].data=torch.from_numpy(weights[i])'''


# initial Niel State
initial_Niel_spin_lattice=np.zeros((L1,L2))
for i in range(L1):
    if i%2==0:
        for j in range(L2):
            if j%2==0:
                initial_Niel_spin_lattice[i,j]=1
            else:
                initial_Niel_spin_lattice[i,j]=-1
    else:
        for j in range(L2):
            if j%2==0:
                initial_Niel_spin_lattice[i,j]=-1
            else:
                initial_Niel_spin_lattice[i,j]=1


#initial random state
def initial_random_spin_lattice():
    particle_number=int(site_number*(1-0.125))
    occupation=random.sample(list(np.arange(0,site_number)),particle_number)
    pick_half_position=random.sample(occupation,int(particle_number/2))
    s=np.zeros(shape=[site_number])
    for k in occupation:
        if k in pick_half_position:
            s[k]=1
        else:
            s[k]=-1
    return s.reshape(L1,L2)


def GRAD(Os_avg,OsEs_avg,Es_avg):
    first_order_gradients=np.zeros(shape=len(OsEs_avg))
    for i in range(len(OsEs_avg)):
        first_order_gradients[i]=OsEs_avg[i]-Es_avg*Os_avg[i]
    return first_order_gradients


@njit(fastmath=False)
def build_covariance_matrix(OO_avg,Os_avg,shift):
    for i in range(len(OO_avg)):
        for j in range(len(OO_avg[i])):
            OO_avg[i,j]=OO_avg[i,j]-Os_avg[i]*Os_avg[j]
    return OO_avg+shift*np.eye(len(OO_avg))


@njit(fastmath=False)
def compute_inverse(S):
    return np.linalg.inv(S)


Energy_per_site_list=[]


running_epoch = 0
for epoch in range(1,100000):
    if rank == 0:
        dt=1e-2
        single_process_working_list=[[1212,net] for i in range(size)]
    else:
        single_process_working_list=None
    process_working_list = comm.scatter(single_process_working_list, root=0)
    if running_epoch == 0:
        if rank == 0:
            spin_lattice = initial_Niel_spin_lattice
            spin_lattice = MC_eq(3000,net,spin_lattice,rank)#warm up the MC sampling
            np.save("startup_spin_lattice",spin_lattice)
            pre_hot_spins = [spin_lattice for _ in range(size)]
            #spin_lattice=np.load('spin_lattice_12_5.npy')
            #with open('spin_lattice_80','rb') as f:
            #    spin_lattice=pickle.load(f)
            #pre_hot_spins = [spin_lattice for _ in range(size)]
            #with open('process_spin_lattice_760','rb') as f:
            #    pre_hot_spins=pickle.load(f)[:]
            #pre_hot_spins=(pre_hot_spins+pre_hot_spins)[:size]
            #pre_hot_spins=[pre_hot_spins[10] for _ in range(size)]
        else:
            pre_hot_spins = None
        spin_lattice = comm.scatter(pre_hot_spins, root=0)
    batch_sample_number=3000 # MC sampling number defined here
    RRR=1
    OsEs_avg_list, Os_avg_list, Es_avg_list, OO_avg_list=[],[],[],[]
    OO_avg_list_1=[]
    for batch in range(RRR):
        Es_list,Os_list,spin_lattice = MC_fliponesite(batch_sample_number,process_working_list[1],spin_lattice,rank)
        OsEs_avg, Os_avg, Es_avg, OO_avg=calculate_parameters(Es_list,Os_list,rank)
        OsEs_avg_list.append(OsEs_avg)
        Os_avg_list.append(Os_avg)
        Es_avg_list.append(Es_avg)
    return_working_list = comm.gather([np.mean(Es_avg_list,0),spin_lattice], root=0)
    if rank==0:
        Os_avg_total=np.empty(shape=(len(Os_avg)))
        OsEs_avg_total=np.empty(shape=(len(Os_avg)))
        OO_avg_total=np.empty(shape=(len(Os_avg),len(Os_avg)))
    else:
        Os_avg_total=None
        OsEs_avg_total=None
        OO_avg_total=None
    comm.Reduce(np.mean(Os_avg_list,0),Os_avg_total,op=MPI.SUM,root=0)
    comm.Reduce(np.mean(OsEs_avg_list,0),OsEs_avg_total,op=MPI.SUM,root=0)
    comm.Reduce(OO_avg,OO_avg_total,op=MPI.SUM,root=0)
    if rank==0:
        Os_avg_total=Os_avg_total/size
        OsEs_avg_total=OsEs_avg_total/size
        OO_avg_total=OO_avg_total/size
    #        OO_avg_total[row]=OO_avg_row 
    if rank == 0:
        #lr=LR_scheduler(0.01,400,epoch+1)
        process_Es_avg=[e[0] for e in return_working_list]
        process_spin_lattice=[e[1] for e in return_working_list]
        Es_avg=np.mean(process_Es_avg)
        with open("energy", "a") as f:
            f.write("Epoch: {}, Epoch Energy:{}, Energy: {}, dt: {}\n".
					format(str(epoch), str(np.mean(Es_avg)/(site_number)), str([e/(site_number) for e in process_Es_avg][::10]),str(dt)))
        Os_avg=Os_avg_total
        OsEs_avg=OsEs_avg_total
        first_order_gradients=GRAD(Os_avg,OsEs_avg,Es_avg)
        # build covariance matrix
        S=build_covariance_matrix(OO_avg_total,Os_avg,shift=dt)
        S_inverse=compute_inverse(S)
        delta = -dt * np.matmul(S_inverse, first_order_gradients)
        delta_flattern_list=[]
        for i,param_number in enumerate(param_number_list):
            start_index=int(np.sum(param_number_list[:i]))
            stop_index=start_index+param_number
            delta_flattern=delta[start_index:stop_index]
            delta_flattern_list.append(delta_flattern)
        params_change=[delta_flattern_list[i].reshape(params_amp_nn[i].shape) for i in range(len(delta_flattern_list))]
        for i in range(len(params_amp_nn)):#update CNN parameters
            params_amp_nn[i].data.add_(torch.from_numpy(params_change[i]))
        if epoch % 20 == 0:
            Params_list = [params_amp_nn[i].cpu().data.numpy() for i in range(len(params_amp_nn))]
            with open("model_chk_" + str(epoch), "wb") as f:
                pickle.dump(Params_list, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open("process_spin_lattice_{}".format(epoch),'wb') as f:
                pickle.dump(process_spin_lattice,f,protocol=pickle.HIGHEST_PROTOCOL)
    running_epoch += 1
