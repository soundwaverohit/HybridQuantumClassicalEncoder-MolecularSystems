import numpy as np
from numba import njit
from multiprocessing import Pool,cpu_count
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def OS_AVG(Os_list):
    return np.mean(Os_list,0)


def OSES_AVG(Os_list,Es_list):
    N = len(Es_list)
    return np.matmul(Es_list,Os_list)/N

#grad_ij=2<Os_ij Es>-2<Os_ij><Es>
def GRAD(Os_avg,OsEs_avg,Es_avg):
    first_order_gradients=np.zeros(shape=len(OsEs_avg))
    for i in range(len(OsEs_avg)):
        first_order_gradients[i]=OsEs_avg[i]-Es_avg*Os_avg[i]
    return first_order_gradients


@njit(fastmath=False)
def OO_AVG_jit(Os_list):
    parameter_number=len(Os_list[0])
    sample_number=len(Os_list)
    S_matrix=np.zeros(shape=(parameter_number,parameter_number))
    for i in range(parameter_number):
        for j in range(i,parameter_number):
            S_matrix[i,j]=np.mean(np.array([Os_list[k][i]*Os_list[k][j]
                                    for k in range(sample_number)]))
    for i in range(parameter_number):
        for j in range(0,i):
            S_matrix[i,j]=S_matrix[j,i]
    return S_matrix


def calculate_parameters(Es_list,Os_list,rank):
    Es_avg=np.mean(Es_list)
    Os_avg=OS_AVG(Os_list)
    OsEs_avg=OSES_AVG(Os_list,Es_list)
    #OO_avg=OO_AVG(Os_list,rank)
    #print("start calculating OO on process {}".format(rank))
    #OO_avg=OO_AVG_jit(np.stack(Os_list))
    Os_list=np.array(Os_list)
    OO_avg=(Os_list.T @ Os_list) / len(Os_list)
    #print("OO calculation completed on process {}".format(rank))
    return OsEs_avg,Os_avg,Es_avg,OO_avg


def calculate_parameters_nompi(Es_list,Os_list):
    Es_avg=np.mean(Es_list)
    Os_avg=OS_AVG(Os_list)
    OsEs_avg=OSES_AVG(Os_list,Es_list)
    #OO_avg=OO_AVG(Os_list,rank)
    if len(Os_list)%cpu_count()==0:
        Os_list_split=np.array_split(Os_list,int(cpu_count()))
        pool=Pool(cpu_count())
        print("start calculating OO")
        OO_avg_process_list=pool.map(OO_AVG_jit,Os_list_split)
        print("OO calculation completed")
        pool.close()
        OO_avg=np.mean(OO_avg_process_list,0)
    else:
        print("cpu number can not be divided by sample number.")
    return OsEs_avg,Os_avg,Es_avg,OO_avg

