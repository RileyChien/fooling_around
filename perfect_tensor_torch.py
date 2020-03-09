"""
Riley Chien 2020

Script for trying to find perfect tensors with quimb & pytorch

Ex: python perfect_tensor_torch.py -n (# pf phys_inds) -d (phys_dim) -v (virt_dim) -s (# steps) -r (learning_rate)

Note: Only takes even values for phys_inds

Note: If virtual dimension is chosen to be 0 it will optimize as one big tensor, if != 0 it will 
form the big tensor as a cyclic MPS of phys_inds 3-leg tensors with virtual bond of dim virt.

Uses the purity related to a Renyi entropy of the reduced state
P = Tr(rho^2)  this is equal to 1 iff the state is pure
Take the sum of P across all balanced partitions

If a satisfying value is attained, save the data of the TN as a npy file
"""

# Imports
from argparse import ArgumentParser, RawTextHelpFormatter

import numpy as np
import quimb as qu
import quimb.tensor as qtn

from quimb.tensor.optimize_pytorch import TNOptimizer
import torch

from itertools import combinations


# Parse tools to call various parameters from command line
parser = ArgumentParser(description="Perfect Tensor Searcher",formatter_class=RawTextHelpFormatter)

parser.add_argument("-n","--phys_inds",\
        help="The number of physical indices (even) (default: 6)",\
	type=int,default=6)

parser.add_argument("-d","--phys_dim",\
        help="The dimension of each physical index (default: 2)",\
	type=int,default=2)

parser.add_argument("-v","--virt_dim",\
        help="The dimension of each virtual index (even) (default: 0)",\
	type=int,default=0)

parser.add_argument("-s","--steps",\
        help="The number of optimization steps (default: 1000)",\
	type=int,default=100)

parser.add_argument("-r","--learning_rate",\
        help="The learning rate (default: 0.01)",\
	type=float,default=0.01)

args=parser.parse_args()

learning_rate = args.learning_rate

phys_inds=args.phys_inds
virt_dim=args.virt_dim
phys_dim=args.phys_dim
gradient_steps = args.steps
break_up=True
if virt_dim==0: 
    break_up=False

# Define our normalizing function
def normalize_state(psi):
    """
    Takes a tensor or tensornetwork and returns it normalized

    Parameters
    ----------
    psi: TensorNetwork

    Returns
    ----------
    psi: TensorNetwork
    """
    return psi / (psi.H @ psi) ** 0.5

if not break_up: # Take as one big tensor
    inds_list =[f'k{i}' for i in range(phys_inds)] # Give each physical index a name ex. 'k1'
    tensor_shape=[phys_dim for i in range(phys_inds)] # Get the size of the tensor
    ket=qtn.tensor_gen.rand_tensor(shape=tensor_shape,inds=inds_list,tags={'KET'}) #Build a random starting tensor
    ame_trial=qtn.TensorNetwork([normalize_state(ket)]) # Take this tensor to be its own tensor network
else: # Or choose to split it up into a network of smaller tensors
    data_tensors=[]
    for i in range(phys_inds):
        data_tensors.append(np.random.rand(phys_dim,virt_dim,virt_dim)) # Make random ndarrays for each tensor
    kets=[]
    # Make tensors naming the physical and virtual indices so they match up
    kets.append(qtn.Tensor(data_tensors[0], inds=(f'k{0}',f'v{phys_inds-1}',f'v{0}'), tags={'KET'})) 
    for i in range(1,phys_inds):
        kets.append(qtn.Tensor(data_tensors[i], inds=(f'k{i}',f'v{i-1}',f'v{i}'), tags={'KET'}))
    # Combine the tensors in to a tensor network with the & sign
    ame_trial = kets[0]
    for i in range(phys_inds-1):
        ame_trial &= kets[i+1]
    # If i split it up then normalize the full network as a ket
    ame_trial=normalize_state(ame_trial)

#list of all partitions
half_partitions=list(combinations(range(phys_inds),int(phys_inds/2)))
half_partitions1=half_partitions[:int(len(half_partitions)/2)]
half_partitions2=half_partitions[int(len(half_partitions)/2):]
half_partitions2.reverse()

def purity_sum(tensor):
    """
    Loss function to pass to TNOptimizer
    computes the purity=Tr(rho_A ^2) and sums across all balanced partitions

    Parameters
    ----------
    tensor: TensorNetwork

    Returns
    ----------
    purities: float
    """
    
    tensorH = tensor.H # Make the conjugate tensor - keeps legs labelled the same
    tensorH.retag_({'KET': 'BRA'}) # Rename 
    #make all bra inds - effectively turn the legs around
    reindex_dict = {}
    for i in range(phys_inds):
        if break_up:
            reindex_dict[f'v{i}']=f'w{i}' # Also reindex virtual bonds if necessary
        reindex_dict[f'k{i}']=f'b{i}'
    tensorH.reindex(index_map=reindex_dict, inplace=True)
    #form density matrix
    rho = tensor & tensorH
    #####

    reduction_reindex_dicts=[]
    for i in range(len(half_partitions2)):
        reduction_reindex_dicts.append({})
        for j in range(int(phys_inds/2)):
            reduction_reindex_dicts[i][f'b{half_partitions2[i][j]}']=f'k{half_partitions2[i][j]}'

    rdms=[]
    for i in range(len(half_partitions1)):
        rdms.append(rho.reindex(index_map = reduction_reindex_dicts[i],inplace=False))
    rdm_copies =[] # make copies of each rdm
    index_flips=[]
    for i in range(len(half_partitions1)):
        index_flips.append({})
        for j in range(int(phys_inds/2)):
            index_flips[i][f'k{half_partitions1[i][j]}']=f'b{half_partitions1[i][j]}'
            index_flips[i][f'b{half_partitions1[i][j]}']=f'k{half_partitions1[i][j]}'
    for i in range(len(half_partitions1)):
        rdm_copies.append(rdms[i].reindex(index_map=index_flips[i],inplace=False))
    # Build networks for each purity to calc, each is two copies of the rdm tracing over all dangling legs
    rdm_squares=[]
    for i in range(len(half_partitions1)):
        rdm_squares.append(rdms[i] & rdm_copies[i])
    purities=[]
    for i in range(len(half_partitions1)):
        purities.append(rdm_squares[i]^all) # contract the networks

    return sum(purities) # This will be the value to minimize

ideal_purity = len(half_partitions1)*phys_dim**(-phys_inds/2)

optmzr = TNOptimizer(
    ame_trial,  # Our initial input, the tensors of which to optimize
    loss_fn=purity_sum, # Our loss function - here minimize the purity across all balanced partitions
    norm_fn=normalize_state, # A normalizing function on our tensors
    learning_rate=learning_rate, # Learning rate - default to 0.01
    loss_target=ideal_purity, # A target value of the loss function, will terminate upon achieving
    progbar=True)

print(f'Aiming for a value: {ideal_purity}')

tensor_opt = optmzr.optimize(gradient_steps) # Do the optimization

if np.isclose(purity_sum(tensor_opt),ideal_purity):
    print(f'Successly found {phys_inds}x{phys_dim} perfect tensor!')
    print(f'Saved as:  {phys_inds}x{phys_dim}_perfect_tensor.npy')
    contracted_tensor = tensor_opt^all
    np.save( f'{phys_inds}x{phys_dim}_perfect_tensor.npy' ,contracted_tensor.data)
else: 
    print('Failed to converge to desired value')