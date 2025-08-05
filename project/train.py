
import torch
import time
from optim_config import OptimConfig
import sys

if __name__ == "__main__":
    
    #Use cuda
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print('using device ' + device)
    device = torch.device(device)

    #Path to load and save data
    data_path = '.'
    #Set training Epoches for Adam
    N_Adam = 50000
    #Set training Epoches for LBFGS
    N_LBFGS = 50000

    start_time = time.time()
    OptimConfig(device,data_path).OptimAndPredi(N_Adam,N_LBFGS)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    
    
    
    
