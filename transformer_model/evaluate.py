import torch
import numpy as np
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.expanduser('~/transformer_model'))
from metricmy import compute_metrics

def evaluate(
    test_loader : DataLoader, 
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    is_print : bool = True,
    ):
    model.eval()
    model.to(device)
    test_loss = 0
    
    pts = []
    gts = []
    batch_count = 0
    
    for batch_idx, (data_0D, data_ctrl, target) in enumerate(test_loader):
        with torch.no_grad():
            # Skip batches with size <= 1 for consistency with training
            if data_0D.size()[0] <= 1:
                continue
                
            # Note: optimizer.zero_grad() is not needed in evaluation with torch.no_grad()
            output = model(data_0D.to(device), data_ctrl.to(device))
            loss = loss_fn(output, target.to(device))
            test_loss += loss.item()
            batch_count += 1
            
            # Flatten the batch and time dimensions, keep feature dimension
            pts.append(output.cpu().numpy().reshape(-1, output.size()[-1]))
            gts.append(target.cpu().numpy().reshape(-1, target.size()[-1]))
    
    # Handle case where no valid batches were processed
    if batch_count == 0:
        print("Warning: No valid batches found in test_loader")
        return float('inf'), float('inf'), float('inf'), float('inf'), float('-inf')
    
    test_loss /= batch_count
    
    # Concatenate all predictions and ground truths
    pts = np.concatenate(pts, axis = 0)
    gts = np.concatenate(gts, axis = 0)
    
    # Compute metrics
    mse, rmse, mae, r2 = compute_metrics(gts, pts, None, is_print)
    
    return test_loss, mse, rmse, mae, r2
