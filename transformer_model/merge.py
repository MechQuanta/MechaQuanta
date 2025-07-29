from typing import Optional, List, Literal, Union
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.expanduser('~/transformer_model'))
from evaluate import evaluate
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from src.nn_env.metric import compute_metrics
from src.nn_env.evaluate import evaluate
from src.nn_env.predict import predict_tensorboard, predict_from_self_tensorboard
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score




def train_per_epoch(
    train_loader : DataLoader, 
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    max_norm_grad : Optional[float] = None,
    ):

    model.train()
    model.to(device)

    train_loss = 0

    for batch_idx, (data_0D, data_ctrl, target) in enumerate(train_loader):
        
        if data_0D.size()[0] <= 1:
            continue
        
        optimizer.zero_grad()
        output = model(data_0D.to(device), data_ctrl.to(device))
        loss = loss_fn(output, target.to(device))
        
        if not torch.isfinite(loss):
            print("train_per_epoch | warning : loss nan occurs")
            break
        else:
            loss.backward()
        
        # use gradient clipping
        if max_norm_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm_grad)

        optimizer.step()

        train_loss += loss.item()

    if scheduler:
        scheduler.step()

    train_loss /= (batch_idx + 1)

    return train_loss

# this validation process will be deprecated
def valid_per_epoch(
    valid_loader : DataLoader, 
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    ):

    model.eval()
    model.to(device)
    valid_loss = 0
    
    for batch_idx, (data_0D, data_ctrl, target) in enumerate(valid_loader):
        with torch.no_grad():
            if data_0D.size()[0] <= 1:
                continue
            
            # Note: optimizer.zero_grad() is not needed in validation with torch.no_grad()
            output = model(data_0D.to(device), data_ctrl.to(device))
            loss = loss_fn(output, target.to(device))
            
            valid_loss += loss.item()
        
    valid_loss /= (batch_idx + 1)

    return valid_loss   

# this version of validation process considers the multi-step prediction performance
def valid_per_epoch_multi_step(
    valid_loader : DataLoader, 
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    ):
    """
    Multi-step validation function.
    Note: This requires implementing multi_step_prediction function.
    """
    model.eval()
    model.to(device)
    valid_loss = 0
    
    seq_len_0D = model.input_0D_seq_len
    pred_len_0D = model.output_0D_pred_len
    
    for batch_idx, (data_0D, data_ctrl, target) in enumerate(valid_loader):        
        with torch.no_grad():
            if data_0D.size()[0] <= 1:
                continue
                
            # TODO: Implement multi_step_prediction function
            # For now, using regular prediction as fallback
            try:
                # Uncomment and implement this when multi_step_prediction is available
                # preds = multi_step_prediction(model, data_0D, data_ctrl, seq_len_0D, pred_len_0D)
                # preds = torch.from_numpy(preds)
                
                # Fallback to regular prediction
                preds = model(data_0D.to(device), data_ctrl.to(device))
                loss = loss_fn(preds, target.to(device))
                valid_loss += loss.item()
            except Exception as e:
                print(f"Error in multi-step validation: {e}")
                # Fallback to regular prediction
                preds = model(data_0D.to(device), data_ctrl.to(device))
                loss = loss_fn(preds, target.to(device))
                valid_loss += loss.item()
        
    valid_loss /= (batch_idx + 1)
    
    return valid_loss   
    
def train(
    train_loader : DataLoader, 
    valid_loader : DataLoader,
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    num_epoch : int = 64,
    verbose : Optional[int] = 8,
    save_best : str = "./weights/best.pt",
    save_last : str = "./weights/last.pt",
    max_norm_grad : Optional[float] = None,
    tensorboard_dir : Optional[str] = None,
    test_for_check_per_epoch : Optional[DataLoader] = None,
    use_multi_step_validation : bool = False,
    ):

    train_loss_list = []
    valid_loss_list = []

    best_epoch = 0
    best_loss = torch.inf
    
    # tensorboard setting
    if tensorboard_dir:
        writer = SummaryWriter(tensorboard_dir)
    else:
        writer = None

    for epoch in tqdm(range(num_epoch), desc = "training process"):

        train_loss = train_per_epoch(
            train_loader, 
            model,
            optimizer,
            scheduler,
            loss_fn,
            device,
            max_norm_grad,
        )
        
        # Choose validation method
        valid_loss = valid_per_epoch(
		valid_loader, 
                model,
                optimizer,
                loss_fn,
                device,
            )

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        if verbose:
            if epoch % verbose == 0:
                print("epoch : {}, train loss : {:.3f}, valid loss : {:.3f},".format(epoch+1, train_loss, valid_loss))
                
                if test_for_check_per_epoch and writer:
                    model.eval()
                    # evaluate metric in tensorboard
                    test_loss, mse, rmse, mae, r2 = evaluate(test_for_check_per_epoch, model, optimizer, loss_fn, device, False)

                    writer.add_scalars('test', 
                                        {
                                            'loss' : test_loss,
                                            'mse':mse,
                                            'rmse':rmse,
                                            'mae':mae,
                                            'r2':r2,
                                        }, 
                                        epoch + 1)
                    
                    fig = predict_from_self_tensorboard(model, test_for_check_per_epoch.dataset, device)
                    
                    # model performance check in tensorboard
                    writer.add_figure('model performance', fig, epoch+1)
                    model.train()
                
        # tensorboard recording (only if writer exists)
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/valid', valid_loss, epoch)
        
        # save the best parameters
        if best_loss > valid_loss:
            best_loss = valid_loss
            best_epoch  = epoch
            torch.save(model.state_dict(), save_best)

        # save the last parameters
        torch.save(model.state_dict(), save_last)
        
    print("training process finished, best loss : {:.3f}, best epoch : {}".format(best_loss, best_epoch))
    
    if writer:
        writer.close()

    return train_loss_list, valid_loss_list
    

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
    
    
def MSE(gt : np.ndarray, pt : np.ndarray):
    return mean_squared_error(gt, pt, squared=True)

def RMSE(gt : np.ndarray, pt : np.ndarray):
    return mean_squared_error(gt, pt, squared=False)

def MAE(gt: np.ndarray, pt: np.ndarray):
    # Using sklearn's implementation for consistency and better handling of edge cases
    return mean_absolute_error(gt, pt)
    # Alternative: return np.mean(np.abs(gt - pt))

def R2(gt : np.ndarray, pt : np.ndarray):
    return r2_score(gt, pt)

def compute_metrics(
    gt : Union[np.ndarray, List], 
    pt : Union[np.ndarray, List], 
    algorithm : Optional[str] = None, 
    is_print : bool = True
):
    # Convert to numpy arrays if they aren't already
    if isinstance(gt, list):
        gt = np.array(gt)
    if isinstance(pt, list):
        pt = np.array(pt)
    
    # Handle different input shapes
    if gt.ndim == 3:
        gt = gt.reshape(-1, gt.shape[2])
        pt = pt.reshape(-1, pt.shape[2])
    elif gt.ndim == 1 and pt.ndim == 1:
        # For 1D arrays, reshape to 2D for consistent handling
        gt = gt.reshape(-1, 1)
        pt = pt.reshape(-1, 1)
    
    # Validate shapes
    if gt.shape != pt.shape:
        raise ValueError(f"Shape mismatch: gt.shape={gt.shape}, pt.shape={pt.shape}")
    
    # Handle empty arrays
    if gt.size == 0:
        print("Warning: Empty arrays provided")
        return np.nan, np.nan, np.nan, np.nan
    
    # Handle arrays with NaN or infinite values
    if np.any(np.isnan(gt)) or np.any(np.isnan(pt)):
        print("Warning: NaN values detected in input arrays")
        # Remove NaN values
        mask = ~(np.isnan(gt).any(axis=1) | np.isnan(pt).any(axis=1))
        gt = gt[mask]
        pt = pt[mask]
        
        if gt.size == 0:
            print("Warning: No valid data after removing NaN values")
            return np.nan, np.nan, np.nan, np.nan
    
    if np.any(np.isinf(gt)) or np.any(np.isinf(pt)):
        print("Warning: Infinite values detected in input arrays")
        # Remove infinite values
        mask = ~(np.isinf(gt).any(axis=1) | np.isinf(pt).any(axis=1))
        gt = gt[mask]
        pt = pt[mask]
        
        if gt.size == 0:
            print("Warning: No valid data after removing infinite values")
            return np.nan, np.nan, np.nan, np.nan
    
    try:
        mse = MSE(gt, pt)
        rmse = RMSE(gt, pt)
        mae = MAE(gt, pt)
        r2 = R2(gt, pt)
        
        if is_print:
            if algorithm:
                print("| {} | mse : {:.3f} | rmse : {:.3f} | mae : {:.3f} | r2-score : {:.3f}".format(
                    algorithm, mse, rmse, mae, r2))
            else:
                print("| mse : {:.3f} | rmse : {:.3f} | mae : {:.3f} | r2-score : {:.3f}".format(
                    mse, rmse, mae, r2))
        
        return mse, rmse, mae, r2
    
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return np.nan, np.nan, np.nan, np.nan
