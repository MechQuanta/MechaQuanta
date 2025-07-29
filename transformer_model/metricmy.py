import numpy as np
from typing import Union, List, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score

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
