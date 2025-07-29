import torch
import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, Any

# Import your existing modules
sys.path.append(os.path.expanduser('~/transformer_model'))
from mData import create_data_loaders, load_excel_data

def emergency_data_diagnosis(excel_file_path: str):
    """Emergency diagnosis to identify critical data issues."""
    
    print("🚨 EMERGENCY DATA DIAGNOSIS 🚨")
    print("="*80)
    
    # 1. Check raw data first
    print("\n1. RAW DATA ANALYSIS")
    print("-" * 40)
    
    try:
        df = load_excel_data(excel_file_path)
        print(f"✓ Data loaded successfully: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check for required columns
        required_cols = ['days', 'k_eff', 'control_rod_position']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"🚨 CRITICAL: Missing columns: {missing_cols}")
            return
        
        # Analyze each column
        for col in required_cols:
            data = df[col]
            print(f"\n{col}:")
            print(f"  Range: [{data.min():.6f}, {data.max():.6f}]")
            print(f"  Mean: {data.mean():.6f}, Std: {data.std():.6f}")
            print(f"  Unique values: {data.nunique()}")
            
            # Check for problematic patterns
            if data.std() < 1e-8:
                print(f"  🚨 CRITICAL: {col} has near-zero variance!")
            
            if data.isnull().any():
                print(f"  ⚠️  WARNING: {col} has {data.isnull().sum()} null values")
                
            if np.isinf(data).any():
                print(f"  🚨 CRITICAL: {col} has infinite values!")
        
    except Exception as e:
        print(f"🚨 CRITICAL: Cannot load data - {e}")
        return
    
    # 2. Check data loaders
    print(f"\n2. DATA LOADER ANALYSIS")
    print("-" * 40)
    
    try:
        train_loader, valid_loader, test_loader, scalers = create_data_loaders(
            file_path=excel_file_path,
            batch_size=16,  # Small batch for testing
            input_seq_len=20,
            output_pred_len=4,
            train_split=0.7,
            valid_split=0.15,
            test_split=0.15,
            random_seed=42
        )
        
        print(f"✓ Data loaders created successfully")
        print(f"Train batches: {len(train_loader)}")
        print(f"Valid batches: {len(valid_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Analyze first batch from each loader
        loaders = [("TRAIN", train_loader), ("VALID", valid_loader), ("TEST", test_loader)]
        
        for name, loader in loaders:
            print(f"\n{name} DATASET:")
            try:
                past_state, past_control, target = next(iter(loader))
                
                print(f"  Shapes: past_state{past_state.shape}, past_control{past_control.shape}, target{target.shape}")
                
                # Critical checks
                for tensor_name, tensor in [("past_state", past_state), ("past_control", past_control), ("target", target)]:
                    print(f"  {tensor_name}:")
                    print(f"    Range: [{tensor.min():.6f}, {tensor.max():.6f}]")
                    print(f"    Mean: {tensor.mean():.6f}, Std: {tensor.std():.6f}")
                    
                    # Critical issues
                    if torch.isnan(tensor).any():
                        print(f"    🚨 CRITICAL: {tensor_name} contains NaN values!")
                    if torch.isinf(tensor).any():
                        print(f"    🚨 CRITICAL: {tensor_name} contains infinite values!")
                    if tensor.std() < 1e-8:
                        print(f"    🚨 CRITICAL: {tensor_name} has near-zero variance!")
                    
                    # Check for extreme values that can cause training issues
                    if tensor.abs().max() > 1000:
                        print(f"    ⚠️  WARNING: {tensor_name} has very large values (max: {tensor.abs().max():.2f})")
                        
            except Exception as e:
                print(f"    🚨 ERROR processing {name} loader: {e}")
    
    except Exception as e:
        print(f"🚨 CRITICAL: Cannot create data loaders - {e}")
        return
    
    # 3. Test simple linear regression baseline
    print(f"\n3. SIMPLE BASELINE TEST")
    print("-" * 40)
    
    try:
        # Get dimensions
        past_state, past_control, target = next(iter(train_loader))
        
        # Flatten inputs
        input_features = torch.cat([
            past_state.flatten(1),  # Flatten sequence dimension
            past_control.flatten(1)
        ], dim=1)
        target_flat = target.flatten(1)
        
        print(f"Input shape: {input_features.shape}")
        print(f"Target shape: {target_flat.shape}")
        
        # Simple linear model
        model = torch.nn.Linear(input_features.shape[1], target_flat.shape[1])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()
        
        print("\nTesting simple linear regression:")
        initial_loss = None
        
        for epoch in range(10):
            epoch_loss = 0
            batch_count = 0
            
            for past_state, past_control, target in train_loader:
                # Prepare inputs
                x = torch.cat([past_state.flatten(1), past_control.flatten(1)], dim=1)
                y = target.flatten(1)
                
                # Forward pass
                pred = model(x)
                loss = loss_fn(pred, y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                
                if batch_count >= 5:  # Limit to first 5 batches for speed
                    break
            
            avg_loss = epoch_loss / batch_count
            if initial_loss is None:
                initial_loss = avg_loss
            
            if epoch % 3 == 0:
                print(f"  Epoch {epoch}: Loss = {avg_loss:.6f}")
        
        # Check if learning occurred
        final_loss = avg_loss
        improvement = (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0
        
        print(f"\nBaseline Results:")
        print(f"  Initial loss: {initial_loss:.6f}")
        print(f"  Final loss: {final_loss:.6f}")
        print(f"  Improvement: {improvement*100:.2f}%")
        
        if improvement < 0.01:  # Less than 1% improvement
            print("  🚨 CRITICAL: Simple baseline cannot learn - fundamental data issues!")
        elif improvement < 0.1:  # Less than 10% improvement
            print("  ⚠️  WARNING: Very little learning - possible data quality issues")
        else:
            print("  ✓ Baseline can learn - transformer issues likely architectural")
            
    except Exception as e:
        print(f"🚨 CRITICAL: Baseline test failed - {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Recommendations
    print(f"\n4. RECOMMENDATIONS")
    print("-" * 40)
    print("Based on the analysis above:")
    print("• If any CRITICAL issues found → Fix data preprocessing first")
    print("• If baseline cannot learn → Check data normalization/scaling")
    print("• If baseline works but transformer fails → Reduce model complexity")
    print("• If extreme values found → Apply better normalization")
    print("• If near-zero variance → Check if target values are meaningful")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    excel_file_path = "expanded_keff_1000_steps.xlsx"
    
    if not os.path.exists(excel_file_path):
        print(f"🚨 CRITICAL: File {excel_file_path} not found!")
        sys.exit(1)
    
    emergency_data_diagnosis(excel_file_path)
