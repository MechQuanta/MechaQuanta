def plot_history(history):
    import matplotlib.pyplot as plt

    # Plotting training and validation loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['valid_loss'], label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plotting detailed test metrics
    plt.subplot(1, 2, 2)
    plt.plot(history['test_epochs'], history['test_mae'], label='Test MAE', marker='o')
    plt.plot(history['test_epochs'], history['test_rmse'], label='Test RMSE', marker='o')
    plt.title('Detailed Test Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Error Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# You would call the functions like this in your main script:
#
# history = train(...)
#
# print("\nFinal Test Metrics:")
# for i, epoch in enumerate(history['test_epochs']):
#    print(f"Epoch {epoch}: MAE = {history['test_mae'][i]:.4f}, R2 = {history['test_r2'][i]:.4f}")
#
# plot_history(history)
