
import torch
from depinn import PhysicsInformedNN
from depinntrainconfig import TrainConfig
import scipy.io

class OptimConfig():
    def __init__(self,device,data_path):
        super(OptimConfig, self).__init__()   
        
        self.device = device
        
        #Path to load and save data
        self.data_path = data_path
        
        train_dict = TrainConfig(self.device,self.data_path).Train_Dict()
        param_dict = TrainConfig(self.device,self.data_path).Param_Dict()

        #Load PINN
        self.model = PhysicsInformedNN(train_dict=train_dict, param_dict=param_dict)
        self.model.to(device)
        
        
    #Optimize and predict
    def OptimAndPredi(self,n_steps_1,n_steps_2):
        #Adam
        Adam_optimizer = torch.optim.Adam(params=self.model.weights + self.model.biases+[self.model.lambda_1],
                                            lr=1e-3,
                                            betas=(0.9, 0.999),
                                            eps=1e-8,
                                            weight_decay=0,
                                            amsgrad=False)
        self.model.train_Adam(Adam_optimizer, n_steps_1, None)
        
        #LBFGS
        LBFGS_optimizer = torch.optim.LBFGS(
            params=self.model.weights + self.model.biases +[self.model.lambda_1],
            lr=1,
            max_iter=n_steps_2,
            tolerance_grad=-1,
            tolerance_change=-1,
            history_size=100,
            line_search_fn=None)
        self.model.train_LBFGS(LBFGS_optimizer, None)    
    
        #Prediction
        Data_X_Pred = scipy.io.loadmat(self.data_path + '/2DIBP_X_Pred.mat')
        self.x_pred = Data_X_Pred['x_pred']
        self.y_pred = Data_X_Pred['y_pred']
        u_pred , v_pred = self.model.predict(self.x_pred,self.y_pred) 
        scipy.io.savemat(self.data_path +'/2DIBP_U_Pred.mat', {'u_pred':u_pred , 'v_pred':v_pred})
        
        
