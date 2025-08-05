class PointKineticsModel:
	def __init__(self,constants):
		self.constants = constants
	def d_by_dt(self,vector):
		power = vector[1]
		rho = vector[2]
		temperature = vector[3]
		demand = vector[4]
		alpha_t = vector[5]
		heat_capacity = vector[6]
		precursors = vector[7:]
		
		dt_dt = 1.0
		dp_dt = (((rho - self.constants.beta)/self.constants.n_gen_time)*power)
		for i in range(self.constants.ndg):
			dp_dt += self.constants.lambda_groups[i].precursors[i]
		if heat_capacity <= 0:
			dtemp_dt = 0.0
		else:
			dtemp_dt = (power - demand)/heat_capacity
		drho_dt = dtemp_dt * alpha_t
		ddemand_dt = 0.0
		dalpha_t_dt = 0.0
		dprecursor_dt = [0.0] * self.constants.ndg
		
		for i in range(self.constants.ndg):
			
		 
