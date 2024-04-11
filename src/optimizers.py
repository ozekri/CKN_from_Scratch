from layering import Parameters
import numpy as np

class SGD:
    """
    Implements stochastic gradient descent (SGD) optimizer.

    Args:
        parameters (list or Parameters): Model parameters or a list of dictionaries containing parameters and their configurations.
        lr (float, optional): Learning rate. Defaults to 0.01.
        momentum (float, optional): Momentum factor. Defaults to 0.
        weight_decay (float, optional): L2 penalty factor. Defaults to 0.

    Raises:
        ValueError: If the input format for parameters is invalid.
    """
    def __init__(self, parameters, lr=0.01, momentum=0, weight_decay=0):
        """
        Initialize the SGD optimizer.
        """
        if isinstance(parameters, list) and all(isinstance(param, dict) for param in parameters):
            self.param_groups = parameters
        elif all(isinstance(param, Parameters) for param in parameters):
            self.param_groups = [{'params': parameters, 'lr': lr}]
        else:
            raise ValueError("Invalid input format. Must be a list of dictionaries or a list of Parameters instances.")

        self.defaults = {'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay}
        self.velocities = [[np.zeros_like(param['params'][i].values) for i in range(len(param['params']))] for param in self.param_groups]

    def step(self):
        """
        Perform a single optimization step.
        """
        for param_group, velocities_group in zip(self.param_groups, self.velocities):
            lr = param_group.get('lr', self.defaults['lr'])
            momentum = param_group.get('momentum', self.defaults['momentum'])
            weight_decay = param_group.get('weight_decay', self.defaults['weight_decay'])

            for param, velocity in zip(param_group['params'], velocities_group):
                if param.values.ndim == 1:
                    param.values = param.values.reshape(-1,1)
                param.gradients += weight_decay * param.values

                velocity *= momentum
                velocity -= lr * param.gradients

                param.values += velocity

    def zero_grad(self):
        """
        Reset gradients of all parameters to zero.
        """
        for param_group in self.param_groups:
            for param in param_group['params']:
                param.gradients.fill(0)




class MultiStepLR:
    """
    Decays the learning rate of an optimizer at given epochs.

    Args:
        optimizer (SGD or Adam or other): Optimizer instance.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float, optional): Factor by which to decay the learning rate. Defaults to 0.1.

    Attributes:
        optimizer (SGD or Adam or other): Optimizer instance.
        milestones (list): List of epoch indices.
        gamma (float): Factor by which to decay the learning rate.
        last_epoch (int): Index of the last epoch. Initialized to -1.

    Raises:
        ValueError: If milestones are not in increasing order.
    """
    def __init__(self, optimizer, milestones, gamma=0.1):
        """
        Initialize the MultiStepLR scheduler.
        """ 
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.last_epoch = -1

    def step(self):
        """
        Decay the learning rate at specified milestones.
        """
        self.last_epoch += 1
        if self.last_epoch in self.milestones:
            for param_group in self.optimizer.param_groups:
                if 'lr' in param_group:
                    param_group['lr'] *= self.gamma




# You can uncomment the end of the code to observe the SGD and scheduling on 2 synthetic examples.

"""
# Example usage :
print("Example 1:")
# Parameters initialization
parameters1 = {'params': [Parameters((3, 3)), Parameters((2, 2))], 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 0.001}
parameters2 = {'params': [Parameters((4, 4)), Parameters((3, 3))], 'lr': 0.05, 'momentum': 0.8, 'weight_decay': 0.0005}

# Optimizer initialization with parameters
optimizer = SGD([parameters1, parameters2])

# Training loop
for epoch in range(15):
    # Simulating gradient updates
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            param.grad = np.random.randn(*param.values.shape)

    # Optimization step
    optimizer.step()

    # Display learning rate for each parameter group
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"Epoch {epoch}: Group {i+1} - Learning Rate - {param_group.get('lr', optimizer.defaults['lr'])}")


##########
print("Example 2:")
# Parameters initialization
parameters1 = [Parameters((3, 3)), Parameters((2, 2))]
parameters2 = [Parameters((4, 4)), Parameters((3, 3))]

# SGD initialization with parameters and their respective learning rates
optimizer = SGD([{'params': parameters1, 'lr': 0.1}, {'params': parameters2, 'lr': 0.01}], momentum=0.9, weight_decay=0.001)

# MultistepLR scheduler
scheduler = MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)

# Training loop
for epoch in range(15):
    # Simulating gradient updates
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            param.grad = np.random.randn(*param.values.shape)

    # Optimization step
    optimizer.step()
    
    # Scheduler update
    scheduler.step()

    # Display learning rate for each parameter group
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"Epoch {epoch}: Group {i+1} - Learning Rate - {param_group['lr']}")
"""