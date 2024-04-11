import numpy as np
from layering import CKNLayer, Linear

class CustomModule:
    """A customized module for neural network model.
    Inspired from PyTorch's nn.Module.

    Attributes:
        params (dict): A dictionary to store the parameters of the module.
        grads (dict): A dictionary to store the gradients of the parameters.
    """
    def __init__(self):
        """
        Initializes the CustomModule with empty parameter dictionaries.
        """
        self.params = {}
        self.grads = {}
    
    def add_param(self, name, shape):
        """Adds a parameter to the module with the given name and shape.

        Args:
            name (str): The name of the parameter.
            shape (tuple): The shape of the parameter.
        """
        self.params[name] = np.random.randn(*shape)
        self.grads[name] = np.zeros_like(self.params[name])
    
    def forward(self, *args):
        """Computes the forward pass of the module.

        This method should be implemented in subclasses.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("forward method must be implemented in subclasses")

    def backward(self, *args):
        """Computes the backward pass of the module.

        This method should be implemented in subclasses.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("backward method must be implemented in subclasses")

    def zero_grad(self):
        """
        Resets all gradients to zero.
        """
        for param in self.grads:
            self.grads[param].fill(0)

    def parameters(self):
        """
        Returns the parameters of the module.
        """
        return self.params

class CKNSequential(CustomModule):
    """
    A sequential container for CKN layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels_list (list): List of integers specifying the number of output channels for each layer.
        filter_sizes (list): List of integers specifying the filter sizes for each layer.
        subsamplings (list): List of integers specifying the subsampling factors for each layer.
        kernel_funcs (list, optional): List of strings specifying the kernel functions for each layer. Default is None.
        kernel_args_list (list, optional): List of kernel arguments for each layer. Default is None.
        **kwargs: Additional keyword arguments.

    Attributes:
        n_layers (int): Number of layers in the sequential container.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels of the last layer.
        ckn_layers (list): List of CKNLayer instances representing the layers in the sequential container.
    """
    def __init__(self, in_channels, out_channels_list, filter_sizes, 
                 subsamplings, kernel_funcs=None, kernel_args_list=None
                 , **kwargs):
        """
        Initializes the CKNSequential with the specified parameters.

        Raises:
            AssertionError: If the dimensions of the provided lists are incompatible.
        """
        
        assert len(out_channels_list) == len(filter_sizes) == len(subsamplings), "incompatible dimensions"
        super(CKNSequential,self).__init__()

        self.n_layers = len(out_channels_list)
        self.in_channels = in_channels
        self.out_channels = out_channels_list[-1]

        ckn_layers = []

        for i in range(self.n_layers):
            if kernel_funcs is None:
                kernel_func = "exp"
            else:
                kernel_func = kernel_funcs[i] 
            if kernel_args_list is None:
                kernel_args = 0.5
            else:
                kernel_args = kernel_args_list[i]
        
            ckn_layer = CKNLayer(in_channels, out_channels_list[i],
                                 filter_sizes[i], subsampling=subsamplings[i],
                                 kernel_func=kernel_func, kernel_args=kernel_args,
                                 **kwargs)
            
            ckn_layers.append(ckn_layer)
            in_channels = out_channels_list[i]
        
        self.ckn_layers = ckn_layers
        print("ckn layers", self.ckn_layers)
    
    def changemode(self,mode='inf'):
        """
        Changes the mode of all layers in the sequential container.
        Could be 'train' for training, or 'inf' for inférence.

        Args:
            mode (str, optional): The mode to set for all layers. Default is 'inf'.
        """
        for layer in self.ckn_layers:
            layer.mode = mode
        print(f'mode changed to {mode}')

    
    def forward_at(self, x, i=0):
        """
        Computes the forward pass at a specific layer index.

        Args:
            x: Input data.
            i (int, optional): Index of the layer to compute the forward pass. Default is 0.

        Returns:
            Output of the forward pass at the specified layer.
        """
        assert x.shape[1] == self.ckn_layers[i].in_channels, "bad dimension"
        return self.ckn_layers[i](x)

    def forward(self, x):
        """
        Computes the forward pass through all layers in the sequential container.

        Args:
            x: Input data.

        Returns:
            Output of the forward pass through all layers.
        """
        for layer in self.ckn_layers:
            x = layer(x)
        return x
    
    def representation(self, x, n=0):
        """
        Computes the representation at a specific layer index.

        Args:
            x: Input data.
            n (int, optional): Index of the layer until which the representation is computed. Default is 0.

        Returns:
            Representation of the input data at the specified layer.
        """
        if n == -1:
            n = self.n_layers
        for i in range(n):
            x = self.forward_at(x, i)
        return x
    
    def normalize(self):
        """
        Normalizes all layers in the sequential container.
        """
        for module in self.ckn_layers:
            module.normalize()
    
    def __call__(self, x):
        return self.forward(x)
    
    def unsup_train_(self, data_loader, n_sampling_patches=100000, top_layers=None):
        """
        Unsupervised training of all layers in the sequential container.

        Args:
            data_loader: Data loader for training patches.
            n_sampling_patches (int, optional): Number of sampling patches. Default is 100000.
            top_layers: Module object representing layers before this layer.
        """
        for i, ckn_layer in enumerate(self.ckn_layers):
            print()
            print('-------------------------------------')
            print('   TRAINING LAYER {}'.format(i + 1))
            print('-------------------------------------')
            n_patches = 0 
            try:
                n_patches_per_batch = (n_sampling_patches + len(data_loader) - 1) // len(data_loader) 
            except:
                n_patches_per_batch = 1000
            patches = np.zeros((n_sampling_patches, ckn_layer.patch_dim))

            for data,_ in data_loader:
                if top_layers is not None:
                    data = top_layers(data)
                data = self.representation(data, i)
                data_patches = ckn_layer.sample_patches(data, n_patches_per_batch)
                size = data_patches.shape[0]
                if n_patches + size > n_sampling_patches:
                    size = n_sampling_patches - n_patches
                    data_patches = data_patches[:size]
                patches[n_patches: n_patches + size] = data_patches
                n_patches += size 
                if n_patches >= n_sampling_patches:
                    break

            patches = patches[:n_patches]
            ckn_layer.unsup_train(patches)
        print("Unsup train CKNSequential end")


class CKNet(CustomModule):
    """
    Full Convolutional Kernel Network (CKNet) and it's classifier, for classification tasks.

    Args:
        nclass (int): Number of classes.
        in_channels (int): Number of input channels.
        out_channels_list (list): List of integers specifying the number of output channels for each layer.
        kernel_sizes (list): List of integers specifying the kernel sizes for each layer.
        subsamplings (list): List of integers specifying the subsampling factors for each layer.
        kernel_funcs (list, optional): List of strings specifying the kernel functions for each layer. Default is None.
        kernel_args_list (list, optional): List of kernel arguments for each layer. Default is None.
        image_size (int, optional): Size of the input image. Default is 32.
        fit_bias (bool, optional): Whether to fit bias terms in the linear classifier. Default is True.
        alpha (float, optional): Regularization strength for the linear classifier. Default is 0.0.
        maxiter (int, optional): Maximum number of iterations for training the linear classifier. Default is 200.
        **kwargs: Additional keyword arguments.

    Attributes:
        features (CKNSequential): Sequential container for the convolutional kernel network layers.
        out_features (int): Number of output features after passing through the convolutional layers.
        nclass (int): Number of classes.
        classifier (Linear): Linear classifier for classification.
    """
    def __init__(self, nclass, in_channels, out_channels_list, kernel_sizes, 
                 subsamplings, kernel_funcs=None, kernel_args_list=None,
                 image_size=32, fit_bias=True, alpha=0.0,
                 maxiter=200, **kwargs):
        """
        Initializes the CKNet with the specified parameters.
        """
        super(CKNet, self).__init__()
        self.features = CKNSequential(
            in_channels, out_channels_list, kernel_sizes, 
            subsamplings, kernel_funcs, kernel_args_list,
            **kwargs)

        out_features = out_channels_list[-1]
        factor = 1
        for s in subsamplings:
            factor *= s
        factor = (image_size - 1) // factor + 1
        self.out_features = factor * factor * out_features
        self.nclass = nclass

        self.classifier = Linear(self.out_features, nclass, fit_bias=fit_bias, alpha=alpha, maxiter=maxiter)
        
    
    def forward(self, input):
        """
        Performs forward pass through the CKNet.

        Args:
            input: Input data.

        Returns:
            Output of the CKNet after passing through the linear classifier.
        """
        features = self.representation(input)
        return self.classifier(features)
    
    def representation(self, input):
        """
        Computes the representation of the input data.

        Args:
            input: Input data.

        Returns:
            Representation of the input data after passing through the convolutional layers.
        """
        features = self.features(input).reshape(input.shape[0], -1)
        return features

    def unsup_train_ckn(self, data_loader, n_sampling_patches=1000000):
        """
        Unsupervised training of the convolutional layers in the CKNet.

        Args:
            data_loader: Data loader for training patches.
            n_sampling_patches (int, optional): Number of sampling patches. Default is 1000000.
        """
        self.features.unsup_train_(data_loader, n_sampling_patches)
    
    def unsup_train_classifier(self, data_loader):
        """
        SUPERVISED Training of the linear classifier in the CKNet.
        the "unsup" here doesn't mean that it's unsupervised, but that it's the "simple" way to train the classifier (with BFGS, not SGD)

        Args:
            data_loader: Data loader for training patches.
        """
        encoded_train, encoded_target = self.predict(
            data_loader, only_representation=True)
        self.classifier.fit(encoded_train, encoded_target)
    
    def predict(self, data_loader, only_representation=False):
        """Makes predictions using the CKNet.

        Args:
            data_loader: Data loader for making predictions.
            only_representation (bool, optional): Whether to return only the representation. Default is False.

        Returns:
            Tuple containing the output and target output (if only_representation is False) or only the output (if only_representation is True).
        """
        n_samples = data_loader.num_samples
        batch_start = 0
        for i, (data, target) in enumerate(data_loader):
            batch_size = data.shape[0]

            if only_representation:
                batch_out = self.representation(data)
            else:
                batch_out = self(data)
            if i == 0:
                output = np.empty((n_samples, batch_out.shape[-1]))
                target_output = np.empty(n_samples)

            output[batch_start:batch_start+batch_size] = batch_out
            target_output[batch_start:batch_start+batch_size] = target
            batch_start += batch_size
        return output, target_output

    def normalize(self):
        """
        Normalizes the convolutional layers in the CKNet.
        """
        self.features.normalize()

    def print_norm(self):
        """
        Prints the norms of the weights of the convolutional and linear layers.
        """
        norms = []
        for module in self.features:
            norms.append(np.sum(module.weight))
        norms.append(np.sum(self.classifier.weight.values))
        print(norms)

    def get_parameters(self):
        """
        Gets the parameters of the CKNet.

        Returns:
            List of parameters of the convolutional and linear layers.
        """
        params = []
        for layer in self.features.ckn_layers:
            layer_params = layer.parameters()  # Get parameters of the layer
            params.extend(layer_params)  # Add layer parameters to the list
        params.extend(self.classifier.parameters())  # Add parameters of the linear layer to the list
        return params


### Models      

class SupCKNetCifar10_2(CKNet):
    """Convolutional Kernel Network (CKN) model for CIFAR-10 classification with 5 layers.

    This class defines a specific CKN model architecture tailored for CIFAR-10 dataset
    classification task. It consists of 2 convolutional layers with varying kernel sizes,
    filter counts, subsampling factors, and kernel functions.

    Args:
        alpha (float): Regularization parameter for L2 penalty. Default is 0.0.
        **kwargs: Additional keyword arguments.

    Attributes:
        filter_sizes (list): List of integers specifying the filter sizes for each layer.
        filters (list): List of integers specifying the number of filters for each layer.
        subsamplings (list): List of integers specifying the subsampling factors for each layer.
        kernel_funcs (list): List of strings specifying the kernel functions for each layer.
        kernel_args_list (list): List of floats specifying the kernel arguments for each layer.
        fit_bias (bool): Boolean indicating whether to fit bias terms. Default is True.
        alpha (float): Regularization parameter for L2 penalty.
        maxiter (int): Maximum number of iterations for optimization.
    """
    def __init__(self, alpha=0.0, **kwargs):
        filter_sizes = [3,3]
        filters = [32,32]
        subsamplings = [2, 6]
        kernel_funcs = ['exp', 'exp']
        kernel_args_list = [1/np.sqrt(3), 1/np.sqrt(3)]
        super(SupCKNetCifar10_2, self).__init__(
            10, 3, filters, filter_sizes, subsamplings, kernel_funcs=kernel_funcs,
            kernel_args_list=kernel_args_list, fit_bias=True, alpha=alpha, maxiter=5000, **kwargs)
    
    def __call__(self, x):
        """
        Forward pass method for the model.

        Args:
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Predicted class probabilities.
        """
        return self.forward(x)
    

class SupCKNetCifar10_3(CKNet):
    """Convolutional Kernel Network (CKN) model for CIFAR-10 classification with 5 layers.

    This class defines a specific CKN model architecture tailored for CIFAR-10 dataset
    classification task. It consists of 5 convolutional layers with varying kernel sizes,
    filter counts, subsampling factors, and kernel functions.

    Args:
        alpha (float): Regularization parameter for L2 penalty. Default is 0.0.
        **kwargs: Additional keyword arguments.

    Attributes:
        filter_sizes (list): List of integers specifying the filter sizes for each layer.
        filters (list): List of integers specifying the number of filters for each layer.
        subsamplings (list): List of integers specifying the subsampling factors for each layer.
        kernel_funcs (list): List of strings specifying the kernel functions for each layer.
        kernel_args_list (list): List of floats specifying the kernel arguments for each layer.
        fit_bias (bool): Boolean indicating whether to fit bias terms. Default is True.
        alpha (float): Regularization parameter for L2 penalty.
        maxiter (int): Maximum number of iterations for optimization.
    """
    def __init__(self, alpha=0.0, **kwargs):
        filter_sizes = [3,3,3]
        filters = [64,64,128]
        subsamplings = [2,2,2]
        kernel_funcs = ['exp','exp','exp']
        kernel_args_list = [0.6,0.6,0.6]
        super(SupCKNetCifar10_3, self).__init__(
            10, 3, filters, filter_sizes, subsamplings, kernel_funcs=kernel_funcs,
            kernel_args_list=kernel_args_list, fit_bias=True, alpha=alpha, maxiter=5000, **kwargs)
    
    def __call__(self, x):
        """
        Forward pass method for the model.

        Args:
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Predicted class probabilities.
        """
        return self.forward(x)


class SupCKNetCifar10_5(CKNet):
    """Convolutional Kernel Network (CKN) model for CIFAR-10 classification with 5 layers.

    This class defines a specific CKN model architecture tailored for CIFAR-10 dataset
    classification task. It consists of 5 convolutional layers with varying kernel sizes,
    filter counts, subsampling factors, and kernel functions.

    Args:
        alpha (float): Regularization parameter for L2 penalty. Default is 0.0.
        **kwargs: Additional keyword arguments.

    Attributes:
        filter_sizes (list): List of integers specifying the filter sizes for each layer.
        filters (list): List of integers specifying the number of filters for each layer.
        subsamplings (list): List of integers specifying the subsampling factors for each layer.
        kernel_funcs (list): List of strings specifying the kernel functions for each layer.
        kernel_args_list (list): List of floats specifying the kernel arguments for each layer.
        fit_bias (bool): Boolean indicating whether to fit bias terms. Default is True.
        alpha (float): Regularization parameter for L2 penalty.
        maxiter (int): Maximum number of iterations for optimization.
    """
    def __init__(self, alpha=0.0, **kwargs):
        filter_sizes = [3, 1, 3, 1, 3]
        filters = [64, 32, 64, 32, 64]
        subsamplings = [2, 1, 2, 1, 3]
        kernel_funcs = ['exp', 'poly', 'exp', 'poly', 'exp']
        kernel_args_list = [0.5, 2, 0.5, 2, 0.5]
        super(SupCKNetCifar10_5, self).__init__(
            10, 3, filters, filter_sizes, subsamplings, kernel_funcs=kernel_funcs,
            kernel_args_list=kernel_args_list, fit_bias=True, alpha=alpha, maxiter=500, **kwargs)
    
    def __call__(self, x):
        """
        Forward pass method for the model.

        Args:
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Predicted class probabilities.
        """
        return self.forward(x)

SUPMODELS = {
    'ckn5': SupCKNetCifar10_5,
    'ckn2': SupCKNetCifar10_2,
    'ckn3': SupCKNetCifar10_3
    }