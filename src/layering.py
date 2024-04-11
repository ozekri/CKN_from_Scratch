import numpy as np
import math
import scipy
from utils import gaussian_filter, conv2d_scipy, spherical_kmeans_, matrix_inverse_sqrt, normalize_np
from loss import CrossEntropyLoss
from kernels import kernels

EPS = 1e-6

######### Parameters class

class Parameters:
    """
    A class representing trainable parameters with associated gradients for use in ML models, for NumPy arrays.
    Inspired from PyTorch's nn.Parameters.
    """
    def __init__(self, shape):
        """
        Initializes a Parameters object with random values and zero gradients of the specified shape.
        
        Args:
            shape (tuple): The shape of the parameter tensor.
        """
        self.values = np.random.randn(*shape)  # Initialize parameter values with random numbers
        self.gradients = np.zeros(shape)  # Initialize gradients with zeros
        self.requires_grad = True  # Indicate that gradients should be computed
    
    def zero_gradients(self):
        """
        Resets the gradients of the parameters to zero.
        """
        self.gradients.fill(0)
    
    def __add__(self, other):
        """
        Performs element-wise addition between two Parameters objects.

        Args:
            other (Parameters): Another Parameters object to be added.

        Returns:
            Parameters: A new Parameters object containing the result of the addition.
        """
        result = Parameters(self.values.shape)
        result.values = self.values + other.values
        return result
    
    def __iadd__(self, other):
        """
        Performs in-place element-wise addition between two Parameters objects.

        Args:
            other (Parameters): Another Parameters object to be added.
        """
        self.values += other.values
        return self
    
    def __sub__(self, other):
        """
        Performs element-wise subtraction between two Parameters objects.

        Args:
            other (Parameters): Another Parameters object to be subtracted.

        Returns:
            Parameters: A new Parameters object containing the result of the subtraction.
        """
        result = Parameters(self.values.shape)
        result.values = self.values - other.values
        return result
    
    def __isub__(self, other):
        """
        Performs in-place element-wise subtraction between two Parameters objects.

        Args:
            other (Parameters): Another Parameters object to be subtracted.
        """
        self.values -= other.values
        return self
    
    def __mul__(self, scalar):
        """
        Performs element-wise multiplication of the Parameters object by a scalar value.

        Args:
            scalar (float or int): The scalar value to multiply the Parameters by.

        Returns:
            Parameters: A new Parameters object containing the result of the multiplication.
        """
        result = Parameters(self.values.shape)
        result.values = self.values * scalar
        return result
    
    def __imul__(self, scalar):
        """
        Performs in-place element-wise multiplication of the Parameters object by a scalar value.

        Args:
            scalar (float or int): The scalar value to multiply the Parameters by.
        """
        self.values *= scalar
        return self
    
    def __truediv__(self, scalar):
        """
        Performs element-wise division of the Parameters object by a scalar value.

        Args:
            scalar (float or int): The scalar value to divide the Parameters by.

        Returns:
            Parameters: A new Parameters object containing the result of the division.
        """
        result = Parameters(self.values.shape)
        result.values = self.values / scalar
        return result
    
    def __itruediv__(self, scalar):
        """
        Performs in-place element-wise division of the Parameters object by a scalar value.

        Args:
            scalar (float or int): The scalar value to divide the Parameters by.
        """
        self.values /= scalar
        return self
    
    def __neg__(self):
        """
        Negates the values of the Parameters object element-wise.

        Returns:
            Parameters: A new Parameters object with negated values.
        """
        result = Parameters(self.values.shape)
        result.values = -self.values
        return result
    
    def __iter__(self):
        """
        Allows iteration over the parameter values.

        Returns:
            iter: An iterator over the parameter values.
        """
        return iter(self.values)

    def __repr__(self):
        """
        Returns a string representation of the Parameters object.

        Returns:
            str: A string representation of the Parameters object.
        """
        return f"Parameters(values={self.values}, requires_grad={self.requires_grad})"
    
    def share_grad(self, other):
        """
        Shares gradients with another Parameters object.

        Args:
            other (Parameters): Another Parameters object whose gradients will be shared.
        """
        self.gradients = other.gradients
        
    @property
    def shape(self):
        """
        Returns the shape of the parameter array.
        
        Returns:
            tuple: The shape of the parameter array.
        """
        return self.values.shape


######### Conv2d module from scratch

class Conv2d_scr:
    """
    A simple 2D convolutional layer implementation.
    It performs a 2D-correlation, and not a 2D-convolution. It is better, because we save a matrix reflexion operation,
    and it does not change a lot of things because as weights will be learnable.
    Inspired from PyTorch's nn.Conv2D.
    """
    def __init__(self, in_channels, out_channels, filter_size, padding=0, dilation=1, groups=1,is_bias=False):
        """
        Initializes a 2D convolutional layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            filter_size (int): Size of the filter/kernel.
            padding (int): Amount of zero-padding to add around the input.
            dilation (int): Spacing between kernel elements.
            groups (int): Number of blocked connections from input channels to output channels.
            is_bias (bool): Indicates whether to include bias parameters.

        Raises:
            AssertionError: If in_channels or out_channels is not divisible by groups.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.is_bias = is_bias
        self.parameters_liste = []
        

        # Validity of the parameters (we have to check that groups is well defined)
        assert in_channels % groups == 0, "in_channels must be divisible by groups"
        assert out_channels % groups == 0, "out_channels must be divisible by groups"
        
        # Initialization of the convolution filter weights
        weight_shape = (out_channels, in_channels // groups, filter_size, filter_size)
        self.weight = Parameters(weight_shape)
        self.parameters_liste.append(self.weight)
        
        if self.is_bias:
            # Initialisation of the bias parameters
            bias_shape = (1, self.in_channels // self.groups, filter_size, filter_size)
            self.bias = Parameters(bias_shape)
            self.parameters_liste.append(self.bias)
        else:
            self.bias=None
        
        # Reset of parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Resets the parameters of the convolutional layer.
        Set bias to zero and distribute weight as a Gaussian
        """
        # Initialize weights from a normal distribution
        stdv = 1. / np.sqrt(self.weight.values.size)
        self.weight.values = np.random.normal(0, stdv, self.weight.values.shape)
        
        if self.is_bias:
            # Initialize bias to zero
            self.bias.values.fill(0)
        
    def forward(self, x):
        """
        Performs forward pass through the convolutional layer.
        This method isn't very useful in the following, as we've implemented this class
        as the basis for another, which will have its own forward method.

        Args:
            x (numpy.ndarray): Input data of shape (batch_size, in_channels, height, width).

        Returns:
            numpy.ndarray: Output feature maps of shape (batch_size, out_channels, height_out, width_out).
        """
        # Apply 2D correlation
        output = np.zeros((x.shape[0], self.out_channels, x.shape[2], x.shape[3]))
        for b in range(x.shape[0]):
            for c_out in range(self.out_channels):
                output[b, c_out] = scipy.signal.correlate2d(x[b], self.weight.values[c_out], mode='valid')
                output[b, c_out] += self.bias.values[c_out]
        return output
    
    def backward(self, x, grad_output):
        """
        Performs backward pass through the convolutional layer.
        This method isn't very useful in the following, as we've implemented this class as the basis for another,
        and we will not use backward propagation as we will learn via an unsupervised way.

        Args:
            x (numpy.ndarray): Input data of shape (batch_size, in_channels, height, width).
            grad_output (numpy.ndarray): Gradient of the loss with respect to the output.

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: Gradient of the loss with respect to the input data.
                - numpy.ndarray: Gradient of the loss with respect to the kernel weights.
                - numpy.ndarray: Gradient of the loss with respect to the bias parameters.
        """
        # Calculate bias gradient
        grad_bias = np.sum(grad_output, axis=(0, 2, 3), keepdims=True)
        
        # Calculate weights gradient
        grad_weight = np.zeros_like(self.weight.values)
        for b in range(x.shape[0]):
            for c_in in range(self.in_channels):
                for c_out in range(self.out_channels):
                    grad_weight[c_out, c_in] += scipy.signal.correlate2d(x[b, c_in], grad_output[b, c_out], mode='valid')
        
        # Calculate input gradient
        grad_input = np.zeros_like(x)
        for b in range(x.shape[0]):
            for c_in in range(self.in_channels):
                for c_out in range(self.out_channels):
                    grad_input[b, c_in] += scipy.signal.correlate2d(grad_output[b, c_out], np.flip(self.weight.values[c_out, c_in]), mode='full')
        
        return grad_input, grad_weight, grad_bias
    
    def parameters(self):
        """
        Returns the list of parameters of the convolutional layer.

        Returns:
            list: A list containing all trainable parameters of the layer.
        """
        return self.parameters_liste
    
    def __repr__(self):
        """
        Returns a string representation of the convolutional layer.

        Returns:
            str: A string representation of the convolutional layer.
        """
        return f'Conv2d_scr({self.in_channels}, {self.out_channels}, filter_size={self.filter_size}, padding={self.padding}, dilation={self.dilation}, groups={self.groups})'


########### Layers of the model

class CKNLayer(Conv2d_scr):
    """
    A CKN layer, inherits from the Conv2d_scr class.
    """
    def __init__(self, in_channels, out_channels, filter_size,
                 subsampling, padding="SAME", dilation=1, groups=1,is_bias=False,
                 subsampling_factor=1/(math.sqrt(2)),
                 kernel_func="exp", kernel_args=[0.5]):
        
        """
        Initializes a CKN layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            filter_size (int): Size of the filter/kernel.
            subsampling (int): Subsampling factor.
            padding (str): Padding mode ('SAME' or 'VALID').
            dilation (int): Spacing between kernel elements.
            groups (int): Number of blocked connections from input channels to output channels.
            is_bias (bool): Indicates whether to include bias parameters.
            subsampling_factor (float): Subsampling factor.
            kernel_func (str): Name of the kernel function.
            kernel_args (list): Arguments for the kernel function.

        Raises:
            AssertionError: If in_channels or out_channels is not divisible by groups.
        """
        
        if padding == "SAME":
            padding = filter_size // 2
        else:
            padding = 0
        super(CKNLayer, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                       filter_size=filter_size,padding=padding,dilation=dilation,
                                       groups=groups,is_bias=is_bias)
        self.normalize()
        self.mode = 'train'
        self.subsampling = subsampling
        self.subsampling_factor = subsampling_factor
        if type(self.filter_size)!= tuple:
            self.filter_size = (self.filter_size,self.filter_size)
        self.patch_dim = self.in_channels * self.filter_size[0] * self.filter_size[1]

        self.kernel_func = kernel_func
        if isinstance(kernel_args, (int, float)):
            kernel_args = [kernel_args]
        if kernel_func == "exp":
            kernel_args = [1./kernel_arg ** 2 for kernel_arg in kernel_args]
        self.kernel_args = kernel_args
        kernel_func = kernels[kernel_func]
        self.kappa = lambda x: kernel_func(x, *self.kernel_args)

        self.ones = np.ones((1, self.in_channels // self.groups, *self.filter_size))
        self.init_pooling_filter() #Initializes the pooling filter.
    
    def normalize(self):
        """
        Normalizes the weight values of the layer.
        """
        norm = np.linalg.norm(self.weight.values.reshape(self.out_channels, -1),
                              ord=2, axis=-1).reshape(-1, 1, 1, 1)
        self.weight.values/(np.clip(norm,a_min=EPS,a_max=None))

    
    def init_pooling_filter(self):
        """
        Initializes a pooling filter in the correct layer size.
        """
        size = 2 * self.subsampling + 1
        pooling_filter = gaussian_filter(size,self.subsampling_factor*self.subsampling).reshape(-1,1) #1D gaussian filter
        pooling_filter = np.dot(pooling_filter,pooling_filter.T) #2D filter
        pooling_filter = np.broadcast_to(pooling_filter,(self.out_channels,1, size, size)) # peut être (1, self.out_channels, size, size)
        self.pooling_filter = pooling_filter
        print("pooling filter size",self.pooling_filter.shape)

    def compute_lintrans(self):
        """
        Computes the linear transformation factor kappa(ZtZ)^(-1/2).

        Returns:
            lintrans: out_channels x out_channels
        """
        lintrans = self.weight.values.reshape(self.out_channels,-1)
        lintrans = lintrans@(lintrans.T)
        lintrans = self.kappa(lintrans)
        lintrans = matrix_inverse_sqrt(lintrans)
        return lintrans
    
    def conv_layer(self,x_in):
        """
        Performs convolutional layer operations.

        Args:
            x_in (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output data.
        """
        if self.bias is not None:
            patch_norm_x = conv2d_scipy(x_in**2,self.ones,bias=None,
                                        stride=1,padding=self.padding,
                                        dilation=self.dilation,
                                        groups=self.groups)
            patch_norm = patch_norm_x - 2*conv2d_scipy(x_in,self.bias,bias=None,
                stride=1,padding=self.padding,dilation=self.dilation,
                groups=self.groups)
            patch_norm = patch_norm + (self.bias.values**2).sum()
            patch_norm = np.sqrt(np.clip(patch_norm,a_min=EPS,a_max=None))
            x_out = conv2d_scipy(x_in, self.weight,bias=None,
                                 stride=1,padding=self.padding,
                                 dilation=self.dilation,
                                 groups=self.groups)
            bias_ = np.multiply(self.weight.values, self.bias.values[:, np.newaxis, np.newaxis, np.newaxis])
            bias_ = bias_.reshape(self.out_channels,-1)
            bias_ = np.sum(bias_,axis=1)
            bias_ = bias_.reshape(1, self.out_channels, 1, 1)
            x_out = x_out - bias_
            x_out = x_out / np.clip(patch_norm,a_min=EPS,a_max=None)
            x_out = patch_norm * self.kappa(x_out)
            return x_out
        
        patch_norm = np.sqrt(np.clip(conv2d_scipy(x_in**2,self.ones,bias=None,
                stride=1,padding=self.padding,dilation=self.dilation,
                groups=self.groups),a_min=EPS,a_max=None))
        x_out = conv2d_scipy(x_in, self.weight, self.bias,(1,1),
                        self.padding, self.dilation, self.groups)
        x_out = x_out / np.clip(patch_norm,a_min=EPS,a_max=None)
        x_out = patch_norm * self.kappa(x_out)
        return x_out


    def _mult_layer(self, x_in, lintrans):
        """
        Multiplication layer (batch matrix multiplication)
        Compute x_out = kappa(ZtZ)^(-1/2) x x_in
        Args:
            x_in: batch_size x in_channels x H x W
            lintrans: in_channels x in_channels
            x_out: batch_size x in_channels x H x W
        """
        batch_size, in_c, H, W = x_in.shape
        x_out = np.matmul(
            np.tile(lintrans, (batch_size, 1, 1)), 
            x_in.reshape(batch_size, in_c, -1))
        return x_out.reshape(batch_size, in_c, H, W)
    

    def pool_layer(self,x_in):
        """
        Pooling layer
        Compute I(z) = \sum_{z'} phi(z') x exp(-\beta_1 ||z'-z||_2^2)
        Args:
            x_in: batch_size x out_channels x H x W
        """
        if self.subsampling <= 1:
            return x_in
        x_out = conv2d_scipy(x_in, self.pooling_filter, bias=None, 
            stride=self.subsampling, padding=self.subsampling, 
            groups=self.out_channels,print_=True)
        return x_out
    
    def forward(self,x_in):
        """
        Forward pass through the CKN layer.

        Args:
            x_in (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output data.
        """
        x_out=self.conv_layer(x_in)
        if self.mode == 'train':
            #x_out = self.spatial_dropout(x_out,0.1)
            #x_out = self.drop_block_np(x_out,dropblock_prob=0.1,block_size=3)
            pass
        x_out=self.pool_layer(x_out)
        lintrans = self.compute_lintrans()
        x_out = self._mult_layer(x_out,lintrans)

        return x_out
    
    def __call__(self, x):
        """
        Calls the forward method.

        Args:
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output data.
        """
        return self.forward(x)
    
    def exctract_2d_patches(self,x):
        """
        Extracts 2D patches from input data.

        Args:
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Extracted patches.
        """
        h,w = self.filter_size
        batch_size, C, _, _ = x.shape

        unfolded_x = np.lib.stride_tricks.sliding_window_view(x, (batch_size, C, h, w))
        unfolded_x = unfolded_x.reshape(-1, self.patch_dim)
        return unfolded_x

    def sample_patches(self, x_in, n_sampling_patches=1000):
        """
        Samples patches from input data.

        Args:
            x_in (numpy.ndarray): Input data.
            n_sampling_patches (int): Number of patches to sample.

        Returns:
            numpy.ndarray: Sampled patches.
        """
        patches = self.exctract_2d_patches(x_in)
        n_sampling_patches = min(patches.shape[0], n_sampling_patches)
        patches = patches[:n_sampling_patches]
        return patches

    def unsup_train(self,patches):
        """
        Performs unsupervised training. of the CKN Layer.

        Args:
            patches (numpy.ndarray): Input data.

        Returns:
            None
        """
        if self.bias is not None:
            print("estimating bias")
            m_patches = patches.mean(0)
            np.copyto(self.bias.values, m_patches.reshape(self.bias.values.shape))
            patches -= m_patches
        patches = normalize_np(patches)
        block_size = None if self.patch_dim < 1000 else 10 * self.patch_dim
        weight = spherical_kmeans_(patches, self.out_channels, block_size=block_size)[0]
        weight = weight.reshape(self.weight.values.shape)
        weight = np.broadcast_to(weight, self.weight.values.shape)
        self.weight.values[:] = weight

    def batch_normalize_forward(self,features, gamma=1, beta=0, epsilon=1e-5):
        """
        Performs batch normalization forward pass.

        Args:
            features (numpy.ndarray): Input features.
            gamma (float): Scaling factor.
            beta (float): Shifting factor.
            epsilon (float): Small constant to avoid division by zero.

        Returns:
            numpy.ndarray: Normalized features.
        """
        # Calculation of batch mean and variance
        batch_mean = np.mean(features, axis=0)
        batch_var = np.var(features, axis=0)

        # Feature standardization
        x_normalized = (features - batch_mean) / np.sqrt(batch_var + epsilon)

        # Scaling and offset
        out = gamma * x_normalized + beta

        return out
    
    def spatial_dropout(self,feature_maps, dropout_prob):
        """
        Applies spatial dropout to feature maps.

        Args:
            feature_maps (numpy.ndarray): Input feature maps. Shape (batch_size, num_maps, height, width)
            dropout_prob (float): Dropout probability (between 0 and 1).

        Returns:
            numpy.ndarray: Feature maps with spatial dropout applied.
        """
        # Generate a spatial dropout mask with the same shape as the feature maps
        dropout_mask = np.random.binomial(n=1, p=1-dropout_prob, size=feature_maps.shape[1:])
        dropout_mask = np.expand_dims(dropout_mask, axis=0)  # Ajouter une dimension de batch
        
        # Apply dropout mask to feature maps
        feature_maps_dropout = feature_maps * dropout_mask
        
        return feature_maps_dropout
    
    def drop_block_np(self,x, dropblock_prob=0.1, block_size=3):
        """
        Applies DropBlock to input data.

        Args:
            x (numpy.ndarray): Input data.
            dropblock_prob (float): DropBlock probability.
            block_size (int): Block size.

        Returns:
            numpy.ndarray: Data with DropBlock applied.
        """
        if dropblock_prob == 0:
            return x

        batch_size, channels, height, width = x.shape
        gamma = (1. - dropblock_prob) / (block_size ** 2)
        mask = np.random.rand(batch_size, channels, height, width) < gamma
        mask = mask.astype(np.float32) # Convert mask to float32

        # Put zeros in the right places in the mask
        mask[:, :, block_size // 2:height - block_size // 2, block_size // 2:width - block_size // 2] = 0
        
        valid_mask = 1 - mask
        normalized_mask = valid_mask / np.mean(valid_mask)
        
        return x * normalized_mask
        
class Linear:
    """
    A linear layer.
    Its role will be to classify features after they have passed through the CKN.
    Inspired from PyTorch's Linear.
    """
    def __init__(self, in_features, out_features, alpha=0.0, fit_bias=True,
                 penalty="l2", maxiter=1000):
        """
        Initializes a linear layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            alpha (float): Regularization strength.
            fit_bias (bool): Indicates whether to fit bias parameters.
            penalty (str): Type of penalty ("l1" or "l2").
            maxiter (int): Maximum number of iterations for optimization.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.fit_bias = fit_bias
        self.penalty = penalty
        self.maxiter = maxiter
        self.parameters_liste = []
        self.weight = Parameters((out_features, in_features))
        self.parameters_liste.append(self.weight)
        if self.fit_bias:
            self.bias = Parameters((out_features,)) # (out_features,1)
            self.bias.values =self.bias.values.reshape(-1,1)
            self.parameters_liste.append(self.bias)
        else:
            self.bias = None
    
    def forward(self, input, scale_bias=1.0):
        """
        Performs forward pass through the linear layer.

        Args:
            input (numpy.ndarray): Input data.
            scale_bias (float): Scaling factor for bias.

        Returns:
            numpy.ndarray: Output data.
        """
        out = np.dot(input,self.weight.values.T)
        if self.bias is not None:
            self.bias.values = self.bias.values.ravel()
            out += scale_bias * self.bias.values
        return out
    
    def backward_propagation(self,X, Y, A):
        """
        Performs backward propagation.

        Args:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): True labels.
            A (numpy.ndarray): Activations.

        Returns:
            tuple: Gradients for weights and biases.
        """
        m = Y.shape[0]
        dZ = A.copy()
        Y = Y.astype(int)
        dZ[np.arange(m), Y] -= 1
        dZ /= m
        dW = np.dot(dZ.T, X)
        if self.alpha != 0.0:
                if self.penalty == "l2":
                    dW += self.alpha * self.weight.values
                if self.penalty == "l1":
                    dW += self.alpha * np.sign(self.weight.values)
        if self.bias is not None:
            db = np.sum(dZ, axis=0, keepdims=True).T
            return dW, db
        else:
            return dW, None
    
    def fit(self,x,y):
        """
        Fits the linear layer to the input data.

        Args:
            x (numpy.ndarray): Input data.
            y (numpy.ndarray): True labels.

        Returns:
            None
        """
        criterion = CrossEntropyLoss()
        print("L2-regularization parameter",self.alpha)
        alpha = self.alpha * x.shape[1]/x.shape[0]
        if self.bias is not None:
            scale_bias = np.sqrt(np.mean(np.mean(x ** 2, axis=-1)))
            alpha *= scale_bias ** 2
        self.real_alpha = alpha
    
        def eval_loss(w):
            """
            Evaluate the loss function during optimization.

            Args:
                w (numpy.ndarray): Flattened array containing the weights and biases.

            Returns:
                float: The value of the loss function.
            """
            w = w.reshape((self.out_features, -1))
            if self.weight.gradients is not None:
                self.weight.gradients = None
            if self.bias.gradients is not None:
                self.bias.gradients = None
            
            if self.bias is None:
                self.weight.values = w.copy()
                y_pred = np.dot(x,self.weight.values.T)    
            if self.bias is not None:
                self.weight.values = w[:, :-1].copy()
                self.bias.values = w[:, -1].copy()
                y_pred = np.dot(x,self.weight.values.T) + scale_bias*self.bias.values.squeeze()
            
            loss = criterion(y_pred, y)
            print("loss",loss.item)
            self.weight.gradients,self.bias.gradients = loss.backward(x)
            if alpha != 0.0:
                if self.penalty == "l2":
                    loss.item += 0.5 * alpha * np.linalg.norm(self.weight.values, ord=2) ** 2
                elif self.penalty == "l1":
                    loss.item += alpha * np.linalg.norm(self.weight.values, ord=1)
            return loss.item
        def eval_grad(w):
            """
            Evaluate the gradients of the loss function during optimization.

            Args:
                w (numpy.ndarray): Flattened array containing the weights and biases.

            Returns:
                numpy.ndarray: The gradients of the loss function with respect to the weights and biases.
            """
            dw = self.weight.gradients.copy()
            if alpha != 0.0:
                if self.penalty == "l2":
                    dw += alpha * self.weight.values
            if self.bias is not None:
                db = self.bias.gradients.copy()
                dw = np.hstack((dw, db.reshape(-1, 1)))
            return dw.ravel().astype("float64")
        
        w_init = self.weight.values.astype("float64")
        if self.bias is not None:
            bias_init = (1. / scale_bias) * self.bias.values.reshape(-1, 1).astype("float64")
            w_init = np.hstack((w_init, bias_init))

        print("w before BFGS",w_init)
        w = scipy.optimize.fmin_l_bfgs_b(
            eval_loss, w_init, fprime=eval_grad, maxiter=self.maxiter, disp=0)
        if isinstance(w, tuple):
            w = w[0]
        print("w after BFGS",w)
        w = w.reshape((self.out_features, -1))
        self.weight.zero_gradients()
        if self.bias is None:
            self.weight.values = w.copy()
        else:
            self.bias.zero_gradients()
            self.weight.values = w[:, :-1].copy()
            self.bias.values = (scale_bias * w[:, -1]).copy().reshape(-1,1)
    
    def __call__(self,x):
        """
        Calls the forward method.

        Args:
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output data.
        """
        return self.forward(x)
    
    def parameters(self):
        """
        Returns the parameters of the linear layer.

        Returns:
            list: List of parameters.
        """
        return self.parameters_liste