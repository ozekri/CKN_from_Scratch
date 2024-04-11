import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.signal as signal

EPS = 1e-6

def visualise_img(X,ctr = True,returnT=False,split=True):
    """Function that plots an image of the dataset.

    Args:
        X (array): 3072-length array of an image, pre-processed.
        ctr (bool, optional): Enhance the contrast of the image by putting the maximum value in the image to 1.0. Defaults to True.
        returnT (bool, optional): Return the image. Defaults to False.

    Returns:
        numpy.ndarray or None: If returnT is True, returns the image array, otherwise returns None.
    """
    Xr,Xg,Xb = np.split(X,3)
    Xr_stack= np.stack(np.split(Xr,32))
    Xr_stack += -np.min(Xr_stack)
    Xr_stack = (1/(np.max(Xr_stack)))*Xr_stack
    
    Xg_stack= np.stack(np.split(Xg,32))
    Xg_stack += -np.min(Xg_stack)
    Xg_stack = (1/(np.max(Xg_stack)))*Xg_stack
    
    Xb_stack= np.stack(np.split(Xb,32))
    Xb_stack += -np.min(Xb_stack)
    Xb_stack = (1/(np.max(Xb_stack)))*Xb_stack
    
    im0 = np.dstack((Xr_stack,Xg_stack,Xb_stack))
    if ctr :
        im0 = (1/(np.max(im0)))*im0

    plt.figure(1,figsize=(0.8,0.8))
    plt.imshow(im0)
    plt.show()
    if returnT :
        return im0
    

def countX(lst, x):
    """
    Count occurrences of x in the list.

    Args:
        lst (list): List to search for occurrences of x.
        x: Element to count occurrences of in the list.

    Returns:
        int: Number of occurrences of x in the list.
    """
    count = 0
    for ele in lst:
        if (ele == x):
            count = count + 1
    return count

def normalize_np(x, p=2, axis=-1):
    """
    Normalize the input array along the specified axis.

    Args:
        x (numpy.ndarray): Input array to be normalized.
        p (int, optional): Order of the norm. Defaults to 2.
        axis (int, optional): Axis along which to normalize. Defaults to -1.
        EPS (float, optional): Small value added to the norm to avoid division by zero. Defaults to 1e-15.

    Returns:
        numpy.ndarray: Normalized array.
    """
    norm = np.linalg.norm(x, ord=p, axis=axis, keepdims=True)
    x /= np.maximum(norm, EPS)
    return x

def count_parameters(model):
    """
    Count the total number of parameters in the model.

    Args:
        model: Model object containing parameters.

    Returns:
        int: Total number of parameters in the model.
    """
    count = 0
    for param in model.get_parameters():
        count += np.prod(param.values.shape)
    return count

def count_len_gen(gen):
    """
    Count the length of a generator.

    Args:
        gen: Generator object.

    Returns:
        int: Length of the generator.
    """
    c=0
    for _ in gen:
        c+=1
    return c


def accuracy_score(true_labels, predicted_labels):
    """Calculate the accuracy score.

    Args:
        true_labels (list): True labels.
        predicted_labels (list): Predicted labels.

    Returns:
        float: Accuracy score.
    """
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Both arrays must have the same length")


    total_samples = len(true_labels)
    correct_predictions = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    accuracy = correct_predictions / total_samples

    return accuracy



def gaussian_filter(size,sigma=None):
    """
    Create a 1D Gaussian filter.

    Args:
        size (int): Size of the filter.
        sigma (float, optional): Standard deviation of the Gaussian distribution. Defaults to None.

    Returns:
        numpy.ndarray: 1D Gaussian filter.
    """
    if size == 1:
        return np.ones(1)
    if sigma is None:
        sigma = (size - 1.) / (2.*math.sqrt(2)) #default sigma
    m = (size - 1) / 2.
    filt = np.arange(-m, m+1)
    filt = np.exp(-filt**2/(2.*sigma**2))
    return filt/np.sum(filt)


def conv2d_scipy(input, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1,mode="valid",print_=False):
        """
        Simulates the torch.nn.functional.conv2d function using scipy.signal.correlate2d.
    
        Adds support for stride, padding, dilation, and groups.

        Args:
            input (ndarray): Input tensor of shape (minibatch, in_channels, iH, iW).
            weight (ndarray): Filters of shape (out_channels, in_channels/groups, kH, kW).
            bias (ndarray, optional): Optional bias tensor of shape (out_channels). Defaults to None.
            stride (tuple, optional): The stride of the convolving kernel. Can be a single number or a tuple (sH, sW). Defaults to (1, 1).
            padding (tuple, optional): Implicit paddings on both sides of the input. Defaults to (0, 0).
            dilation (tuple, optional): The spacing between kernel elements. Can be a single number or a tuple (dH, dW). Defaults to (1, 1).
            groups (int, optional): Split input into groups. Both in_channels and out_channels should be divisible by the number of groups. Defaults to 1.
            mode (str, optional): Mode of correlation. Defaults to "valid".

        Returns:
            ndarray: The 2D convolution array.
        """
        # Get dimensions of input, weight, and bias
        if type(weight) != np.ndarray:
            weight = weight.values
        batch_size, in_channels, in_height, in_width = input.shape
        out_channels, _, kernel_height, kernel_width = weight.shape

        # Reshape dilation if it is not tuple
        if type(dilation) !=tuple:
            dilation = (dilation,dilation)

        # Reshape stride if it is not tuple
        if type(stride) !=tuple:
            stride = (stride,stride)
        
        # Reshape padding if it is not tuple
        if type(padding) !=tuple:
            padding= (padding,padding)

        # Apply padding to input if necessary
        if padding != (0, 0):
            input = np.pad(input, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant')

        # Compute output dimensions considering dilation
        stride_height, stride_width = stride
        dilation_height, dilation_width = dilation
        out_height = ((in_height + 2 * padding[0] - dilation_height * (kernel_height - 1) - 1) // stride_height) + 1
        out_width = ((in_width + 2 * padding[1] - dilation_width * (kernel_width - 1) - 1) // stride_width) + 1


        # Initialize output array
        output = np.zeros((batch_size, out_channels, out_height, out_width))

        # Perform convolution operation
        for b in range(batch_size):
            for g in range(groups):
                input_group = input[b, g * (in_channels // groups): (g + 1) * (in_channels // groups)]
                weight_group = weight[g * (out_channels // groups): (g + 1) * (out_channels // groups)]
                
                for c_in in range(in_channels // groups):
                    for c_out in range(out_channels // groups):
                        output[b, g * (out_channels // groups) + c_out] += support_correlate2d(input_group[c_in], weight_group[c_out, c_in],mode,dilation,stride) #g mod

        # Add bias if provided
        if bias is not None:
            output += bias  

        return output


def support_correlate2d(input, kernel, mode, dilation, stride=(1, 1)):
    """Adds support for dilation, stride, and padding to conv2d_scipy, inspired from scipy.signal.correlate2d.

    Args:
        input (ndarray): Input image.
        kernel (ndarray): Kernel for correlation.
        mode (str): Mode for correlation.
        dilation (tuple): Dilation factors (dilation_y, dilation_x).
        stride (tuple, optional): Stride of the sliding window (stride_y, stride_x). Defaults to (1, 1).

    Returns:
        ndarray: Correlation result.
    """
    # Calculate the shape of the dilated kernel
    dilated_shape = (
        kernel.shape[0] + (kernel.shape[0] - 1) * (dilation[0] - 1),
        kernel.shape[1] + (kernel.shape[1] - 1) * (dilation[1] - 1)
    )
    
    # Create a dilated kernel
    dilated_kernel = np.zeros(dilated_shape)
    dilated_kernel[::dilation[0], ::dilation[1]] = kernel

    # Apply padding to the input

    # Perform correlation with the dilated kernel
    result = signal.correlate2d(input, dilated_kernel, mode=mode)

    # Apply stride
    result = result[::stride[0], ::stride[1]]

    return result
    

def spherical_kmeans_(x, n_clusters, max_iters=100, block_size=None, verbose=True, init=None):
    """Performs spherical k-means clustering on the input data.

    Args:
        x (ndarray): Input data of shape (n_samples, n_features).
        n_clusters (int): The number of clusters to form.
        max_iters (int, optional): The maximum number of iterations. Defaults to 100.
        block_size (int, optional): The size of the block for computation. Defaults to None.
        verbose (bool, optional): Whether to print progress information. Defaults to True.
        init (ndarray, optional): Initial cluster centroids. Defaults to None.

    Returns:
        tuple: A tuple containing:
        ndarray: The cluster centroids of shape (n_clusters, n_features).
        ndarray: The index of the cluster each sample belongs to, of shape (n_samples,).

    """
    n_samples, n_features = x.shape
    if init is None:
        print("n_samples (kmeans)",n_samples)
        print("n_clusters",n_clusters)
        indices = np.random.choice(n_samples, n_clusters, replace=False)
        clusters = x[indices]
    else:
        clusters = init

    prev_sim = np.inf
    tmp = np.empty(n_samples)
    assign = np.empty(n_samples, dtype=np.int64)
    if block_size is None or block_size == 0:
        block_size = n_samples

    for n_iter in range(max_iters):
        # assign data points to clusters
        for i in range(0, n_samples, block_size):
            end_i = min(i + block_size, n_samples)
            cos_sim = np.dot(x[i: end_i], clusters.T)
            tmp[i: end_i] = np.max(cos_sim, axis=1)
            assign[i: end_i] = np.argmax(cos_sim, axis=1)

        sim = np.mean(tmp)
        if (n_iter + 1) % 10 == 0 and verbose:
            print("Spherical kmeans iter {}, objective value {}".format(
                n_iter + 1, sim))

        # update clusters
        for j in range(n_clusters):
            index = assign == j
            if np.sum(index) == 0:
                idx = np.argmin(tmp)
                clusters[j] = x[idx]
                tmp[idx] = 1.
            else:
                xj = x[index]
                c = np.mean(xj, axis=0)
                clusters[j] = c / np.linalg.norm(c)

        if np.abs(prev_sim - sim) / (np.abs(sim) + 1e-20) < 1e-4:
            break
        prev_sim = sim

    return clusters, assign


class MatrixInverseSqrt:
    """
    Matrix inverse square root for a symmetric definite positive matrix
    """
    
    @staticmethod
    def forward(input, eps=1e-2):
        """
        Forward pass of the matrix inverse square root operation.

        Args:
            input (ndarray): Input symmetric definite positive matrix.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-2.

        Returns:
            ndarray: The result of the matrix inverse square root operation.
        """
        e, v = np.linalg.eigh(input)
        e = np.maximum(e, 0)  # Clamp eigenvalues to be non-negative
        e_sqrt = np.sqrt(e + eps)
        e_rsqrt = 1.0 / e_sqrt
        output = np.dot(v, np.dot(np.diag(e_rsqrt), v.T))
        return output
    
    @staticmethod
    def backward(ctx,grad_output):
        """
        Backward pass of the matrix inverse square root operation.

        Args:
            ctx (object): Context object containing saved variables.
            grad_output (ndarray): Gradient of the loss with respect to the output.

        Returns:
            tuple: A tuple containing:
                ndarray: Gradient of the loss with respect to the input.
                None: The gradient with respect to the saved variables (unused).
        """
        e_sqrt, v = ctx.saved_variables
        ei = np.tile(e_sqrt, (v.shape[0], 1))
        ej = np.tile(e_sqrt[:, np.newaxis], (1, v.shape[1]))
        f = 1.0 / ((ei + ej) * ei * ej)
        grad_input = -np.dot(v, np.dot(f * np.dot(v.T, np.dot(grad_output, v)), v.T))
        return grad_input, None

def matrix_inverse_sqrt(input, eps=1e-2):
    """
    Wrapper function for MatrixInverseSqrt.

    Args:
        input (ndarray): Input symmetric definite positive matrix.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-2.

    Returns:
        ndarray: The result of the matrix inverse square root operation.
    """
    return MatrixInverseSqrt.forward(input, eps)


#### Vanilla Kmeans

class KMeans0:
    def __init__(self,k=3,dist_type='euc',tol=1e-4):
        self.k = k
        self.centroids = None
        self.dist_type=dist_type
        self.tol=tol
    
    
    @staticmethod
    def dist(data_point,centroids,type):
        if type == 'euc':
            return np.sqrt(np.sum((centroids-data_point)**2,axis=1))
        if type == 'cos':
            return -(data_point@centroids.T)/(np.linalg.norm(data_point)*np.linalg.norm(centroids.T))


    
    def fit(self,X,max_iter=100):
        self.centroids = np.random.uniform(np.amin(X,axis=0),np.amax(X,axis=0),size=(self.k,X.shape[1]))

        for _ in range(max_iter):
            y=[]
            for data_point in X:
                distances = KMeans0.dist(data_point,self.centroids,self.dist_type)
                cluster_num = np.argmin(distances)
                y.append(cluster_num)
            y=np.array(y)

            cluster_indices=[]

            for i in range(self.k):
                cluster_indices.append(np.argwhere(y==i))
            
            cluster_centers=[]

            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices],axis=0)[0])
            
            if np.max(self.centroids - np.array(cluster_centers))<self.tol:
                break
            else:
                self.centroids = np.array(cluster_centers)
            
        return y