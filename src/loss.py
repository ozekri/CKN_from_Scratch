import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        """
        Initialize the Cross Entropy Loss object.

        Attributes:
            item (float or None): The value of the loss.
        """
        self.item = None
    
    def __call__(self, output, target):
        """
        Compute the Cross Entropy Loss between the output and target.

        Args:
            output (numpy.ndarray): The output of the model.
            target (numpy.ndarray): The target labels.

        Returns:
            CrossEntropyLoss: The Cross Entropy Loss object.
        """
        self.output = output
        target = target.astype(int)
        self.target = target
        # Check if the target is one-hot encoded
        if len(target.shape) > 1 and target.shape[1] > 1:
            target = target
        else:
            num_classes = np.max(target) + 1
            target = np.eye(num_classes)[target]

        # Compute softmax probabilities with a trick for numerical stability
        exp_scores = np.exp(output - np.max(output, axis=1, keepdims=True))
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # Select the probabilities corresponding to the target classes
        target_probabilities = probabilities[np.arange(len(output)), self.target]

        # Compute the average loss over examples
        self.loss_value = -np.mean(np.log(target_probabilities + 1e-15)) # Add a small value to avoid division by zero
        self.probabilities = probabilities
        self.item = self.loss_value
        
        return self

    def backward(self,x):
        """
        Compute the gradients of the loss function with respect to the model parameters.

        Args:
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Gradients of the loss function with respect to the model parameters.
        """
        gradient = self.probabilities.copy()
        gradient[np.arange(len(self.probabilities)), self.target] -= 1
        gradient /= len(self.probabilities)
        
        dW = np.dot(gradient.T,x)
        db = np.sum(gradient, axis=0, keepdims=True).T
        return dW, db


LOSS = {
    'ce': CrossEntropyLoss,
}


# You can uncomment the end of the code to observe this loss decreasing on synthetic examples. 

"""
import matplotlib.pyplot as plt
# Dummy dataset
X = np.array([[0.2, 0.3], [0.4, 0.5], [0.7, 0.9], [0.8, 0.2]])
y = np.array([0, 1, 1, 0])  # Binary classification

# Initialize weights randomly
W = np.random.randn(2, 2)
b = np.zeros((1, 2))

# Hyperparameters
learning_rate = 0.1
num_epochs = 100

# Loss function
loss_function = CrossEntropyLoss()

# Training loop
losses = []
for epoch in range(num_epochs):
    # Forward pass
    logits = np.dot(X, W) + b
    loss = loss_function(logits, y)
    losses.append(loss.item)

    # Backward pass
    dW, db = loss_function.backward(X)
    W -= learning_rate * dW
    b -= learning_rate * np.squeeze(db)  # Adjust the shape of db


    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item}')

# Plot the loss curve
plt.plot(range(1, num_epochs + 1), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

"""