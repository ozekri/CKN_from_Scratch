import argparse
from model import SUPMODELS
from loss import LOSS
import numpy as np
import pandas as pd
import optimizers
from utils import accuracy_score, countX, count_parameters

from timeit import default_timer as timer


def create_dataset(data_path):
    """
    Load dataset from CSV files and reshape them into appropriate shapes.

    Args:
        data_path (str): Path to the directory containing dataset CSV files.

    Returns:
        None: Global variables Xtr, Xte, Ytr, Xval, and Yval are updated with the loaded dataset.
    """
    global Xtr
    global Xte
    global Ytr
    global Xval
    global Yval
    #global Xtr_val
    #global Ytr_val

    # Load training data
    Xtr = np.array(pd.read_csv(f'{data_path}Xtr.csv',header=None,sep=',',usecols=range(3072),encoding='unicode_escape'))
    Xtr = np.array(Xtr).reshape(5000,3,32,32)
    print("X_train shape",Xtr.shape)   
    # Load training labels
    Ytr = np.array(pd.read_csv(f'{data_path}Ytr.csv',sep=',',usecols=[1])).squeeze()

    # Load testing data
    Xte = np.array(pd.read_csv(f'{data_path}Xte.csv',header=None,sep=',',usecols=range(3072)))
    Xte = np.array(Xte).reshape(2000,3,32,32)

    # Some slicing used during the experimentation phase..
    #Xtr = np.array(Xtr).reshape(5000, 3,32, 32).swapaxes(1,2).swapaxes(2,3)
    #Xte = np.array(Xte).reshape(2000, 3,32, 32).swapaxes(1,2).swapaxes(2,3)

    #Xval,Yval =Xtr[1000:1020], Ytr[1000:1020]
    #Xtr_val,Ytr_val = Xtr[4500:], Ytr[4500:]
    #Xtr_val,Ytr_val = Xtr[4900:], Ytr[4900:]
    #Xtr,Ytr =Xtr[0:4500],Ytr[0:4500]
    #Xtr,Ytr =Xtr[0:500],Ytr[0:500]
    #Xtr,Ytr =Xtr[0:100],Ytr[0:100]
    #Xval,Yval =Xtr[10:20], Ytr[10:20]
    #Xtr,Ytr =Xtr[0:10],Ytr[0:10]

class NumpyDataLoader:
    """
    A data loader for iterating over batches of data stored in NumPy arrays.
    """
    def __init__(self, dataset, batch_size=1, shuffle=False):
        """
        Initialize the NumpyDataLoader object.

        Args:
            dataset (tuple or list): Tuple or list containing features and labels.
            batch_size (int, optional): Batch size. Defaults to 1.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(self.dataset[0][0])
        self.num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.is_labeled = isinstance(dataset[0], tuple) and len(dataset[0]) == 2

    def __len__(self):
        """
        Get the number of batches.

        Returns:
            int: Number of batches.
        """
        return self.num_batches

    def __iter__(self):
        """
        Iterate over batches.

        Yields:
            tuple or ndarray: Batch features and labels if labeled, otherwise batch features.
        """
        for i in range(self.num_batches):
            batch_indices = self.indices[i * self.batch_size : (i + 1) * self.batch_size]
            if self.is_labeled:
                batch_features = np.array([self.dataset[0][0][idx] for idx in batch_indices])
                batch_labels = np.array([self.dataset[0][1][idx] for idx in batch_indices])
                yield batch_features, batch_labels
            else:
                yield np.array([self.dataset[0][idx] for idx in batch_indices])

def create_dataloader_generator(X, y=None, batch_size=1, shuffle=True):
    """
    Creates a DataLoader similar to PyTorch's with data and labels using NumPy.

    Args:
        X (numpy.ndarray): Data.
        y (numpy.ndarray, optional): Labels. Default None.
        batch_size (int, optional): Mini-batch size. Default 1.
        shuffle (bool, optional): Shuffle data at each epoch. Default True.

    Returns:
        Generator: A generator that yields mini-batches of data and labels (if y is provided).
    """
    num_samples = len(X)
    indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(indices)

    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_X = X[batch_indices]
        if y is not None:
            batch_y = y[batch_indices]
            yield batch_X, batch_y
        else:
            yield batch_X

def create_dataloader_np(batch_size,shuffle=True):
    """
    Create NumPy data loaders for training and testing datasets.

    Args:
        batch_size (int): The batch size.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

    Returns:
        tuple: A tuple containing the training and testing data loaders.
    """
    dataloader_train = create_dataloader_generator(Xtr,Ytr, batch_size=batch_size, shuffle=shuffle)
    dataloader_test = create_dataloader_generator(Xte, batch_size=batch_size, shuffle=shuffle)
    return dataloader_train, dataloader_test


def load_args():
    """
    Load command-line arguments for the CKN Kaggle Kernel Methods Challenge 2023/2024.

    Returns:
        argparse.Namespace: The parsed arguments.

    Example:
        # Load command-line arguments
        args = load_args()
    """
    parser = argparse.ArgumentParser(
        description="CKN for Kaggle Kernel Methods Challenge 2023/2024")
    parser.add_argument('--datapath', type=str, default='../data/',
                        help='path to the dataset')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--model', default='ckn3', choices=list(SUPMODELS.keys()), help='which model to use')
    parser.add_argument(
        '--sampling-patches', type=int, default=150000, help='number of subsampled patches for initilization')
    parser.add_argument('--lr', default=0.004, type=float, help='initial learning rate')
    parser.add_argument('--epochs', default=30, type=int, help='number of epochs')
    parser.add_argument('--alpha', default=0.001, type=float, help='regularization parameter')
    parser.add_argument('--loss', default='ce', choices=list(LOSS.keys()), help='loss function')
    args = parser.parse_args()
    
    return args

def sup_train(model, dataloader, args):
    """
    Train the supervised CKN model.

    Args:
        model (CKNet): The supervised CKN model.
        dataloader (dict): A dictionary containing the data loaders for training, testing, and validation.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        float: The best validation accuracy achieved during training.
    """
    criterion = LOSS[args.loss]()
    alpha = args.alpha * model.out_features / args.batch_size
    optimizer = optimizers.SGD([{'params': model.classifier.parameters(), 'weight_decay': alpha}], lr=args.lr, momentum=0.9)
    #lr_scheduler = optimizers.MultiStepLR(optimizer, [15], gamma=0.1)
    model.unsup_train_ckn(
        dataloader['init'], args.sampling_patches)
    epoch_loss = None
    best_loss = float('inf')
    best_acc = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch + 1, args.epochs))
        print('-' * 10)
        if epoch ==15:
            optimizer.param_groups[0]['lr'] = 0.1*args.lr
        print(f"Learning Rate - {optimizer.param_groups[0].get('lr', optimizer.defaults['lr'])}")
        if epoch >= 0:
            tic = timer()
            model.unsup_train_classifier(
                dataloader['train'])
            model.normalize()
            toc = timer()
            print('Last layer trained, elapsed time: {:.2f}s'.format(toc - tic))
            optimizer.param_groups[-1]['weight_decay'] = model.classifier.real_alpha
        running_loss = 0.0
        running_acc = 0
        tic = timer()    
        for data, target in dataloader['train']:
            size = data.shape[0]
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            print("LOSS",loss.item)
            pred = np.argmax(output, axis=1)
            print("PRED",pred)
            model.classifier.weight.gradients,model.classifier.bias.gradients = loss.backward(model.representation(data))
            optimizer.step()
            model.normalize()
            running_loss += loss.item * size
            running_acc += np.sum(pred == target)
        toc = timer()

        #model.features.changemode('inf')
        #output = model(Xtr_val)
        #Ytrr_valpred = np.argmax(output, axis=1)
        #print("Accuracy score", accuracy_score(Ytr_val,Ytrr_valpred))

        model.features.changemode('train')

        epoch_loss = running_loss / (args.batch_size*len(dataloader['train']))
        epoch_acc = running_acc / (args.batch_size*len(dataloader['train']))
        
        print('{} Loss: {:.4f} Acc: {:.2f}% Elapsed time: {:.2f}s'.format(
                'train', epoch_loss, epoch_acc * 100, toc - tic))

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_loss = epoch_loss
            best_epoch = epoch
    print('Best epoch: {}'.format(best_epoch + 1))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val loss: {:4f}'.format(best_loss))

    return best_acc


def main():
    """
    Main function for training and evaluating the supervised CKN model.

    It loads the command-line arguments, creates the dataset, initializes the model, 
    trains the model, and evaluates its performance.

    Example:
        main()
    """
    args = load_args()
    print(args)
    create_dataset(args.datapath)
    print(Xtr.shape,Ytr.shape)

    print("Batch Size",args.batch_size)
    init_dset = NumpyDataLoader([(Xtr,Ytr)],batch_size=args.batch_size,shuffle=True)
    train_dset = NumpyDataLoader([(Xtr,Ytr)],batch_size=args.batch_size)

    model = SUPMODELS[args.model](alpha=args.alpha)
    print(model)
    print(len(Xte))
    nb_params = count_parameters(model)
    print('number of paramters: {}'.format(nb_params))
    #test_dset = NumpyDataLoader([Xval],batch_size=10)
    #train_val = NumpyDataLoader([(Xval,Yval)],batch_size=args.batch_size)
    #data_loader = {'train': train_dset, 'test': test_dset,'val' : train_val,'init' :init_dset}
    data_loader = {'train': train_dset,'init' :init_dset}
    tic = timer()
    score = sup_train(model,data_loader,args)
    toc = timer()
    training_time = (toc - tic) / 60
    print("Final accuracy: {:6.2f}%, elapsed time: {:.2f}min".format(score * 100, training_time))


    model.features.changemode('inf')

    """
    output = model(Xval)
    Yt_valpred = np.argmax(output, axis=1)
    print(Yt_valpred)
    print(Yval)
    print(accuracy_score(Yval,Yt_valpred))
    """

    
    output2 = model(Xte)
    Yte = np.argmax(output2, axis=1)
    for cl in range(0,10):
        print('class {} has occurred {} times'.format(cl, countX(Yte, cl)))

    Yte = {'Prediction' : Yte}
    dataframe = pd.DataFrame(Yte)
    dataframe.index += 1

    dataframe.to_csv('Yte_pred2.csv',index_label='Id')

if __name__ == '__main__':
    main()