import numpy as np
import argparse
import logging
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .csv file')
parser.add_argument('validation_input', type=str,
                    help='path to validation input .csv file')
parser.add_argument('train_out', type=str,
                    help='path to store prediction on training data')
parser.add_argument('validation_out', type=str,
                    help='path to store prediction on validation data')
parser.add_argument('metrics_out', type=str,
                    help='path to store training and testing metrics')
parser.add_argument('num_epoch', type=int,
                    help='number of training epochs')
parser.add_argument('hidden_units', type=int,
                    help='number of hidden units')
parser.add_argument('init_flag', type=int, choices=[1, 2],
                    help='weight initialization functions, 1: random')
parser.add_argument('learning_rate', type=float,
                    help='learning rate')
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')


def args2data(parser):
    """
    Parse argument, create data and label.
    :return:
    X_tr: train data (numpy array)
    y_tr: train label (numpy array)
    X_te: test data (numpy array)
    y_te: test label (numpy array)
    out_tr: predicted output for train data (file)
    out_te: predicted output for test data (file)
    out_metrics: output for train and test error (file)
    n_epochs: number of train epochs
    n_hid: number of hidden units
    init_flag: weight initialize flag -- 1 means random, 2 means zero
    lr: learning rate
    """

    # # Get data from arguments
    out_tr = parser.train_out
    out_te = parser.validation_out
    out_metrics = parser.metrics_out
    n_epochs = parser.num_epoch
    n_hid = parser.hidden_units
    init_flag = parser.init_flag
    lr = parser.learning_rate

    X_tr = np.loadtxt(parser.train_input, delimiter=',')
    y_tr = X_tr[:, 0].astype(int)
    X_tr[:, 0] = 1.0 #add bias terms

    X_te = np.loadtxt(parser.validation_input, delimiter=',')
    y_te = X_te[:, 0].astype(int)
    X_te[:, 0]= 1.0 #add bias terms


    return (X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics,
            n_epochs, n_hid, init_flag, lr)



def shuffle(X, y, epoch):
    """
    Permute the training data for SGD.
    :param X: The original input data in the order of the file.
    :param y: The original labels in the order of the file.
    :param epoch: The epoch number (0-indexed).
    :return: Permuted X and y training data for the epoch.
    """
    np.random.seed(epoch)
    N = len(y)
    ordering = np.random.permutation(N)
    return X[ordering], y[ordering]

def random_init(shape):
    """
    Randomly initialize a numpy array of the specified shape
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    # DO NOT CHANGE THIS
    np.random.seed(np.prod(shape))

    # Implement random initialization here
    return np.random.uniform(low=-0.1,high=0.1,size=shape)


def zero_init(shape):
    """
    Initialize a numpy array of the specified shape with zero
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    return np.zeros(shape)

def load_csv_dataset(file):
    """
    Loads raw data and returns a ndarray containing the data.

    Parameters:
        file (str): File path to the dataset csv file.

    Returns:
        An N x M np.ndarray.
    """
    dataset = np.loadtxt(file, delimiter=',', comments=None, encoding='utf-8')
    return dataset


class NN(object):
    def __init__(self, lr, n_epoch, weight_init_fn, input_size, hidden_size, output_size):
        """
        Initialization
        :param lr: learning rate
        :param n_epoch: number of training epochs
        :param weight_init_fn: weight initialization function
        :param input_size: number of units in the input layer
        :param hidden_size: number of units in the hidden layer
        :param output_size: number of units in the output layer
        """
        self.lr = lr
        self.n_epoch = n_epoch
        self.weight_init_fn = weight_init_fn
        self.n_input = input_size
        self.n_hidden = hidden_size
        self.n_output = output_size

        # initialize weights and biases for the models
        # HINT: pay attention to bias here
        weight1 = weight_init_fn([hidden_size, input_size-1])
        self.w1 = np.insert(weight1,0,0,axis=1)
        weight2 = weight_init_fn([output_size, hidden_size])
        self.w2 = np.insert(weight2,0,0,axis=1)

        # initialize parameters for adagrad
        # self.epsilon = 1e-5
        # self.grad_sum_w1 =
        # self.grad_sum_w2 =

        # feel free to add additional attributes


def print_weights(nn):
    """
    An example of how to use logging to print out debugging infos.

    Note that we use the debug logging level -- if we use a higher logging
    level, we will log things with the default logging configuration,
    causing potential slowdowns.

    Note that we log NumPy matrices on separate lines -- if we do not do this,
    the arrays will be turned into strings even when our logging is set to
    ignore debug, causing potential massive slowdowns.
    :param nn: your model
    :return:
    """
    logging.debug(f"shape of w1: {nn.w1.shape}")
    logging.debug(nn.w1)
    logging.debug(f"shape of w2: {nn.w2.shape}")
    logging.debug(nn.w2)


def forward(X, nn):
    """
    Neural network forward computation.
    Follow the pseudocode!
    :param X: input data
    :param nn: neural network class
    :return: output probability
    """
    a = linear(nn.w1,X)
    z = sigmoid(a)
    z = np.insert(z, 0, 1)
    b = linear(nn.w2, z)
    y_predicted = softmax(b)
    return y_predicted, a, b, z

def backward(x, Y, y_predicted, nn, z):
    """
    Neural network backward computation.
    Follow the pseudocode!
    :param X: input data
    :param y: label
    :param y_hat: prediction
    :param nn: neural network class
    :return:
    d_w1: gradients for w1
    d_w2: gradients for w2
    """
    alpha = nn.w1
    beta = nn.w2
    beta_star = np.delete(beta, 0, 1)
    z_star = np.delete(z, 0, 1)

    db = dldb(Y, y_predicted)
    # print("db is: ")
    # print(db)
    dbeta = dldbeta(db, z)
    # print("dbeta is: ")
    # print(dbeta)
    dz = dldz(db, beta_star)
    # print("dz is: ")
    # print(dz)
    da = dlda(dz, dzda(z_star))
    # print("da is: ")
    # print(da)
    dalpha = dldalpha(da, x)
    # print("dalpha is: ")
    # print(dalpha)
    return dalpha, dbeta


def test(X, y, nn):
    """
    Compute the label and error rate.
    :param X: input data
    :param y: label
    :param nn: neural network class
    :return:
    labels: predicted labels
    error_rate: prediction error rate
    """
    labels = []
    error = 0
    for i in range(X.shape[0]):
        y_predicted, a, b, z = forward(X[i],nn)
        # print("y hat in testing is: ")
        # print(y_predicted)
        m = np.unravel_index(y_predicted.argmax(), y_predicted.shape)
        labels.append(m[0])
    for i in range(len(labels)):
        if labels[i] != y[i]:
            error +=1
    error_rate = error/len(y)
    return labels,error_rate


def train(X_tr, y_tr, nn, x_valid, y_valid):
    """
    Train the network using SGD for some epochs.
    :param X_tr: train data
    :param y_tr: train label
    :param nn: neural network class
    """
    s_alpha = np.zeros_like(nn.w1)
    s_beta = np.zeros_like(nn.w2)
    epsilon = 1e-5
    for i in range(nn.n_epoch):
        x_shuffled, Y_shuffled = shuffle(X_tr, y_tr, i)
        for j in range(x_shuffled.shape[0]):
            x = x_shuffled[j]
            Y = Y_shuffled[j]
            x = np.array([x])
            Y = np.array([Y])
            y_predicted, a, b, z = forward(x, nn)
            y_predicted = np.array([y_predicted])
            a = np.array([a])
            b = np.array([b])
            z = np.array([z])

            # print("a is: ")
            # print(a)
            # print("z is: ")
            # print(z)
            # print("b is: ")
            # print(b)
            # print("y hat is: ")
            # print(y_predicted)
            dalpha, dbeta = backward(x, Y, y_predicted,nn,z)
            s_alpha = s_alpha + dalpha * dalpha
            s_beta = s_beta + dbeta * dbeta
            nn.w1 = nn.w1 - (lr / np.sqrt(s_alpha + epsilon)) * dalpha
            nn.w2 = nn.w2 - (lr / np.sqrt(s_beta + epsilon)) * dbeta
            # print("new alpha is: ")
            # print(nn.w1)
            # print("new beta is: ")
            # print(nn.w2)
        ce_train = 0
        for w in range(x_shuffled.shape[0]):
            y_predicted_train,a,b,z = forward(np.array([x_shuffled[w]]),nn)
            ce_train += cross_entropy(np.array([Y_shuffled[w]]),y_predicted_train)
        ce_valid = 0
        for q in range(x_valid.shape[0]):
            y_predicted_val, a, b, z = forward(np.array([x_valid[q]]), nn)
            ce_valid += cross_entropy(np.array([y_valid[q]]),y_predicted_val)
        train_entropy.append(ce_train/x_shuffled.shape[0])
        valid_entropy.append(ce_valid/x_valid.shape[0])

def sigmoid(p):
    return 1 / (1+np.exp(-p))

def linear(m: np.ndarray, n:np.ndarray):
    return (m.dot(n.T)).T

def softmax(b:np.ndarray):
    return np.exp(b) / np.sum(np.exp(b), axis=0)

def cross_entropy(Y, y_hat):
    N = Y.shape[0]
    ce = -np.sum(Y*np.log(y_hat))/N
    return ce

def dldb(Y, y_hat):
    return y_hat - Y

def dldbeta(db,z):
    return np.dot(db.T, z)

def dldz(db,beta):
    return np.dot(db,beta)

def dlda(dz,dadz):
    return dz * dadz

def dldalpha(da,X):
    return da.T * X

def dzda(z):
    return z*(1-z)

def write_labels(path, labels):
    '''
    write the label files
    :param data: np array of data
    :param path: output path
    :param majority_label: most common label
    :return: None
    '''
    with open(path, mode='w') as file:
        for label in labels:
            file.write(str(label) + '\n')
    file.close()

def write_metrics(train_entroy, valid_entropy, error_train, error_valid, metrics):
    '''
    write the metrics file
    :param error_train: error rate of training set
    :param error_test: error rate of testing set
    :param metrics: path to the metrics file
    :return: None
    '''
    with open(metrics, mode='w') as file:
        for i in range(len(train_entroy)):
            file.write("epoch=" + str(i+1) + " crossentropy(train): " + str(train_entroy[i]) + "\n")
            file.write("epoch=" + str(i + 1) + " crossentropy(validation): " + str(valid_entropy[i]) + "\n")
        file.write("error(train): " + str(error_train) + '\n')
        file.write("error(validation): " + str(error_valid))





if __name__ == "__main__":

    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)04d} %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    logging.debug('*** Debugging Mode ***')
    # Note: You can access arguments like learning rate with args.learning_rate

    # initialize training / test data and labels
    path_train = args.train_input
    train_set = load_csv_dataset(path_train)

    path_validation = args.validation_input
    valid_set = load_csv_dataset(path_validation)

    Y_train = train_set[:,0]
    x_train = np.delete(train_set,0,1)
    x_train = np.insert(x_train, 0, 1, axis=1)

    Y_valid = valid_set[:,0]
    x_valid = np.delete(valid_set,0,1)
    x_valid = np.insert(x_valid, 0, 1, axis=1)

    # one hot encoder
    Y_train = Y_train.astype(int)
    Y_encoded = np.zeros((Y_train.size, 10))
    Y_encoded[np.arange(Y_train.size), Y_train] = 1

    Y_valid = Y_valid.astype(int)
    Y_encoded_valid = np.zeros((Y_valid.size, 10))
    Y_encoded_valid[np.arange(Y_valid.size), Y_valid] = 1

    lr = args.learning_rate
    num_epoch = args.num_epoch
    hidden_units = args.hidden_units
    init_flag = args.init_flag

    global train_entropy
    global valid_entropy

    train_entropy = []
    valid_entropy = []

    # Build model
    if init_flag == 2:
        my_nn = NN(lr,num_epoch,zero_init,len(x_train[1]),hidden_units,10)
    else:
        my_nn = NN(lr, num_epoch, random_init, len(x_train[1]), hidden_units, 10)

    # train model with train set
    train(x_train,Y_encoded,my_nn,x_valid,Y_encoded_valid)

    # test model and get predicted labels and errors
    labels_train,error_rate_train = test(x_train,Y_train,my_nn)

    # Build model
    # if init_flag == 2:
    #     my_nn = NN(lr,num_epoch,zero_init,len(x_train[1]),hidden_units,10)
    # else:
    #     my_nn = NN(lr, num_epoch, random_init, len(x_train[1]), hidden_units, 10)
    # train with validation set
    # train(x_valid,Y_encoded_valid,my_nn,"valid")

    # test with validation set
    labels_valid,error_rate_valid = test(x_valid,Y_valid,my_nn)

    # print("train entropy: ")
    # print(train_entropy)
    #
    # print("valid entropy: ")
    # print(valid_entropy)

    # write predicted label and error into file
    train_out = args.train_out
    valid_out = args.validation_out

    write_labels(train_out,labels_train)
    write_labels(valid_out,labels_valid)

    metrics_out = args.metrics_out
    write_metrics(train_entropy,valid_entropy,error_rate_train,error_rate_valid,metrics_out)



