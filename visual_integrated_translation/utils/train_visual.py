
from create_visual_dataset import *


# sigmoid function for the input x:  if x is a vector -> element-wise sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))


# feedforward network for visual weighing. Returns a score vector of each target words
# (normalized by sigmoid) for the given visual contexts
def feedforward(weight_matrix, visual_vec, bias_vec=[], activation=sigmoid):
    if bias_vec != []:
        return activation(np.dot(weight_matrix,visual_vec)+bias_vec)
    else:
        return activation(np.dot(weight_matrix,visual_vec))


# training
def train_visual(target_id_file, visual_vec_file, target_vocab_file, target_vocab_size, learning_rate=0.05, num_epochs=10000):

    ########## prepare data set ##########
    # create target vocabulary dict and data set
    with open(target_vocab_file, "r") as f:
        vocab_dict = dict(enumerate(f.read().splitlines())) # key=id, value=word
    data_set = get_data_set(target_id_file, visual_vec_file, len(vocab_dict))
    X = data_set[0] # visual context vectors
    T = data_set[1] # target sentences


    ########## initialize parameters ##########
    # length of visual context vectors
    visual_vec_size = len(X[0])
    # initialize weights randomly with mean 0
    W = 2*np.random.random((target_vocab_size,visual_vec_size)) - 1
    # initialize biases
    bias_vec = np.zeros(target_vocab_size)


    ########## start learning ##########
    for iter in range(num_epochs):
        error = 0
        for i in range(0, len(X)):
            # output vector with sigmoid activation
            vis_vec = X[i]
            tar_vec = T[i]
            out_vec = feedforward(W, vis_vec, bias_vec)

            # cross entropy loss
            #error += - sum([tar_vec[j]*log(out_vec[j]) + (1-tar_vec[j])*log(1-out_vec[j]) for j in range(0, len(out_vec)-1)])

            # update weights and biases by partial derivatives
            for c in range(0, len(vocab_dict)):
                for j in range(0, len(W[c])):
                    W[c][j] -= learning_rate * vis_vec[j]*(out_vec[c] - tar_vec[c]) # d(entroy_loss)/d(weight) = (output - desired_output)*connected_input_to_the_weight
                    bias_vec[c] -= learning_rate * (out_vec[c] - tar_vec[c]) # d(entropy_loss)/d(bias) = (output - desired_output)

        # if iter % 1000 == 0:
        #     print "cross entropy loss at epoch {0}:".format(str(iter)), error/len(X)

    return W, bias_vec


