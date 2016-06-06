
import argparse
import sys
from utils.train_visual import *


parser = argparse.ArgumentParser()
parser.add_argument('--target_id_file', default='data/english_to_japanese/train.ids30.ja', type=str,
                   help='Path to the target sentence file (token ids) in one-sentence-per-line format')
parser.add_argument('--visual_vec_file', default='data/english_to_japanese/visual_train.txt', type=str,
                    help='Path to the visual context file in one-vector-per-line format')
parser.add_argument('--target_vocab_file', default='data/english_to_japanese/vocab30.ja', type=str,
                    help='Path to the target vocab file in one-word-per-line format')
parser.add_argument('--target_vocab_size', default=30, type=int,
                    help='Number of the target words')
parser.add_argument('--learning_rate', type=float, default=0.05,
                    help='Learning rate')
parser.add_argument('--num_epochs', type=int, default=10000,
                    help='Number of epochs')
args = parser.parse_args()



# softmax function for the given input vector x
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# optimize weights and biases
print "Number of epochs: " + str(args.num_epochs)
W, bias_vec = train_visual(args.target_id_file, args.visual_vec_file, args.target_vocab_file, args.target_vocab_size,
                           learning_rate=args.learning_rate, num_epochs=args.num_epochs)

# create vocabulary dict for getting back ids to words
with open(args.target_vocab_file, "r") as f:
    vocab_dict = dict(enumerate(f.read().splitlines())) # key=id, value=word



# user input
print "Type a visual context vector, e.g. 1 0 1 for size 3"
sys.stdout.write("> ")
sys.stdout.flush()
visual_vec_input = sys.stdin.readline()

# start decoding
while visual_vec_input:
    # create visual context vector
    visual_vec = np.array([int(num) for num in visual_vec_input.split()])

    # compute score vector
    out_vec = feedforward(W, visual_vec, bias_vec)

    # create dict for computed scores for sorted result
    result_dict = dict(enumerate(out_vec))

    # print scores
    for id, score in sorted(result_dict.items(), key=lambda x:x[1], reverse=True):
        if id in vocab_dict:
            print score, vocab_dict[id]

    print "> ",
    sys.stdout.flush()
    visual_vec_input = sys.stdin.readline()
