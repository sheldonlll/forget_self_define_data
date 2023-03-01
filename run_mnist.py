from __future__ import print_function
import argparse
import numpy as np
import numpy.random as npr
import time
import os
import sys
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms


# Format time for printing purposes
def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s


# Setup basic CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        if args.no_dropout:
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
        else:
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))

        if not args.no_dropout:
            x = F.dropout(x, training=self.training)

        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Train model for one epoch
#
# example_stats: dictionary containing statistics accumulated over every presentation of example
#
def train(args, model, device, trainset, optimizer, epoch, example_stats):
    train_loss = 0
    correct = 0
    total = 0
    batch_size = args.train_batch_size

    model.train()

    # Get permutation to shuffle trainset
    trainset_permutation_inds = npr.permutation(
        np.arange(len(trainset.train_labels)))

    for batch_idx, batch_start_ind in enumerate(
            range(0, len(trainset.train_labels), batch_size)):

        # Get trainset indices for batch
        batch_inds = trainset_permutation_inds[batch_start_ind:
                                               batch_start_ind + batch_size]

        # Get batch inputs and targets, transform them appropriately
        transformed_trainset = []
        for ind in batch_inds:
            transformed_trainset.append(trainset.__getitem__(ind)[0])
        inputs = torch.stack(transformed_trainset)
        targets = torch.LongTensor(
            np.array(trainset.train_labels)[batch_inds].tolist())

        # Map to available device
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward propagation, compute loss, get predictions
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        _, predicted = torch.max(outputs.data, 1)

        # Update statistics and loss
        acc = predicted == targets
        for j, index in enumerate(batch_inds):

            # Get index in original dataset (not sorted by forgetting)
            index_in_original_dataset = train_indx[index]

            # Compute missclassification margin
            output_correct_class = outputs.data[
                j, targets[j].item()]  # output for correct class
            sorted_output, _ = torch.sort(outputs.data[j, :])
            if acc[j]:
                # Example classified correctly, highest incorrect class is 2nd largest output
                output_highest_incorrect_class = sorted_output[-2]
            else:
                # Example misclassified, highest incorrect class is max output
                output_highest_incorrect_class = sorted_output[-1]
            margin = output_correct_class.item(
            ) - output_highest_incorrect_class.item()

            # Add the statistics of the current training example to dictionary
            index_stats = example_stats.get(index_in_original_dataset,
                                            [[], [], []])
            # print(f"loss: {loss.item()}")
            # index_stats[0].append(loss[j].item())
            index_stats[0].append(loss.item())
            index_stats[1].append(acc[j].sum().item())
            index_stats[2].append(margin)
            example_stats[index_in_original_dataset] = index_stats

        # Update loss, backward propagate, update optimizer
        loss = loss.mean()
        train_loss += loss.item()
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write(
            '| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%' %
            (epoch, args.epochs, batch_idx + 1,
             (len(trainset) // batch_size) + 1, loss.item(),
             100. * correct.item() / total))
        sys.stdout.flush()

        # Add training accuracy to dict
        index_stats = example_stats.get('train', [[], []])
        index_stats[1].append(100. * correct.item() / float(total))
        example_stats['train'] = index_stats


# Evaluate model predictions on heldout test data
#
# example_stats: dictionary containing statistics accumulated over every presentation of example
#
def test(args, model, device, testset, example_stats):
    test_loss = 0
    correct = 0
    total = 0
    test_batch_size = args.test_batch_size

    model.eval()

    for batch_idx, batch_start_ind in enumerate(
            range(0, len(testset.test_labels), test_batch_size)):

        # Get batch inputs and targets
        transformed_testset = []
        for ind in range(
                batch_start_ind,
                min(
                    len(testset.test_labels),
                    batch_start_ind + test_batch_size)):
            transformed_testset.append(testset.__getitem__(ind)[0])
        inputs = torch.stack(transformed_testset)
        targets = torch.LongTensor(
            np.array(testset.test_labels)[batch_start_ind:batch_start_ind +
                                          test_batch_size].tolist())

        # Map to available device
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward propagation, compute loss, get predictions
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss = loss.mean()
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Add test accuracy to dict
    acc = 100. * correct.item() / total
    index_stats = example_stats.get('test', [[], []])
    index_stats[1].append(100. * correct.item() / float(total))
    example_stats['test'] = index_stats
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %
          (epoch, loss.item(), acc))

class Linear_SVM(nn.Module):
    def __init__(self, n_features, n_classes):
        super(Linear_SVM, self).__init__()
        self.fc = nn.Linear(n_features, n_classes)
  
    def forward(self, x):
        x = x.view(x.size(0), -1)
        outputs = self.fc(x)
        return outputs

parser = argparse.ArgumentParser(description='training MNIST')
parser.add_argument(
    '--epochs',
    type=int,
    default=20,
    metavar='N',
    help='number of epochs to train (default: 200)')
parser.add_argument(
    '--lr',
    type=float,
    default=0.02,
    metavar='LR',
    help='learning rate (default: 0.01)')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--sorting_file',
    default="none",
    help='name of a file containing order of examples sorted by a certain metric (default: "none", i.e. not sorted)'
)
parser.add_argument(
    '--remove_n',
    type=int,
    default=0,
    help='number of sorted examples to remove from training')
parser.add_argument(
    '--keep_lowest_n',
    type=int,
    default=0,
    help='number of sorted examples to keep that have the lowest metric score, equivalent to start index of removal; if a negative number given, remove random draw of examples'
)
parser.add_argument(
    '--input_dir',
    default='mnist_results/',
    help='directory where to read sorting file from')
parser.add_argument(
    '--output_dir', required=True, help='directory where to save results')
parser.add_argument(
    "--n_samples", required=True, help="n_samples", type = int
)
parser.add_argument(
    "--n_features", required=True, help="n_features", type = int
)
parser.add_argument(
    "--n_informative", required=True, help="n_informative", type = int
)
parser.add_argument(
    "--n_redundant", required=True, help="n_redundant", type = int
)
parser.add_argument(
    "--ratio", required=True, help="ratio", type = float
)
parser.add_argument(
    "--random_por", required=True, help="random_por", type = float
)
parser.add_argument(
    "--n_classes", required=True, help="n_classes", type = int
)
parser.add_argument(
    "--n_clusters_per_class", required=True, help="n_clusters_per_class", type = int
)
parser.add_argument(
    "--random_state", required=True, help="random_state"
)
parser.add_argument(
    "--train_batch_size", required=True, help="train_batch_size", type = int
)
parser.add_argument(
    "--test_batch_size", required=True, help="test_batch_size", type = int
)
parser.add_argument(
    "--test_size", required=True, help="test_size", type = float
)
parser.add_argument(
    "--train_ratio", required=True, help="train_ratio", type = float
)
threshold = True
no_cuda = False
shuffle = True

# Enter all arguments that you want to be in the filename of the saved output
ordered_args = [
    'seed', 'sorting_file', 'remove_n',
    'keep_lowest_n'
]

# Parse arguments and setup name of output file with forgetting stats
args = parser.parse_args()
args_dict = vars(args)
print(args_dict)
save_fname = '__'.join(
    '{}_{}'.format(arg, args_dict[arg]) for arg in ordered_args)

# Set appropriate devices
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# Set random seed for initialization
def set_seed(seed = 1):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    npr.seed(seed)

set_seed(args.seed)

os.makedirs(args.output_dir, exist_ok=True)

def self_define_data(n_samples, n_features, n_informative, n_redundant, n_classes, n_clusters_per_class, \
                                  shuffle, random_state):
    from sklearn.datasets import make_classification
    from torch.utils.data import TensorDataset
    if random_state == "None":
        random_state = None
    else:
        random_state = int(random_state)
    samples, labels = make_classification(n_samples = n_samples, n_features = n_features, n_informative = n_informative, n_redundant = n_redundant, \
                             n_classes = n_classes, n_clusters_per_class = n_clusters_per_class, shuffle = shuffle, \
                              random_state = random_state)
    print(samples)
    dataset = TensorDataset(torch.FloatTensor(samples), torch.FloatTensor(labels))
    
    return dataset

def shuffle_dataset(dataset, times):
    from torch import randperm
    for t in range(times):
        lenth = randperm(len(dataset)).tolist() # 生成乱序的索引
        shuffle_data = torch.utils.data.Subset(dataset, lenth)#生成乱序子集
    return shuffle_data

def random_dataset(dataset, train_batch_size, nums, shuffle, random_por):
    set_seed()
    train_dataset = dataset
    

    trainsize = int(random_por * len(train_dataset))
    valsize = int(( 1 - random_por ) * len(train_dataset))

    random_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [trainsize, valsize])
    print("shuffling dataset..........")
    shuffle_dataset(random_dataset, 1)
    print(" We get a new random dataset with {} samples".format(len(random_dataset)))

    return random_dataset

dataset = self_define_data(args.n_samples,\
              args.n_features, args.n_informative, args.n_redundant,\
              args.n_classes, args.n_clusters_per_class,\
              True, args.random_state)

trainsize = int(args.train_ratio * len(dataset))
testsize = int(( 1 - args.train_ratio ) * len(dataset)) + 1  
# print(trainsize)
# print(testsize)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [trainsize, testsize])
# print(train_dataset.__dir__())
# print(train_dataset.__len__())
# print(train_dataset.indices.__dir__())
# print(train_dataset.indices.__len__())
# print(train_dataset.indices)
# print(test_dataset.indices)
if threshold == False:
  train_dataset = random_dataset(train_dataset, args["train_batch_size"], args["num_workers"], args["shuffle"], args["random_por"])

train_dataset.train_labels = train_dataset.dataset.tensors[1][train_dataset.indices]
train_dataset.train_data = train_dataset.dataset.tensors[0][train_dataset.indices]
test_dataset.test_labels = test_dataset.dataset.tensors[1][test_dataset.indices]
test_dataset.test_data = test_dataset.dataset.tensors[0][test_dataset.indices]
print(len(train_dataset.train_labels))
print(len(train_dataset.train_data))
print(len(test_dataset.test_labels))
print(len(test_dataset.test_data))
# Get indices of examples that should be used for training
if args.sorting_file == 'none':
    train_indx = np.array(range(len(train_dataset.train_labels)))
else:
    try:
        with open(
                os.path.join(args.input_dir, args.sorting_file) + '.pkl',
                'rb') as fin:
            ordered_indx = pickle.load(fin)['indices']
    except IOError:
        with open(os.path.join(args.input_dir, args.sorting_file),
                  'rb') as fin:
            ordered_indx = pickle.load(fin)['indices']

    # Get the indices to remove from training
    elements_to_remove = np.array(
        ordered_indx)[args.keep_lowest_n:args.keep_lowest_n + args.remove_n]

    # Remove the corresponding elements
    train_indx = np.setdiff1d(
        range(len(train_dataset.train_labels)), elements_to_remove)

# Remove remove_n number of examples from the train set at random
if args.keep_lowest_n < 0:
    train_indx = npr.permutation(np.arange(len(
        train_dataset.train_labels)))[:len(train_dataset.train_labels) - args.remove_n]

# Reassign train data and labels
train_dataset.train_data = train_dataset.train_data[train_indx, :]
train_dataset.train_labels = np.array(train_dataset.train_labels)[train_indx].tolist()

print('Training on ' + str(len(train_dataset.train_labels)) + ' examples')

# Setup model and optimizer
model = Linear_SVM(n_features = args.n_features, n_classes = args.n_classes).to(device)
##优化器（SGD）
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

# Setup loss
criterion = torch.nn.MultiMarginLoss().to(device)

# Initialize dictionary to save statistics for every example presentation
example_stats = {}

elapsed_time = 0
for epoch in range(args.epochs):
    start_time = time.time()

    train(args, model, device, train_dataset, optimizer, epoch, example_stats)
    test(args, model, device, test_dataset, example_stats)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

    # Save the stats dictionary
    fname = os.path.join(args.output_dir, save_fname)
    with open(fname + "__stats_dict.pkl", "wb") as f:
        pickle.dump(example_stats, f)

    # Log the best train and test accuracy so far
    with open(fname + "__best_acc.txt", "w") as f:
        f.write('train test \n')
        f.write(str(max(example_stats['train'][1])))
        f.write(' ')
        f.write(str(max(example_stats['test'][1])))