import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

def load_config():
    with open('../config.json') as config_file:
        config = json.load(config_file)
    return config


def box_print(msg):
    print("=" * max(len(msg), 100))
    print(msg)
    print("=" * max(len(msg), 100))

def print_plots(train_acc_list, train_loss_list, test_acc_list, test_loss_list, directory_path, _time=''):
    """
    Prints two plot that describes our processes of learning through an NLLL loss function and the accuracy measure.
    Args:
        train_acc_list: Contains the accuracy measure tracking through the training phase.
        train_loss_list: Contains the loss measure tracking through the training phase.
        test_acc_list: Contains the accuracy measure tracking through the evaluation phase.
        test_loss_list: Contains the loss measure tracking through the evaluation phase.
        _time: The time id to recognize the plot output.
    Returns:
        Saves the plot in a png file.
    """

    # sns.set_style("whitegrid")

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    x_train = [a for a in range(len(train_loss_list))]
    x_test = [a for a in range(len(test_loss_list))]

    ax[0].plot(x_train, train_loss_list, label='Loss Train')
    ax[0].plot(x_test, test_loss_list, label='Loss Test')

    ax[0].legend()
    ax[0].set_title('Loss Convergence')
    ax[0].set_xlabel('Num of Epochs')
    ax[0].set_ylabel('Loss')

    ax[1].plot(x_train, train_acc_list, label='Train acc')
    ax[1].plot(x_test, test_acc_list, label='Test acc')
    ax[1].legend()
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Num of Epochs')
    ax[1].set_ylabel('ACC')
    fig.savefig('{}/figures/plots_{}.png'.format(directory_path, _time))


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss
