import copy
from timeit import default_timer as timer

import IPython
import numpy as np
import tensorly as tl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from tensorly.decomposition import partial_tucker
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import VBMF
from consts import *


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_ranks(layer):
    weights = layer.weight.data
    unfold_0 = tl.base.unfold(weights, 0)
    unfold_1 = tl.base.unfold(weights, 1)
    _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
    _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
    ranks = [diag_0.shape[0], diag_1.shape[1]]
    return ranks


def tucker_decomposition_conv_layer(layer, ranks=None):
    tl.set_backend('pytorch')

    if ranks == None:
        ranks = estimate_ranks(layer)

    (core, factors), rec_errors = \
        partial_tucker(layer.weight,
                       modes=[0, 1], rank=ranks, init='svd')
    last, first = factors

    first_layer = torch.nn.Conv2d(in_channels=first.shape[0],
                                  out_channels=first.shape[1], kernel_size=1,
                                  stride=1, padding=0, dilation=layer.dilation, bias=False)

    core_layer = torch.nn.Conv2d(in_channels=core.shape[1],
                                 out_channels=core.shape[0], kernel_size=layer.kernel_size,
                                 stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
                                 bias=False)

    last_layer = torch.nn.Conv2d(in_channels=last.shape[1],
                                 out_channels=last.shape[0], kernel_size=1, stride=1,
                                 padding=0, dilation=layer.dilation, bias=True)

    first_layer.weight.data = \
        torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core

    new_layers = [first_layer, core_layer, last_layer]
    # print(ranks)
    return nn.Sequential(*new_layers)


def optimize_model(model, optim_conv_layer):
    conv1_layer = model.conv1

    model.conv1 = optim_conv_layer(conv1_layer)

    for i in range(2):
        layer1_conv1 = model.layer1[i].conv1
        model.layer1[i].conv1 = optim_conv_layer(layer1_conv1)
        layer1_conv2 = model.layer1[i].conv2
        model.layer1[i].conv2 = optim_conv_layer(layer1_conv2)

    for i in range(2):
        layer2_conv1 = model.layer2[i].conv1
        model.layer2[i].conv1 = optim_conv_layer(layer2_conv1)
        layer2_conv2 = model.layer2[i].conv2
        model.layer2[i].conv2 = optim_conv_layer(layer2_conv2)

    for i in range(2):
        layer3_conv1 = model.layer3[i].conv1
        model.layer3[i].conv1 = optim_conv_layer(layer3_conv1)
        layer3_conv2 = model.layer3[i].conv2
        model.layer3[i].conv2 = optim_conv_layer(layer3_conv2)

    for i in range(2):
        layer4_conv1 = model.layer4[i].conv1
        model.layer4[i].conv1 = optim_conv_layer(layer4_conv1)
        layer4_conv2 = model.layer4[i].conv2
        model.layer4[i].conv2 = optim_conv_layer(layer4_conv2)

    return model


def tucker_wrapper():
    rank_i = 0

    def optim_conv_layer(layer):
        nonlocal rank_i
        result = tucker_decomposition_conv_layer(layer, RANKS[rank_i])
        rank_i += 1
        return result
    return optim_conv_layer


def train(model, train_loader):
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / (i + 1)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")


def test(model, test_loader) -> float:
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def train_test(name: str, model, train_loader, test_loader):
    print(f'Training {name} for {NUM_EPOCHS} epochs')
    start = timer()
    train(model, train_loader)
    end = timer()
    accuracy = test(model, test_loader)
    print(
        f'Done, accuracy = {accuracy:.4f}%, training time = {end - start:.4f}s')


def main():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = CIFAR10(root='./data', train=True,
                            download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False,
                           download=True, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=100, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=100,
                             shuffle=False, num_workers=2)

    model_original = models.resnet18(pretrained=True)
    n_params_original = count_parameters(model_original)

    model_tuckerified = copy.deepcopy(model_original)
    print('Model optimization...')
    model_tuckerified = optimize_model(model_tuckerified, tucker_wrapper())
    n_params_tuckerified = count_parameters(model_tuckerified)

    print(f'No. parameters original = {n_params_original}')
    print(
        f'No. parameters optimized = {n_params_tuckerified} ({(n_params_tuckerified / n_params_original)*100:.4f}%)')

    train_test('original model', model_original, train_loader, test_loader)
    train_test('tuckerified model', model_tuckerified,
               train_loader, test_loader)


if __name__ == '__main__':
    main()
