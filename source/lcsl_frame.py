import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import mallorn_generator as mg
import torch.early_stopping as early_stopping
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torch.utils.data.dataloader as dataloader
import torch.utils.data.sampler as sampler
import torch.utils.data.distributed as distributed
import torch.utils.data.distributed.distributed_sampler as distributed_sampler
import torch.utils.data.distributed.distributed_sampler as distributed_sampler


class lcsl_frame(nn.Module):
    '''
    lcsl_frame is a training framework for machine learning classifiers that focus on rare labels.
    '''
    def __init__(self, model, dataset, epochs=100, learning_rate=0.001):
        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
        self.test_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, train_loader, test_loader):
        for epoch in range(self.epochs):
            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            for inputs, labels in self.test_loader:
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)
                print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        return self.model

    def test(self, test_loader):
        for inputs, labels in test_loader:
            outputs = self.forward(inputs)
            loss = self.criterion(outputs, labels)
    
    def update_train_loader(self, train_loader):
        self.train_loader = train_loader

    def update_test_loader(self, test_loader):
        self.test_loader = test_loader
    
    def call_augmentor(self, object):
        simulated_data = mg.Mallorn(object, simulate_num = 10)
        return simulated_data
    
    

    


