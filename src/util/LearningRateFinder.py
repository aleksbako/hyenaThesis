import torch
import matplotlib.pyplot as plt
from util import DEVICE
import numpy as np
class LearningRateFinder:
    def __init__(self, model, criterion, optimizer, train_loader):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.lrs = []
        self.losses = []
        self.best_loss = float('inf')
        self.prev_loss = float('inf')

    def find_lr(self, init_lr=1e-7, final_lr=1e-1, num_iter=100):
        lr_schedule = np.logspace(np.log10(init_lr), np.log10(final_lr), num_iter)
        self.model.train()

        for i, (inputs, targets) in enumerate(self.train_loader):
            if i >= num_iter:
                break
            
            # Move to GPU if available
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # Set learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_schedule[i]

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            # Record the learning rate and loss
            self.lrs.append(lr_schedule[i])
            self.losses.append(loss.item())

            # Track the best loss
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()

            # Stop if the loss starts increasing significantly
            if loss.item() > self.prev_loss * 4:
                break
            self.prev_loss = loss.item()

    def plot_lr_finder(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.lrs, self.losses, label="Loss")
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.grid()
        plt.legend()
        plt.show()
