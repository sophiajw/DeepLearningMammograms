from random import shuffle
import numpy as np
from torch.optim.lr_scheduler import StepLR
import math

import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []
        self.best_val_acc = 0
        self.best_model = None

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        if torch.cuda.is_available():
            model.cuda()
        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################
        #exp_decay = math.exp(-0.01)
        #exp_decay=0.7
       # scheduler = StepLR(optim, step_size=2)
        
        for epoch in range(num_epochs):
            train_scores = []
            
            
            for iter, (data, labels) in enumerate(train_loader):
               # print('test')
                data, labels = data.to(device), labels.to(device)
                output = model(data)
                loss = self.loss_func(output, labels)
                loss.backward()
                loss = loss.item()
                optim.step()
                optim.zero_grad()
                self.train_loss_history.append(loss)

                _, train_preds = torch.max(output, 1)
                train_targets_mask = labels >= 0
                train_scores.append(np.mean((train_preds == labels)[train_targets_mask].data.cpu().numpy()))

                if log_nth == 0:
                    print('[Iteration {}/{}] TRAIN loss: {}'.format(iter + 1, iter_per_epoch, loss))
                elif iter % log_nth == (log_nth-1):
                    print('[Iteration {}/{}] TRAIN loss: {}'.format(iter+1,iter_per_epoch, loss))

            train_accuracy = np.mean(train_scores)
            self.train_acc_history.append(train_accuracy)

            print('[Epoch {}/{}] TRAIN loss: {}, acc: {}'.format(epoch+1, num_epochs, loss, train_accuracy))

            val_scores = []
            model.eval()
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs = model.forward(val_inputs)
                _, val_preds = torch.max(val_outputs, 1)
                val_targets_mask = val_targets >= 0
                val_scores.append(np.mean((val_preds == val_targets)[val_targets_mask].data.cpu().numpy()))

            model.train()
            #scheduler.step()

            #self.val_acc_history.append(np.mean(val_scores))
            val_acc = np.mean(val_scores)
            self.val_acc_history.append(val_acc)
            val_loss = self.loss_func(val_outputs, val_targets).item()
            self.val_loss_history.append(val_loss)

            

            print('[Epoch {}/{}] VAL loss: {}, acc: {}'.format(epoch + 1, num_epochs, val_loss, np.mean(val_scores)))
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model = model
                
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')
