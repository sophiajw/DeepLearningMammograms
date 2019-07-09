import os
import matplotlib.pyplot as plt
import torch
import numpy as np

from exercise_code.classifiers.classification_cnn import ClassificationCNN
from exercise_code.data_utils import get_CIFAR10_datasets, OverfitSampler, rel_error
from exercise_code.classifiers.classification_cnn import ClassificationCNN
from exercise_code.solver_my import Solver

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# get dataset
dir_path = os.path.dirname(os.path.realpath(__file__))
train_data, val_data, test_data, mean_image = get_CIFAR10_datasets(dir_path)
print("Train size: %i" % len(train_data))
print("Val size: %i" % len(val_data))
print("Test size: %i" % len(test_data))


num_epochs = 50

#Arrays for the tuning process
batch_size = [32,64,128]
learning_rates = [1e-2,1e-3,1e-4]
weight_decay = [0.0, 0.001, 0.01]

#### maybe other interesting parameters
# reg=...
# log_nth=...


for batch in batch_size:
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch, shuffle=False, num_workers=4)
    for lr in learning_rates:
        for weight in weight_decay:
            model = ClassificationCNN()
            solver = Solver(optim_args={"lr": lr, 
                                            #"betas": (0.9, 0.999),
                                            #"eps": 1e-8,
                                            "weight_decay": weight})

            print("learning rate: ", lr, "weight decay: ", weight, "batch size: ", batch)
            solver.train(model, train_loader, val_loader, log_nth=1000, num_epochs=num_epochs)

            np.save("logs/train_loss_{}_{}_{}".format(batch, lr, weight), solver.train_loss_history)
            np.save("logs/val_loss_{}_{}_{}".format(batch, lr, weight), solver.val_loss_history)
            np.save("logs/train_acc_{}_{}_{}".format(batch, lr, weight), solver.train_acc_history)
            np.save("logs/val_acc_{}_{}_{}".format(batch, lr, weight), solver.train_acc_history)
            best_model = solver.best_model
            best_model.save("models/classification_{}_{}_{}_{}.model".format(solver.best_val_acc, batch, lr, weight))