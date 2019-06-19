import os
import matplotlib.pyplot as plt
import torch

from exercise_code.classifiers.classification_cnn import ClassificationCNN
from exercise_code.data_utils import get_CIFAR10_datasets, OverfitSampler, rel_error
from exercise_code.classifiers.classification_cnn import ClassificationCNN
from exercise_code.solver import Solver

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


# load data in DataLoader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=100, shuffle=False, num_workers=4)

# train the model
model = ClassificationCNN()
model.to(device)
solver = Solver(optim_args={"lr": 1e-3})
solver.train(model, train_loader, val_loader, log_nth=1000, num_epochs=10)

# plot results
plt.subplot(2, 1, 1)
plt.plot(solver.train_loss_history, 'o')
plt.plot(solver.val_loss_history, 'o')
plt.xlabel('iteration')
plt.ylabel('loss')

plt.subplot(2, 1, 2)
plt.plot(solver.train_acc_history, '-o')
plt.plot(solver.val_acc_history, '-o')
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()