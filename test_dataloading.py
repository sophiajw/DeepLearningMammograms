import os
import torch

# import exercise_code.data_utils as data_utils
from exercise_code.data_utils import load_mammography_data
from exercise_code.classifiers.classification_mammograms import ClassificationCNN
from exercise_code.solver import Solver

# dir_path = os.path.dirname(os.path.realpath(__file__))
# dir_path = os.path.join(dir_path, 'data')
# txt_file = 'train.txt'
# data = data_utils.load_mammography_data(os.path.join(dir_path, txt_file))

train_data = load_mammography_data('data/train.txt')
val_data = load_mammography_data('data/val.txt')
test_data = load_mammography_data('data/test.txt')

train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=1)

model = ClassificationCNN()
# model.to(device)
solver = Solver(optim_args={"lr": 1e-3})
solver.train(model, train_loader, val_loader, log_nth=1000, num_epochs=10)