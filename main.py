
import torch
import numpy as np

from deeplearner.data_utils import load_mammography_data
from deeplearner.classifiers.classification_mammograms import ClassificationMammograms
from deeplearner.solver import Solver

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load dataset
#train_data = load_mammography_data('/content/gdrive/My Drive/CaseStudies/data/train_aug.txt')
#val_data = load_mammography_data('/content/gdrive/My Drive/CaseStudies/data/val_aug.txt')
#test_data = load_mammography_data('/content/gdrive/My Drive/CaseStudies/data/test_aug.txt')
train_data = load_mammography_data('/content/gdrive/My Drive/CaseStudies/data/train.txt')
val_data = load_mammography_data('/content/gdrive/My Drive/CaseStudies/data/val.txt')
test_data = load_mammography_data('/content/gdrive/My Drive/CaseStudies/data/test.txt')


print("loaded the dataset")
print("Train size: %i" % len(train_data))
print("Val size: %i" % len(val_data))
print("Test size: %i" % len(test_data))


num_epochs = 70

batch_size = [32,64]
learning_rates = [1e-2,1e-3,1e-4]
weight_decay = [0.0, 0.001, 0.01]


for batch in batch_size:
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch, shuffle=False, num_workers=4)
    for lr in learning_rates:
        for weight in weight_decay:
            model = ClassificationMammograms()
            solver = Solver(optim_args={"lr": lr,
                                        "weight_decay": weight})

            print("learning rate:", lr, "weight decay:", weight, "batch size:", batch)
            solver.train(model, train_loader, val_loader, log_nth=1000, num_epochs=num_epochs)

            np.save("/content/gdrive/My Drive/CaseStudies/logs_new/train_loss_{}_{}_{}".format(batch, lr, weight), solver.train_loss_history)
            np.save("/content/gdrive/My Drive/CaseStudies/logs_new/val_loss_{}_{}_{}".format(batch, lr, weight), solver.val_loss_history)
            np.save("/content/gdrive/My Drive/CaseStudies/logs_new/train_acc_{}_{}_{}".format(batch, lr, weight), solver.train_acc_history)
            np.save("/content/gdrive/My Drive/CaseStudies/logs_new/val_acc_{}_{}_{}".format(batch, lr, weight), solver.val_acc_history)
            best_model = solver.best_model
            best_model.save("/content/gdrive/My Drive/CaseStudies/models_new/classification_{}_{}_{}_{}.model".format(solver.best_val_acc, batch, lr, weight))
            
            
# if we want to have the solver,adam stuff in there           
#            for batch in batch_size:
#    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True, num_workers=4)
#    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch, shuffle=False, num_workers=4)
#    for lr in learning_rates:
#        for weight in weight_decay:
#            model = ClassificationMammograms()
#            solver = Solver(optim_args={"lr": lr, 
#                                            #"betas": (0.9, 0.999),
#                                            #"eps": 1e-8,
#                                            "weight_decay": weight})
#
#            print("learning rate:", lr, "weight decay:", weight, "batch size:", batch)
#            solver.train(model, train_loader, val_loader, log_nth=1000, num_epochs=num_epochs)
