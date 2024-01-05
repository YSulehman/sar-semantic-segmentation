import os
import torch
import pickle
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix


class Evaluate:
    def __init__(self, model, model_pth, test_loader, rgb_dict, device, save_dir):
        self.model = model
        self.model_pth = model_pth
        self.test_loader = test_loader
        self.rgb_dict = rgb_dict
        self.device = device
        self.save_dir = save_dir

    def perform_evaluation(self):
        self.model.load_state_dict(torch.load(self.model_pth))
        self.model.eval()
        true_labels = []
        predicted_labels = []

        with torch.no_grad():
            num_correct = 0
            total_test = 0
            for i, (images, labels) in enumerate(self.test_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)

                predictions = outputs.argmax(dim=1)
                num_correct += predictions.eq(labels).sum.item()
                total_test += labels.numel()

                true_labels.extend(labels.view(-1).cpu().numpy())
                predicted_labels.extend(predictions.view(-1).cpu().numpy())

            print(f'Test accuracy: {100. * (num_correct / total_test)}')
            # compute confusion matrix and save
            true_labels = np.array(true_labels)
            predicted_labels = np.array(predicted_labels)
            self._confusion_matrix(true_labels, predicted_labels, self.save_dir)

    def _confusion_matrix(self, true_labels, predicted_labels, save_direc):
        print('computing confusion matrix...')
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        file = os.path.join(save_direc, 'confusion_mat')
        with open(file, 'wb') as f:
            pickle.dump(conf_matrix, f)
        print(f'confusion matrix saved to: {file}')

    def _predicted_masks(self, predictions):
        """
        takes a batch of predictions (batch, patch_size, patch_size) and converts each to rgb representation
        """
        label_colours = list(self.rgb_dict.keys())
        colour_map = ListedColormap(label_colours)

        predicted_mask = colour_map(predictions.cpu().numpy())
