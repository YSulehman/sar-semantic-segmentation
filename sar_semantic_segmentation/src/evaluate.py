import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.metrics import confusion_matrix


class Evaluate:
    def __init__(self, model, model_pth, test_loader, rgb_dict, device, save_dir):
        """
        :param model:
        :param model_pth: string -> path to trained model
        :param test_loader:
        :param rgb_dict: dictionary rgb_tuple: label_idx
        :param device:
        :param save_dir:
        """
        self.model = model
        self.model_pth = model_pth
        self.test_loader = test_loader
        self.rgb_dict = rgb_dict
        self.device = device
        self.save_dir = save_dir

        #print(self.rgb_dict)

    def perform_evaluation(self):
        self.model.load_state_dict(torch.load(self.model_pth))
        self.model.to(self.device)
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
                num_correct += predictions.eq(labels).sum().item()
                total_test += labels.numel()

                # true_labels.extend(labels.view(-1).cpu().numpy())
                # predicted_labels.extend(predictions.view(-1).cpu().numpy())

                if i == 8:
                    self._predicted_masks(labels[7, :, :], 7)
                #     # self._predicted_masks(labels[1, :, :], 1)
                #     for j in range(predictions.shape[0]):
                #        #print('predictions', predictions[j, :, :].min(), predictions[j, :, :].max())
                #        #print('labels', labels[j, :, :].min(), labels[j, :, :].max())
                #        self._predicted_masks(predictions[j, :, :], j)

            print(f'Test accuracy: {100. * (num_correct / total_test)}')
            # compute confusion matrix and save
            # true_labels = np.array(true_labels)
            # predicted_labels = np.array(predicted_labels)
            # self._confusion_matrix(true_labels, predicted_labels, self.save_dir)

    def _confusion_matrix(self, true_labels, predicted_labels, save_direc):
        print('computing confusion matrix...')
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        file = os.path.join(save_direc, 'confusion_mat')
        with open(file, 'wb') as f:
            pickle.dump(conf_matrix, f)
        print(f'confusion matrix saved to: {file}')

    def _predicted_masks(self, predictions, index):
        """
        takes a batch of predictions (batch, patch_size, patch_size) and converts each to rgb representation
        """

        label_colours = np.array(list(self.rgb_dict.keys()))

        # colour_map = ListedColormap(label_colours, name='custom_map')

        predicted_mask = label_colours[predictions.cpu().numpy()]
        predicted_mask = predicted_mask.astype(np.uint8)

        # predicted_mask = colour_map(predictions.cpu().numpy()).astype(np.uint8)

        # dealing with transparency
        fig, ax = plt.subplots()

        # display the mask without interpolation
        ax.imshow(predicted_mask, interpolation='none')

        # set a non-transparent background colour
        ax.set_facecolor('white')

        # plt.imshow(predicted_mask, cmap=colour_map)
        #plt.imshow(predicted_mask, interpolation=None) -> normally without accounting for transparency
        #plt.savefig('./test_true' + str(index) + '.png')
        plt.savefig('./testing_true' + str(index) + '.png', bbox_inches='tight', pad_inches=0.1)
