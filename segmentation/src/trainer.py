import os
import torch

class Trainer:
    def __init__(self, model, train_loader, validation_loader, loss, optimiser, epochs, device, save_dir):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.loss = loss
        self.optimiser = optimiser
        self.epochs = epochs
        self.device = device
        self.save_dir = save_dir
        self.file = None

    def train_model(self):
        for epoch in range(self.epochs):
            # training step
            loss, num_correct, num_trained = self._train_step(self.train_loader)

            # validation step
            val_correct, num_val = self._validation_step(self.validation_loader)

            print(f'Epoch number {epoch + 1}, Loss: {loss/ len(self.train_loader)}, Training accuracy: {100. * (num_correct/num_trained)}')
            print(f'Epoch number {epoch + 1}, Validation accuracy: {100. * (val_correct / num_val)}')
            print('---')
        # save the trained model
        file = os.path.join(self.save_dir, 'trained_model_v1.pth')
        self.file = file
        torch.save(self.model.state_dict(), file)
        print(f'model has been saved to {self.file}')

    def _train_step(self, train_loader):
        self.model.train()
        train_loss = 0.0
        correct = 0
        total_train = 0
        for i, (images,labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            #forward pass
            self.optimiser.zero_grad()
            outputs = self.model(images)

            predictions = outputs.argmax(dim=1)
            num_correct = predictions.eq(labels).sum().item()
            num_pixels = labels.numel()

            #update total correct and total trained here
            correct += num_correct
            total_train += num_pixels

            training_loss = self._calculate_loss(outputs, labels)
            #update train_loss
            train_loss += training_loss.item()

            #backprop and parameter update
            training_loss.backward()
            self.optimiser.step()

        #return training_loss, num_correct, num_pixels
        return train_loss, correct, total_train

    def _validation_step(self, validation_loader):
        self.model.eval()
        # train_loss = 0.0
        correct = 0
        total_val = 0
        with torch.no_grad():
            for i, (images,labels) in enumerate(validation_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                #print(outputs.shape)

                predictions = outputs.argmax(dim=1)
                num_correct = predictions.eq(labels).sum().item()
                num_pixels = labels.numel()

                correct += num_correct
                total_val += num_pixels
        return correct, total_val

    def _calculate_loss(self, predicted, truth):
        truth = truth.long()
        #truth
        ls = self.loss(predicted, truth)

        return ls
