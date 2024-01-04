import os
import torch

class Trainer:
    def __init__(self, model, train_loader, loss, optimiser, epochs, device, save_dir):
        self.model = model.to(device)
        self.loader = train_loader
        self.loss = loss
        self.optimiser = optimiser
        self.epochs = epochs
        self.device = device
        self.save_dir = save_dir
        self.file = None

    def train_model(self):
        train_loss = 0.0
        correct = 0
        total_train = 0
        for epoch in range(self.epochs):
            loss, num_correct, num_pixels = self._train_step(self.loader)

            train_loss += loss.item()
            correct += num_correct
            total_train += num_pixels

            print(f'Epoch number {epoch + 1}, Loss: {train_loss}, Training accuracy: {100. * (correct/total_train)}')

        #save the trained model
        file = os.path.join(self.save_dir, 'trained_model.pth')
        self.file = file
        torch.save(self.model.state_dict(), file)


    def _train_step(self, train_loader):
        for i, (images,labels) in enumerate(train_loader):
            #specify device?
            images, labels = images.to(self.device), labels.to(self.device)

            #forward pass
            self.optimiser.zero_grad()
            outputs = self.model(images)

            predictions = outputs.argmax(dim=1)
            num_correct = predictions.eq(labels).sum.item()
            num_pixels = labels.numel()

            training_loss = self._calculate_loss(labels, outputs)

            #backprop and parameter update
            training_loss.backward()
            self.optimiser.step()

        return training_loss, num_correct, num_pixels

    def _calculate_loss(self, truth, predicted):
        ls = self.loss(truth, predicted)

        return ls
