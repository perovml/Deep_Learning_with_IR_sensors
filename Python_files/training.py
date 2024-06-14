import random
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import time



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

#Dataaugmentations:
def RandomHorizontalFlip(sample, prob, dim = 2):
    number = random.random()
    state = False
    if number <= prob:
        state = True
        return torch.flip(sample, (dim,)), state
    else:
        return sample, state

def RandomVerticalFlip(sample, prob, dim = 3):
    number = random.random()
    state = False
    if number <= prob:
        state = True
        return torch.flip(sample, (dim,)), state
    else:
        return sample, state

def AddValue(frame):
    tensor_ones = torch.ones(frame.shape)
    integer_ = random.randint(0, 2)
    float_ = random.random()
    return frame + tensor_ones.to(device)*(integer_ + float_)

class AddGaussianNoise:
    def __init__(self, mean=0., std=0.5):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()).to(device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



def apply_augmentation(inputs, add_noise = True, dim = (2, 3), add_val = False, std = 0.05):
    noise = AddGaussianNoise(mean=0., std=std)
    inputs, state_hor = RandomHorizontalFlip(inputs, 0.5, dim[0])
    inputs, state_ver = RandomVerticalFlip(inputs, 0.5, dim[1])
    if add_noise:
        inputs = noise(inputs)
    if add_val:
        inputs = AddValue(inputs)
    return inputs, (state_hor, state_ver)

def flip_labels(labels, states):
    state_hor, state_ver = states
    if state_hor:
        labels = torch.flip(labels, (1,))
    if state_ver:
        labels = torch.flip(labels, (2,))
    return labels

def network_training(net, train_loader, optimizer, criterion, 
                     num_epochs, code_word, path_to_save, interpolate = True, patience=20, add_noise = True, factor=0.4, norm_grid = False, add_val = True, std = 0.01):
    epoch_acc_max = 0
    patience_counter = 0  # Early stopping counter
    start_time = time.time()
    loss_batch = []
    loss_epoch = []
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=patience//2, factor=factor)
    model_path = f'{path_to_save}/model_{code_word}'
    current_lr = optimizer.param_groups[0]['lr']
    print(f'current learning rate:{current_lr}')
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_loss_epoch = 0.0
        net.train()
        running_corrects = 0
        #to log learning rate:
        if patience_counter >= patience//2:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'current learning rate:{current_lr}')       
        #wandb.log({"learning_rate": current_lr})
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, _ = apply_augmentation(inputs.to(device), add_noise = add_noise, std = std)
            if norm_grid:
                inputs = optimized_gridwise_normalize(inputs)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Ensure labels are of type LongTensor
            labels = labels.long()
            #zero the parameter gradients
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = net(inputs).to(device)
                _, preds = torch.max(outputs, 1)
                #print(outputs)
                #print(labels)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            # print training statistics
            running_loss += loss.item()
            running_loss_epoch += loss.item()
            running_corrects += torch.sum(preds == labels)
            if i % 200 == 199:    # print every 200 mini-batches
                loss_batch += [running_loss / 200]
                running_loss = 0.0
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        scheduler.step(epoch_acc)
        epoch_loss = running_loss_epoch/ len(train_loader.dataset)
        loss_epoch.append(epoch_loss)
        #wandb.log({'epoch': epoch, 'loss': epoch_loss})
        running_loss_epoch = 0.0
        if epoch_acc > epoch_acc_max:
            epoch_acc_max = epoch_acc
            patience_counter = 0
            torch.save(net.state_dict(), model_path)
        else:
            patience_counter += 1
        print(f'Epoch {epoch}: {epoch_acc}')
        if patience_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch}.')
            break  # Stop training if patience limit is reached
    training_time = time.time() - start_time
    print(f'time of training: {training_time}')
    print(f'max accuracy: {epoch_acc_max}')
    #log the model
    #wandb.log_artifact(model_path, type="model")
    
    return net, loss_batch, loss_epoch

def optimized_gridwise_normalize(batch):
    # Calculate the mean and std dev for each image in the batch
    # Reshape to [batch_size, -1] to treat each image as a flat array for mean and std computation
    batch_flat = batch.view(batch.size(0), -1)
    
    # Compute mean and std along the flattened dimension
    means = batch_flat.mean(dim=1).view(batch.size(0), 1, 1, 1)
    stds = batch_flat.std(dim=1, unbiased=False).view(batch.size(0), 1, 1, 1)
    
    # Normalize the batch using broadcasting
    normalized_batch = (batch - means) / (stds + 1e-8)
    
    return normalized_batch


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


def network_training_detect(net, train_loader, criterion, optimizer, num_epochs, code_word, path_to_save, interpolate = True, sequence = False, add_noise = False, patience = 20, std = 0.02, factor = 0.3):
    patience_counter = 0  # Early stopping counter
    epoch_recall_max = 0
    start_time = time.time()
    learning_rate_list = []
    loss_batch = []
    loss_epoch = []
    if sequence:
        dim_aug = (3, 4)
    else:
        dim_aug = (2, 3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=patience//2, factor = factor)
    current_lr = optimizer.param_groups[0]['lr']
    print(f'current learning rate:{current_lr}')
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_loss_epoch = 0.0
        running_corrects = 0
        running_true_positives = 0
        running_false_negatives = 0
        running_true_negatives = 0
        running_false_positives = 0
        total_pixels = 0  # total number of pixels processed
        net.train()
        current_lr = optimizer.param_groups[0]['lr']
        learning_rate_list.append(current_lr)
        if patience_counter >= patience//2:
            print(f'current learning rate:{current_lr}')  
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, states = apply_augmentation(inputs.to(device), add_noise = add_noise, dim = dim_aug, std = std)
            if interpolate == True:
                inputs = F.interpolate(inputs, size=(16, 16), mode='bilinear')
            inputs = inputs.to(device)
            #to flip the target mask as well as input:
            labels = flip_labels(labels, states)
            #print(f'labels after flipping {labels[0]}')
            labels = labels.float().to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = net(inputs)
                loss = criterion(outputs.squeeze(1), labels)
                loss.backward()
                optimizer.step()

            # calculate running loss & accuracy
            running_loss += loss.item()
            running_loss_epoch += loss.item()

            preds = (outputs > 0.5).squeeze(1)  # threshold predictions
            running_corrects += torch.sum(preds == labels.bool())
            running_true_positives += torch.sum((preds == 1) & (labels.bool() == 1))
            running_false_negatives += torch.sum((preds == 0) & (labels.bool() == 1))
            running_true_negatives += torch.sum((preds == 0) & (labels.bool() == 0))
            running_false_positives += torch.sum((preds == 1) & (labels.bool() == 0))
            total_pixels += labels.numel()

            if i % 200 == 199:  # print every 200 mini-batches
                loss_batch.append(running_loss / 200)
                running_loss = 0.0
        epoch_recall = running_true_positives / (running_true_positives + running_false_negatives)
        epoch_acc = (running_true_positives + running_true_negatives) / (running_true_positives + running_true_negatives + running_false_positives + running_false_negatives)
        scheduler.step(epoch_recall)
        loss_epoch.append(running_loss_epoch / len(train_loader.dataset))
        print(f'EPOCH {epoch}: Accuracy: {epoch_acc:.4f}')
        print(f'Recall: {epoch_recall} ')
        print(f'tp: {running_true_positives}, fn: {running_false_negatives}, fp: {running_false_positives}, tn: {running_true_negatives}')
        print(f'Epoch loss: {running_loss_epoch / len(train_loader.dataset)}')
        running_loss_epoch = 0.0
        if epoch_recall > epoch_recall_max:
            epoch_recall_max = epoch_recall
            torch.save(net.state_dict(), f'{path_to_save}/model_{code_word}.pt')
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch}.')
            break  # Stop training if patience limit is reached

    training_time = time.time() - start_time
    print(f'Time of training: {training_time:.2f} seconds')
    print(f'Max recall: {epoch_recall_max:.4f}')
    return net, loss_batch, loss_epoch, learning_rate_list
    

