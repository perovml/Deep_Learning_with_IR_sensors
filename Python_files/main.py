import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import pickle
import os
from data_processing import create_dataset, dataset_to_loader
from architectures import CNN_int
from training_evaluation import network_training, evaluation_of_net, visualize_misclassified

def main():
     # Load configuration
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    # Extracting configurations
    path_to_data = config['path_to_data']
    task = config['task']
    interpolate = config['interpolate']
    batch_size = config['batch_size']
    model_type = config['model_type']
    num_epochs = config['num_epochs']
    code_word = config['code_word']
    path_to_save = config['path_to_save']
    load_pretrained_model = config['load_pretrained_model']
    pretrained_model_path = config['pretrained_model_path']
    load_pretrained_dataloaders = config['load_pretrained_dataloaders']
    pretrained_dataloader_paths = config['pretrained_dataloader_paths']
    pickle_dataloaders = config['pickle_dataloaders']

    # Class number configuration
    class_number = 2 if task == 'binary_covid' else 4

    # Data Loaders
    if load_pretrained_dataloaders:
        train_loader, test_loader = load_data_loaders(pretrained_dataloader_paths)
    else:
        train_set, test_set, imbalance_coef = create_dataset(path=path_to_data, task=task, interpolate=interpolate)    
        train_loader = dataset_to_loader(train_set, batch_size, imbalance_coef, balancing=True)
        test_loader = dataset_to_loader(test_set, 1, imbalance_coef, balancing=False)
        if pickle_dataloaders:
            pickle_data_loaders(train_loader, test_loader, code_word)

    # Model setup
    model = CNN_int(class_number=class_number)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    # Load existing model if available
    if load_pretrained_model and os.path.exists(pretrained_model_path):
        model.load_state_dict(torch.load(pretrained_model_path))
    else:
        # Training
        model, loss_batch, loss_epoch = network_training(model, train_loader, optimizer, criterion, num_epochs, code_word, path_to_save)

    # Evaluation
    accuracy, labels, predictions, misclassified_samples = evaluation_of_net(model, test_loader, class_number)

def pickle_data_loaders(train_loader, test_loader, code_word):
    with open(f'loaders/train_loader_{code_word}.pkl', 'wb') as f:
        pickle.dump(train_loader, f)
    with open(f'loaders/test_loader_{code_word}.pkl', 'wb') as f:
        pickle.dump(test_loader, f)

def load_data_loaders(code_word):
    with open(f'loaders/train_loader_{code_word}.pkl', 'rb') as f:
        train_loader = pickle.load(f)
    with open(f'loaders/test_loader_{code_word}.pkl', 'rb') as f:
        test_loader = pickle.load(f)
    return train_loader, test_loader

if __name__ == "__main__":
    main()