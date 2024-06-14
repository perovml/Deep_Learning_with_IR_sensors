import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, auc, f1_score
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


def evaluation_of_net(net, test_loader, class_number, interpolate=False, norm_grid = False ):
    net.eval()
    labels = []
    predictions = []
    misclassified_samples = []
    misclassified_count = 0
    probas = []

    for inputs, label in test_loader:
        if interpolate:
            inputs = interpolate(inputs, size=(16, 16), mode='bilinear')
        inputs = inputs.to(device)
        if norm_grid:
            inputs = optimized_gridwise_normalize(inputs)
        label = label.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)

        # Aggregate outputs for PR-AUC calculation
        probas.extend(outputs.cpu().detach().numpy())

        # Extend labels and predictions for the entire batch
        labels.extend(label.cpu().numpy())
        predictions.extend(predicted.cpu().numpy())

        # Batch processing for misclassified samples
        misclassified_indices = np.where(predicted.cpu().numpy() != label.cpu().numpy())[0]
        misclassified_count += len(misclassified_indices)
        for index in misclassified_indices:
            misclassified_samples.append((label[index].item(), predicted[index].item(), inputs[index].cpu().numpy()))

    correctly_classified = (len(labels) - misclassified_count) * 100 / len(labels)
    print(f'- accuracy: {correctly_classified:.2f} %')
    print(f'- misclassification: {100 - correctly_classified:.2f} %')

    # Confusion Matrix and Class-specific Accuracies
    cm = confusion_matrix(labels, predictions)
    if class_number == 2:
        TN, FP, FN, TP = cm.ravel()
        # Calculate sensitivity and specificity
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
    
        # Calculate Balanced Accuracy
        balanced_accuracy = (sensitivity + specificity) / 2
        print(f'- Balanced Accuracy: {balanced_accuracy * 100:.2f} %')
        # F1 Score Calculation
        f1 = f1_score(labels, predictions)
        print(f'- F1 Score: {f1:.2f}')
    
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    average_class_accuracy = class_accuracies.sum()/ class_accuracies.shape[0]
    print(f'- average class accuracy: {average_class_accuracy * 100:.5f} %')
    for i, acc in enumerate(class_accuracies):
        print(f"Class {i} Accuracy: {acc:.2f}")

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.rcParams.update({'font.size': 16})
    mpl.rc('image', cmap='cividis')
    cmp = ConfusionMatrixDisplay(cm, display_labels=[i for i in range(class_number)])
    cmp.plot(ax=ax)
    plt.show()

    # PR-AUC Calculation
    y_real = label_binarize(labels, classes=[i for i in range(class_number)])
    y_proba = np.array(probas)

    precision = dict()
    recall = dict()
    pr_auc = dict()
    for i in range(class_number):
        precision[i], recall[i], _ = precision_recall_curve(y_real[:, i], y_proba[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

    # Averaging PR-AUC scores across classes
    average_pr_auc = np.mean(list(pr_auc.values()))
    print(f'- average PR-AUC: {average_pr_auc}')

    return correctly_classified, average_class_accuracy , pr_auc, labels, predictions, misclassified_samples

def distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def evaluation_of_detection_simple(net, test_loader, device, margin_of_error=0, interpolate=False):
    net.eval()
    total_pixels = 0
    correct_pixels = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    image_level_correct = 0
    misclassified_samples = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            original_inputs = inputs.clone()
            if interpolate:
                inputs = F.interpolate(inputs, size=(16, 16), mode='bilinear')

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            outputs = torch.sigmoid(outputs).squeeze(1)
            preds = (outputs > 0.5)

            total_pixels += torch.numel(labels)
            correct_pixels += (preds == labels).sum().item()
            true_positives += ((preds == 1) & (labels == 1)).sum().item()
            false_positives += ((preds == 1) & (labels == 0)).sum().item()
            false_negatives += ((preds == 0) & (labels == 1)).sum().item()

            # Image-level accuracy with margin of error for each image in the batch
            for idx in range(preds.size(0)):
                diff = torch.abs(preds[idx].float() - labels[idx].float())
                if torch.all(diff <= margin_of_error):
                    image_level_correct += 1
                else:
                    misclassified_samples.append((original_inputs[idx].squeeze(), preds[idx].squeeze(), labels[idx].squeeze()))

    pixel_accuracy = correct_pixels / total_pixels
    pixel_recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    pixel_precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    image_level_accuracy = image_level_correct / len(test_loader.dataset)  # Adjusted to divide by total number of images

    print(f"Pixel-Level Accuracy: {pixel_accuracy:.4f}")
    print(f"Pixel-Level Recall: {pixel_recall:.4f}")
    print(f"Pixel-Level Precision: {pixel_precision:.4f}")
    print(f"Image-Level Accuracy: {image_level_accuracy:.4f}")

    return pixel_accuracy, pixel_recall, pixel_precision, image_level_accuracy, misclassified_samples


def is_correctly_classified(pred, label, original_input, temp_threshold, device, slack):
    """
    Check if the predicted peaks are correctly classified based on temperature difference
    and spatial proximity, ensuring unique matches.
    """
    count_targets = (label == 1).sum().item()  # Count of target peaks
    count_detections = (pred == 1).sum().item()  # Count of detected peaks
    #if (count_targets + 1 < count_detections) or (count_targets > count_detections):
    #    return False, False  # Early return if the counts don't match

    # Initialize a mask to track matched predictions
    matched_preds = torch.zeros_like(pred, dtype=torch.bool)
    
    # This will store the minimum distance match for each target
    best_match = {}

    for i in range(label.shape[0]):  
        for j in range(label.shape[1]):
            if label[i, j] == 1:  # Target peak
                min_distance = float('inf')
                best_idx = None
                min_i, max_i = max(0, i-slack), min(label.shape[0], i+slack+1)
                min_j, max_j = max(0, j-slack), min(label.shape[1], j+slack+1)
                
                for mi in range(min_i, max_i):
                    for mj in range(min_j, max_j):
                        if pred[mi, mj] == 1 and not matched_preds[mi, mj]:
                            #distance to choose the closest prediction only
                            distance = (mi - i)**2 + (mj - j)**2
                            temp_diff = abs(original_input[mi, mj] - original_input[i, j])
                            if temp_diff <= temp_threshold and distance < min_distance:
                                min_distance = distance
                                best_idx = (mi, mj)
                
                if best_idx:
                    matched_preds[best_idx[0], best_idx[1]] = True
                    best_match[(i, j)] = best_idx

    # Check if all targets have been matched uniquely
    targets_detected = len(best_match)
    correctly_classified = (targets_detected == count_targets and count_targets == count_detections)
    return correctly_classified



def evaluation_of_detection_joint(net, test_loader, device, margin_of_error=0, interpolate=False, temp_threshold=0.5, sequence = False, slack = 1):
    net.eval()

    # Initialize metrics
    total_pixels = 0
    correct_pixels = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    image_level_correct_relaxed = 0
    image_level_correct = 0
    misclassified_samples = []
    labels = []
    mis_idxs = []
    tolerables = 0
    tolearable = False

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            if sequence:
                original_inputs = inputs[:,-1,:,:,:].clone().squeeze(1)
            else:
                original_inputs = inputs.clone().squeeze(1)
            if interpolate:
                inputs = F.interpolate(inputs, size=(16, 16), mode='bilinear')

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            outputs = torch.sigmoid(outputs).squeeze(1)

            # Threshold outputs to get binary predictions
            preds = (outputs > 0.5).to(device)#1,8,8

             # Image-level accuracy with margin of error
            diff = torch.abs(preds.float() - labels.float())

            image_level_correct += torch.all(diff <= margin_of_error).item()

            for idx in range(preds.size(0)):  # Loop through batch
                pred = preds[idx]  #8,8
                label = labels[idx] #8,8
                original_input = original_inputs[idx].to(device) #8,8
                diff = torch.abs(pred.float() - label.float())
                if torch.all(diff <= margin_of_error):
                    image_level_correct += 1
                
                correctly_classified = is_correctly_classified(pred, label, original_input, temp_threshold, slack)

                if correctly_classified:
                    image_level_correct_relaxed += 1
                else:
                    misclassified_samples.append((i, original_input.cpu(), pred.cpu(), label.cpu()))
                    mis_idxs.append(i)
            # Update pixel-level metrics
            correct_pixels += torch.sum(preds == labels).item()
            total_pixels += labels.numel()

            # Update true positives, false positives, and false negatives
            true_positives += torch.sum((preds == 1) & (labels == 1)).item()
            false_positives += torch.sum((preds == 1) & (labels == 0)).item()
            false_negatives += torch.sum((preds == 0) & (labels == 1)).item()

    pixel_accuracy = correct_pixels / total_pixels
    pixel_recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0
    pixel_precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0
    image_level_accuracy = image_level_correct / len(test_loader.dataset)
    image_level_accuracy_relaxed = image_level_correct_relaxed / len(test_loader.dataset)
    
    print(f"Pixel-Level Accuracy: {pixel_accuracy}")
    print(f"Pixel-Level Recall: {pixel_recall}")
    print(f"Pixel-Level Precision: {pixel_precision}")
    print(f"Image-Level Accuracy: {image_level_accuracy}")
    print(f"Image-Level Accuracy Relaxed: {image_level_accuracy_relaxed}")

    return {
        'pixel_accuracy': pixel_accuracy,
        'pixel_recall': pixel_recall,
        'pixel_precision': pixel_precision,
        'image_level_accuracy': image_level_accuracy,
        'image_level_accuracy_relaxed': image_level_accuracy_relaxed,
        'misclassified_samples': misclassified_samples
    }



def visualize_misclassified(misclass_samples, interpolate = True):
    count = 0
    for target, sample in misclass_samples:
        if interpolate == True:
            sample = sample.reshape(16,16)
            font_size = 5
        else:
            sample = sample.reshape(8,8)
            font_size = 8   
        fig, ax = plt.subplots()
        im = ax.imshow(sample)
        for (j,i),label in np.ndenumerate(sample):
            ax.text(i,j,round(label, 1),ha='center',va='center', fontsize = font_size)
        if target == 1:
            ax.set_title(f"False Negative")
        else: ax.set_title(f"False Positive")
        fig.tight_layout()
        count += 1
        if count %3 == 0:
            plt.show()
            
def plot_grid(sample):
    _, _, tensor_grid, predicted_mask, target_mask = sample
    tensor_grid, predicted_mask, target_mask = tensor_grid.cpu().numpy(), predicted_mask.cpu().numpy(), target_mask.cpu().numpy()
    # Create a combined visualization
    plt.figure(figsize=(12, 4))
    
    # Plot tensor grid with numbers
    plt.subplot(1, 2, 1)
    plt.imshow(tensor_grid, cmap='viridis')
    for i in range(tensor_grid.shape[0]):
        for j in range(tensor_grid.shape[1]):
            plt.text(j, i, f'{tensor_grid[i, j]:.2f}', ha="center", va="center", color="w")
    plt.colorbar()
    plt.title("Tensor Grid with Values")

    # Create RGB image for visualization
    combined_rgb = np.zeros(tensor_grid.shape + (3,))  # RGB format
    
    # Setting red color for predicted mask (True values)
    combined_rgb[predicted_mask, 0] = 1  # Red channel
    
    # Setting green color for target mask (1 values)
    combined_rgb[target_mask == 1, 1] = 1  # Green channel
    
    plt.subplot(1, 2, 2)
    # Plot the combined RGB mask
    plt.imshow(combined_rgb)
    plt.title("Distinguished Predicted (Red) and Target (Green) Masks")
    
    plt.tight_layout()
    plt.show()