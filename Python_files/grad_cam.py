import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt


# Grad-CAM class definition
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model  # The CNN model
        self.target_layer = target_layer  # The target convolutional layer for Grad-CAM
        self.gradients = None  # To store gradients
        self.activations = None  # To store activations
        self.register_hooks()  # Register forward and backward hooks

    def register_hooks(self):
        # Hook to capture gradients
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        # Hook to capture activations
        def forward_hook(module, input, output):
            self.activations = output

        # Attaching hooks to the target layer
        layer = dict([*self.model.named_modules()])[self.target_layer]
        layer.register_forward_hook(forward_hook)
        layer.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, class_idx):
        # Forward pass through the model
        output = self.model(input_tensor)
        # Handling the case of output being a tuple
        if isinstance(output, tuple):
            output = output[0]

        # Zero gradients
        self.model.zero_grad()
        # Create one-hot encoding for the target class
        one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_()
        one_hot_output[0][class_idx] = 1
        # Backward pass for the target class
        output.backward(gradient=one_hot_output, retain_graph=True)

        # Calculating weights using gradients
        gradients = self.gradients.data[0]
        activations = self.activations.data[0]
        if activations.dim() == 3:  # [channels, height, width]
            # Add a batch dimension
            activations = activations.unsqueeze(0)
            gradients = gradients.unsqueeze(0)
        b, k, u, v = activations.size()
        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)

        # Creating the weighted activation map
        gradcam = F.relu((weights*activations).sum(1, keepdim=True))
        # Resizing the activation map to match the input image size
        gradcam = F.interpolate(gradcam, input_tensor.shape[2:], mode='bilinear', align_corners=False)
        # Generating the heatmap
        heatmap = gradcam.squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        return heatmap

    
def apply_grad_cam(model, img_tensor, target_layer, target_class = None):
    # Forward pass to get logits
    logits = model(img_tensor)
    # Applying softmax to get probabilities
    probabilities = F.softmax(logits, dim=1)
    # Getting the class index with the highest probability (if target_class not specified)
    class_idx = target_class if target_class is not None else torch.argmax(probabilities, dim=1).item()
    #class_idx -= 2
    # Creating Grad-CAM object and generating heatmap
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam.generate_heatmap(img_tensor, class_idx)
    #heatmap = cv2.resize(heatmap, (img_tensor.shape[-2], img_tensor.shape[-1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_grey = heatmap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    img_numpy = img_tensor.squeeze().cpu().numpy()  # Shape (H, W)
    img_scaled = (img_numpy - img_numpy.min()) / (img_numpy.max() - img_numpy.min()) * 255.0
    img_scaled = img_scaled.astype(np.uint8)  # Convert to 8-bit format
    img_scaled = np.uint8(255 * img_scaled)
    
    # Superimposing the heatmap on the scaled image
    superimposed_img = cv2.addWeighted(heatmap, 0.4, np.stack([img_scaled]*3, axis=-1), 0.6, 0)

    # Saving the output image
    #cv2.imwrite('grad_cam_output.jpg', superimposed_img)
    #cv2.imwrite('heatmap.jpg', heatmap_grey)
    #cv2.imwrite('original.jpg', img_scaled)
    print(f'prediction: {class_idx}')
    return img_scaled, heatmap_grey, superimposed_img

def apply_grad_cam(model, img_tensor, target_layer, target_class=None, ratio = 0.5):
    """
    Apply Gradient-weighted Class Activation Mapping (Grad-CAM).

    Parameters:
    - model: The model to interpret.
    - img_tensor: The input tensor for the model.
    - target_layer: The target layer for Grad-CAM.
    - target_class: The target class (optional).

    Returns:
    - img_scaled: The original image scaled for visualization.
    - heatmap: The heatmap.
    - superimposed_img: The original image with the heatmap superimposed.
    """
    # Forward pass to get logits and probabilities
    logits = model(img_tensor)
    probabilities = F.softmax(logits, dim=1)
    class_idx = target_class if target_class is not None else torch.argmax(probabilities, dim=1).item()

    # Generating heatmap with Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam.generate_heatmap(img_tensor, class_idx)
    heatmap = np.uint8(255 * heatmap)  # Scaling for visualization
    heatmap_to_color = 255 - heatmap
    #colored_heatmap = cv2.applyColorMap(heatmap_to_color, cv2.COLORMAP_HOT)  # Applying colormap to heatmap

    # Processing original image for superimposition
    img_numpy = img_tensor.squeeze().cpu().numpy()
    #img_numpy = img_numpy.transpose(1, 2, 0)  # Assuming PyTorch tensor format: CxHxW to HxWxC
    img_scaled = (img_numpy - img_numpy.min()) / (img_numpy.max() - img_numpy.min())
    img_scaled = np.uint8(255 * img_scaled)
    img_to_color = img_scaled#255 - img_scaled
    colored_image = cv2.applyColorMap(img_to_color, cv2.COLORMAP_HOT) 
    
    cv2.COLORMAP_AUTUMN
    # Superimposing the heatmap on the scaled image
    superimposed_img = cv2.addWeighted(cv2.cvtColor(colored_image, cv2.COLOR_RGB2BGR), ratio, np.stack([heatmap]*3, axis=-1), 1-ratio, 0)
    print(f'prediction: {class_idx}')

    return colored_image, heatmap, superimposed_img, class_idx




def display_images(original_img, heatmap, superimposed_img, path, index_, label, prediction, save = True):
    """
    Displays the original image, heatmap, and superimposed image side by side.
    
    Parameters:
    - original_img: The original image as a numpy array.
    - heatmap: The heatmap as a numpy array.
    - superimposed_img: The image with heatmap superimposed as a numpy array.
    """
    # Convert images to RGB if they are grayscale
    if len(original_img.shape) == 2 or original_img.shape[2] == 1:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    if len(heatmap.shape) == 2 or heatmap.shape[2] == 1:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB)
    if len(superimposed_img.shape) == 2 or superimposed_img.shape[2] == 1:
        superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_GRAY2RGB)
    
    # Save original image with colormap applied
    if save: 
        plt.imsave(f'{path}/original_{index_}_lbl{label}_prd{prediction}.png', original_img)
        # Save heatmap
        #plt.imsave(f'{path}/heatmap_{index_}_lbl{label}_prd{prediction}.png', heatmap)
        # Save superimposed image
        #plt.imsave(f'{path}superimposed_{index_}_lbl{label}_prd{prediction}.png', superimposed_img)
    
    # Display images
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original_img, interpolation='nearest')
    #axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    axs[1].imshow(heatmap, interpolation='nearest')
    #axs[1].set_title('Gradients')
    axs[1].axis('off')
    
    axs[2].imshow(superimposed_img, interpolation='nearest')
    #axs[2].set_title('Image with gradients')
    axs[2].axis('off')
    if save:
        plt.savefig(f'{path}/grad_cam_output_{index_}_lbl{label}_prd{prediction}.png')
    plt.show()

def display_and_save_grad_cam(model, test_set, index_, ratio, save, path_to_folder):
    data_sample = test_set[index_]
    img_tensor, label = data_sample
    img_tensor = img_tensor.unsqueeze(0)
    target_layer = 'conv4'
    
    original_img, heatmap, superimposed, prediction = apply_grad_cam(model, img_tensor, target_layer, None, ratio = ratio)  
    print(f'label: {label}')
    display_images(original_img, heatmap, superimposed, path_to_folder, index_, label, prediction, save = save)




    