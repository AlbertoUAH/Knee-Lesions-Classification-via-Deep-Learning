# -- Utils file (functions + constants)
# -- Author: Fernandez Hernandez, Alberto
# -- Date: 2022 - 01 - 29

# -- Libraries
import numpy          as np
import torch.nn       as nn
import torch
import cv2
import re

# -- Constants
MAX_PIXEL_VALUE = 255
FIG_HEIGHT      = 500
FIG_WIDTH       = 500
LABEL_DICT      = {
	"Abnormal":      0,
	"ACL Tear":      1,
	"Meniscus Tear": 2
}

# -- HSV dict with color ranges
HSV_RANGES = {
    'red': [
        {
            'lower': np.array([0, 39, 64]),
            'upper': np.array([20, 255, 255])
        },
        {
            'lower': np.array([161, 39, 64]),
            'upper': np.array([180, 255, 255])
        }
    ],
    # yellow is a minor color
    'yellow': [
        {
            'lower': np.array([21, 39, 64]),
            'upper': np.array([40, 255, 255])
        }
    ],
    # green is a major color
    'green': [
        {
            'lower': np.array([41, 39, 64]),
            'upper': np.array([80, 255, 255])
        }
    ]
}

# -- Python Classes
# -- MRNet model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Build 2D model
        # IMPORTANT: Input shape of torch model must be (N,C,H,W) -> BatchSize, Channel, Heigth, Width
        # Source   : https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # 3 pre-trained alexnet models (one for each plane)
        self.axial     = models.alexnet(pretrained=True, progress=False).features
        self.sagittal  = models.alexnet(pretrained=True, progress=False).features
        self.coronal   = models.alexnet(pretrained=True, progress=False).features

        # Disect the networks to access their last convolutional layer
        self.features_conv_axial    = self.axial[:12]
        self.features_conv_sagittal = self.sagittal[:12]
        self.features_conv_coronal  = self.coronal[:12]

        # Get the max pool of the features stem
        self.max_pool    = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.avg_pool_axial     = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_sagittal  = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_coronal   = nn.AdaptiveAvgPool2d(1)

        # Placeholders for the gradients
        self.gradients_axial    = None
        self.gradients_sagittal = None
        self.gradients_coronal  = None

        # Finally, define 3-outputs-dense layer
        self.fc = nn.Sequential(
            nn.Linear(in_features=3 * 256, out_features=3)
        )

    # Hook for the gradients of the activations
    # Axial
    def activations_hook_axial(self, grad):
        self.gradients_axial = grad
    # Sagittal
    def activations_hook_sagittal(self, grad):
        self.gradients_sagittal = grad
    # Coronal
    def activations_hook_coronal(self, grad):
        self.gradients_coronal = grad

    def forward(self, x):
        # Remove first dimension for each image (on each plane)
        images = [torch.squeeze(img, dim=0) for img in x]
        # Extract features from alexnet models
        image1 = self.features_conv_axial(images[0])
        image2 = self.features_conv_sagittal(images[1])
        image3 = self.features_conv_coronal(images[2])

        # Register the hook (for each plane)
        h_axial    = image1.register_hook(self.activations_hook_axial)
        h_sagittal = image2.register_hook(self.activations_hook_sagittal)
        h_coronal  = image3.register_hook(self.activations_hook_coronal)

        # Apply remaining pooling (MaxPool2d & AdaptiveAvgPool2d)
        image1 = self.max_pool(image1)
        image2 = self.max_pool(image2)
        image3 = self.max_pool(image3)

        # Convert image dimension from [slices, 256, 1, 1] to [slices, 256]
        image1 = self.avg_pool_axial(image1).view(image1.size(0), -1)
        image2 = self.avg_pool_sagittal(image2).view(image2.size(0), -1)
        image3 = self.avg_pool_coronal(image3).view(image3.size(0), -1)

        # Find maximum value across slices, reducing images to [1, 256]
        image1 = torch.max(image1, dim=0, keepdim=True)[0]
        image2 = torch.max(image2, dim=0, keepdim=True)[0]
        image3 = torch.max(image3, dim=0, keepdim=True)[0]

        # Stack 3 images together
        output = torch.cat([image1, image2, image3], dim=1)

        # Feed the output to last dense layer
        output = self.fc(output)
        return output

    # Method for the gradient extraction
    def get_activations_gradient(self):
        return [self.gradients_axial, self.gradients_sagittal, self.gradients_coronal]

    # Method for the activation extraction
    def get_activations(self, x):
        images = [torch.squeeze(img, dim=0) for img in x]
        return [self.features_conv_axial(images[0]),
                self.features_conv_sagittal(images[1]),
                self.features_conv_coronal(images[2])]

# -- Python Methods/Functions
# -- Function to prepare image for model input
def prepare_data(img):
    img = img / MAX_PIXEL_VALUE
    img = np.stack((img,)*3, axis=1)
    img = torch.FloatTensor(img)
    return img

# -- Function to build GradCAM heatmap
def build_grad_cam(model, img, label, img_dim=256):
    # Source: https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
    # Source: https://www.kaggle.com/minnieliang/gradcam-melanoma
    # Source: https://glassboxmedicine.com/2020/05/29/grad-cam-visual-explanations-from-deep-networks/
    # First, let's make the forward pass through the network with the image chosen
    # Set the evaluation mode
    model.eval()
    torch.set_grad_enabled(True)

    # Get the most likely prediction of the model
    pred = model(img)
    proba= torch.sigmoid(pred.squeeze())[label] * 100

    # Second, get the gradient of the output with respect to the parameters of the model (ACL tear)
    pred[:, label].backward()
    # Pull the gradients out of the model (ACL tear)
    gradients_axial, gradients_sagittal, gradients_coronal = model.get_activations_gradient()
    # Pool the gradients across the channels (alpha values): Global Average Pooling
    pooled_gradients_axial    = torch.mean(gradients_axial, dim=[0, 2, 3])
    pooled_gradients_sagittal = torch.mean(gradients_sagittal, dim=[0, 2, 3])
    pooled_gradients_coronal  = torch.mean(gradients_coronal, dim=[0, 2, 3])

    # Get the activations of the last convolutional layer
    activations = [activation.detach() for activation in model.get_activations(img)]
    activation_axial, activation_sagittal, activation_coronal = activations

    # Weight the channels by corresponding gradients
    for i in range(img_dim):
        activation_axial[:, i, :, :]    *= pooled_gradients_axial[i]
        activation_sagittal[:, i, :, :] *= pooled_gradients_sagittal[i]
        activation_coronal[:, i, :, :]  *= pooled_gradients_coronal[i]

    # Average the channels of the activations
    heatmap_axial    = torch.mean(activation_axial, dim=1).squeeze()
    heatmap_sagittal = torch.mean(activation_sagittal, dim=1).squeeze()
    heatmap_coronal  = torch.mean(activation_coronal, dim=1).squeeze()

    # Apply ReLU function: max(x, 0)
    heatmap_axial    = np.maximum(heatmap_axial.cpu(), 0)
    heatmap_sagittal = np.maximum(heatmap_sagittal.cpu(), 0)
    heatmap_coronal  = np.maximum(heatmap_coronal.cpu(), 0)

    # Normalize between 0 and 1
    heatmap_axial    /= torch.max(heatmap_axial)
    heatmap_sagittal /= torch.max(heatmap_sagittal)
    heatmap_coronal  /= torch.max(heatmap_coronal)

    heatmap_axial     = [cv2.resize(heatmap.numpy(), (img_dim, img_dim)) \
                            for heatmap in heatmap_axial]
    heatmap_sagittal  = [cv2.resize(heatmap.numpy(), (img_dim, img_dim)) \
                            for heatmap in heatmap_sagittal]
    heatmap_coronal   = [cv2.resize(heatmap.numpy(), (img_dim, img_dim)) \
                            for heatmap in heatmap_coronal]

    return heatmap_axial, heatmap_sagittal, heatmap_coronal, proba

# -- Function to filter Green, Yellow and Red colors (HSV image)
def create_mask(hsv_img, colors):
    mask = np.zeros((hsv_img.shape[0], hsv_img.shape[1]), dtype=np.uint8)

    for color in colors:
        for color_range in HSV_RANGES[color]:
            mask += cv2.inRange(
                hsv_img,
                color_range['lower'],
                color_range['upper']
            )

    return mask

# -- Function to superimposed original image with GradCAM heatmap
def superimpose_img_heatmap(img, heatmap):
    img              = img[:,0,:,:]
    heatmap          = torch.FloatTensor(heatmap)
    superimposed_img = []

    for img_slice, heatmap_slice in zip(img, heatmap):
        img_slice    = np.stack((img_slice.numpy(),)*3, axis=2)
        heatmap_slice= heatmap_slice.numpy()
        heatmap_slice= cv2.resize(heatmap_slice, (img_slice.shape[0], img_slice.shape[1]))
        heatmap_slice= np.uint8(MAX_PIXEL_VALUE * heatmap_slice)
        heatmap_slice= cv2.applyColorMap(heatmap_slice, cv2.COLORMAP_JET)
        heatmap_slice= cv2.cvtColor(heatmap_slice, cv2.COLOR_BGR2RGB)
        hsv          = cv2.cvtColor(heatmap_slice, cv2.COLOR_RGB2HSV)

        mask         = create_mask(hsv, ['red', 'yellow', 'green'])

        heatmap_slice= cv2.bitwise_and(heatmap_slice, heatmap_slice, mask=mask)
        superimposed_img.append(heatmap_slice * 0.4 + (img_slice * MAX_PIXEL_VALUE)*0.6)
    return np.array(superimposed_img)








