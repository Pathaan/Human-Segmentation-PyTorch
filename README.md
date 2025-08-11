# Human Segmentation using PyTorch
## Problem Statement
The goal of this project is to extract the human portion from an image. Our dataset contains 350 training images along with 350 corresponding ground truth mask images. Additionally, we have a train.csv file, which serves as the annotation file for both the training images and their respective masks.

The task is to build a model that can accurately segment humans from input images. Once trained, the model will be integrated into a Flask-based application, providing an endpoint where a user can upload an image and receive a segmented output highlighting the human portion.

## Proposed Solution
There are multiple approaches to building an image segmentation model:

Build from scratch using U-Net architecture – U-Net is widely used for segmentation tasks, making it a strong candidate for this project.

Transfer Learning – Leverage pre-trained segmentation models and fine-tune them for our dataset.

We will adopt the transfer learning approach for efficiency and performance.

## Key Learnings from the Project
Deep understanding of image segmentation concepts.

In-depth knowledge of the U-Net architecture.

Custom dataset preparation using PyTorch.

Building and deploying a Flask application.

Dockerizing the application for portability.

Using Amazon Elastic Container Registry (ECR) to store Docker images.

Setting up GitHub Actions for CI/CD deployment.

Deploying the application on an AWS EC2 instance.


## Project 
Create Environment
```
python -m create venv venv
```
Activate your Environment
```
venv/Scripts/activate
```
Upgrade pip
```
python -m pip install --upgrade pip
```
Create Ipykernel
```
python -m ipykernel install --user --name=segmentation
```
## UNET Architecture

<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/9f15c0d8-312f-4a75-b0d3-204d39b98de1" />

U-Net is a type of neural network specifically designed for image segmentation, which involves dividing an image into meaningful parts, such as identifying tumors in medical scans. Its architecture is shaped like a "U" and is highly effective in scenarios with limited labeled data, making it popular in medical imaging and similar fields.

U-Net consists of three main components:

Contracting Path (Encoder):

Uses several small 3×3 convolutional filters to extract features from the image.

Applies ReLU activations for non-linearity and uses 2×2 max pooling to reduce the image size, focusing on essential features.

Bottleneck:

This is the lowest point of the “U” where the image information is most compressed and abstract, bridging the encoder and decoder.

Expansive Path (Decoder):

Utilizes upsampling (increasing the image size) to regain the original resolution.

Incorporates "skip connections" that pass info from encoder layers directly to corresponding decoder layers, preserving fine spatial details lost during downsampling.

Additional convolution layers refine the segmentation output.

How it works:

The input image is fed into the encoder, which progressively reduces its spatial dimensions while expanding the number of feature channels, capturing higher-level patterns.

At the bottleneck, the most compressed representation of the data is reached.

The decoder then reconstructs the feature maps back to the original image size, using skip connections to merge encoder features at each level, ensuring high localization accuracy.

Finally, a 1×1 convolutional layer classifies each pixel, generating a segmentation map (such as distinguishing object from background).

Implementation overview:

The standard U-Net model uses encoder blocks (with convolutions, activations, and max pooling), followed by decoder blocks (with upsampling, merging via skip connections, and further convolutions).

The model is built in Python using TensorFlow/Keras as illustrated in the source, and can be adapted for different input shapes and segmentation classes.

In summary, U-Net excels at pixel-level segmentation tasks, handling both the extraction of broad image features and the precise recovery of spatial details through its symmetrical encoder-decoder architecture and skip connections.

## How does the U-Net architecture improve medical image segmentation tasks
<img width="3331" height="1890" alt="image" src="https://github.com/user-attachments/assets/2703c622-85b5-4ab5-a75c-90167d37485e" />
<img width="1770" height="682" alt="image" src="https://github.com/user-attachments/assets/9a388e4b-42b9-4dbc-8870-521a925429c3" />

U-Net architecture improves medical image segmentation by enabling highly accurate, pixel-level classification of medical images, even with limited annotated data, through its unique encoder-decoder structure and the use of skip connections.

Key ways U-Net enhances medical image segmentation:

Precise Localization: The skip connections directly link each layer in the encoder (downsampling path) to its counterpart in the decoder (upsampling path), retaining high-resolution spatial details that are crucial for distinguishing fine boundaries such as those between tumors and healthy tissue.

High Accuracy with Small Datasets: By combining high- and low-level features and employing extensive data augmentation, U-Net achieves exceptional results even when labeled images are scarce, which is common in medical imaging.

Fully Convolutional and Efficient: The absence of fully connected layers lets U-Net handle variable image sizes efficiently, making it faster to train and requiring fewer computational resources than many alternatives.

Overlap-Tile Strategy: U-Net can segment large medical images by dividing them into smaller tiles with overlapping regions, ensuring consistent predictions across entire images without requiring excessive GPU memory.

Versatility: The architecture generalizes well beyond its original biomedical applications to other segmentation tasks, helping it remain the dominant solution in medical imaging.

By combining all these features, U-Net outperforms earlier models for medical segmentation—delivering fast, accurate, and robust performance suitable for real-world clinical applications.
