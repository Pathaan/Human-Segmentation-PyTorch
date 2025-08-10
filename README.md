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



