#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18
# Adapted for helpthx https://github.com/helpthx
# Modity:   2021-01-01 	

from __future__ import print_function

import copy
import os.path as osp

import click
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.hub
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms

from grad_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)

# if a model includes LSTM, such as in image captioning,
# torch.backends.cudnn.enabled = False


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def get_classtable():
    classes = []
    with open("/content/gdrive/My Drive/UnB/TCC-1/TCC1-1-dataset-final/skin.txt") as lines:
        for line in lines:
            line = line.strip().split(" ", 1)[1]
            line = line.split(", ", 1)[0].replace(" ", "_")
            classes.append(line)
    return classes


def preprocess(image_path):
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))


def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)


# torchvision models
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


@click.group()
@click.pass_context
def main(ctx):
    print("Mode:", ctx.invoked_subcommand)


@main.command()
@click.option("-i", "--image-paths", type=str, multiple=True, required=True)
@click.option("-a", "--arch", type=click.Choice(model_names), required=True)
@click.option("-t", "--target-layer", type=str, required=True)
@click.option("-k", "--topk", type=int, default=3)
@click.option("-o", "--output-dir", type=str, default="./results")
@click.option("--cuda/--cpu", default=True)
def demo1(image_paths, target_layer, arch, topk, output_dir, cuda):
    """
    Visualize model responses given multiple images
    """

    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model from torchvision
    PRE_MODEL_DIR = '/content/gdrive/My Drive/UnB/TCC-1/TCC1-1-dataset-final/restnet_model152_trained_exp7.pt'

    model_name = 'resnet'
    num_classes = 9
    feature_extract = False

    model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    if train_on_gpu:
        state = torch.load(PRE_MODEL_DIR)
    else:
        state = torch.load(PRE_MODEL_DIR, map_location='cpu')

    # Loading weights in restnet architecture
    model.load_state_dict(state['state_dict'])

    model.to(device)
    model.eval()

    # Images
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    images = torch.stack(images).to(device)

    """
    Common usage:
    1. Wrap your model with visualization classes defined in grad_cam.py
    2. Run forward() with images
    3. Run backward() with a list of specific classes
    4. Run generate() to export results
    """

    # =========================================================================
    print("Vanilla Backpropagation:")

    bp = BackPropagation(model=model)
    probs, ids = bp.forward(images)

    for i in range(topk):
        # In this example, we specify the high confidence classes
        bp.backward(ids=ids[:, [i]])
        gradients = bp.generate()

        # Save results as image files
        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            save_gradient(
                filename=osp.join(
                    output_dir,
                    "{}-{}-vanilla-{}.png".format(j, arch, classes[ids[j, i]]),
                ),
                gradient=gradients[j],
            )

    # Remove all the hook function in the "model"
    bp.remove_hook()

    # =========================================================================
    print("Deconvolution:")

    deconv = Deconvnet(model=model)
    _ = deconv.forward(images)

    for i in range(topk):
        deconv.backward(ids=ids[:, [i]])
        gradients = deconv.generate()

        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            save_gradient(
                filename=osp.join(
                    output_dir,
                    "{}-{}-deconvnet-{}.png".format(j, arch, classes[ids[j, i]]),
                ),
                gradient=gradients[j],
            )

    deconv.remove_hook()

    # =========================================================================
    print("Grad-CAM/Guided Backpropagation/Guided Grad-CAM:")

    gcam = GradCAM(model=model)
    _ = gcam.forward(images)

    gbp = GuidedBackPropagation(model=model)
    _ = gbp.forward(images)

    for i in range(topk):
        # Guided Backpropagation
        gbp.backward(ids=ids[:, [i]])
        gradients = gbp.generate()

        # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            # Guided Backpropagation
            save_gradient(
                filename=osp.join(
                    output_dir,
                    "{}-{}-guided-{}.png".format(j, arch, classes[ids[j, i]]),
                ),
                gradient=gradients[j],
            )

            # Grad-CAM
            save_gradcam(
                filename=osp.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, arch, target_layer, classes[ids[j, i]]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )

            # Guided Grad-CAM
            save_gradient(
                filename=osp.join(
                    output_dir,
                    "{}-{}-guided_gradcam-{}-{}.png".format(
                        j, arch, target_layer, classes[ids[j, i]]
                    ),
                ),
                gradient=torch.mul(regions, gradients)[j],
            )


@main.command()
@click.option("-i", "--image-paths", type=str, multiple=True, required=True)
@click.option("-o", "--output-dir", type=str, default="./results")
@click.option("--cuda/--cpu", default=True)
@click.option("-c", "--classes-target", required=True)
def demo2(image_paths, output_dir, cuda, classes_target):
    """
    Generate Grad-CAM at different layers of ResNet-152
    """

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model
    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model from torchvision
    PRE_MODEL_DIR = '/content/gdrive/My Drive/UnB/TCC-1/TCC1-1-dataset-final/restnet_model152_trained_exp7.pt'

    model_name = 'resnet'
    num_classes = 9
    feature_extract = False

    model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    if train_on_gpu:
        state = torch.load(PRE_MODEL_DIR)
    else:
        state = torch.load(PRE_MODEL_DIR, map_location='cpu')

    # Loading weights in restnet architecture
    model.load_state_dict(state['state_dict'])
  
    model.to(device)
    model.eval()

    # The four residual layers
    target_layers = ["relu", "layer1", "layer2", "layer3", "layer4"]
    target_class = int(classes_target)

#     {'actinic-keratosis': 0,
#  'basal-cell-carcinoma': 1,
#  'dermatofibroma': 2,
#  'hemangioma': 3,
#  'intraepithelial-carcinoma': 4,
#  'malignant-melanoma': 5,
#  'melanocytic-nevus': 6,
#  'pyogenic-granuloma': 7,
#  'squamous-cell-carcinoma': 8}

    # Images
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    images = torch.stack(images).to(device)

    gcam = GradCAM(model=model)
    probs, ids = gcam.forward(images)
    ids_ = torch.LongTensor([[target_class]] * len(images)).to(device)
    gcam.backward(ids=ids_)

    for target_layer in target_layers:
        print("Generating Grad-CAM @{}".format(target_layer))

        # Grad-CAM
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print(
                "\t#{}: {} ({:.5f})".format(
                    j, classes[target_class], float(probs[ids == target_class])
                )
            )

            save_gradcam(
                filename=osp.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, "resnet152", target_layer, classes[target_class]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )


@main.command()
@click.option("-i", "--image-paths", type=str, multiple=True, required=True)
@click.option("-o", "--output-dir", type=str, default="./results")
@click.option("--cuda/--cpu", default=True)
@click.option("-c", "--classes-target", required=True)
def demo40(image_paths, output_dir, cuda, classes_target):
    """
    Generate Grad-CAM at last convolucional for each block layers of ResNet-152
    """

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model
    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model from torchvision
    PRE_MODEL_DIR = '/content/gdrive/My Drive/UnB/TCC-1/TCC1-1-dataset-final/restnet_model152_trained_exp7.pt'

    model_name = 'resnet'
    num_classes = 9
    feature_extract = False

    model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    if train_on_gpu:
        state = torch.load(PRE_MODEL_DIR)
    else:
        state = torch.load(PRE_MODEL_DIR, map_location='cpu')

    # Loading weights in restnet architecture
    model.load_state_dict(state['state_dict'])
  
    model.to(device)
    model.eval()

    # The the last convolucinal layer for each block
    target_layers = ["layer1.0.conv3", "layer1.1.conv3", "layer1.2.conv3", "layer2.0.conv3", "layer2.1.conv3",
                     "layer2.2.conv3", "layer2.3.conv3", "layer2.4.conv3", "layer2.5.conv3", "layer2.6.conv3",
                     "layer2.7.conv3", "layer3.0.conv3", "layer3.1.conv3", "layer3.2.conv3", "layer3.3.conv3",
                     "layer3.4.conv3", "layer3.5.conv3", "layer3.6.conv3", "layer3.7.conv3", "layer3.8.conv3",
                     "layer3.9.conv3", "layer3.10.conv3", "layer3.11.conv3", "layer3.12.conv3", "layer3.13.conv3",
                     "layer3.14.conv3", "layer3.15.conv3", "layer3.16.conv3", "layer3.17.conv3", "layer3.18.conv3",
                     "layer3.19.conv3", "layer3.20.conv3", "layer3.21.conv3", "layer3.22.conv3", "layer3.23.conv3",
                     "layer3.24.conv3", "layer3.25.conv3", "layer3.26.conv3", "layer3.27.conv3", "layer3.28.conv3",
                     "layer3.29.conv3", "layer3.30.conv3", "layer3.31.conv3", "layer3.32.conv3", "layer3.33.conv3",
                     "layer3.34.conv3", "layer3.35.conv3", "layer4.0.conv3", "layer4.1.conv3", "layer4.2.conv3"]

    target_class = int(classes_target)

#     {'actinic-keratosis': 0,
#  'basal-cell-carcinoma': 1,
#  'dermatofibroma': 2,
#  'hemangioma': 3,
#  'intraepithelial-carcinoma': 4,
#  'malignant-melanoma': 5,
#  'melanocytic-nevus': 6,
#  'pyogenic-granuloma': 7,
#  'squamous-cell-carcinoma': 8}

    # Images
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    images = torch.stack(images).to(device)

    gcam = GradCAM(model=model)
    probs, ids = gcam.forward(images)
    ids_ = torch.LongTensor([[target_class]] * len(images)).to(device)
    gcam.backward(ids=ids_)

    for target_layer in target_layers:
        print("Generating Grad-CAM @{}".format(target_layer))

        # Grad-CAM
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print(
                "\t#{}: {} ({:.5f})".format(
                    j, classes[target_class], float(probs[ids == target_class])
                )
            )

            save_gradcam(
                filename=osp.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, "resnet152", target_layer, classes[target_class]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )


@main.command()
@click.option("-i", "--image-paths", type=str, multiple=True, required=True)
@click.option("-k", "--topk", type=int, default=3)
@click.option("-o", "--output-dir", type=str, default="./results")
@click.option("--cuda/--cpu", default=True)
def demo3(image_paths, topk, output_dir, cuda):
    """
    Generate Grad-CAM with original models
    """

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Third-party model from my other repository, e.g. Xception v1 ported from Keras
    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model from torchvision
    PRE_MODEL_DIR = '/content/gdrive/My Drive/UnB/TCC-1/TCC1-1-dataset-final/restnet_model152_trained_exp7.pt'

    model_name = 'resnet'
    num_classes = 9
    feature_extract = False

    model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    if train_on_gpu:
        state = torch.load(PRE_MODEL_DIR)
    else:
        state = torch.load(PRE_MODEL_DIR, map_location='cpu')

    # Loading weights in restnet architecture
    model.load_state_dict(state['state_dict'])

    model.to(device)
    model.eval()

    # Check available layer names
    print("Layers:")
    for m in model.named_modules():
        print("\t", m[0])

    # Here we choose the last convolution layer
    target_layer = "layer4.2.conv3"

    # Preprocessing
    def _preprocess(image_path):
        raw_image = cv2.imread(image_path)
        raw_image = cv2.resize(raw_image, (224,224))
        image = torch.FloatTensor(raw_image[..., ::-1].copy())
        image -= torch.FloatTensor([0.485, 0.456, 0.406])
        image /= torch.FloatTensor([0.229, 0.224, 0.225])
        image = image.permute(2, 0, 1)
        return image, raw_image

    # Images
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = _preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    images = torch.stack(images).to(device)

    print("Grad-CAM:")

    gcam = GradCAM(model=model)
    probs, ids = gcam.forward(images)

    for i in range(topk):

        # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            # Grad-CAM
            save_gradcam(
                filename=osp.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, "xception_v1", target_layer, classes[ids[j, i]]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )


@main.command()
@click.option("-i", "--image-paths", type=str, multiple=True, required=True)
@click.option("-a", "--arch", type=click.Choice(model_names), required=True)
@click.option("-k", "--topk", type=int, default=5)
@click.option("-s", "--stride", type=int, default=1)
@click.option("-b", "--n-batches", type=int, default=128)
@click.option("-o", "--output-dir", type=str, default="./results")
@click.option("--cuda/--cpu", default=True)
def demo4(image_paths, arch, topk, stride, n_batches, output_dir, cuda):
    """
    Generate occlusion sensitivity maps
    """

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model from torchvision
   # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model from torchvision
    PRE_MODEL_DIR = '/content/gdrive/My Drive/UnB/TCC-1/TCC1-1-dataset-final/restnet_model152_trained_exp7.pt'

    model_name = 'resnet'
    num_classes = 9
    feature_extract = False

    model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    if train_on_gpu:
        state = torch.load(PRE_MODEL_DIR)
    else:
        state = torch.load(PRE_MODEL_DIR, map_location='cpu')

    # Loading weights in restnet architecture
    model.load_state_dict(state['state_dict'])
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    # Images
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    images = torch.stack(images).to(device)

    print("Occlusion Sensitivity:")

    patche_sizes = [10, 15, 25, 35, 45, 90]

    logits = model(images)
    probs = F.softmax(logits, dim=1)
    probs, ids = probs.sort(dim=1, descending=True)

    for i in range(topk):
        for p in patche_sizes:
            print("Patch:", p)
            sensitivity = occlusion_sensitivity(
                model, images, ids[:, [i]], patch=p, stride=stride, n_batches=n_batches
            )

            # Save results as image files
            for j in range(len(images)):
                print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

                save_sensitivity(
                    filename=osp.join(
                        output_dir,
                        "{}-{}-sensitivity-{}-{}.png".format(
                            j, arch, p, classes[ids[j, i]]
                        ),
                    ),
                    maps=sensitivity[j],
                )


if __name__ == "__main__":
    main()
