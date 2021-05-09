import Augmentor as aug
import glob
import os
import numpy as np
import cv2 
import PIL
from Augmentor.Operations import Operation

class Lightning(Operation):
    def __init__(self, probability, intensity_low=0.7, intensity_high=1.2):
        Operation.__init__(self, probability)
        # Init classes variables with default values 
        # Default values treshold intent to create a optimal range
        # Imagens cant be too dark or too brigher
        self.intensity_low = intensity_low
        self.intensity_high = intensity_high

    def perform_operation(self, images):
        for i, image in enumerate(images):
            image = np.array(image.convert('RGB'))
            row, col, _ = image.shape
            light_intensity = np.random.randint(int(self.intensity_low * 100),
                                          int(self.intensity_high * 100))
    
            light_intensity /= 100

            gaussian = 100 * np.random.random((row, col, 1))
            gaussian = np.array(gaussian, dtype=np.uint8)
            gaussian = np.concatenate((gaussian, gaussian, gaussian), axis=2)
            image = cv2.addWeighted(image, light_intensity, gaussian, 0.25, 0)

            image = PIL.Image.fromarray(image)
            images[i] = image
  
        return images

# Multiplier used to set the final augmented images number
MULTIPLIER=29

# Default dir where we can find the train dataset
TRAIN_DIRECTORY_DATASET = '/home/helpthx/Desktop/TCC-1/TCC-1-UnB/dataset-split/val/*'

folders = []
for f in glob.glob(TRAIN_DIRECTORY_DATASET):
    if os.path.isdir(f):
        folders.append(os.path.abspath(f))

print('Classes found {}'.format([os.path.split(x)[1] for x in folders]))
print('Numb: ', len([os.path.split(x)[1] for x in folders]))

# Dictionari to hold the abspath and class's name
pipelines = {}

for folder in folders:
    pipelines[os.path.split(folder)[1]] = (aug.Pipeline(source_directory=folder,
                                                        output_directory='resnet-augmented'))
							

classes_count = []
for p in pipelines.values():
    print("Class '{}' has {} samples".format(p.augmentor_images[0].class_label,
                                           len(p.augmentor_images)))
  
    classes_count.append(len(p.augmentor_images))

# Instantiating Lighthing Class with 50 % probability 
lightning = Lightning(probability=0.5)

for p in pipelines.values():
  # 50 % of rotation the imagem with max left and max right
  p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
  
  # 40 % of zoom inside the imagem with a 90 % cover area
  p.zoom_random(probability=0.4, percentage_area=0.9)
  
  # 70 % of mirror vertical imagem for 50 % left or rigth 
  p.flip_left_right(probability=0.7)

  # 50 % of mirror horizontal
  p.flip_top_bottom(probability=0.5)

  # Appling some distortion in the imagem
  p.random_distortion(probability=0.8, grid_width=5, grid_height=5, magnitude=15)
  
  # Custom lightning of 50 %
  p.add_operation(lightning)

  # Rezise all the imagens size for default restnet 224x224
  p.resize(probability=1.0, width=224, height=224)

# If a equal sampling of the lesions is needed
# Mind that the final MULTIPLIER can scale many times if True

SAME_SAMPLING = False
for p in pipelines.values():
    if SAME_SAMPLING:
        diff = max(classes_count) - len(p.augmentor_images)
        p.sample((len(p.augmentor_images) + diff)*MULTIPLIER + diff)
    else:
        p.sample(len(p.augmentor_images)*MULTIPLIER)


