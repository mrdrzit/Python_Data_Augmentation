import pandas as pd
import numpy as np
import imgaug.augmenters as iaa
import os
from imgaug.augmentables import Keypoint, KeypointsOnImage
from matplotlib import pyplot as plt
from PIL import Image

image_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images"))
# Load the image files using tkinter
images_path = list([os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.png') or file.endswith('.jpg')])
images_names = list([file for file in os.listdir(image_folder) if file.endswith('.png') or file.endswith('.jpg')])
csv_files_path = list([os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith('.csv')])

# images_path = filedialog.askopenfilenames(title='Select the images', filetypes=[('Image Files', '*.jpg *.png *.jpeg')])
# csv_files_path = filedialog.askopenfilenames(title='Select the csv files', filetypes=[('CSV Files', '*.csv')])

# The read dataframe function already imports the data as a dataframe, so it probably easier to create a copy of the dataframe and modify
def read_csv(csv_files_path):
    dataframes = []
    for csv_file in csv_files_path: 
        if "CollectedData" in csv_file:
            df = pd.read_csv(csv_file, header=1)
            dataframes.append(df.iloc[:, 2:])
    return dataframes

class Animal:
    """
    Class to store the data for each animal
    """

    def __init__(self):
        self.name = None
        self.image_data = []
        self.csv = []

    def populate_bodyparts(self, df):
       for index in range(2, len(df)):
           name = df.iloc[index, 0].strip()
           data = df.iloc[[0,1,index]]
           self.image_data.append({name: data})


animal_aug_list = []
csv_list = read_csv(csv_files_path)

for csv_data_frame in csv_list:
    animal = Animal()
    animal.populate_bodyparts(csv_data_frame)
    animal.csv.append(csv_data_frame)
    animal_aug_list.append(animal)

for image, image_name in zip(images_path, images_names):
    img_to_aug = np.asarray(Image.open(image))
    points_dataframe = animal.csv[0].set_index(animal.csv[0].columns[0], inplace=False)

    points = points_dataframe.loc[image_name]

    focinho_pos = [float(points['Focinho']), float(points['Focinho.1'])]
    orelhaE_pos = [float(points['OrelhaE']), float(points['OrelhaE.1'])]
    orelhaD_pos = [float(points['OrelhaD']), float(points['OrelhaD.1'])]
    centro_pos = [float(points['Centro']), float(points['Centro.1'])]
    rabo_pos = [float(points['Rabo']), float(points['Rabo.1'])]
    original_x_keypoints = [focinho_pos[0], orelhaE_pos[0], orelhaD_pos[0], centro_pos[0], rabo_pos[0]]
    original_y_keypoints = [focinho_pos[1], orelhaE_pos[1], orelhaD_pos[1], centro_pos[1], rabo_pos[1]]

    keypoints = KeypointsOnImage([
        Keypoint(x=focinho_pos[0], y=focinho_pos[1]),
        Keypoint(x=orelhaE_pos[0], y=orelhaE_pos[1]),
        Keypoint(x=orelhaD_pos[0], y=orelhaD_pos[1]),
        Keypoint(x=centro_pos[0], y=centro_pos[1]),
        Keypoint(x=rabo_pos[0], y=rabo_pos[1])
    ], shape=img_to_aug.shape)

    rotate_image = iaa.Rotate(20)
    blur_image = iaa.imgcorruptlike.MotionBlur(severity=5)
    x_rotated = []
    y_rotated = []
    x_blurred = []
    y_blurred = []

    # Augment the image and the keypoints
    rotated_image, rotated_kps = rotate_image(image=img_to_aug, keypoints=keypoints)
    blurred_image, blurred_kps = blur_image(image=img_to_aug, keypoints=keypoints)

    for i in range(0, len(blurred_kps)):
        x_blurred.append(blurred_kps.keypoints[i].x)
        y_blurred.append(blurred_kps.keypoints[i].y)

    for i in range(0, len(rotated_kps)):
        x_rotated.append(rotated_kps.keypoints[i].x)
        y_rotated.append(rotated_kps.keypoints[i].y)

    fig, ax = plt.subplots(2, 2, figsize=(15, 8))
    ax[0][0].imshow(img_to_aug)
    ax[0][0].plot(original_x_keypoints, original_y_keypoints, 'ro', markersize=3)
    ax[0][0].set_title('Original Image')
    ax[1][0].imshow(rotated_image)
    ax[1][0].plot(x_rotated, y_rotated, 'ro', markersize=3)
    ax[1][0].set_title('Rotated Image')
    ax[0][1].imshow(img_to_aug)
    ax[0][1].plot(original_x_keypoints, original_y_keypoints, 'ro', markersize=3)
    ax[0][1].set_title('Original Image')
    ax[1][1].imshow(blurred_image)
    ax[1][1].plot(x_blurred, y_blurred, 'ro', markersize=3)
    ax[1][1].set_title('Blurred Image')
    plt.show()

    pass


