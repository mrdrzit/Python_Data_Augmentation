import pandas as pd
import numpy as np
import imgaug.augmenters as iaa
import os
from imgaug.augmentables import Keypoint, KeypointsOnImage
from matplotlib import pyplot as plt
from PIL import Image

image_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images"))
# Load the image files using tkinter
images_path = list(
    [
        os.path.join(image_folder, file)
        for file in os.listdir(image_folder)
        if file.endswith(".png") or file.endswith(".jpg")
    ]
)
images_names = list([file for file in os.listdir(image_folder) if file.endswith(".png") or file.endswith(".jpg")])
csv_files_path = list([os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith(".csv")])


# The read dataframe function already imports the data as a dataframe, so its probably easier to create a copy of the dataframe and modify it
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
            data = df.iloc[[0, 1, index]]
            self.image_data.append({name: data})


animal_aug_list = []
csv_list = read_csv(csv_files_path)

colortemperature = 6000
sigma = 3
gamma = 1.5
sharpen_alpha = 0.7
sharpen_lightness = 1.3
saturation = 2
desaturation = 0.5


for csv_data_frame in csv_list:
    animal = Animal()
    animal.populate_bodyparts(csv_data_frame)
    animal.csv.append(csv_data_frame)
    animal_aug_list.append(animal)

for image, image_name in zip(images_path, images_names):
    img_to_aug = np.asarray(Image.open(image))
    points_dataframe = animal.csv[0].set_index(animal.csv[0].columns[0], inplace=False)

    points = points_dataframe.loc[image_name]

    focinho_pos = [float(points["Focinho"]), float(points["Focinho.1"])]
    orelhaE_pos = [float(points["OrelhaE"]), float(points["OrelhaE.1"])]
    orelhaD_pos = [float(points["OrelhaD"]), float(points["OrelhaD.1"])]
    centro_pos = [float(points["Centro"]), float(points["Centro.1"])]
    rabo_pos = [float(points["Rabo"]), float(points["Rabo.1"])]
    original_x_keypoints = [focinho_pos[0], orelhaE_pos[0], orelhaD_pos[0], centro_pos[0], rabo_pos[0]]
    original_y_keypoints = [focinho_pos[1], orelhaE_pos[1], orelhaD_pos[1], centro_pos[1], rabo_pos[1]]

    keypoints = KeypointsOnImage(
        [
            Keypoint(x=focinho_pos[0], y=focinho_pos[1]),
            Keypoint(x=orelhaE_pos[0], y=orelhaE_pos[1]),
            Keypoint(x=orelhaD_pos[0], y=orelhaD_pos[1]),
            Keypoint(x=centro_pos[0], y=centro_pos[1]),
            Keypoint(x=rabo_pos[0], y=rabo_pos[1]),
        ],
        shape=img_to_aug.shape,
    )

    white_balance = iaa.ChangeColorTemperature(colortemperature)
    blur_image = iaa.imgcorruptlike.MotionBlur(sigma)
    change_gamma = iaa.GammaContrast(gamma)
    sharpen_image = iaa.Sharpen(alpha=sharpen_alpha, lightness=sharpen_lightness)
    multiply_saturation = iaa.MultiplySaturation(saturation)
    remove_saturation = iaa.RemoveSaturation(desaturation)

    half_white_balance = iaa.ChangeColorTemperature(colortemperature * 2)
    half_blur_image = iaa.imgcorruptlike.MotionBlur(round(sigma / 2))
    half_change_gamma = iaa.GammaContrast(gamma / 2)
    half_sharpen_image = iaa.Sharpen(alpha=sharpen_alpha / 2, lightness=sharpen_lightness / 2)
    half_multiply_saturation = iaa.MultiplySaturation(saturation / 2)
    half_remove_saturation = iaa.RemoveSaturation(desaturation / 2)

    x_balanced = []
    y_balanced = []
    x_blurred = []
    y_blurred = []
    x_gamma = []
    y_gamma = []
    x_sharpened = []
    y_sharpened = []
    x_multiplied_saturation = []
    y_multiplied_saturation = []
    x_removed_saturation = []
    y_removed_saturation = []

    x_half_balanced = []
    y_half_balanced = []
    x_half_blurred = []
    y_half_blurred = []
    x_half_gamma = []
    y_half_gamma = []
    x_half_sharpened = []
    y_half_sharpened = []
    x_half_multiplied_saturation = []
    y_half_multiplied_saturation = []
    x_half_removed_saturation = []
    y_half_removed_saturation = []

    # Augment the image and the keypoints
    balanced_image, balanced_kps = white_balance(image=img_to_aug, keypoints=keypoints)
    blurred_image, blurred_kps = blur_image(image=img_to_aug, keypoints=keypoints)
    changend_gamma_image, changed_gamma_keypoints = change_gamma(image=img_to_aug, keypoints=keypoints)
    sharpened_image, sharpened_keypoints = sharpen_image(image=img_to_aug, keypoints=keypoints)
    multiplied_saturation_image, multiplied_saturation_keypoints = multiply_saturation(
        image=img_to_aug, keypoints=keypoints
    )
    removed_saturation_image, removed_saturation_keypoints = remove_saturation(image=img_to_aug, keypoints=keypoints)

    half_balanced_image, half_balanced_kps = half_white_balance(image=img_to_aug, keypoints=keypoints)
    half_blurred_image, half_blurred_kps = half_blur_image(image=img_to_aug, keypoints=keypoints)
    half_changend_gamma_image, half_changed_gamma_keypoints = half_change_gamma(image=img_to_aug, keypoints=keypoints)
    half_sharpened_image, half_sharpened_keypoints = half_sharpen_image(image=img_to_aug, keypoints=keypoints)
    half_multiplied_saturation_image, half_multiplied_saturation_keypoints = half_multiply_saturation(
        image=img_to_aug, keypoints=keypoints
    )
    half_removed_saturation_image, half_removed_saturation_keypoints = half_remove_saturation(
        image=img_to_aug, keypoints=keypoints
    )

    image_list = [
        balanced_image,
        blurred_image,
        changend_gamma_image,
        sharpened_image,
        multiplied_saturation_image,
        removed_saturation_image,
        half_balanced_image,
        half_blurred_image,
        half_changend_gamma_image,
        half_sharpened_image,
        half_multiplied_saturation_image,
        half_removed_saturation_image,
    ]

    aug_list = [
        'balanced_image',
        'blurred_image',
        'changend_gamma_image',
        'sharpened_image',
        'multiplied_saturation_image',
        'removed_saturation_image',
        'half_balanced_image',
        'half_blurred_image',
        'half_changend_gamma_image',
        'half_sharpened_image',
        'half_multiplied_saturation_image',
        'half_removed_saturation_image'
    ]
    

    # Save the augmented images
    for i, image_to_save in enumerate(image_list):
        augmentation = aug_list[i]
        path_to_save = os.path.join(image_folder, "augmented_images", f"{image_name}_{augmentation}.png")
        image_to_save = Image.fromarray(image_to_save)
        image_to_save.save(path_to_save)
        print(f"Image saved at {path_to_save}")

    if (
        len(blurred_kps)
        == len(balanced_kps)
        == len(changed_gamma_keypoints)
        == len(sharpened_keypoints)
        == len(multiplied_saturation_keypoints)
        == len(removed_saturation_keypoints)
        == len(half_blurred_kps)
        == len(half_balanced_kps)
        == len(half_changed_gamma_keypoints)
        == len(half_sharpened_keypoints)
        == len(half_multiplied_saturation_keypoints)
        == len(half_removed_saturation_keypoints)
    ):
        for i in range(0, len(blurred_kps)):
            x_blurred.append(blurred_kps.keypoints[i].x - 5)
            y_blurred.append(blurred_kps.keypoints[i].y)

            x_balanced.append(balanced_kps.keypoints[i].x)
            y_balanced.append(balanced_kps.keypoints[i].y)

            x_gamma.append(changed_gamma_keypoints.keypoints[i].x)
            y_gamma.append(changed_gamma_keypoints.keypoints[i].y)

            x_sharpened.append(sharpened_keypoints.keypoints[i].x)
            y_sharpened.append(sharpened_keypoints.keypoints[i].y)

            x_multiplied_saturation.append(multiplied_saturation_keypoints.keypoints[i].x)
            y_multiplied_saturation.append(multiplied_saturation_keypoints.keypoints[i].y)

            x_removed_saturation.append(removed_saturation_keypoints.keypoints[i].x)
            y_removed_saturation.append(removed_saturation_keypoints.keypoints[i].y)

            x_half_blurred.append(half_blurred_kps.keypoints[i].x - 4)
            y_half_blurred.append(half_blurred_kps.keypoints[i].y)

            x_half_balanced.append(half_balanced_kps.keypoints[i].x)
            y_half_balanced.append(half_balanced_kps.keypoints[i].y)

            x_half_gamma.append(half_changed_gamma_keypoints.keypoints[i].x)
            y_half_gamma.append(half_changed_gamma_keypoints.keypoints[i].y)

            x_half_sharpened.append(half_sharpened_keypoints.keypoints[i].x)
            y_half_sharpened.append(half_sharpened_keypoints.keypoints[i].y)

            x_half_multiplied_saturation.append(half_multiplied_saturation_keypoints.keypoints[i].x)
            y_half_multiplied_saturation.append(half_multiplied_saturation_keypoints.keypoints[i].y)

            x_half_removed_saturation.append(half_removed_saturation_keypoints.keypoints[i].x)
            y_half_removed_saturation.append(half_removed_saturation_keypoints.keypoints[i].y)
    else:
        print("The number of keypoints is different for each augmentation")
        break

    fig, ax = plt.subplots(3, 6, figsize=(15, 8))
    ax[0][0].imshow(img_to_aug)
    ax[0][0].plot(original_x_keypoints, original_y_keypoints, "ro", markersize=3)
    ax[0][0].set_title("Original Image")
    ax[1][0].imshow(balanced_image)
    ax[1][0].plot(x_balanced, y_balanced, "ro", markersize=3)
    ax[1][0].set_title(f"balanced Image - {colortemperature}K")
    ax[0][1].imshow(img_to_aug)
    ax[0][1].plot(original_x_keypoints, original_y_keypoints, "ro", markersize=3)
    ax[0][1].set_title("Original Image")
    ax[1][1].imshow(blurred_image)
    ax[1][1].plot(x_blurred, y_blurred, "ro", markersize=3)
    ax[1][1].set_title(f"Blurred Image - Sigma = {sigma}")
    ax[0][2].imshow(img_to_aug)
    ax[0][2].plot(original_x_keypoints, original_y_keypoints, "ro", markersize=3)
    ax[0][2].set_title("Original Image")
    ax[1][2].imshow(changend_gamma_image)
    ax[1][2].plot(x_gamma, y_gamma, "ro", markersize=3)
    ax[1][2].set_title(f"Changed Gamma Image - Gamma = {gamma}")
    ax[0][3].imshow(img_to_aug)
    ax[0][3].plot(original_x_keypoints, original_y_keypoints, "ro", markersize=3)
    ax[0][3].set_title("Original Image")
    ax[1][3].imshow(sharpened_image)
    ax[1][3].plot(x_sharpened, y_sharpened, "ro", markersize=3)
    ax[1][3].set_title(f"Sharpened Image - Alpha, Lightness = {sharpen_alpha, sharpen_lightness}")
    ax[0][4].imshow(img_to_aug)
    ax[0][4].plot(original_x_keypoints, original_y_keypoints, "ro", markersize=3)
    ax[0][4].set_title("Original Image")
    ax[1][4].imshow(multiplied_saturation_image)
    ax[1][4].plot(x_multiplied_saturation, y_multiplied_saturation, "ro", markersize=3)
    ax[1][4].set_title(f"Multiplied Saturation Image - Saturation = {saturation}")
    ax[0][5].imshow(img_to_aug)
    ax[0][5].plot(original_x_keypoints, original_y_keypoints, "ro", markersize=3)
    ax[0][5].set_title("Original Image")
    ax[1][5].imshow(removed_saturation_image)
    ax[1][5].plot(x_removed_saturation, y_removed_saturation, "ro", markersize=3)
    ax[1][5].set_title(f"Removed Saturation Image - Desaturation = {desaturation}")
    ax[2][0].imshow(half_balanced_image)
    ax[2][0].plot(x_half_balanced, y_half_balanced, "ro", markersize=3)
    ax[2][0].set_title(f"Half balanced Image - {colortemperature*2}K")
    ax[2][1].imshow(half_blurred_image)
    ax[2][1].plot(x_half_blurred, y_half_blurred, "ro", markersize=3)
    ax[2][1].set_title(f"Half Blurred Image - Sigma = {sigma/2}")
    ax[2][2].imshow(half_changend_gamma_image)
    ax[2][2].plot(x_half_gamma, y_half_gamma, "ro", markersize=3)
    ax[2][2].set_title(f"Half Changed Gamma Image - Gamma = {gamma/2}")
    ax[2][3].imshow(half_sharpened_image)
    ax[2][3].plot(x_half_sharpened, y_half_sharpened, "ro", markersize=3)
    ax[2][3].set_title(f"Half - Alpha, Lightness = {sharpen_alpha/2, sharpen_lightness/2}", wrap=True)
    ax[2][4].imshow(half_multiplied_saturation_image)
    ax[2][4].plot(x_half_multiplied_saturation, y_half_multiplied_saturation, "ro", markersize=3)
    ax[2][4].set_title(f"Half - Saturation = {saturation/2}", wrap=True)
    ax[2][5].imshow(half_removed_saturation_image)
    ax[2][5].plot(x_half_removed_saturation, y_half_removed_saturation, "ro", markersize=3)
    ax[2][5].set_title(f"Half - Desaturation = {desaturation/2}")

    # plt.tight_layout()
    # plt.show()

    pass
