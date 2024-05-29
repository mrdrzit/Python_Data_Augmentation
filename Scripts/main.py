import pandas as pd
import numpy as np
import imgaug.augmenters as iaa
from matplotlib import pyplot as plt
from PIL import Image
from tkinter import filedialog
from pathlib import Path

# Load the image files using tkinter
images_path = ["F:\Matheus\GITHUB\Python_Data_Augmentation\images\img1091.png", 
    "F:\Matheus\GITHUB\Python_Data_Augmentation\images\img1654.png", 
    "F:\Matheus\GITHUB\Python_Data_Augmentation\images\img3635.png", 
    "F:\Matheus\GITHUB\Python_Data_Augmentation\images\img4638.png", 
    "F:\Matheus\GITHUB\Python_Data_Augmentation\images\img5948.png", 
    "F:\Matheus\GITHUB\Python_Data_Augmentation\images\img6303.png", 
    "F:\Matheus\GITHUB\Python_Data_Augmentation\images\img7761.png", 
    "F:\Matheus\GITHUB\Python_Data_Augmentation\images\img8630.png"
]

csv_files_path = ["F:\Matheus\GITHUB\Python_Data_Augmentation\images\CollectedData_matheus copy 2.csv",
                  "F:\Matheus\GITHUB\Python_Data_Augmentation\images\CollectedData_matheus copy.csv",
                  "F:\Matheus\GITHUB\Python_Data_Augmentation\images\CollectedData_matheus.csv"
]

# images_path = filedialog.askopenfilenames(title='Select the images', filetypes=[('Image Files', '*.jpg *.png *.jpeg')])
# csv_files_path = filedialog.askopenfilenames(title='Select the csv files', filetypes=[('CSV Files', '*.csv')])

# The read dataframe function already imports the data as a dataframe, so it probably easier to create a copy of the dataframe and modify
def read_csv(csv_files_path):
    dataframes = []
    for csv_file in csv_files_path:
        df = pd.read_csv(csv_file)
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


img_to_aug = np.asarray(Image.open(images_path[0]))

blur_image = iaa.imgcorruptlike.MotionBlur(severity=5)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img_to_aug)
ax[0].set_title('Original Image')
ax[1].imshow(blur_image.augment_image(img_to_aug))
ax[1].set_title('Augmented Image')
plt.show()

pass


