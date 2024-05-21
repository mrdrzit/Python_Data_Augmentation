import pandas as pd
import numpy as np
import imgaug 
from tkinter import filedialog
from pathlib import Path

# Load the image files using tkinter
images_path = filedialog.askopenfilenames(title='Select the images', filetypes=[('Image Files', '*.jpg *.png *.jpeg')])
csv_files_path = filedialog.askopenfilenames(title='Select the csv files', filetypes=[('CSV Files', '*.csv')])

# TODO create a function to load the csv files into a dataframe where the header is the bodyparts row of the csv, 
# the first row is the name of the image file and the columns are the x and y coordinates of the bodyparts in the same order as the header
# For this you will need the pandas library to read the csv file and create the dataframe
# Hint to read the csv:
# https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

# Hint to create the dataframe:
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html

# The read dataframe function already imports the data as a dataframe, so it probably easier to create a copy of the dataframe and modify it
# into a second variable that will be used to store the data in the format described above