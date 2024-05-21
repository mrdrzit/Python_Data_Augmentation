# Image Augmentation and DeepLabCut Training Repository

## Repository Overview

This repository provides a comprehensive script for augmenting images and their corresponding keypoints, and subsequently using the augmented data to retrain a DeepLabCut model. The script includes steps for data preparation, augmentation, and model retraining, ensuring a streamlined workflow for enhancing image datasets and improving model performance.

### Repository Structure

- `scripts/`
  - `data_preparation.py`: Script for loading images and keypoints from a CSV file.
  - `augmentation_pipeline.py`: Script defining and applying the augmentation pipeline to images and keypoints.
  - `save_augmented_data.py`: Script for saving augmented images and creating a new CSV file.
  - `retrain_model.py`: Script for retraining the DeepLabCut model using the augmented data.
- `data/`
  - `raw_images/`: Directory for storing the original images.
  - `augmented_images/`: Directory for storing the augmented images.
  - `labels.csv`: CSV file containing the paths to images and their corresponding keypoints.
  - `augmented_labels.csv`: Generated CSV file with paths and keypoints of augmented images.

### Steps to Use This Repository

1. **Import Necessary Libraries**

   ```python
   import pandas as pd  # For handling dataframes
   import numpy as np  # For numerical operations
   import cv2  # For image processing
   import albumentations as A  # For image augmentation
   import os  # For handling file paths and directories
   ```

2. **Load Dataset**

   - Read the `labels.csv` file that contains the image paths and keypoints.
   - Initialize lists to store image data and labels.
   - Iterate through the CSV, read each image, and store its data and keypoints.

3. **Define Augmentation Pipeline**

   - Define a sequence of augmentations including horizontal flip, rotation, scaling, and brightness adjustment using `albumentations`.

4. **Apply Augmentation and Adjust Keypoints**

   - Iterate over the loaded images and keypoints.
   - Convert keypoints to the format required by the augmentation library.
   - Apply the augmentations and adjust the keypoints accordingly.
   - Store the augmented images and keypoints.

5. **Save Augmented Images and Create New CSV**

   - Create a directory for augmented images.
   - Save each augmented image and prepare a new CSV file with updated paths and keypoints.
   - Save the new CSV file.

6. **Verify and Retrain**

   - Import the DeepLabCut library.
   - Load the augmented data and retrain the model using DeepLabCut.

### Usage Instructions

- Coming soon

### Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Ensure your code adheres to the project's coding standards and includes appropriate documentation.

### License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

This repository aims to simplify the process of augmenting image datasets and improving the performance of DeepLabCut models. By following the steps outlined, users can efficiently prepare their data, apply augmentations, and retrain their models for enhanced accuracy and robustness.
