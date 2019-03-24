# Pulmonary nodule classification

Automatic classification on Pulmonary Nodules of Computed Tomography Image - Dicom.

# Necessary libraries
1. hyperopt
2. hyperas
3. sklearn
4. keras_metrics
5. tqdm

# Initial ideas
1. Data exploration (initialy, just visualizations) on the Features dataset;
2. Create a classifier (CNN) using the images as inputs;
3. Create a classifier that uses the Features and Images as Input of the network;
4. Create a CNN archtecture wich receives the images as input and injects the Features after the flatten layer.

# Workflow
1. Apply the Dicom Window on images;
2. Download the images from de database - Solid nodules and Solid nodules with attributes;
3. Crop the images using Phatch, configuring que percentage as 37.5% - It results on 64x64 images;
4. Modify the import_images.py to read and prepare the images as numpy arrays;
5. Use the numpy arrays prepared on **4** to be the inputs of the network;
6. Evaluate the model metrics using Accuracy, Specificity, Sensitivity and AUC.
