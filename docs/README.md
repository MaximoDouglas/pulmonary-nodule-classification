# Pulmonary nodule classification

Automatic classification on Pulmonary Nodules of Computed Tomography Image.

# Initial ideas
1. Data exploration (initialy, just visualizations) on the Features dataset;
2. Create a classifier just for the features;
  
  - Features of the 3mm-10mm nodules;
  - Features of the 5mm-30mm nodules;

3. Execute some Data cleaning and/or transformation on the Features dataset (if it shows up as a good thing).
4. Create a classifier that uses the pixels on the images;
5. Create a classifier (CNN) using the images as inputs;
6. Create a classifier that uses the Features and Pixels as Input on a dense archtecture;
7. Create a CNN archtecture wich receives the images as input and injects the Features after the flatten layer.

Obs.: The steps 1 and 2 will be executed in parallel.
