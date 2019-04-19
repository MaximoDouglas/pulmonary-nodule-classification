# Pulmonary nodule classification

Classification model on Pulmonary Nodules of Computed Tomography Image - Dicom.

# Future Pipeline
2. Download the images of solid nodules and its features from the database;
3. Crop the images using Phatch, configuring que percentage as 37.5% - It results on 64x64 images;
4. Generate numpies for model optimization;
5. Optimize model (images only) using the numpies generated on item **4**;
6. Optimize model (images with features injection) using the numpies generated on item **4**;
7. Validate models obtained on items **5** and **6**;
8. Compare results.

# Pipeline
2. Download the images of solid nodules and its features from the database;
3. Crop the images using Phatch, configuring que percentage as 37.5% - It results on 64x64 images;
4. Generate numpies for model optimization;
5. Optimize model (images only) using the numpies generated on item **4**;
6. Validate model obtained on item **5** 
7. Validate model obtained on item **5** using features injection;
8. Compare results.
