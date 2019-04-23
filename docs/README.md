# Pulmonary nodule classification

Classification model on Pulmonary Nodules of Computed Tomography Image - Dicom.

## Pipeline
1. Download the images of solid nodules and its features from the database;
2. Crop the images using Phatch, configuring que percentage as 37.5% - It results on 64x64 images;
3. Generate numpies for model optimization;
4. Optimize model (images only) using the numpies generated on item **4**;
5. Validate model obtained on item **5** 
6. Validate model obtained on item **5** using features injection;
7. Compare results.

## Next Steps:
1. Generate numpies for model optimization;
2. Optimize model (images with features injection) using the numpies generated on item **1**;
3. Validate models obtained on item **2**;

## Bibliographic revision schedule
1. NoduleX - Causey, L.;
2. Modelo Computacional para Classificação de Nódulos Pulmonares Utilizando Redes Neurais Convolucionais - Lins, Lucas.;
3. 3D multi-view convolutional neural networks for lung nodule classification - Kang, G.;
4. An introduction to ROC analysis - Fawcett, T.;
5. Performance evaluation of classification algorithms by k-fold and leave-one-out cross validation - Wong, T.;
6. [A Conceptual Explanation of Bayesian Hyperparameter Optimization for Machine Learning](https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f);
7. [Deep Learning with Magnetic Resonance and Computed Tomography Images](https://towardsdatascience.com/deep-learning-with-magnetic-resonance-and-computed-tomography-images-e9f32273dcb5).
