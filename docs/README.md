# Pulmonary nodule classification

Classification model on Pulmonary Nodules of Computed Tomography Image - Dicom.

## Pipeline
1. Download the images of solid nodules and its features from the database;
2. Crop the images using Phatch, configuring que percentage as 37.5% - It results on 64x64 images;
3. Generate numpies for model optimization;
4. Optimize model (images only) using the numpies generated on item **3**;
5. Validate model obtained on item **4** 
6. Validate model obtained on item **4** using features injection;
7. Compare results.

## Next Steps:
- [ ] Code for optimize model with features injection;

## Literature read schedule
[ ] Causey, L. "Highly accurate model for prediction of lung nodule malignancy with CT scans";
[ ] Lins, L. "Modelo Computacional para Classificação de Nódulos Pulmonares Utilizando Redes Neurais Convolucionais";
[ ] Kang, G. "3D multi-view convolutional neural networks for lung nodule classification";
[ ] Fawcett, T. "An introduction to ROC analysis";
[ ] Wong, T. "Performance evaluation of classification algorithms by k-fold and leave-one-out cross validation";
[ ] Koehrsen, W. ["A Conceptual Explanation of Bayesian Hyperparameter Optimization for Machine Learning"](https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f);
[ ] Reinhold, J. ["Deep Learning with Magnetic Resonance and Computed Tomography Images"](https://towardsdatascience.com/deep-learning-with-magnetic-resonance-and-computed-tomography-images-e9f32273dcb5).
