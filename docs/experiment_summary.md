## Experiment glossary
This is a 'glossary' file, which goal is to describe in a short way what are the constraints of each experiment code file.

### 001_cnn_optimizer.ipynb
- location: `../../notebooks/`
- description: notebook for CNN hyperparameters optimization using random search
- constraints:
    - balanced dataset: segmented nodule images
    - input: images
    - output: hyperparameters of the model with the best AUC

### 002_features_extraction_and_fusion.ipynb
- location: `../../notebooks/`
- description: notebook for deep features extraction and fusion with radiomics
- constraints:
    - balanced dataset: segmented nodule images
    - input: images
    - output: deep features of the 3D nodule images, fused or not with the respective radiomics of the nodules.

### 003_genetic_algorithms_optimization.py
- location: `../../local_env_codes/`
- description: script to optimize a SVM (with random search) for a set of features and optimize the set of features for the optimized SVM (with G.A.)
- constraints:
    - balanced dataset: deep and/or radiomic features of the nodules
    - input: structured (.csv) table with the features of the nodules
    - output: result of a 10-fold cross-validation of the optimized SVM for the optimized subset of features.