## Pulmonary Nodules Classification

### Project structure
```
./
|__ data
    |__ images
        |__ benigno
        |__ maligno
    |__ features
        |__ radiomics
        |__ deep_features_with_radiomics
        |__ legacy_deep_features_with_radiomics

|__ notebooks
    |__ 001_experiment_short_description
    |__ 002_experiment_short_description
    ...
    |__ n_experiment_short_description

|__ local_env_codes
    |__ 003_genetic_algorithms_optimization

|__ docs
    |__ experiment_summary.md
    |__ requirements.txt

|__ results
    |__ convX_denseY_dropZ_denseW_dropK
        |__ genetic_algorithms
            |__ rocs
            |__ terminal_outputs
```
### Running optimization
Example:
```
python local_env_codes/003_genetic_algorithms_optimization.py -r results/conv64_dense128_drop04406_dense32_drop14918/genetic_algorithms/rocs/ -f data/features/deep_features_with_radiomics/conv64_dense128_drop04406_dense32_drop14918/dense1_shape_features.csv > results/conv64_dense128_drop04406_dense32_drop14918/genetic_algorithms/terminal_outputs/dense1_shape_features.txt
```
