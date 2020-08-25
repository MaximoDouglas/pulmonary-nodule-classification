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
python local_env_codes/004_select_k_best_chi2_k5_optimization.py -f data/features/deep_features_with_radiomics/conv96_dense64_drop241_dense24_drop236/ -r results/conv96_dense64_drop241_dense24_drop236/select_k_best_chi2_k5/rocs/ > results/conv96_dense64_drop241_dense24_drop236/select_k_best_chi2_k5/terminal_outputs/output.txt
```
