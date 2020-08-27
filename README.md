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
    |__ 001_<experiment title>
    |__ 002_<experiment title>
    ...
    |__ n_<experiment title>

|__ local_env_codes
    |__ 001_<optmization title>
    |__ 002_<optmization title>
    ...
    |__ n_<optmization short description>

|__ docs
    |__ experiment_summary.md
    |__ requirements.txt

|__ results
    |__ convX_denseY_dropZ_denseW_dropK
        |__ 001_<optmization title>
            |__ rocs
            |__ terminal_outputs
        |__ 002_<optmization title>
        ...
        |__ n_experiment_short_description
```

### Running optimization
Example genetic algorithms optimization (it needs to be executed for each of the radiomic files):
```
python local_env_codes/001_genetic_algorithms_optimization.py -r results/conv64_dense128_drop04406_dense32_drop14918/001_genetic_algorithms_optimization/rocs/ -f data/features/deep_features_with_radiomics/conv64_dense128_drop04406_dense32_drop14918/dense2_texture_features.csv > results/conv64_dense128_drop04406_dense32_drop14918/001_genetic_algorithms_optimization/terminal_outputs/dense2_texture_features.txt
```

Example of execution for other types of optimization:
```
python local_env_codes/005_select_percentile_10_chi2.py -f data/features/deep_features_with_radiomics/conv96_dense64_drop241_dense24_drop236/ -r results/conv96_dense64_drop241_dense24_drop236/005_select_percentile_10_chi2/rocs/ > results/conv96_dense64_drop241_dense24_drop236/005_select_percentile_10_chi2/terminal_outputs/output.txt
```
