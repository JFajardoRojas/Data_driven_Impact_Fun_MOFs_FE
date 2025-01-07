# MOF Functionalization Free Energy Prediction Models

This repository contains machine learning models developed to predict the change in free-energy of Metal-Organic Frameworks (MOFs) upon functionalization. The repository includes the following data and models:

## Repository Contents

### 1. **Training Data**
The training data used for the development of each model is included in the repository. These datasets contain energy-related features that were used to train the models.

- **Trivial Model Data**: This dataset includes energy information such as ∆U, T∆S, and other energy-related features.
- **Proper Model 1 Data**: This dataset contains only the components of the potential energy as features.
- **Proper Model 2 Data**: This dataset uses ∆E of strain as a surrogate for ∆U as the unique energy information in the features.
- **Proper Model 3 Data**: This dataset includes only information about the parent MOF and its functionalization scheme as features.

### 2. **Trained Models**
The repository contains the trained machine learning models for predicting the change in free-energy in MOFs upon functionalization. These models have been trained using the datasets mentioned above.

- **Trivial Model**: A baseline model that includes a variety of energy-related features, including ∆U and T∆S.
- **Model 1**: A model focused solely on potential energy components.
- **Model 2**: A model that uses ∆E of strain as a surrogate for ∆U in the feature space.
- **Model 3**: A model that uses the parent MOF structure and functionalization scheme for predictions.

### 3. **Scalers**
Each model was trained with specific scalers to normalize the input data. The corresponding scalers used for each model are included in the repository.

## How to Use

### Requirements
To use or test the models, you will need to install the following Python packages:
- `scikit-learn`
- `pandas`
- `numpy`
- `xgboost`
- `matplotlib`
- `joblib`
