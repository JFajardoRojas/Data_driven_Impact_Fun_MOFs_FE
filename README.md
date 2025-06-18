# ML Data for: Data-driven Insights on the Impact of functionalization on Metal-Organic Framework (MOF) Free Energies
Fernando Fajardo-Rojas, Ryther Anderson, Mingwei Li, Remco Chang, Diego A. Gómez-Gualdrón

This repository contains the complete datasets used in the analysis presented in the publication, the machine learning models developed to predict the change in free-energy of Metal-Organic Frameworks (MOFs) upon functionalization, example codes to complete the free energy calculations, and template codes to train XGBoost using the datasets for different levels of descriptor complexity.

## Repository Contents

### **Complementary_code**
This folder contains the training and usage template python codes to train XGBoost using the datasets available in this repository (general_training_model.py, model_use.py).
It also has a solder **Simulation_Eample** with all the files needed to reproduce the free-energy calculations in LAMMPS. There is a README file in the folder explaining its usage.

### **Datasets**
This folder contains 2 .csv files. These have all the relevant data for free energy analysis in the 5133 MOFs (GENERAL_data_DimBridge.csv), and the relevant data for the topological change via functionalization in 62 polymorphic families (POLYMORPHS_data_DimBridge.csv)

### **dSE_model**
This folder contains the complete dataset used in the model development (data_set_model_dSE.csv), the final dataset division 80/20 (train_data_dSE.csv / test_data_dSE.csv), the final model trained (xgb_model_dSE.json), and its scaler file (scaler_dSE.pkl).
This model uses as descriptors information about the parent MOF, the functionalization, and the change in strain energy.

### **dUdecouple_model**
This folder contains the complete dataset used in the model development (data_set_model_dUdecouple.csv), the final dataset division 80/20 (train_data_dUdecouple.csv / test_data_dUdecouple.csv), the final model trained (xgb_model_dUdecouple.json), and its scaler file (scaler_dUdecouple.pkl).
This model uses as descriptors information about the parent MOF, the functionalization, and the energetic components of the change in potential energy (∆U).

### **prediction_model**
This folder contains the complete dataset used in the model development (data_set_model_prediction.csv), the final dataset division 80/20 (train_data_prediction.csv / test_data_prediction.csv), the final model trained (xgb_model_prediction.json), and its scaler file (scaler_prediction.pkl).
This model uses as descriptors information about the parent MOF and the functionalization.

### **trivial**
This folder contains the complete dataset used in the model development (data_set_model_trivial.csv), the final dataset division 80/20 (train_data_trivial.csv / test_data_trivial.csv), the final model trained (xgb_model_trivial.json), and its scaler file (scaler_trivial.pkl).
This model uses as descriptors information about the parent MOF, the functionalization, change in potential energy (∆U), and change in entropy (T∆S).

## How to Use

### Requirements
To use or test the models, you will need to install the following Python packages
- `scikit-learn 1.4.2`
- `numpy 2.1.0`
- `xgboost 2.1.3`
- `matplotlib (Compatible Python 3.11.9)`
- `joblib`
