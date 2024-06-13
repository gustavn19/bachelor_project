# Bachelor 
#### Repository for the bachelor project: Investigating the suitability of performance metrics for medical generative image models

The repository includes the following files:

- chexpert_model.py --> Used to train InceptionV3 model on new data
- data_processing.ipynb --> Used to do general data processing both prior to modelling and evaluation.
- eval_model.py --> Used to evaluate InceptionV3 model
- fid_experiment.py --> Structure for performing experiment 1 and 2 of the project
- fid_experiment_ny.py --> Structure for performing experiment 3
- nih_image_load.ipynb --> Used to process NIH data
- bachelor_environment.yml --> Conda environment for performing experiments

The datafiles can be found through the links in the report (and below), and the NIH-InceptionV3 weights, obtained and used in the project, are availabe in the following google drive: https://drive.google.com/drive/folders/1O54bDzbLatTM1bXEPxMEFr8af1E4KK-r?usp=sharing

## External resources used in the project:

- The weights and project from RadImageNet: https://github.com/BMEII-AI/RadImageNet/blob/main/README.md
- The original CheXpert dataset and project: https://stanfordmlgroup.github.io/competitions/chexpert/
    - The resized CheXpert dataset: https://www.kaggle.com/datasets/willarevalo/chexpert-v10-small 
- The NIH ChestXray dataset: https://nihcc.app.box.com/v/ChestXray-NIHCC
