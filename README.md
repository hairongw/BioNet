# Biologically-informed deep neural networks provide quantitative assessment of intratumoral heterogeneity in post-treatment glioblastoma
![Picture1](https://github.com/hairongw/BioNet/assets/30871667/e2d4f3c6-592e-443a-924a-fe1452a8531d)

**BioNet**: A biologically-informed neural network model, to predict regional distributions of three tissue-specific gene modules: proliferating tumor (Pro), reactive/inflammatory cells (Inf), and infiltrated brain tissue (Neu) by using texture features extracted from multiparametric Magnetic Resonance Imaging (MRI). BioNet offers valuable insights into the integration of multiple implicit and qualitative biological domain knowledge. BioNet performs significantly better than a range of existing methods on cross-validation and blind test datasets. 

BioNet consists of two networks: BioNet_Neu to predict **Neu** using MRI; BioNet_ProInf to simultaneously predict **Pro** and **Inf** using MRI. BioNet_Neu is a feedforward neural network pre-trained using a large number of unlabeled samples with noisy Neu labels informed by biological knowledge, and fine-tuned using biopsy samples with data augmentation. It also corporates Monte Carlo dropout to enable uncertainty quantification for the predictions. The role of BioNet_Neu is to stratify unlabeled samples with high predictive certainty, which were then incorporated into the training of BioNet_ProInf. BioNet_ProInf is a multitask semi-supervised learning model. The architecture consists of a shared block and task-specific blocks. The loss function combines a prediction loss and a knowledge attention loss that penalizes the violation of the knowledge-based relationships on unlabeled samples.

This is a PyTorch implementation of [this paper](https://www.biorxiv.org/content/10.1101/2022.12.20.521086v3.full.pdf).

## Dataset
- **MRI**: Due to restrictions on data privacy, we are not allowed to upload raw MRI images. To ensure that shared data remains non-identifiable, we have uploaded texture features that were extracted from raw MRIs. These texture features are the input of BioNet. The datasets for Cohort A and Cohort B are accessible on Figshare through [the project page](https://figshare.com/projects/Texture_features_of_Multiparametric_MRI_-_Recurrent_Glioblastoma/193223). The detailed methodology for this texture feature extraction is comprehensively outlined in [Hu et al. (2021)](https://www.nature.com/articles/s41598-021-83141-z). For further reference and implementation, the extraction codes can be found in the folder Texture Feature Extraction.

- **RNAseq and Enrichment Analysis**: published in [Al-Dalahmah et al. (2023)](https://www.nature.com/articles/s41467-023-38186-1).

## Main results
![Picture3](https://github.com/hairongw/BioNet/assets/30871667/cd3ad781-6cbb-4878-82b8-ffff9c1673bc)

Competing methods:
- Multitask Adversarial Autoencoder (MTL-AAE): This model was originally proposed in [Latif et al. (2022)](https://www.computer.org/csdl/journal/ta/2022/02/09052467/1iFLxG1NYkM). For implementation details, please refer to the folder Competing Methods.
- Multitask Learning Neural Network (MTL-NN): For implementation details, please refer to the folder Competing Methods.
- Feed-forward Neural Network (NN): For implementation details, please refer to the folder Competing Methods.
- A Unified Semi-supervised Learning Algorithm (AdaMatch): This model was originally proposed in [Berthelot et al. (2022)](https://openreview.net/pdf?id=Q5uh1Nvv5dm). We implemented AdaMatch by using the code provided in the original paper. The GitHub repository can be found [here](https://github.com/google-research/adamatch).
- Support Vector Regression (SVR): We implemented SVR using the [sklearn.svm.SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) module from the scikit-learn package.
- Random Forest (RF): We implemented RF using the [sklearn.ensemble.RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) module from the scikit-learn package.

## Prediction Maps
![Picture4](https://github.com/hairongw/BioNet/assets/30871667/01a8a6bc-f828-4fdf-88cf-ef1f14de5e16)





