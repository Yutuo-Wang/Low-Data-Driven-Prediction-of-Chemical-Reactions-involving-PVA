# Low-Data-Driven-Prediction-of-Chemical-Reactions-involving-PVA
![20250810](https://github.com/user-attachments/assets/88a1c5e8-31b0-46cb-b1d9-9f4a9115dbb9)

This is the official repository of papers titled “Low-Data-Driven Prediction of Chemical Reactions involving PVA: A Quantum Chemistry-inspired Multiscale Deep Learning Framework”. This work uses a revolutionary model that can mimic the bonding changes that occur during the reaction process, and can use a relatively small training set to highly accurately predict the results involving PVA reactions without pre-training or additional features. Materials scientists and chemists will benefit from this, because the model has strong generalization ability, which can not only effectively predict the reaction involving PVA, but also can be extended to the modeling and analysis of chemical reactions in other material systems, providing a universal solution for intelligent optimization of different chemical systems. 
 
The files related to the model are in this repository.

Please feel free to email yutuo7@126.com with any questions about this work.

# Tips
## Create and activate the Conda environment
Create a new Conda environment and activate it.
```
sudo apt update
conda create -n pva_env python==3.8
conda activate pva_env
```
## Install necessary dependency packages
Verify Installation.
```
conda install numpy pandas scikit-learn matplotlib
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge transformers rdkit pyscf tqdm
```
## Verifying the installation
Make sure that all packages are properly installed and the environment is configured correctly.
```
conda list
```
