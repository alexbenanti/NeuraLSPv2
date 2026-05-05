# NeuraLSP: A Neural Spectral Preconditioner for Accelerating PDE Solvers (Anonymous NeurIPS Submission)
This repository is an implementation of the paper: NeuraLSP: A Neural Spectral Preconditioner for Accelerating PDE Solvers
**Anonymous repository for double-blind review.**
This repo will be de-anonymized upon acceptance.

## Overview
NeuraLSP is a neural preconditioning approach for accelerating Conjugate Gradient (CG) by learning a left singular subspace surrogate used within a rigorous solver framework.

## Repository Structure 
* ```src/```: helper functions such as data generation, models, etc.
* ```train_models.py```: trains neural models on selected PDEs
* ```comparison_test.py```: compares captured energy between subspace loss and NLSS loss
* ```scalability_ablation.py```: runs scalability experiments for varying sizes of $N$
* ```results/```: contains main results that were discussed in the paper

## Getting the code
- Download: use the “Download Repository” button on the anonymous repository page and unzip locally.

## Installation
We recommend creating a fresh environment.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
```

## Dependencies
Install PyTorch then:
```bash
pip install -r requirements.txt
```
NOTE: torch-scatter may require a wheel matching your PyTorch/CUDA version. If this gives an error, simply comment it out from requirement.txt (however, GNN experiments will not run if you do this)

## Training 
To train the models for all PDEs run:
```bash
python train_models.py
```
This may take a while, but it only needs to be done once, as all models are saved via checkpoints after training for each PDE. Also, please note that ```MLP_Nested``` refers to our NLSS loss and ```MLP_Unnested``` refers to the subspace loss as referenced in the paper. 

## Perform Main Experiments
After the models are trained, you can perform the experiments done in the main body of the paper related to solve time by running main.py
```bash
python main.py
```
## Perform Captured-Energy Comparison Experiments 
To reproduce the results we presented for comparing captured energy of subspace loss vs. NLSS loss, please run the following code: 
```bash
python comparison_test.py
```

## Scalability Ablation
Finally, to run the scalability ablation, please run the following:
```bash
python scalability_ablation.py
```

## Running on Smaller Scale Problems 
The code defaults to run on the $N=64$ problem with $K=72$. This can be changed to whatever size problem you like; however, you must change the values of ```N``` and ```K_VECTORS``` in ```main.py``` and ```rank_sweep.py```, and ```train_models.py```. Also, you should ensure that the maximal rank in ```RANKS``` does NOT exceed whichever value you selected for ```K_VECTORS```.
