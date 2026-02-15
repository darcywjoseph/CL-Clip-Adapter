# CL-Clip-Adapter
Code relating to the implementation of a continual learning adapter for pretrain CLIP.

```bash
├── README.md
├── data
├── src
├── __init__.py
├── basic
│   ├── __init__.py
│   ├── dataset.py
│   └── run_basic_experiment.py
├── model
│   ├── __init__.py
│   ├── model.py
│   ├── model_utils.py
│   └── train.py
└── mtil
    ├── __init__.py
    ├── dataset_config.py
    ├── dataset_utils.py
    └── run_mtil_experiment.py
├── pyproject.toml
├── poetry.lock
└── .gitignore
```

## Set Up Environment

Install Poetry with pipx

```bash
sudo apt install pipx
pipx install --force poetry==2.2.1
pipx ensurepath
```
Install and activate the Poetry environment.

```bash
poetry install
eval $(poetry env activate)
pip install ftfy regex tqdm wheel
pip install git+https://github.com/openai/CLIP.git
```
Incase of failure, add depencies via:

```bash
poetry add <package-name>
```

## Setup StanfordCars dataset (torchvision method broken)

Download zip from kaggle - https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset
Move the zip file to data/stanford_cars

```bash
cd data/
mkdir stanford_cars
cd stanford_cars
unzip archive.zip
```

download from cars_test_annos_withlabels (1).mat https://www.kaggle.com/code/subhangaupadhaya/pytorch-stanfordcars-classification/input?select=cars_test_annos_withlabels+%281%29.mat

Rename to cars_test_annos_withlabels.mat and drag into data/stanford_cars/car_devkit/devkit/

## Repeat Investigation Results

### Run Synthetic feature Shift Experiment 

```bash
cd src/
python -m basic.run_basic_experiment --use_contrastive --use_ewc
```
The --use_constrative flag can be used to toggle contrastive loss on and off. This was used in ablation.
The --use_ewc flag can be used to toggle ewc on or off. This was investigative extension. 
The switch either function off just exclude the flag from the command. 

### Run MTIL Benchmark Experiment 

```bash
python src/run_mtil_experiment.py --use_contrastive --use_ewc
```