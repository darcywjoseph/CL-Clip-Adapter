# CL-Clip-Adapter
Code relating to the implementation of a continual learning adapter for pretrain CLIP.

```bash
├── README.md
├── src
│   ├── dataset.py
│   ├── experiments.py
│   ├── model.py
│   └── train.py
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
```
Incase of failure, add depencies via:

```bash
poetry add <package-name>
```
## Repeat Investigation Results

Optionally, you can build the synthetci datasets for the Synthetic Feature Shift Experiment and save to view. 
Otherwise this will dont within the experiment run script with no saving.
```bash
python src/dataset.py
```

### Run Synthetic feature Shift Experiment 

```bash
python src/run_basic_experiment.py --use_contrastive --use_ewc 
```
The --use_constrative flag can be used to toggle contrastive loss on and off. This was used in ablation.
The --use_ewc flag can be used to toggle ewc on or off. This was investigative extension. 
The switch either function off just exclude the flag from the command. 

### Run MTIL Benchmark Experiment 

```bash
python src/run_mtil_experiment.py 
```