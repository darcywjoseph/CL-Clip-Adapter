# CL-Clip-Adapter
Code relating to the implementation of a continual learning adapter for pretrain CLIP.

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
