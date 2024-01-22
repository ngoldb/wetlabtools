# wetlabtools
This repo contains useful software tools for wet lab taks. For example you can easily plot data from FPLC after export from Unicorn or you can plot your CD and SEC-MALS data for your records. Most code is from the great MP - I just tidied up some mess and packaged the code.

## Installation
To install the package, first clone the repo to your local machine and install the environment from the yml file:
```
cd ~
git clone https://github.com/ngoldb/wetlabtools.git
cd ~/wetlabtools
conda env create -f wetlabtools.yml
```
For some reason installing the environment only works with anaconda, but not with miniconda...

Activate the environment and install the downloaded package using pip (make sure you are in the directory above the package):
```
conda activate wetlabtools
cd ../
pip install -e wetlabtools
```
If you like to add the environment to your jupyter lab, run the following commands:
```
conda activate wetlabtools
python -m ipykernel install --user --name=wetlabtools
```

## Examples
A jupyter notebook with examples is included here: `examples/examples.ipynb`.
