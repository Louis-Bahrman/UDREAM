# UDREAM: Unsupervised Dereverberation guided by a Reverberation Model

This repository contains a Python program, `dereverberate.py`, to  dereverberate an audio file using our proposed supervision paradigms.
It also contains training code to apply this framework to new models and datasets.

## Installation

1. Clone this repository

```
git clone https://github.com/Louis-Bahrman/UDREAM.git
cd UDREAM
```

2. Install required dependencies

```
conda env create -f environment.yaml
conda activate hybrid_wssd
```

3. Download checkpoints

```
wget 'https://zenodo.org/records/19672464/files/UDREAM_checkpoints.zip'
unzip UDREAM_checkpoints.zip

wget https://zenodo.org/records/19880589/files/PhaseInv_checkpoints.zip
unzip PhaseInv_checkpoints.zip
```

## Usage

See:
```
python dereverberate.py -h
```

## Training new models or datasets

See [framework_details.md](framework_details.md)

## Citing

If you use this work in your research or business, please cite it using the following BibTeX entry:

```
@ARTICLE{11425772,
  author={Bahrman, Louis and Rodrigues, Marius and Fontaine, Mathieu and Richard, Gaël},
  journal={IEEE Transactions on Audio, Speech and Language Processing},
  title={U-DREAM: Unsupervised Dereverberation Guided by a Reverberation Model},
  year={2026},
  volume={34},
  number={},
  pages={1552-1563},
  keywords={Reverberation;Acoustics;Training;Convolution;Data models;Weak supervision;Time-frequency analysis;Time-domain analysis;Europe;Predictive models;Dereverberation;hybrid deep learning;reverberation modeling;unsupervised learning},
  doi={10.1109/TASLPRO.2026.3671615}}
```
