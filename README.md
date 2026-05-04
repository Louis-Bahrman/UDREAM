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

If you use this work in your research or business, please cite it:

- For the variants with strong supervision, the reverberation model, and the unsupervised methods:

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

- For the best-performing weakly-supervised phase-invariant methods:

```
@INPROCEEDINGS{11462939,
  author={Rodrigues, Marius and Bahrman, Louis and Badeau, Roland and Richard, Gaël},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Is Phase Really Needed for Weakly-Supervised Dereverberation?},
  year={2026},
  volume={},
  number={},
  pages={17422-17426},
  keywords={Filtering;System-on-chip;Filters;Band-pass filters;Feedback;Filter banks;Circuits;Application specific integrated circuits;Circuits and systems;Wireless Access in Vehicular Environments;Speech dereveberation;reverberation modeling;phase retrieval;unsupervised learning},
  doi={10.1109/ICASSP55912.2026.11462939}}
```
