# MS-APPT: Multi-Scale Affinity Protein-Protein Transformer

A deep learning model for predicting protein-protein binding affinity from sequence alone.

## Installation

### Using Conda
```bash
conda env create -f environment.yml
conda activate ms-appt
```

### Using pip
```bash
pip install -r requirements.txt
```

## Quick Start

### Training
```bash
python train.py --config config.yml --train-data train.csv
```

### Prediction
```bash
python predict.py --checkpoint checkpoints/best_model.pt --input test.csv --output predictions.csv
```

### Biological Interface Analysis
```bash
# Analyze interaction with biological interpretation
python analyze_interaction.py --checkpoint checkpoints/best_model.pt --sequences MKTAYI... MGSSHH... --save-plots

# From FASTA file
python analyze_interaction.py --checkpoint checkpoints/best_model.pt --fasta proteins.fasta --save-plots
```

## Project Structure
```
ms-appt/
├── src/
│   ├── data/           # Data preprocessing and loading
│   ├── models/         # Model architectures
│   ├── training/       # Training pipeline
│   ├── evaluation/     # Metrics and evaluation
│   └── utils/          # Utilities
├── train.py            # Main training script
├── predict.py          # Inference script
├── config.yml          # Configuration file
└── CLAUDE.md           # Detailed methodology
```

## Model Architecture

MS-APPT uses:
- ESM-2 protein language model for embeddings
- Multi-scale convolutional layers
- Self-attention and cross-attention mechanisms
- Deep MLP for final prediction

### Biological Interpretability
- Attention-based interface prediction
- Hotspot residue identification
- Physicochemical property analysis
- Interaction type classification

See `CLAUDE.md` for detailed methodology.

## Configuration

Edit `config.yml` to modify:
- Model architecture parameters
- Training hyperparameters
- Data preprocessing settings
- Hardware configuration

## Citation

If you use this code, please cite:
```
@software{ms_appt,
  title={MS-APPT: Multi-Scale Affinity Protein-Protein Transformer},
  year={2024}
}
```