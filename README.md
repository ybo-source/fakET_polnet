# Micrograph Style Transfer Pipeline

A comprehensive pipeline for applying style transfer to cryo-electron tomography micrographs using the faket package. This tool processes simulated micrographs, applies style transfer, and reconstructs tomograms for training machine learning models.

## Overview

This pipeline performs the following main operations:
- Projects 3D volumes into 2D micrograph tilt series
- Applies style transfer using the faket package
- Reconstructs stylized micrographs back into 3D tomograms
- Prepares training data for downstream analysis

## Prerequisites

### Required Software
- Python 3.7+
- [faket](https://github.com/paloha/faket.git) package
- CUDA-capable GPU (recommended)
- RELION (for some utilities)
- IMOD (for tomogram reconstruction)

### Python Dependencies
- numpy
- pandas
- scipy
- mrcfile
- scikit-image
- argparse
- And other standard scientific Python libraries

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required Python packages:
```bash
pip install numpy pandas scipy mrcfile scikit-image
```

3. Install faket following the official instructions from [the faket repository](https://github.com/paloha/faket.git)

4. Ensure the `svnet` module is available in your Python path (this appears to be a custom module for this project)

## Usage

### Basic Command

```bash
python main.py /path/to/base_directory
```

### Complete Example

```bash
python main.py /path/to/base_directory \
    --micrograph_index 0 \
    --style_index 1 \
    --simulation_index 0 \
    --faket_index 0 \
    --train_dir_index 0 \
    --tilt_start -60 \
    --tilt_end 60 \
    --tilt_step 3 \
    --detector_snr 0.15 0.20 \
    --faket_command "python -m faket.style_transfer.cli" \
    --cuda_device 0
```

### Command Line Arguments

#### Required Arguments
- `base_dir`: Base directory containing simulation and style directories

#### Index Parameters
- `--micrograph_index`: Micrograph index (default: 0)
- `--style_index`: Style index (default: 0)
- `--micrograph_directory_index`: Micrograph directory index (default: 0)
- `--simulation_index`: Simulation index (default: 0)
- `--faket_index`: Faket index (default: 0)
- `--train_dir_index`: Train directory index (default: 0)
- `--static_index`: Static index (default: 0)

#### Tilt Series Parameters
- `--tilt_start`: Tilt series start angle (default: -60)
- `--tilt_end`: Tilt series end angle (default: 60)
- `--tilt_step`: Tilt series step size (default: 3)

#### Style Transfer Parameters
- `--detector_snr`: Detector SNR range (default: 0.15 0.20)
- `--denoised`: Use denoised style micrographs
- `--random_faket`: Use random faket style transfer (default: True)
- `--simulation_name`: Simulation name (default: "all_v_czii")

#### Faket Configuration
- `--faket_command`: Command to run faket style transfer (default: "python -m faket.style_transfer.cli")
- `--cuda_device`: CUDA device to use (default: "0")

## Directory Structure

The pipeline expects the following directory structure:

```
base_directory/
├── simulation_dir_{index}/
├── style_tomograms_{index}/

```

## Pipeline Steps

1. **Directory Validation**: Checks that required simulation and style directories exist
2. **Style Projection**: Projects 3D style tomograms into 2D micrographs
3. **Label Transformation**: Processes simulation labels for training
4. **Micrograph Projection**: Projects 3D simulations into 2D micrograph tilt series
5. **Style Transfer**: Applies faket style transfer to micrographs
6. **Tomogram Reconstruction**: Reconstructs stylized micrographs back into 3D tomograms
7. **Data Organization**: Organizes output for training pipelines

## Customization

### Using Different Faket Commands

If faket is installed in a custom location or requires special invocation:

```bash
python pipeline.py /path/to/base_dir \
    --faket_command "/path/to/faket/env/bin/python -m faket.style_transfer.cli"
```

### Multiple GPU Usage

To use a different GPU:

```bash
python pipeline.py /path/to/base_dir --cuda_device 1
```

## Output

The pipeline generates:
- Style-transferred micrographs in `faket_mics_style_transfer_*/`
- Reconstructed tomograms in `reconstructed_tomograms_*/`
- Training-ready data in `train_directory_*/static_*/`
- SNR lists for quality control

## Troubleshooting

### Common Issues

1. **Module not found errors**: Ensure all dependencies are installed and the `svnet` module is in your Python path
2. **CUDA out of memory**: Reduce batch size or use a smaller model
3. **Missing directories**: Verify the directory structure matches expectations
4. **faket command not found**: Use the `--faket_command` parameter to specify the correct path

### Debug Mode

For verbose output, you can modify the script to add debug prints or use the existing print statements to track progress.

## Citation

If you use this software in your research, please cite the relevant papers:
- faket: [faket publication]
- RELION: [RELION publication]
- IMOD: [IMOD publication]

## License

[Add your license information here]

## Support

For questions and issues, please open an issue on the GitHub repository or contact [your email/contact information].
