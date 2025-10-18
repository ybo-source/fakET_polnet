import sys
import os
import json
import subprocess
import time
import numpy as np
import shutil
import ssl
import pandas as pd
from ast import literal_eval
from svnet.utils import get_absolute_paths,check_mrc_files
from svnet.utils import project_micrographs,reconstruct_micrographs_only_recon3D,project_style_micrographs
from svnet.utils import transform_directory_structure,copy_style_micrographs,compare_tomograms
from svnet.utils import analyze_json_files,visualize_results,print_style_stats
from svnet.utils import label_transform,find_labels_table,get_tomos_motif_list_paths

ssl._create_default_https_context = ssl._create_unverified_context

# Set your base directory here
base_dir = "/Users/yusufberkoruc/Desktop/Master_thesis/all_pipeline_directory"
#sys.path.append(os.path.join(base_dir, "svnet"))

# Index_parameters
micrograph_index = 0
style_index = 0
micrograph_directory_index = 0
simulation_index = 0
faket_index = 0
train_dir_index = 0
static_index = 0
tilt_range = (-60, 60, 3)
detector_snr = [0.15, 0.20]
denoised = False
use_csv_pairs = False
csv_path = None
random_faket = True
seq_end = len(np.arange(*tilt_range))
print(f"seq_end is {seq_end}")


# Simulation parameters
simulation_name = "all_v_czii"
simulation_base_dir = f"{base_dir}/simulation_dir_{simulation_index}"
simulation_dirs = [os.path.join(simulation_base_dir, simulation_name)]
labels_table = find_labels_table(simulation_dirs)

in_csv_list = sorted(get_tomos_motif_list_paths(simulation_base_dir))
out_dir = f"{base_dir}/train_directory_{train_dir_index}/overlay_pdb_{simulation_index}"
csv_dir_list = [os.path.join(d, "csv") for d in get_absolute_paths(simulation_base_dir)]

out_base_dir_style = f"{base_dir}/style_micrographs_output_{style_index}"
style_tomo_dir = f"{base_dir}/style_micrographs_{style_index}"
style_dir = f"{base_dir}/faket_data/style_micrographs_{style_index}"

# Check if the style_path exists
if os.path.exists(out_base_dir_style):
    print("style path already exists")
else:
    os.makedirs(out_base_dir_style,exist_ok =True)
    project_style_micrographs(style_tomo_dir, out_base_dir_style,tilt_range = tilt_range,ax = "Y",cluster_run = False,projection_threshold=100)
    copy_style_micrographs(out_base_dir_style, style_dir,copy_flag = False)

# Label transformation
if not os.path.exists(out_dir):
    label_transform(in_csv_list, out_dir, csv_dir_list, labels_table, mapping_flag=False)

# Project micrographs
out_base_dir_micrographs = f"{base_dir}/micrograph_directory_{micrograph_directory_index}/micrographs_output_dir_{micrograph_index}"
if not os.path.exists(out_base_dir_micrographs):
    snr_list = project_micrographs(
        out_base_dir_micrographs, simulation_dirs, tilt_range=tilt_range,
        detector_snr=detector_snr, micrograph_threshold=30, reconstruct_3d=False,
        add_misalignment=True, simulation_threshold=1
    )


STYLE_DIR = f"{base_dir}/faket_data/style_micrographs_{style_index}"


# Define directories
CLEAN_DIR = f"{base_dir}/micrograph_directory_{micrograph_directory_index}/micrographs_output_dir_{micrograph_index}/Micrographs" 
OUTPUT_DIR = f"{base_dir}/micrograph_directory_{micrograph_directory_index}/faket_mics_style_transfer_{faket_index}"

# Load modules and activate environment
setup_cmd = """
module load miniforge3
export CONDA_PKGS_DIRS=/mnt/lustre-grete/usr/u15206/conda_pkgs
source activate /user/yusufberk.oruc/u15206/.conda/envs/faketGPU-clean
export PYTHONPATH=/mnt/lustre-grete/usr/u15206/faket:$PYTHONPATH
"""

if use_csv_pairs and csv_path is not None:
    try:
        df = pd.read_csv(csv_path)
        df['styles'] = df['styles'].apply(literal_eval)
        style_pairs = dict(zip(df['content'], df['styles']))
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {e}")
else:
    if style_index is None:
        raise ValueError("style_index must be provided when not using CSV pairs")

if STYLE_DIR.endswith("style_tilt_series"):
    try:
        style_pairs = {key.split("/")[-1]: value for key, value in style_pairs.items()}
        style_pairs = {key: "_".join(value[0].split("_")[0:3]) + ".mrc" for key, value in style_pairs.items()}
        print(f"Style Pairs Experimental: {style_pairs}")
    except Exception as e:
        print(f"Error processing style pairs: {e}")
else:
    try:
        if denoised:
            style_pairs = {key.split("/")[-1]: value for key, value in style_pairs.items()}
            style_pairs = {key: value[0].replace("style_mics", "denoised") for key, value in style_pairs.items()}
            print(f"Style Pairs Denoised: {style_pairs}")
        else:
            style_pairs = {key.split("/")[-1]: value for key, value in style_pairs.items()}
            style_pairs = {key.split("/")[-1]: value for key, value in style_pairs.items()}
            print(f"Style Pairs Projected: {style_pairs}")
    except Exception as e:
        print(f"Error processing style pairs: {e}")


# Find all clean tomograms
clean_tomograms = subprocess.run(f"find {CLEAN_DIR} -name 'tomo_mics_clean_*.mrc'", 
                                shell=True, capture_output=True, text=True).stdout.splitlines()
for CLEAN_TOMOGRAM in clean_tomograms:
    TOMOGRAM_DIR = CLEAN_TOMOGRAM.split('/')[-2]
    CLEAN_ID = CLEAN_TOMOGRAM.split('_')[-1].replace('.mrc', '')
    print(f"Clean ID: {CLEAN_ID}")
    print(f"Looking for: tomo_mics_{CLEAN_ID}_*.mrc")
    NOISY_TOMOGRAM_CMD = f"find {CLEAN_DIR}/{TOMOGRAM_DIR} -name 'tomo_mics_{CLEAN_ID}_*.mrc' | head -n 1"
    NOISY_TOMOGRAM = subprocess.run(NOISY_TOMOGRAM_CMD, shell=True, capture_output=True, text=True).stdout.strip()

    if not NOISY_TOMOGRAM:
        print(f"Noisy tomogram not found for {TOMOGRAM_DIR}. Skipping...")
        continue
    if random_faket:
        STYLE_TOMOGRAM_CMD = f"find {STYLE_DIR} -name '*.mrc' | shuf -n 1"
        STYLE_TOMOGRAM = subprocess.run(STYLE_TOMOGRAM_CMD, shell=True, capture_output=True, text=True).stdout.strip()
    else:
        if STYLE_DIR.endswith("style_tilt_series"):
            CLEAN_TOMOGRAM_NAME = CLEAN_TOMOGRAM.split("/")[-1]
            STYLE_TOMOGRAM_NAME = style_pairs[CLEAN_TOMOGRAM_NAME]
            print(STYLE_TOMOGRAM_NAME)
            STYLE_TOMOGRAM = os.path.join(STYLE_DIR, STYLE_TOMOGRAM_NAME)
        else:
            if denoised:
                CLEAN_TOMOGRAM_NAME = CLEAN_TOMOGRAM.split("/")[-1]
                STYLE_TOMOGRAM_NAME = style_pairs[CLEAN_TOMOGRAM_NAME]
                print(STYLE_TOMOGRAM_NAME)
                STYLE_TOMOGRAM = os.path.join(STYLE_DIR, STYLE_TOMOGRAM_NAME)
            else:
                CLEAN_TOMOGRAM_NAME = CLEAN_TOMOGRAM.split("/")[-1]
                STYLE_TOMOGRAM_NAME = style_pairs[CLEAN_TOMOGRAM_NAME][0]
                print(STYLE_TOMOGRAM_NAME)
                STYLE_TOMOGRAM = os.path.join(STYLE_DIR, STYLE_TOMOGRAM_NAME)

    OUTPUT_TOMOGRAM = f"{OUTPUT_DIR}/tomo_style_transfer_{TOMOGRAM_DIR}.mrc"
    print(f"Output Tomogram: {OUTPUT_TOMOGRAM}")
    if os.path.exists(OUTPUT_TOMOGRAM):
        continue
    print(f"Processing: Clean={CLEAN_TOMOGRAM}, Noisy={NOISY_TOMOGRAM}, Style={STYLE_TOMOGRAM}")

    style_transfer_cmd = f"""
    {setup_cmd}
    CUDA_VISIBLE_DEVICES=0 python3 -m faket.style_transfer.cli {CLEAN_TOMOGRAM} {STYLE_TOMOGRAM} \
        --init {NOISY_TOMOGRAM} \
        --output {OUTPUT_TOMOGRAM} --devices cuda:0 --random-seed 0 --min-scale 630 \
        --end-scale 630 --seq_start 0 --seq_end {seq_end} --style-weights 1.0 --content-weight 1.0 --tv-weight 0 --iterations 5 \
        --initial-iterations 1 --save-every 2 --step-size 0.15 --avg-decay 0.99 --style-scale-fac 1.0 --pooling max --content_layers 8 \
        --content_layers_weights 100 --model_weights pretrained
    """
    if os.path.exists(OUTPUT_TOMOGRAM):
        continue
    else:
        subprocess.run(style_transfer_cmd, shell=True, executable="/bin/bash")

print("Style transfer completed for one index!")

base_dir_Micrographs = os.path.join(out_base_dir_micrographs, "Micrographs")
base_dir_TEM = os.path.join(out_base_dir_micrographs, "TEM")
base_dir_faket = f"{base_dir}/micrograph_directory_{micrograph_directory_index}/faket_mics_style_transfer_{faket_index}"
snr_list_dir = f"{base_dir}/micrograph_directory_{micrograph_directory_index}/snr_list_dir"
os.makedirs(snr_list_dir, exist_ok=True)

tomograms = os.listdir(base_dir_TEM)
        # Sorting function
sorted_tomograms = sorted(tomograms, key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2])))
TEM_paths = [os.path.join(base_dir_TEM,tomogram) for tomogram in sorted_tomograms]
Micrograph_paths = [os.path.join(base_dir_Micrographs, f) for f in os.listdir(base_dir_Micrographs)]
Micrographs_sorted = sorted(os.listdir(base_dir_Micrographs),key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2])))
Micrograph_paths = [os.path.join(base_dir_Micrographs,tomogram) for tomogram in Micrographs_sorted]

print(f"Micrograph Paths: {Micrograph_paths}")
print(f"TEM Paths: {TEM_paths}")
snr_list = []
for path in Micrograph_paths:
    if os.path.exists(path):
        for micrograph_file in os.listdir(path):
            if micrograph_file.split("_")[-2] != "clean":
                snr = micrograph_file.split("_")[-1].split(".")
                snr = snr[0] + "." + snr[1]
                snr_list.append(float(snr))
with open(os.path.join(snr_list_dir, f"snr_list_{micrograph_index}.json"), "w") as file:
    json.dump(snr_list, file)

faket_paths = [f for f in os.listdir(base_dir_faket) if f.endswith(".mrc")]
if faket_paths:
    # Sorting function
    old_faket = True
    if old_faket:
        sorted_tomograms_faket = sorted(faket_paths,key=lambda x:int(x.split("_")[-1].split(".")[0]))
        print(sorted_tomograms_faket)
    else:
        sorted_tomograms_faket = sorted(faket_paths, key=lambda x: (int(x.split('_')[4]), int(x.split('_')[5].split('.')[0])))
        print(sorted_tomograms_faket)
    
    source_dir = f"{base_dir}/reconstructed_tomograms_{micrograph_index}"
    target_dir_faket = f"{base_dir}/train_directory_{train_dir_index}/static_{micrograph_index}/ExperimentRuns_faket"
    target_dir_basic = f"{base_dir}/train_directory_{train_dir_index}/static_{micrograph_index}/ExperimentRuns_basic"
    faket_paths = [os.path.join(base_dir_faket,tomogram) for tomogram in sorted_tomograms_faket]
    if not os.path.exists(target_dir_faket):
        reconstruct_micrographs_only_recon3D(TEM_paths, faket_paths, source_dir, snr_list, custom_mic=True, micrograph_threshold=100)
        reconstruct_micrographs_only_recon3D(TEM_paths, faket_paths, source_dir, snr_list, custom_mic=False, micrograph_threshold=100)
        transform_directory_structure(source_dir, target_dir_faket, target_dir_basic, copy_flag=False)
        try:
            shutil.rmtree(source_dir)
            print(f"Successfully deleted the directory: {source_dir}")
        except Exception as e:
            print(f"Error deleting directory {source_dir}: {e}")
else:
    print("No faket tomogram found.")