import sys
import os
import json
import subprocess
import time
import numpy as np
import shutil
import ssl
import pandas as pd
import argparse
from ast import literal_eval
from svnet.utils import get_absolute_paths,check_mrc_files
from svnet.utils import project_micrographs,reconstruct_micrographs_only_recon3D,project_style_micrographs
from svnet.utils import transform_directory_structure,copy_style_micrographs,compare_tomograms
from svnet.utils import analyze_json_files,visualize_results,print_style_stats
from svnet.utils import label_transform,find_labels_table,get_tomos_motif_list_paths

ssl._create_default_https_context = ssl._create_unverified_context

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run the complete pipeline with configurable parameters')
    
    # Required argument
    parser.add_argument('base_dir', type=str, help='Base directory containing simulation and style directories')
    
    # Index parameters
    parser.add_argument('--micrograph_index', type=int, default=0, help='Micrograph index')
    parser.add_argument('--style_index', type=int, default=0, help='Style index')
    parser.add_argument('--micrograph_directory_index', type=int, default=0, help='Micrograph directory index')
    parser.add_argument('--simulation_index', type=int, default=0, help='Simulation index')
    parser.add_argument('--faket_index', type=int, default=0, help='Faket index')
    parser.add_argument('--train_dir_index', type=int, default=0, help='Train directory index')
    parser.add_argument('--static_index', type=int, default=0, help='Static index')
    
    # Tilt range parameters
    parser.add_argument('--tilt_start', type=int, default=-60, help='Tilt series start angle')
    parser.add_argument('--tilt_end', type=int, default=60, help='Tilt series end angle')
    parser.add_argument('--tilt_step', type=int, default=3, help='Tilt series step size')
    
    # Other parameters
    parser.add_argument('--detector_snr', type=float, nargs=2, default=[0.15, 0.20], help='Detector SNR range')
    parser.add_argument('--denoised', action='store_true', help='Use denoised style micrographs')
    parser.add_argument('--random_faket', action='store_true', default=True, help='Use random faket style transfer')
    parser.add_argument('--simulation_name', type=str, default="all_v_czii", help='Simulation name')
    
    # Faket parameters
    parser.add_argument('--faket_gpu', type=int, default=0, help='GPU device ID for faket')
    parser.add_argument('--faket_iterations', type=int, default=5, help='Number of iterations for faket style transfer')
    parser.add_argument('--faket_step_size', type=float, default=0.15, help='Step size for faket')
    parser.add_argument('--faket_min_scale', type=int, default=630, help='Minimum scale for faket')
    parser.add_argument('--faket_end_scale', type=int, default=630, help='End scale for faket')
    
    return parser.parse_args()

def validate_directories(base_dir, simulation_index, style_index):
    """Validate that required directories exist"""
    simulation_dir = f"{base_dir}/simulation_dir_{simulation_index}"
    style_tomo_dir = f"{base_dir}/style_tomograms_{style_index}"
    style_dir = f"{base_dir}/faket_data/style_micrographs_{style_index}"
    
    if not os.path.exists(simulation_dir):
        raise ValueError(f"Simulation directory not found: {simulation_dir}")
    
    # Check if either style_tomo_dir or style_dir exists
    style_tomo_exists = os.path.exists(style_tomo_dir)
    style_dir_exists = os.path.exists(style_dir)
    
    if not style_tomo_exists and not style_dir_exists:
        raise ValueError(f"Neither style tomogram directory {style_tomo_dir} nor style directory {style_dir} found. At least one must exist.")
    
    print(f"Found simulation directory: {simulation_dir}")
    if style_tomo_exists:
        print(f"Found style tomogram directory: {style_tomo_dir}")
    if style_dir_exists:
        print(f"Found style directory: {style_dir}")
    
    return simulation_dir, style_tomo_dir, style_dir, style_tomo_exists, style_dir_exists

def main():
    args = parse_arguments()
    
    # Set base directory from arguments
    base_dir = args.base_dir
    
    # Validate that required directories exist
    simulation_dir, style_tomo_dir, style_dir, style_tomo_exists, style_dir_exists = validate_directories(base_dir, args.simulation_index, args.style_index)
    
    # Set parameters from arguments
    micrograph_index = args.micrograph_index
    style_index = args.style_index
    micrograph_directory_index = args.micrograph_directory_index
    simulation_index = args.simulation_index
    faket_index = args.faket_index
    train_dir_index = args.train_dir_index
    static_index = args.static_index
    tilt_range = (args.tilt_start, args.tilt_end, args.tilt_step)
    detector_snr = args.detector_snr
    denoised = args.denoised
    random_faket = args.random_faket
    simulation_name = args.simulation_name
    
    # Faket parameters
    faket_gpu = args.faket_gpu
    faket_iterations = args.faket_iterations
    faket_step_size = args.faket_step_size
    faket_min_scale = args.faket_min_scale
    faket_end_scale = args.faket_end_scale
    
    # Calculate seq_end
    seq_end = len(np.arange(*tilt_range))
    print(f"seq_end is {seq_end}")

    # Simulation parameters
    simulation_base_dir = f"{base_dir}/simulation_dir_{simulation_index}"
    simulation_dirs = [os.path.join(simulation_base_dir, simulation_name)]
    labels_table = find_labels_table(simulation_dirs)

    in_csv_list = sorted(get_tomos_motif_list_paths(simulation_base_dir))
    out_dir = f"{base_dir}/train_directory_{train_dir_index}/overlay_{simulation_index}"
    csv_dir_list = [os.path.join(d, "csv") for d in get_absolute_paths(simulation_base_dir)]

    out_base_dir_style = f"{base_dir}/style_micrographs_output_{style_index}"

    # Handle style directory logic
    if style_dir_exists:
        print("Style directory already exists, skipping projection and copy.")
    elif style_tomo_exists:
        # Only run projection if style_tomo_dir exists and style_dir doesn't exist
        print("Style tomogram directory found but style directory doesn't exist. Running projection...")
        os.makedirs(out_base_dir_style, exist_ok=True)
        project_style_micrographs(style_tomo_dir, out_base_dir_style, tilt_range=tilt_range, ax="Y", cluster_run=False, projection_threshold=100)
        copy_style_micrographs(out_base_dir_style, style_dir, copy_flag=False)
        print(f"Style projection completed and copied to: {style_dir}")
    else:
        # This should not happen due to validation, but just in case
        raise ValueError("No style directory available")

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

    STYLE_DIR = style_dir  # Use the validated style directory

    # Define directories
    CLEAN_DIR = f"{base_dir}/micrograph_directory_{micrograph_directory_index}/micrographs_output_dir_{micrograph_index}/Micrographs" 
    OUTPUT_DIR = f"{base_dir}/micrograph_directory_{micrograph_directory_index}/faket_mics_style_transfer_{faket_index}"

    # Create OUTPUT_DIR if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all clean tomograms
    clean_tomograms = subprocess.run(f"find {CLEAN_DIR} -name 'tomo_mics_clean_*.mrc'", 
                                    shell=True, capture_output=True, text=True).stdout.splitlines()
    
    if not clean_tomograms:
        print(f"No clean tomograms found in {CLEAN_DIR}")
        return
    
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
            # If not random, you might want to implement specific style selection logic here
            STYLE_TOMOGRAM_CMD = f"find {STYLE_DIR} -name '*.mrc' | head -n 1"
            STYLE_TOMOGRAM = subprocess.run(STYLE_TOMOGRAM_CMD, shell=True, capture_output=True, text=True).stdout.strip()

        OUTPUT_TOMOGRAM = f"{OUTPUT_DIR}/tomo_style_transfer_{TOMOGRAM_DIR}.mrc"
        print(f"Output Tomogram: {OUTPUT_TOMOGRAM}")
        
        if os.path.exists(OUTPUT_TOMOGRAM):
            print(f"Output tomogram already exists, skipping: {OUTPUT_TOMOGRAM}")
            continue
            
        print(f"Processing: Clean={CLEAN_TOMOGRAM}, Noisy={NOISY_TOMOGRAM}, Style={STYLE_TOMOGRAM}")

        # Build faket command without environment setup
        style_transfer_cmd = [
            "python3", "-m", "faket.style_transfer.cli",
            CLEAN_TOMOGRAM, STYLE_TOMOGRAM,
            "--init", NOISY_TOMOGRAM,
            "--output", OUTPUT_TOMOGRAM,
            "--devices", f"cuda:{faket_gpu}",
            "--random-seed", "0",
            "--min-scale", str(faket_min_scale),
            "--end-scale", str(faket_end_scale),
            "--seq_start", "0",
            "--seq_end", str(seq_end),
            "--style-weights", "1.0",
            "--content-weight", "1.0",
            "--tv-weight", "0",
            "--iterations", str(faket_iterations),
            "--initial-iterations", "1",
            "--save-every", "2",
            "--step-size", str(faket_step_size),
            "--avg-decay", "0.99",
            "--style-scale-fac", "1.0",
            "--pooling", "max",
            "--content_layers", "8",
            "--content_layers_weights", "100",
            "--model_weights", "pretrained"
        ]
        
        # Set CUDA_VISIBLE_DEVICES environment variable
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(faket_gpu)
        
        try:
            print(f"Running faket command: {' '.join(style_transfer_cmd)}")
            subprocess.run(style_transfer_cmd, env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running style transfer for {CLEAN_TOMOGRAM}: {e}")
            continue
        except FileNotFoundError:
            print("Error: faket not found. Please make sure faket is installed and available in your PATH")
            continue

    print("Style transfer completed for one index!")

    base_dir_Micrographs = os.path.join(out_base_dir_micrographs, "Micrographs")
    base_dir_TEM = os.path.join(out_base_dir_micrographs, "TEM")
    base_dir_faket = f"{base_dir}/micrograph_directory_{micrograph_directory_index}/faket_mics_style_transfer_{faket_index}"
    snr_list_dir = f"{base_dir}/micrograph_directory_{micrograph_directory_index}/snr_list_dir"
    os.makedirs(snr_list_dir, exist_ok=True)

    # Check if directories exist before processing
    if not os.path.exists(base_dir_TEM) or not os.path.exists(base_dir_Micrographs):
        print(f"Required directories not found: {base_dir_TEM} or {base_dir_Micrographs}")
        return

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

if __name__ == "__main__":
    main()
