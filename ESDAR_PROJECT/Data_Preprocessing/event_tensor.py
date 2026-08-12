import os
import numpy as np
import cv2

# ====================== Configurable Parameters ======================
INPUT_NPY_FOLDER = r"F:\ESDAR\ESDAR_PROJECT\event_out"  # Folder containing raw event .npy files
OUT_TENSOR_FOLDER = r"F:\ESDAR\ESDAR_PROJECT\tensor_output"  # Output folder for tensorized .npy files
OUT_VIS_FOLDER = r"F:\ESDAR\ESDAR_PROJECT\tensor_output\tensor_visual"     # Output folder for visualization without temporal axis
EPS = 1e-8
# ========================================================

def event_spatiotemporal_tensorize(raw_event_np):
    H, W, C = raw_event_np.shape
    assert C == 2, "Event .npy must be in [H, W, 2] dual-channel format"
    tensor_out = np.zeros_like(raw_event_np, dtype=np.float32)
    # Global normalization for positive and negative event channels separately (accumulative normalization as in papers)
    for ch in range(2):
        channel_data = raw_event_np[:, :, ch]
        max_val = np.max(channel_data)
        # Normalize to [0, 1] to avoid division-by-zero errors
        tensor_out[:, :, ch] = channel_data / np.maximum(max_val, EPS)
    return tensor_out

def tensor_visualization(tensor_2ch, save_path):
    H, W, _ = tensor_2ch.shape
    canvas = np.zeros((H, W, 3), dtype=np.uint8)  # Pure black background
    pos_channel = tensor_2ch[:, :, 1]
    neg_channel = tensor_2ch[:, :, 0]
    # Positive events mapped to red channel
    canvas[:, :, 2] = (pos_channel * 255).astype(np.uint8)
    # Negative events mapped to green channel
    canvas[:, :, 1] = (neg_channel * 255).astype(np.uint8)
    cv2.imwrite(save_path, canvas)

def batch_tensor_process():
    # Create output directories
    os.makedirs(OUT_TENSOR_FOLDER, exist_ok=True)
    os.makedirs(OUT_VIS_FOLDER, exist_ok=True)
    # Filter all .npy files
    npy_list = [f for f in os.listdir(INPUT_NPY_FOLDER) if f.endswith(".npy")]
    if len(npy_list) == 0:
        print("No .npy files found in the input folder!")
        return
    total = len(npy_list)
    print(f"Detected {total} event .npy files in total, starting tensorization...\n")
    for idx, npy_name in enumerate(npy_list, 1):
        npy_path = os.path.join(INPUT_NPY_FOLDER, npy_name)
        base_name = os.path.splitext(npy_name)[0]
        try:
            # 1. Load raw event tensor
            raw_event = np.load(npy_path)
            # 2. Perform spatiotemporal tensorization (temporal compression + normalization)
            tensor_norm = event_spatiotemporal_tensorize(raw_event)
            # 3. Save standardized tensorized .npy
            tensor_save_path = os.path.join(OUT_TENSOR_FOLDER, f"{base_name}_tensor.npy")
            np.save(tensor_save_path, tensor_norm)
            # 4. Generate visualization without temporal axis
            vis_save_path = os.path.join(OUT_VIS_FOLDER, f"{base_name}_vis.png")
            tensor_visualization(tensor_norm, vis_save_path)
            print(f"[{idx}/{total}] Completed: {npy_name}")
        except Exception as e:
            print(f"[{idx}/{total}] Failed {npy_name} : {str(e)}")
    print(f"Standardized tensor directory: {OUT_TENSOR_FOLDER}")
    print(f"Visualization (no temporal axis) directory: {OUT_VIS_FOLDER}")

if __name__ == "__main__":
    batch_tensor_process()