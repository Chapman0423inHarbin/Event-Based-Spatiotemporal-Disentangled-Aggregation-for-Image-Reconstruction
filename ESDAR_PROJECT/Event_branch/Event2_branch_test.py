import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import re
import numpy as np
from PIL import Image
import cv2
import lpips
import time
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.ops import deform_conv2d

# ====================== Deformable Convolution Guidance Module (identical to training) ======================
class DCMModule(nn.Module):
    def __init__(self, ch=16, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.ch = ch

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_kernel = torch.cat([sobel_x, sobel_y], dim=0).repeat(ch, 1, 1, 1)
        self.register_buffer('sobel_kernel', sobel_kernel)

        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.var_conv = nn.Conv2d(ch, ch, kernel_size=(1, 5), padding=(0, 2))
        self.conv_main = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.mask_branch = nn.Sequential(
            nn.Conv2d(ch, kernel_size * kernel_size, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.offset_branch = nn.Conv2d(ch, 2 * kernel_size * kernel_size, kernel_size=1)
        self.deform_weight = nn.Parameter(torch.empty(ch, ch, kernel_size, kernel_size))
        self.deform_bias = nn.Parameter(torch.zeros(ch))
        nn.init.kaiming_normal_(self.deform_weight, mode='fan_out', nonlinearity='relu')
        self.res_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        B, C, H, W = x.shape
        grad_raw = F.conv2d(x, self.sobel_kernel, padding=1, groups=C)
        gx = grad_raw[:, 0::2, :, :]
        gy = grad_raw[:, 1::2, :, :]
        grad_map = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        grad_feat = F.relu(grad_map)

        mean = self.avg_pool(x)
        mean_sq = self.avg_pool(x ** 2)
        var_map = mean_sq - mean ** 2
        var_feat = F.relu(self.var_conv(var_map))

        main_conv = self.conv_main(x)
        feat_map = main_conv + grad_feat + var_feat

        mask = self.mask_branch(var_feat)
        offset = self.offset_branch(var_feat)

        deform_out = deform_conv2d(
            feat_map, offset, self.deform_weight, self.deform_bias,
            stride=1, padding=self.padding, dilation=1, mask=mask
        )
        out = F.relu(deform_out + self.res_weight * x)
        return out

# ====================== Adaptive ProD Recurrent Block (identical to training) ======================
class DynamicProDBlock(nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        self.branch_a = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.branch_b = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, loop_times=3):
        feat = x
        for _ in range(loop_times):
            scale_map = self.branch_a(feat)
            b_feat = self.branch_b(feat)
            feat = self.relu(b_feat * scale_map + feat)
        return feat

# ====================== Event2 Event Stream Backbone (identical to training) ======================
class Event2Branch(nn.Module):
    def __init__(self, in_channels=16, base_ch=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1)
        self.conv_1d = nn.Conv2d(base_ch, base_ch, kernel_size=(1, 5), padding=(0, 2))
        self.conv_mid = nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1)
        self.dcm_block = DCMModule(ch=base_ch)
        self.prod_block = DynamicProDBlock(ch=base_ch)

    def forward(self, x):
        identity = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x + identity
        x = F.relu(self.conv_1d(x))
        x = F.relu(self.conv_mid(x))
        x = self.dcm_block(x)
        x = self.prod_block(x, loop_times=3)
        return x

# ====================== Reconstruction Head (identical to training) ======================
class Event2ReconHead(nn.Module):
    def __init__(self, in_ch=16):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_ch, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, feat):
        return self.decoder(feat)

# ====================== Test Dataset: matched by prefix numeric ID ======================
class Event2TestDataset(Dataset):
    def __init__(self, feat_dir, label_dir, img_size=320):
        self.feat_dir = feat_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.pairs = []

        if not os.path.isdir(feat_dir):
            print(f"[Error] Feature directory does not exist: {feat_dir}")
            return
        if not os.path.isdir(label_dir):
            print(f"[Error] Label directory does not exist: {label_dir}")
            return

        label_dict = {}
        label_files = [f for f in os.listdir(label_dir) if f.lower().endswith((".png", ".jpg"))]
        for fname in label_files:
            nums = re.findall(r"\d+", os.path.splitext(fname)[0])
            if nums:
                label_dict[nums[0]] = fname  # use first numeric group as prefix ID, consistent with training

        feat_files = [f for f in os.listdir(feat_dir) if f.lower().endswith(".npy")]
        debug_count = 0
        for fname in sorted(feat_files):
            nums = re.findall(r"\d+", os.path.splitext(fname)[0])
            if nums:
                file_id = nums[0]
                if file_id in label_dict:
                    self.pairs.append((fname, label_dict[file_id]))
                else:
                    if debug_count < 5:
                        print(f"[Debug-Unmatched] Feature file: {fname} | Extracted ID: {file_id} | No such ID in labels")
                        debug_count += 1

        print(f"[Debug] Feature files: {len(feat_files)} | Label files: {len(label_files)} | Matched pairs: {len(self.pairs)}")
        if len(self.pairs) > 0:
            print(f"[Debug] First 3 matched pairs:")
            for f, l in self.pairs[:3]:
                print(f"    Feature: {f}  <->  Label: {l}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        feat_fname, label_fname = self.pairs[idx]
        feat_np = np.load(os.path.join(self.feat_dir, feat_fname)).astype(np.float32)
        feat_tensor = torch.from_numpy(feat_np)
        feat_tensor = F.interpolate(
            feat_tensor.unsqueeze(0), size=(self.img_size, self.img_size),
            mode='bilinear', align_corners=False
        ).squeeze(0)

        label_pil = Image.open(os.path.join(self.label_dir, label_fname)).convert("L")
        label_pil = label_pil.resize((self.img_size, self.img_size), Image.BILINEAR)
        label_np = np.array(label_pil, dtype=np.float32) / 255.0
        label_tensor = torch.from_numpy(label_np).unsqueeze(0)
        return feat_tensor, label_tensor, feat_fname

# ====================== Main Inference Test Function ======================
def test_event2_branch():
    # Path configuration
    WEIGHT_PATH = r"F:\ESDAR\ESDAR_PROJECT\Event_branch\Event2_branch\model_Event2_branch\event2_branch_best.pth"
    TEST_DIR = r"F:\ESDAR\ESDAR_PROJECT\Event_branch\Event2_branch\Event1_branch_test_before_feat_all"
    LABEL_DIR = r"F:\ESDAR\ESDAR_PROJECT\gray_output_label"
    OUT_FEAT_DIR = r"F:\ESDAR\ESDAR_PROJECT\Event_branch\Event2_branch\Event2_branch_test_feat"
    OUT_VIS_DIR = r"F:\ESDAR\ESDAR_PROJECT\Event_branch\Event2_branch\Event2_branch_test_vis"

    os.makedirs(OUT_FEAT_DIR, exist_ok=True)
    os.makedirs(OUT_VIS_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset and dataloader
    test_dataset = Event2TestDataset(TEST_DIR, LABEL_DIR)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    total_sample = len(test_dataset)

    if total_sample == 0:
        print("No matched feature-label pairs found. Please verify the paths and filenames")
        return
    print(f"Starting inference, total test samples: {total_sample}")

    # Initialize network (identical to training)
    backbone = Event2Branch(in_channels=16, base_ch=16).to(device)
    recon_head = Event2ReconHead(in_ch=16).to(device)

    # Load full weights (backbone + reconstruction head)
    checkpoint = torch.load(WEIGHT_PATH, map_location=device, weights_only=True)
    backbone.load_state_dict(checkpoint["backbone"])
    recon_head.load_state_dict(checkpoint["recon_head"])
    backbone.eval()
    recon_head.eval()

    # Evaluation metrics
    mse_crit = nn.MSELoss().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = lpips.LPIPS(net='alex', spatial=False).to(device)

    total_mse = 0.0
    total_ssim = 0.0
    total_lpips = 0.0

    def tensor2cv(t):
        arr = t.squeeze().cpu().numpy() * 255
        return arr.astype(np.uint8)

    # Convert 16-channel feature to 4×4 grid visualization
    def feat16_to_grid(feat_tensor):
        feat = feat_tensor.cpu().numpy()
        C, H, W = feat.shape
        vis_list = []
        for c in range(C):
            ch = feat[c]
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max - ch_min < 1e-8:
                ch_norm = np.zeros_like(ch)
            else:
                ch_norm = (ch - ch_min) / (ch_max - ch_min) * 255
            vis_list.append(ch_norm.astype(np.uint8))
        rows = [np.hstack(vis_list[i*4:(i+1)*4]) for i in range(4)]
        return np.vstack(rows)

    with torch.no_grad():
        for feat_in, label_gt, fname_list in test_loader:
            feat_in = feat_in.to(device)
            label_gt = label_gt.to(device)
            feat_out = backbone(feat_in)
            pred_img = recon_head(feat_out)

            # Process and save sample by sample
            for i in range(feat_in.shape[0]):
                fname_stem = os.path.splitext(fname_list[i])[0]
                time.sleep(0.01)  # IO buffer

                # Save output feature tensor (16 channels)
                try:
                    feat_np = feat_out[i].cpu().numpy()
                    np.save(os.path.join(OUT_FEAT_DIR, f"{fname_stem}_out_feat.npy"), feat_np)
                except Exception as e:
                    print(f"Warning: failed to save feature for {fname_stem}: {e}")

                # Save feature visualization (4×4 grid)
                try:
                    vis_grid = feat16_to_grid(feat_out[i])
                    cv2.imwrite(os.path.join(OUT_VIS_DIR, f"{fname_stem}_feat_grid.png"), vis_grid)
                except Exception as e:
                    print(f"Warning: failed to save visualization for {fname_stem}: {e}")

                # Per-sample metric computation
                pred_single = pred_img[i:i+1]
                label_single = label_gt[i:i+1]
                mse_val = mse_crit(pred_single, label_single).item()
                ssim_val = ssim_metric(pred_single, label_single).item()

                pred3 = pred_single.repeat(1, 3, 1, 1)
                gt3 = label_single.repeat(1, 3, 1, 1)
                lpips_val = lpips_fn(pred3, gt3).mean().item()

                total_mse += mse_val
                total_ssim += ssim_val
                total_lpips += lpips_val

                print(f"Processed: {fname_list[i]} | MSE={mse_val:.4f} SSIM={ssim_val:.4f} LPIPS={lpips_val:.4f}")

    avg_mse = total_mse / total_sample
    avg_ssim = total_ssim / total_sample
    avg_lpips = total_lpips / total_sample

    print("\n===== Overall Test Set Average Metrics =====")
    print(f"Avg MSE: {avg_mse:.6f}")
    print(f"Avg SSIM: {avg_ssim:.4f}")
    print(f"Avg LPIPS: {avg_lpips:.4f}")
    print(f"\nOutput feature tensor path: {OUT_FEAT_DIR}")
    print(f"Feature visualization path: {OUT_VIS_DIR}")

if __name__ == "__main__":
    test_event2_branch()