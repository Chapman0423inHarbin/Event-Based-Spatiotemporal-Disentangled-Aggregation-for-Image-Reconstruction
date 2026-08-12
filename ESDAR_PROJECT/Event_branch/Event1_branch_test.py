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

# ====================== Adaptive ProD Recurrent Block ======================
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

# ====================== Fixed DCM (Deformable Convolution Guidance) Module ======================
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

        # Explicit average pooling layer to ensure identical parameters for both calls
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
        # Branch a: gradient map processing
        grad_raw = F.conv2d(x, self.sobel_kernel, padding=1, groups=C)
        gx = grad_raw[:, 0::2, :, :]
        gy = grad_raw[:, 1::2, :, :]
        grad_map = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        grad_feat = F.relu(grad_map)

        # Branch b: variance map processing (unified pooling layer)
        mean = self.avg_pool(x)
        mean_sq = self.avg_pool(x ** 2)
        var_map = mean_sq - mean ** 2
        var_feat = F.relu(self.var_conv(var_map))

        # Branch c: feature mapping
        main_conv = self.conv_main(x)
        feat_map = main_conv + grad_feat + var_feat

        # Branches d/e: deformable convolution parameters
        mask = self.mask_branch(var_feat)
        offset = self.offset_branch(var_feat)

        # Branch f: deformable convolution + residual connection
        deform_out = deform_conv2d(
            feat_map, offset, self.deform_weight, self.deform_bias,
            stride=1, padding=self.padding, dilation=1, mask=mask
        )
        out = F.relu(deform_out + self.res_weight * x)
        return out

# ====================== Deformable TCN Block ======================
class DeformableTCNBlock(nn.Module):
    def __init__(self, ch=16, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.res_weight_branch = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, ch, kernel_size=1),
            nn.Sigmoid()
        )
        self.offset_branch = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, 2 * kernel_size * kernel_size, kernel_size=3, padding=1)
        )
        self.mask_branch = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, kernel_size * kernel_size, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.conv_weight = nn.Parameter(torch.empty(ch, ch, kernel_size, kernel_size))
        self.conv_bias = nn.Parameter(torch.zeros(ch))
        nn.init.kaiming_normal_(self.conv_weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        res_weight = self.res_weight_branch(x)
        offset = self.offset_branch(x)
        mask = self.mask_branch(x)
        deform_out = deform_conv2d(
            x, offset, self.conv_weight, self.conv_bias,
            stride=1, padding=self.padding, dilation=1, mask=mask
        )
        out = F.relu(deform_out + x * res_weight)
        return out

# ====================== Event Stream Backbone Network ======================
class EventStreamBranch(nn.Module):
    def __init__(self, in_channels=2, base_ch=16, tcn_loop=4):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1)
        self.conv_1d = nn.Conv2d(base_ch, base_ch, kernel_size=(1, 5), padding=(0, 2))
        self.conv_mid = nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1)
        self.dcm_block = DCMModule(ch=base_ch)
        self.prod_block = DynamicProDBlock(ch=base_ch)
        self.tcn_blocks = nn.Sequential(*[
            DeformableTCNBlock(ch=base_ch) for _ in range(tcn_loop)
        ])
        self.bn_out = nn.BatchNorm2d(base_ch, track_running_stats=False)

    def forward(self, x, return_intermediate=False):
        x = F.relu(self.conv_in(x))
        feat_before_conv1d = x
        x = self.conv_1d(x)
        x = self.conv_mid(x)
        x = self.dcm_block(x)
        x = self.prod_block(x, loop_times=3)
        x = self.tcn_blocks(x)
        out = self.bn_out(x)
        if return_intermediate:
            return out, feat_before_conv1d
        return out

# ====================== Event Reconstruction Head ======================
class EventReconHead(nn.Module):
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

# ====================== Test Dataset ======================
class EventTestDataset(Dataset):
    def __init__(self, event_dir, label_dir, img_size=320):
        self.event_dir = event_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.pairs = []

        if not os.path.isdir(event_dir):
            print(f"[Error] Event directory does not exist: {event_dir}")
            return
        if not os.path.isdir(label_dir):
            print(f"[Error] Label directory does not exist: {label_dir}")
            return

        label_dict = {}
        label_files = [f for f in os.listdir(label_dir) if f.lower().endswith((".png", ".jpg"))]
        for fname in label_files:
            nums = re.findall(r"\d+", os.path.splitext(fname)[0])
            if nums:
                label_dict[nums[-1]] = fname

        event_files = [f for f in os.listdir(event_dir) if f.lower().endswith(".npy")]
        for fname in sorted(event_files):
            nums = re.findall(r"\d+", os.path.splitext(fname)[0])
            if nums and nums[-1] in label_dict:
                self.pairs.append((fname, label_dict[nums[-1]]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        event_fname, label_fname = self.pairs[idx]
        event_np = np.load(os.path.join(self.event_dir, event_fname)).astype(np.float32)
        if event_np.ndim == 3 and event_np.shape[2] == 2:
            event_np = event_np.transpose(2, 0, 1)
        event_tensor = torch.from_numpy(event_np)
        event_tensor = F.interpolate(
            event_tensor.unsqueeze(0), size=(self.img_size, self.img_size),
            mode='bilinear', align_corners=False
        ).squeeze(0)

        label_pil = Image.open(os.path.join(self.label_dir, label_fname)).convert("L")
        label_pil = label_pil.resize((self.img_size, self.img_size), Image.BILINEAR)
        label_np = np.array(label_pil, dtype=np.float32) / 255.0
        label_tensor = torch.from_numpy(label_np).unsqueeze(0)
        return event_tensor, label_tensor, event_fname

# ====================== Main Inference Test Function ======================
def test_event_branch():
    WEIGHT_PATH = r"F:\ESDAR\ESDAR_PROJECT\Event_branch\model_Event1_branch\event_branch_best.pth"
    TEST_DIR = r"F:\ESDAR\ESDAR_PROJECT\event_out_all"
    LABEL_DIR = r"F:\ESDAR\ESDAR_PROJECT\gray_output_label"
    OUT_FEAT_NPY = r"F:\ESDAR\ESDAR_PROJECT\Event_branch\Event1_branch_test_feat"
    OUT_VIS_IMG = r"F:\ESDAR\ESDAR_PROJECT\Event_branch\Event1_branch_test_vis"
    BEFORE_FEAT_DIR = r"F:\ESDAR\ESDAR_PROJECT\Event_branch\Event1_branch_test_before_feat"
    BEFORE_VIS_DIR = r"F:\ESDAR\ESDAR_PROJECT\Event_branch\Event1_branch_test_before_vis"

    def safe_makedirs(path):
        if not os.path.exists(path):
            os.makedirs(path)
    safe_makedirs(OUT_FEAT_NPY)
    safe_makedirs(OUT_VIS_IMG)
    safe_makedirs(BEFORE_FEAT_DIR)
    safe_makedirs(BEFORE_VIS_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_dataset = EventTestDataset(TEST_DIR, LABEL_DIR)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    total_sample = len(test_dataset)

    if total_sample == 0:
        print("No matched event-label pairs found. Please verify the paths and filenames")
        return
    print(f"Starting inference, total test samples: {total_sample}")

    backbone = EventStreamBranch(in_channels=2, base_ch=16, tcn_loop=4).to(device)
    recon_head = EventReconHead(in_ch=16).to(device)

    checkpoint = torch.load(WEIGHT_PATH, map_location=device, weights_only=True)
    backbone.load_state_dict(checkpoint["backbone"])
    recon_head.load_state_dict(checkpoint["recon_head"])
    backbone.eval()
    recon_head.eval()

    mse_crit = nn.MSELoss().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = lpips.LPIPS(net='alex', spatial=False).to(device)

    total_mse = 0.0
    total_ssim = 0.0
    total_lpips = 0.0

    def tensor2cv(t):
        arr = t.squeeze().cpu().numpy() * 255
        return arr.astype(np.uint8)

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
        for event_in, label_gt, fname_list in test_loader:
            event_in = event_in.to(device)
            label_gt = label_gt.to(device)
            feat_16ch, before_feat = backbone(event_in, return_intermediate=True)
            pred_img = recon_head(feat_16ch)

            for i in range(event_in.shape[0]):
                fname_stem = os.path.splitext(fname_list[i])[0]
                time.sleep(0.01)

                # Save backbone feature
                try:
                    feat_np = feat_16ch[i].cpu().numpy()
                    np.save(os.path.join(OUT_FEAT_NPY, f"{fname_stem}_feat.npy"), feat_np)
                except Exception as e:
                    print(f"Warning: failed to save backbone feature for {fname_stem}: {e}")

                # Save intermediate feature after conv_in
                try:
                    bf_np = before_feat[i].cpu().numpy()
                    np.save(os.path.join(BEFORE_FEAT_DIR, f"{fname_stem}_before_conv1d.npy"), bf_np)
                except Exception as e:
                    print(f"Warning: failed to save intermediate feature npy for {fname_stem}: {e}")

                # Save intermediate feature visualization
                try:
                    grid = feat16_to_grid(before_feat[i])
                    cv2.imwrite(os.path.join(BEFORE_VIS_DIR, f"{fname_stem}_before_conv1d.png"), grid)
                except Exception as e:
                    print(f"Warning: failed to save intermediate visualization for {fname_stem}: {e}")

                # Metric computation (fixed bug where label was missing in SSIM call)
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

                # Comparison visualization
                try:
                    ev = tensor2cv(event_in[i][0:1])
                    pr = tensor2cv(pred_img[i])
                    gt = tensor2cv(label_gt[i])
                    concat = np.hstack([ev, pr, gt])
                    cv2.imwrite(os.path.join(OUT_VIS_IMG, f"{fname_stem}.png"), concat)
                except Exception as e:
                    print(f"Warning: failed to save comparison image for {fname_stem}: {e}")

                print(f"Processed: {fname_list[i]} | MSE={mse_val:.4f} SSIM={ssim_val:.4f} LPIPS={lpips_val:.4f}")

    avg_mse = total_mse / total_sample
    avg_ssim = total_ssim / total_sample
    avg_lpips = total_lpips / total_sample

    print("\n===== Overall Test Set Average Metrics =====")
    print(f"Avg MSE: {avg_mse:.6f}")
    print(f"Avg SSIM: {avg_ssim:.4f}")
    print(f"Avg LPIPS: {avg_lpips:.4f}")
    print(f"\nBackbone feature path: {OUT_FEAT_NPY}")
    print(f"Reconstruction visualization path: {OUT_VIS_IMG}")
    print(f"Post-conv_in feature path: {BEFORE_FEAT_DIR}")
    print(f"Post-conv_in visualization path: {BEFORE_VIS_DIR}")

if __name__ == "__main__":
    test_event_branch()