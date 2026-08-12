import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
import cv2
import lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision import transforms

# ====================== Adaptive ProD Recurrent Block (identical to training) ======================
class DynamicProDBlock(nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        # Branch A generates spatial attention scaling factors
        self.branch_a = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        # Branch B convolution
        self.branch_b = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, loop_times=3):
        feat = x
        for _ in range(loop_times):
            scale_map = self.branch_a(feat)
            b_feat = self.branch_b(feat)
            feat = self.relu(b_feat * scale_map + feat)  # residual connection
        return feat

# ====================== Backbone Network (identical to training, includes ProD blocks and BN) ======================
class StaticImageAuxBranch(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.pool_max = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool_avg = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.path3_conv1 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.path3_conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.up_1x1 = nn.Conv2d(8, 16, kernel_size=1)
        self.prod_a1 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.prod_cycle = DynamicProDBlock(ch=16)
        self.prod_a2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.prod_b = nn.Conv2d(16, 16, kernel_size=1)
        self.bn_fire = nn.BatchNorm2d(8, track_running_stats=False)
        self.bn_prod = nn.BatchNorm2d(16, track_running_stats=False)
        self.drop = nn.Dropout2d(0.05)
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))
        self.w3 = nn.Parameter(torch.tensor(1.0))
        self.w4 = nn.Parameter(torch.tensor(1.0))
        self.w5 = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, return_intermediate=False):
        x = F.relu(self.conv1(x))
        x = self.drop(x)
        p1 = self.pool_max(x)
        p2 = self.pool_avg(x)
        p3 = F.relu(self.path3_conv1(x))
        p3 = self.drop(p3)
        p3 = F.relu(self.path3_conv2(p3))
        p3 = self.drop(p3)
        p3 = p3 + x
        # ========== Target export position: after three-path weighted fusion, before BN ==========
        x_fused = self.w1 * p1 + self.w2 * p2 + self.w3 * p3
        x = self.bn_fire(x_fused)
        # ========================================================
        x = self.up_1x1(x)
        pa = F.relu(self.prod_a1(x))
        pa = self.prod_cycle(pa, loop_times=3)
        pa = F.relu(self.prod_a2(pa))
        pa = self.drop(pa)
        pb = self.prod_b(x)
        pb = self.drop(pb)
        x = self.w4 * pa + self.w5 * pb
        out = self.bn_prod(x)
        if return_intermediate:
            return out, x_fused  # return final output + intermediate fusion feature
        return out

# ====================== Reconstruction Head (identical to training) ======================
class AuxBranchReconHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, feat):
        return self.decoder(feat)

# ====================== Test Dataset ======================
class TestGrayDataset(Dataset):
    def __init__(self, test_dir, label_dir, transform=None):
        self.test_dir = test_dir
        self.label_dir = label_dir
        self.transform = transform
        self.file_list = sorted([
            f for f in os.listdir(test_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            and os.path.exists(os.path.join(label_dir, f))
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        test_img = Image.open(os.path.join(self.test_dir, fname)).convert("L")
        label_img = Image.open(os.path.join(self.label_dir, fname)).convert("L")
        if self.transform:
            test_img = self.transform(test_img)
            label_img = self.transform(label_img)
        return test_img, label_img, fname

# ====================== Main Inference Test Function ======================
def test_static_branch():
    # Path configuration
    WEIGHT_PATH = r"F:\ESDAR\ESDAR_PROJECT\Static_branch\model\static_aux_best.pth"
    TEST_DIR = r"F:\ESDAR\ESDAR_PROJECT\gray_output_all"
    LABEL_DIR = r"F:\ESDAR\ESDAR_PROJECT\gray_output_label"
    OUT_FEAT_NPY = r"F:\ESDAR\ESDAR_PROJECT\Static_branch\test_output_feat"
    OUT_VIS_IMG = r"F:\ESDAR\ESDAR_PROJECT\Static_branch\test_output_vis"
    # Additional intermediate feature save path
    BEFORE_FEAT_DIR = r"F:\ESDAR\ESDAR_PROJECT\Static_branch\test_output_before_feat"
    BEFORE_VIS_DIR = r"F:\ESDAR\ESDAR_PROJECT\Static_branch\test_output_before_vis"

    os.makedirs(OUT_FEAT_NPY, exist_ok=True)
    os.makedirs(OUT_VIS_IMG, exist_ok=True)
    os.makedirs(BEFORE_FEAT_DIR, exist_ok=True)
    os.makedirs(BEFORE_VIS_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
    ])
    test_dataset = TestGrayDataset(TEST_DIR, LABEL_DIR, transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    total_sample = len(test_dataset)

    if total_sample == 0:
        print("No matched image pairs found. Please verify the folder paths")
        return
    print(f"Starting inference, total test samples: {total_sample}")

    backbone = StaticImageAuxBranch(in_channels=1).to(device)
    recon_head = AuxBranchReconHead().to(device)

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

    # Convert 8-channel feature to visualization grid (2 rows × 4 columns)
    def feat_to_grid(feat_tensor):
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
        row1 = np.hstack(vis_list[:4])
        row2 = np.hstack(vis_list[4:])
        return np.vstack([row1, row2])

    def tensor2cv(t):
        arr = t.squeeze().cpu().numpy() * 255
        return arr.astype(np.uint8)

    with torch.no_grad():
        for test_in, label_gt, fname_list in test_loader:
            test_in = test_in.to(device)
            label_gt = label_gt.to(device)
            # Enable intermediate feature return
            feat_16ch, before_bn_feat = backbone(test_in, return_intermediate=True)
            pred_img = recon_head(feat_16ch)

            for i in range(test_in.shape[0]):
                fname_stem = os.path.splitext(fname_list[i])[0]
                # Save backbone 16-channel final feature
                feat_np = feat_16ch[i].cpu().numpy()
                np.save(os.path.join(OUT_FEAT_NPY, f"{fname_stem}_feat.npy"), feat_np)

                # ========== Save 8-channel feature after three-path fusion and before BN ==========
                before_feat = before_bn_feat[i]
                np.save(os.path.join(BEFORE_FEAT_DIR, f"{fname_stem}_before_bn.npy"), before_feat.cpu().numpy())
                # Save visualization grid image
                vis_grid = feat_to_grid(before_feat)
                cv2.imwrite(os.path.join(BEFORE_VIS_DIR, f"{fname_stem}_before_bn.png"), vis_grid)
                # ======================================================

                # Per-sample metric computation
                pred_single = pred_img[i:i + 1]
                label_single = label_gt[i:i + 1]
                mse_val = mse_crit(pred_single, label_single).item()
                ssim_val = ssim_metric(pred_single, label_single).item()

                pred3 = pred_single.repeat(1, 3, 1, 1)
                gt3 = label_single.repeat(1, 3, 1, 1)
                lpips_val = lpips_fn(pred3, gt3).mean().item()

                total_mse += mse_val
                total_ssim += ssim_val
                total_lpips += lpips_val

                # Save input-prediction-gt comparison visualization
                in_cv = tensor2cv(test_in[i])
                pred_cv = tensor2cv(pred_img[i])
                gt_cv = tensor2cv(label_gt[i])
                concat_img = np.hstack([in_cv, pred_cv, gt_cv])
                cv2.imwrite(os.path.join(OUT_VIS_IMG, f"{fname_stem}.png"), concat_img)

                print(f"Processed: {fname_list[i]} | MSE={mse_val:.4f} SSIM={ssim_val:.4f} LPIPS={lpips_val:.4f}")

    avg_mse = total_mse / total_sample
    avg_ssim = total_ssim / total_sample
    avg_lpips = total_lpips / total_sample

    print("\n===== Overall Test Set Average Metrics =====")
    print(f"Avg MSE: {avg_mse:.6f}")
    print(f"Avg SSIM: {avg_ssim:.4f}")
    print(f"Avg LPIPS: {avg_lpips:.4f}")
    print(f"\nFinal feature path: {OUT_FEAT_NPY}")
    print(f"Reconstruction visualization path: {OUT_VIS_IMG}")
    print(f"Three-path fusion feature path: {BEFORE_FEAT_DIR}")
    print(f"Fusion feature visualization path: {BEFORE_VIS_DIR}")

if __name__ == "__main__":
    test_static_branch()