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
from torchmetrics.image import StructuralSimilarityIndexMeasure

# ====================== Coordinate Attention Module (identical to training) ======================
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_h * a_w
        return out

# ====================== Learnable Weighted Fusion Module (identical to training) ======================
class LearnableWeightedFusion(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.5)

    def forward(self, x1, x2):
        w = torch.sigmoid(self.alpha)
        out = w * x1 + (1 - w) * x2
        return out

# ====================== Generator: Multi-branch Fusion Reconstruction Network (identical to training) ======================
class FusionReconNet(nn.Module):
    def __init__(self, base_ch=16):
        super().__init__()
        self.proj_d = nn.Conv2d(8, base_ch, kernel_size=1, padding=0)
        self.fusion_ab = LearnableWeightedFusion(base_ch)
        self.fusion_cd = LearnableWeightedFusion(base_ch)
        self.ca_fusion = CoordAtt(inp=base_ch * 2, oup=base_ch * 2)
        self.conv_reduce = nn.Conv2d(base_ch * 2, base_ch, kernel_size=1, padding=0)
        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat_a, feat_b, feat_c, feat_d):
        feat_e = self.fusion_ab(feat_a, feat_b)
        feat_d_proj = self.proj_d(feat_d)
        feat_cd = self.fusion_cd(feat_c, feat_d_proj)
        feat_f = F.relu(feat_cd)
        concat_ef = torch.cat([feat_e, feat_f], dim=1)
        ca_feat = self.ca_fusion(concat_ef)
        fused_feat = self.conv_reduce(ca_feat)
        out_img = self.sigmoid(self.out_conv(fused_feat))
        return out_img, fused_feat

# ====================== Test Dataset: four-branch matching by numeric ID ======================
class FusionTestDataset(Dataset):
    def __init__(self, feat_a_dir, feat_b_dir, feat_c_dir, feat_d_dir, label_dir, img_size=320):
        self.feat_a_dir = feat_a_dir
        self.feat_b_dir = feat_b_dir
        self.feat_c_dir = feat_c_dir
        self.feat_d_dir = feat_d_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.samples = []

        for d in [feat_a_dir, feat_b_dir, feat_c_dir, feat_d_dir, label_dir]:
            if not os.path.isdir(d):
                print(f"[Error] Directory not found: {d}")
                return

        def get_id_dict(dir_path):
            id_dict = {}
            files = [f for f in os.listdir(dir_path) if f.lower().endswith(".npy")]
            for fname in files:
                nums = re.findall(r"\d+", os.path.splitext(fname)[0])
                if nums:
                    id_dict[nums[0]] = fname
            return id_dict

        dict_a = get_id_dict(feat_a_dir)
        dict_b = get_id_dict(feat_b_dir)
        dict_c = get_id_dict(feat_c_dir)
        dict_d = get_id_dict(feat_d_dir)

        label_dict = {}
        label_files = [f for f in os.listdir(label_dir) if f.lower().endswith((".png", ".jpg"))]
        for fname in label_files:
            nums = re.findall(r"\d+", os.path.splitext(fname)[0])
            if nums:
                label_dict[nums[0]] = fname

        common_ids = set(dict_a.keys()) & set(dict_b.keys()) & set(dict_c.keys()) & set(dict_d.keys()) & set(label_dict.keys())
        common_ids = sorted(list(common_ids))

        for fid in common_ids:
            self.samples.append((
                dict_a[fid], dict_b[fid], dict_c[fid], dict_d[fid], label_dict[fid]
            ))

        print(f"[Debug] Feature a: {len(dict_a)} | Feature b: {len(dict_b)} | Feature c: {len(dict_c)} | Feature d: {len(dict_d)} | Labels: {len(label_dict)}")
        print(f"[Debug] Total matched samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        a_f, b_f, c_f, d_f, lab_f = self.samples[idx]

        def load_feat(fpath):
            feat = np.load(fpath).astype(np.float32)
            t = torch.from_numpy(feat)
            t = F.interpolate(t.unsqueeze(0), size=(self.img_size, self.img_size),
                              mode='bilinear', align_corners=False).squeeze(0)
            return t

        feat_a = load_feat(os.path.join(self.feat_a_dir, a_f))
        feat_b = load_feat(os.path.join(self.feat_b_dir, b_f))
        feat_c = load_feat(os.path.join(self.feat_c_dir, c_f))
        feat_d = load_feat(os.path.join(self.feat_d_dir, d_f))

        label_pil = Image.open(os.path.join(self.label_dir, lab_f)).convert("L")
        label_pil = label_pil.resize((self.img_size, self.img_size), Image.BILINEAR)
        label_np = np.array(label_pil, dtype=np.float32) / 255.0
        label_tensor = torch.from_numpy(label_np).unsqueeze(0)
        return feat_a, feat_b, feat_c, feat_d, label_tensor, a_f

# ====================== Main Inference Test Function ======================
def test_fusion():
    # Path configuration
    WEIGHT_PATH = r"F:\ESDAR\ESDAR_PROJECT\Fusion_network\model_fusion\fusion_best.pth"
    FEAT_A = r"F:\ESDAR\ESDAR_PROJECT\Event_branch\Event1_branch\Event1_branch_test_feat"
    FEAT_B = r"F:\ESDAR\ESDAR_PROJECT\Static_branch\test_output_feat"
    FEAT_C = r"F:\ESDAR\ESDAR_PROJECT\Event_branch\Event2_branch\Event2_branch_test_feat"
    FEAT_D = r"F:\ESDAR\ESDAR_PROJECT\Static_branch\test_output_before_feat"
    LABEL_DIR = r"F:\ESDAR\ESDAR_PROJECT\gray_output_label"
    OUT_PNG_DIR = r"F:\ESDAR\ESDAR_PROJECT\Fusion_network\Fusion_network_test_png"

    os.makedirs(OUT_PNG_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset and dataloader
    test_dataset = FusionTestDataset(FEAT_A, FEAT_B, FEAT_C, FEAT_D, LABEL_DIR)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    total_sample = len(test_dataset)

    if total_sample == 0:
        print("No matched feature-label pairs found. Please verify the paths and filenames")
        return
    print(f"Starting inference, total test samples: {total_sample}")

    # Initialize network (identical to training)
    net_g = FusionReconNet(base_ch=16).to(device)

    # Load full generator weights
    checkpoint = torch.load(WEIGHT_PATH, map_location=device, weights_only=True)
    net_g.load_state_dict(checkpoint["net_g"])
    net_g.eval()

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

    with torch.no_grad():
        for feat_a, feat_b, feat_c, feat_d, label_gt, fname_list in test_loader:
            feat_a = feat_a.to(device)
            feat_b = feat_b.to(device)
            feat_c = feat_c.to(device)
            feat_d = feat_d.to(device)
            label_gt = label_gt.to(device)

            pred_img, _ = net_g(feat_a, feat_b, feat_c, feat_d)

            # Save and compute metrics per sample
            for i in range(feat_a.shape[0]):
                fname_stem = os.path.splitext(fname_list[i])[0]

                # Save reconstructed image
                try:
                    pred_cv = tensor2cv(pred_img[i])
                    cv2.imwrite(os.path.join(OUT_PNG_DIR, f"{fname_stem}_recon.png"), pred_cv)
                except Exception as e:
                    print(f"Warning: failed to save image for {fname_stem}: {e}")

                # Per-sample metrics
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
    print(f"\nReconstruction output path: {OUT_PNG_DIR}")

if __name__ == "__main__":
    test_fusion()