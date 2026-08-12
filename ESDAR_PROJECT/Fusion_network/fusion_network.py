import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import os
import re
import numpy as np
from PIL import Image
import torch.optim as optim
import lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure

# ====================== Coordinate Attention Module ======================
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

# ====================== Learnable Weighted Fusion Module ======================
class LearnableWeightedFusion(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.5)

    def forward(self, x1, x2):
        w = torch.sigmoid(self.alpha)
        out = w * x1 + (1 - w) * x2
        return out

# ====================== Generator: Multi-branch Fusion Reconstruction Network (channel alignment fixed) ======================
class FusionReconNet(nn.Module):
    def __init__(self, base_ch=16):
        super().__init__()
        # Channel projection: upsample 8-channel shallow feature d to 16 channels
        self.proj_d = nn.Conv2d(8, base_ch, kernel_size=1, padding=0)
        self.fusion_ab = LearnableWeightedFusion(base_ch)
        self.fusion_cd = LearnableWeightedFusion(base_ch)
        self.ca_fusion = CoordAtt(inp=base_ch * 2, oup=base_ch * 2)
        self.conv_reduce = nn.Conv2d(base_ch * 2, base_ch, kernel_size=1, padding=0)
        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat_a, feat_b, feat_c, feat_d):
        # Weighted fusion of a and b yields e
        feat_e = self.fusion_ab(feat_a, feat_b)
        # d is projected to match channels, then fused with c + ReLU yields f
        feat_d_proj = self.proj_d(feat_d)
        feat_cd = self.fusion_cd(feat_c, feat_d_proj)
        feat_f = F.relu(feat_cd)
        # Coordinate attention based fusion of e and f
        concat_ef = torch.cat([feat_e, feat_f], dim=1)
        ca_feat = self.ca_fusion(concat_ef)
        fused_feat = self.conv_reduce(ca_feat)
        # Output reconstructed image
        out_img = self.sigmoid(self.out_conv(fused_feat))
        return out_img, fused_feat

# ====================== Discriminator (Adversarial Branch) ======================
class Discriminator(nn.Module):
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 4, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)

# ====================== Dataset: Four-branch features matched by numeric ID ======================
class FusionDataset(Dataset):
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

        common_ids = set(dict_a.keys()) & set(dict_b.keys()) & set(dict_c.keys()) & set(dict_d.keys()) & set(
            label_dict.keys())
        common_ids = sorted(list(common_ids))

        for fid in common_ids:
            self.samples.append((
                dict_a[fid], dict_b[fid], dict_c[fid], dict_d[fid], label_dict[fid]
            ))

        print(
            f"[Debug] Feature a: {len(dict_a)} | Feature b: {len(dict_b)} | Feature c: {len(dict_c)} | Feature d: {len(dict_d)} | Labels: {len(label_dict)}")
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
        return feat_a, feat_b, feat_c, feat_d, label_tensor

# ====================== Main Training Function ======================
def train_fusion():
    # Path configuration
    FEAT_A = r"F:\ESDAR\ESDAR_PROJECT\Event_branch\Event1_branch\Event1_branch_test_feat"
    FEAT_B = r"F:\ESDAR\ESDAR_PROJECT\Static_branch\test_output_feat"
    FEAT_C = r"F:\ESDAR\ESDAR_PROJECT\Event_branch\Event2_branch\Event2_branch_test_feat"
    FEAT_D = r"F:\ESDAR\ESDAR_PROJECT\Static_branch\test_output_before_feat"
    LABEL_DIR = r"F:\ESDAR\ESDAR_PROJECT\gray_output_label"
    SAVE_DIR = r"F:\ESDAR\ESDAR_PROJECT\fusion_network\model_fusion"
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = 320
    batch_size = 4
    lr = 1e-4
    epoch_num = 20
    val_ratio = 0.1

    # Dataset and split
    full_dataset = FusionDataset(FEAT_A, FEAT_B, FEAT_C, FEAT_D, LABEL_DIR)
    if len(full_dataset) == 0:
        raise ValueError("No matched samples found. Please check file paths and naming.")

    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(42))
    print(f"Training set: {train_size} | Validation set: {val_size}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # Network initialization
    net_g = FusionReconNet(base_ch=16).to(device)
    net_d = Discriminator(in_ch=1).to(device)

    # Losses and metrics
    mse_loss = nn.MSELoss().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = lpips.LPIPS(net='alex', spatial=False).to(device)

    # Loss weights
    w_mse = 0.2
    w_ssim = 1.0
    w_lpips = 1.2
    w_adv = 0.001

    # Optimizers
    opt_g = optim.AdamW(net_g.parameters(), lr=lr, weight_decay=1e-5)
    opt_d = optim.AdamW(net_d.parameters(), lr=lr * 0.1, weight_decay=1e-5)
    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epoch_num)
    scheduler_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epoch_num)

    best_ssim = 0.0
    best_weight_path = os.path.join(SAVE_DIR, "fusion_best.pth")
    last_weight_path = os.path.join(SAVE_DIR, "fusion_last.pth")

    for epoch in range(epoch_num):
        net_g.train()
        net_d.train()
        total_g_loss = 0.0
        total_d_loss = 0.0

        for feat_a, feat_b, feat_c, feat_d, label_gt in train_loader:
            feat_a = feat_a.to(device)
            feat_b = feat_b.to(device)
            feat_c = feat_c.to(device)
            feat_d = feat_d.to(device)
            label_gt = label_gt.to(device)

            # ========== Train Discriminator ==========
            opt_d.zero_grad()
            fake_img, _ = net_g(feat_a, feat_b, feat_c, feat_d)
            pred_real = net_d(label_gt)
            pred_fake = net_d(fake_img.detach())

            loss_d_real = mse_loss(pred_real, torch.ones_like(pred_real))
            loss_d_fake = mse_loss(pred_fake, torch.zeros_like(pred_fake))
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            opt_d.step()

            # ========== Train Generator ==========
            opt_g.zero_grad()
            fake_img, _ = net_g(feat_a, feat_b, feat_c, feat_d)
            pred_fake_g = net_d(fake_img)

            # Reconstruction losses
            loss_mse = mse_loss(fake_img, label_gt)
            ssim_val = ssim_metric(fake_img, label_gt)
            loss_ssim = 1.0 - ssim_val
            loss_lpips = lpips_fn(fake_img.repeat(1, 3, 1, 1), label_gt.repeat(1, 3, 1, 1)).mean()

            # Adversarial loss
            loss_adv = mse_loss(pred_fake_g, torch.ones_like(pred_fake_g))

            loss_g = w_mse * loss_mse + w_ssim * loss_ssim + w_lpips * loss_lpips + w_adv * loss_adv
            loss_g.backward()
            opt_g.step()

            total_g_loss += loss_g.item()
            total_d_loss += loss_d.item()

        avg_g_loss = total_g_loss / len(train_loader)
        avg_d_loss = total_d_loss / len(train_loader)
        scheduler_g.step()
        scheduler_d.step()

        # ========== Validation ==========
        net_g.eval()
        val_mse_sum = 0.0
        val_ssim_sum = 0.0
        val_lpips_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for feat_a, feat_b, feat_c, feat_d, label_gt in val_loader:
                feat_a = feat_a.to(device)
                feat_b = feat_b.to(device)
                feat_c = feat_c.to(device)
                feat_d = feat_d.to(device)
                label_gt = label_gt.to(device)

                pred_img, _ = net_g(feat_a, feat_b, feat_c, feat_d)
                val_mse_sum += mse_loss(pred_img, label_gt).item()
                val_ssim_sum += ssim_metric(pred_img, label_gt).item()
                val_lpips_sum += lpips_fn(pred_img.repeat(1, 3, 1, 1), label_gt.repeat(1, 3, 1, 1)).mean().item()
                val_count += 1

        avg_val_mse = val_mse_sum / val_count
        avg_val_ssim = val_ssim_sum / val_count
        avg_val_lpips = val_lpips_sum / val_count

        print(f"==== Epoch [{epoch + 1:02d}/{epoch_num}] ====")
        print(f"Train G_loss: {avg_g_loss:.6f} | D_loss: {avg_d_loss:.6f}")
        print(f"Val MSE: {avg_val_mse:.6f} | SSIM: {avg_val_ssim:.4f} | LPIPS: {avg_val_lpips:.4f}\n")

        if avg_val_ssim > best_ssim:
            best_ssim = avg_val_ssim
            torch.save({
                "net_g": net_g.state_dict(),
                "net_d": net_d.state_dict(),
            }, best_weight_path)
            print(f"Best weights updated, best SSIM: {best_ssim:.4f}\n")

    # Save final weights
    torch.save({
        "net_g": net_g.state_dict(),
        "net_d": net_d.state_dict(),
    }, last_weight_path)
    print(f"Training finished. Best weights: {best_weight_path} | Global best SSIM: {best_ssim:.4f}")
    return net_g

# ====================== Entry Point ======================
if __name__ == "__main__":
    train_fusion()