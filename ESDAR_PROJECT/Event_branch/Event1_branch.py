import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import re
import numpy as np
from PIL import Image
import torch.optim as optim
import lpips
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

# ====================== Deformable Convolution Guidance Module (DCM) ======================
class DCMModule(nn.Module):
    def __init__(self, ch=16, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.ch = ch

        # Fixed Sobel gradient operator (channel-wise, computes x/y gradients per channel)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_kernel = torch.cat([sobel_x, sobel_y], dim=0).repeat(ch, 1, 1, 1)
        self.register_buffer('sobel_kernel', sobel_kernel)

        # Branch b: variance map with 1×5 1D convolution
        self.var_conv = nn.Conv2d(ch, ch, kernel_size=(1, 5), padding=(0, 2))

        # Branch c: main 3×3 convolution
        self.conv_main = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

        # Branch d: generates convolution weight modulation mask
        self.mask_branch = nn.Sequential(
            nn.Conv2d(ch, kernel_size * kernel_size, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # Branch e: generates convolution offset field
        self.offset_branch = nn.Conv2d(ch, 2 * kernel_size * kernel_size, kernel_size=1)

        # Base weights and bias for deformable convolution
        self.deform_weight = nn.Parameter(torch.empty(ch, ch, kernel_size, kernel_size))
        self.deform_bias = nn.Parameter(torch.zeros(ch))
        nn.init.kaiming_normal_(self.deform_weight, mode='fan_out', nonlinearity='relu')

        # Learnable residual weight
        self.res_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        B, C, H, W = x.shape

        # ========== Branch a: input gradient map → ReLU → gradient feature ==========
        grad_raw = F.conv2d(x, self.sobel_kernel, padding=1, groups=C)
        gx = grad_raw[:, 0::2, :, :]
        gy = grad_raw[:, 1::2, :, :]
        grad_map = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        grad_feat = F.relu(grad_map)

        # ========== Branch b: input variance map → 1×5 conv → ReLU → variance feature ==========
        # Local variance computation in 3×3 windows
        mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        mean_sq = F.avg_pool2d(x ** 2, kernel_size=3, stride=1, padding=1)
        var_map = mean_sq - mean ** 2
        var_feat = F.relu(self.var_conv(var_map))

        # ========== Branch c: 3×3 conv + gradient + variance → feature map ==========
        main_conv = self.conv_main(x)
        feat_map = main_conv + grad_feat + var_feat

        # ========== Branch d: variance feature → convolution mask ==========
        mask = self.mask_branch(var_feat)

        # ========== Branch e: variance feature → convolution offset ==========
        offset = self.offset_branch(var_feat)

        # ========== Branch f: deformable conv on feature map + input residual → output ==========
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

# ====================== Event Stream Backbone (with DCM) ======================
class EventStreamBranch(nn.Module):
    def __init__(self, in_channels=2, base_ch=16, tcn_loop=4):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1)
        self.conv_1d = nn.Conv2d(base_ch, base_ch, kernel_size=(1, 5), padding=(0, 2))
        self.conv_mid = nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1)
        # DCM module inserted between conv_mid and prod_block
        self.dcm_block = DCMModule(ch=base_ch)
        self.prod_block = DynamicProDBlock(ch=base_ch)
        self.tcn_blocks = nn.Sequential(*[
            DeformableTCNBlock(ch=base_ch) for _ in range(tcn_loop)
        ])
        self.bn_out = nn.BatchNorm2d(base_ch, track_running_stats=False)

    def forward(self, x):
        x = F.relu(self.conv_in(x))
        x = self.conv_1d(x)
        x = self.conv_mid(x)
        x = self.dcm_block(x)
        x = self.prod_block(x, loop_times=3)
        x = self.tcn_blocks(x)
        out = self.bn_out(x)
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

# ====================== Dataset: matched by numeric ID ======================
class EventStreamDataset(Dataset):
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

        # Build label ID dictionary
        label_dict = {}
        label_files = [f for f in os.listdir(label_dir) if f.lower().endswith((".png", ".jpg"))]
        for fname in label_files:
            nums = re.findall(r"\d+", os.path.splitext(fname)[0])
            if nums:
                label_dict[nums[-1]] = fname

        # Match event files by numeric ID
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
        return event_tensor, label_tensor

# ====================== Main Training Function ======================
def train_event_branch(
    train_folder,
    val_folder,
    label_folder,
    save_model_dir,
    batch_size,
    lr,
    epoch_num,   # Aligned with entry call parameters
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    train_dataset = EventStreamDataset(train_folder, label_folder)
    val_dataset = EventStreamDataset(val_folder, label_folder)

    if len(train_dataset) == 0:
        raise ValueError("No matched pairs in training set. Please check file paths and naming.")
    if len(val_dataset) == 0:
        raise ValueError("No matched pairs in validation set. Please check file paths and naming.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    backbone = EventStreamBranch(in_channels=2, base_ch=16, tcn_loop=4).to(device)
    recon_head = EventReconHead(in_ch=16).to(device)
    full_model = nn.Sequential(backbone, recon_head)

    # Triple supervision: MSE + SSIM + LPIPS
    mse_criterion = nn.MSELoss().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_loss_fn = lpips.LPIPS(net='alex', spatial=False).to(device)

    w_mse = 0.2
    w_ssim = 1.0
    w_lpips = 1.2

    optimizer = optim.AdamW(full_model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)

    best_ssim = 0.0
    os.makedirs(save_model_dir, exist_ok=True)
    best_weight_path = os.path.join(save_model_dir, "event_branch_best.pth")
    last_weight_path = os.path.join(save_model_dir, "event_branch_last.pth")

    for epoch in range(epoch_num):
        full_model.train()
        train_total_loss = 0.0
        for event_in, label_img in train_loader:
            event_in = event_in.to(device)
            label_img = label_img.to(device)
            pred = full_model(event_in)

            loss_mse = mse_criterion(pred, label_img)
            ssim_val = ssim_metric(pred, label_img)
            loss_ssim = 1.0 - ssim_val
            pred3 = pred.repeat(1, 3, 1, 1)
            gt3 = label_img.repeat(1, 3, 1, 1)
            loss_lpips = lpips_loss_fn(pred3, gt3).mean()

            total_loss = w_mse * loss_mse + w_ssim * loss_ssim + w_lpips
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_total_loss += total_loss.item()

        avg_train_loss = train_total_loss / len(train_loader)
        scheduler.step()

        # Validation phase
        full_model.eval()
        val_mse_sum = 0.0
        val_ssim_sum = 0.0
        val_lpips_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for event_in, label_img in val_loader:
                event_in = event_in.to(device)
                label_img = label_img.to(device)
                pred = full_model(event_in)
                val_mse_sum += mse_criterion(pred, label_img).item()
                ssim_v = ssim_metric(pred, label_img).item()
                val_ssim_sum += ssim_v
                pred3 = pred.repeat(1, 3, 1, 1)
                gt3 = label_img.repeat(1, 3, 1, 1)
                val_lpips_sum += lpips_loss_fn(pred3, gt3).mean()
                val_count += 1

        avg_mse = val_mse_sum / val_count
        avg_ssim = val_ssim_sum / val_count
        avg_lpips = val_lpips_sum / val_count

        print(f"==== Epoch [{epoch+1:02d}/{epoch_num}] LR={scheduler.get_last_lr()[0]:.6f} ====")
        print(f"Train Avg Loss: {avg_train_loss:.6f}")
        print(f"Val MSE: {avg_mse:.6f} | Val SSIM: {avg_ssim:.4f} | Val LPIPS: {avg_lpips:.4f}\n")

        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            torch.save({
                "backbone": backbone.state_dict(),
                "recon_head": recon_head.state_dict()
            }, best_weight_path)
            print(f"Best weights saved | Best SSIM = {best_ssim:.4f}\n")

    torch.save({
        "backbone": backbone.state_dict(),
        "recon_head": recon_head.state_dict()
    }, last_weight_path)
    print(f"Training finished. Best weights: {best_weight_path} | Global best SSIM = {best_ssim:.4f}")
    return backbone

# ====================== Entry Point ======================
if __name__ == "__main__":
    TRAIN_PATH = r"F:\ESDAR\ESDAR_PROJECT\event_out_train"
    VAL_PATH = r"F:\ESDAR\ESDAR_PROJECT\event_out_val"
    LABEL_PATH = r"F:\ESDAR\ESDAR_PROJECT\gray_output_label"
    MODEL_SAVE_DIR = r"F:\ESDAR\ESDAR_PROJECT\Event_branch\model_Event1_branch"

    trained_backbone = train_event_branch(
        train_folder=TRAIN_PATH,
        val_folder=VAL_PATH,
        label_folder=LABEL_PATH,
        save_model_dir=MODEL_SAVE_DIR,
        batch_size=4,
        lr=1e-4,
        epoch_num=20,
    )

    test_tensor = torch.randn(1, 2, 320, 320).to("cuda" if torch.cuda.is_available() else "cpu")
    feat_out = trained_backbone(test_tensor)
    print(f"\nEvent backbone output feature shape: {feat_out.shape}")