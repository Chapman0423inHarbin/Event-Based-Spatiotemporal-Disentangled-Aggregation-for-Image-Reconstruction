import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch.optim as optim
import lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure

# ====================== Adaptive ProD Recurrent Block ======================
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

# ====================== Static Image Auxiliary Branch ======================
class StaticImageAuxBranch(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        # FireNet three-path architecture
        self.pool_max = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool_avg = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.path3_conv1 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.path3_conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.up_1x1 = nn.Conv2d(8, 16, kernel_size=1)
        self.prod_a1 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.prod_cycle = DynamicProDBlock(ch=16)
        self.prod_a2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.prod_b = nn.Conv2d(16, 16, kernel_size=1)
        # Revised
        self.bn_fire = nn.BatchNorm2d(8, track_running_stats=False)
        self.bn_prod = nn.BatchNorm2d(16, track_running_stats=False)
        self.drop = nn.Dropout2d(0.05)
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))
        self.w3 = nn.Parameter(torch.tensor(1.0))
        self.w4 = nn.Parameter(torch.tensor(1.0))
        self.w5 = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.drop(x)
        p1 = self.pool_max(x)
        p2 = self.pool_avg(x)
        p3 = F.relu(self.path3_conv1(x))
        p3 = self.drop(p3)
        p3 = F.relu(self.path3_conv2(p3))
        p3 = self.drop(p3)
        p3 = p3 + x
        x = self.w1*p1 + self.w2*p2 + self.w3*p3
        x = self.bn_fire(x)
        x = self.up_1x1(x)
        pa = F.relu(self.prod_a1(x))
        pa = self.prod_cycle(pa, loop_times=3)
        pa = F.relu(self.prod_a2(pa))
        pa = self.drop(pa)
        pb = self.prod_b(x)
        pb = self.drop(pb)
        x = self.w4*pa + self.w5*pb
        out = self.bn_prod(x)
        return out

# ====================== Reconstruction Head ======================
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

# ====================== Dataset ======================
class DegradeGrayDataset(Dataset):
    def __init__(self, input_dir, label_dir, transform=None):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.transform = transform
        self.file_list = sorted([
            f for f in os.listdir(input_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            and os.path.exists(os.path.join(label_dir, f))
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        in_img = Image.open(os.path.join(self.input_dir, fname)).convert("L")
        label_img = Image.open(os.path.join(self.label_dir, fname)).convert("L")
        if self.transform:
            in_img = self.transform(in_img)
            label_img = self.transform(label_img)
        return in_img, label_img

# ====================== Main Training Function ======================
def train_static_aux_branch(
    train_folder,
    val_folder,
    label_folder,
    save_model_dir,
    batch_size=4,
    lr=8e-5,
    epoch_num=10,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    from torchvision import transforms
    train_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
    ])
    train_dataset = DegradeGrayDataset(train_folder, label_folder, transform=train_transform)
    val_dataset = DegradeGrayDataset(val_folder, label_folder, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    backbone = StaticImageAuxBranch(in_channels=1).to(device)
    recon_head = AuxBranchReconHead().to(device)
    full_model = nn.Sequential(backbone, recon_head)

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
    best_weight_path = os.path.join(save_model_dir, "static_aux_best.pth")
    last_weight_path = os.path.join(save_model_dir, "static_aux_last.pth")

    for epoch in range(epoch_num):
        full_model.train()
        train_total_loss = 0.0
        for in_img, label_img in train_loader:
            in_img = in_img.to(device)
            label_img = label_img.to(device)
            pred = full_model(in_img)

            loss_mse = mse_criterion(pred, label_img)
            ssim_val = ssim_metric(pred, label_img)
            loss_ssim = 1.0 - ssim_val
            # [0,1] → [-1,1] as required by LPIPS input
            pred3 = pred.repeat(1, 3, 1, 1)
            gt3 = label_img.repeat(1, 3, 1, 1)
            loss_lpips = lpips_loss_fn(pred3, gt3).mean()

            total_loss = w_mse * loss_mse + w_ssim * loss_ssim + w_lpips * loss_lpips
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
            for in_img, label_img in val_loader:
                in_img = in_img.to(device)
                label_img = label_img.to(device)
                pred = full_model(in_img)
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
            # Save best weights
            torch.save({
                "backbone": backbone.state_dict(),
                "recon_head": recon_head.state_dict()
            }, best_weight_path)
            print(f"Best weights saved to: {best_weight_path} | Best SSIM = {best_ssim:.4f}\n")

    torch.save({
        "backbone": backbone.state_dict(),
        "recon_head": recon_head.state_dict()
    }, last_weight_path)
    print(f"Training completed successfully!")
    print(f"Best backbone weights: {best_weight_path}")
    print(f"Last epoch backbone weights: {last_weight_path}")
    print(f"Global best validation SSIM = {best_ssim:.4f}")
    return backbone

if __name__ == "__main__":
    TRAIN_PATH = r"F:\ESDAR\ESDAR_PROJECT\gray_output_train"
    VAL_PATH = r"F:\ESDAR\ESDAR_PROJECT\gray_output_val"
    LABEL_PATH = r"F:\ESDAR\ESDAR_PROJECT\gray_output_label"
    MODEL_SAVE_DIR = r"F:\ESDAR\ESDAR_PROJECT\Static_branch\model"

    trained_backbone = train_static_aux_branch(
        train_folder=TRAIN_PATH,
        val_folder=VAL_PATH,
        label_folder=LABEL_PATH,
        save_model_dir=MODEL_SAVE_DIR,
        batch_size=4,
        lr=1e-4,
        epoch_num=20
    )

    test_tensor = torch.randn(1, 1, 320, 320).to("cuda" if torch.cuda.is_available() else "cpu")
    feat_out = trained_backbone(test_tensor)
    print(f"\nBackbone output feature shape: {feat_out.shape}")