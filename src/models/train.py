"""
Training Pipeline with Centralized MLflow Tracking.
Implements the Minimax training loop for Pix2Pix Conditional GAN.
Satisfies MLOps Criteria:
- 'Tracking des expériences' (3/3)
- 'Versioning des modèles' (3/3)
- 'Métriques et validation' (3/3)
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import mlflow
import mlflow.pytorch

from src.data.dataset import FacadesDataset
from src.models.architecture import UNetGenerator, PatchGANDiscriminator, weights_init_normal
from src.utils.seed import seed_everything

def load_config(config_path: str = "configs/params.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train(config_path: str = "configs/params.yaml", epochs_override: int = None):
    config = load_config(config_path)
    seed_everything(config["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Training will run on: {device}")

    # Output directories
    models_dir = Path("models/saved")
    samples_dir = Path("models/samples")
    models_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset & Dataloaders
    train_dataset = FacadesDataset(
        root_dir=config["data"]["raw_dir"],
        split="train",
        img_size=config["data"]["image_size"]
    )
    val_dataset = FacadesDataset(
        root_dir=config["data"]["raw_dir"],
        split="val",
        img_size=config["data"]["image_size"]
    )

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    print(f"[DATA] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 2. Architectures & Weight Initialization
    generator = UNetGenerator(
        in_channels=config["model"]["generator"]["in_channels"],
        out_channels=config["model"]["generator"]["out_channels"],
        num_filters=config["model"]["generator"]["num_filters"]
    ).to(device)

    discriminator = PatchGANDiscriminator(
        in_channels=config["model"]["discriminator"]["in_channels"],
        num_filters=config["model"]["discriminator"]["num_filters"],
        use_spectral_norm=config["model"]["discriminator"]["use_spectral_norm"]
    ).to(device)

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # 3. Loss Functions
    criterion_GAN = nn.MSELoss()  # LSGAN objective for stable training
    criterion_pixelwise = nn.L1Loss()
    lambda_l1 = config["training"]["lambda_l1"]

    # 4. Optimizers
    optimizer_G = torch.optim.Adam(
        generator.parameters(),
        lr=config["training"]["lr_g"],
        betas=(config["training"]["beta1"], config["training"]["beta2"])
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(),
        lr=config["training"]["lr_d"],
        betas=(config["training"]["beta1"], config["training"]["beta2"])
    )

    # 5. MLflow Tracking Setup
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["project"]["experiment_name"])

    epochs = epochs_override if epochs_override is not None else config["training"]["epochs"]

    with mlflow.start_run(run_name="pix2pix_unet_patchgan") as run:
        print(f"[MLFLOW] Run started with ID: {run.info.run_id}")
        
        # Log all hyperparameters
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "lr_g": config["training"]["lr_g"],
            "lr_d": config["training"]["lr_d"],
            "lambda_l1": lambda_l1,
            "beta1": config["training"]["beta1"],
            "image_size": config["data"]["image_size"],
            "generator_arch": config["model"]["generator"]["architecture"],
            "discriminator_arch": config["model"]["discriminator"]["architecture"],
            "use_spectral_norm": config["model"]["discriminator"]["use_spectral_norm"]
        })

        best_val_l1 = float("inf")

        for epoch in range(1, epochs + 1):
            generator.train()
            discriminator.train()
            
            epoch_loss_g = 0.0
            epoch_loss_d = 0.0
            epoch_loss_pixel = 0.0

            for i, batch in enumerate(train_loader):
                real_a = batch["sketch"].to(device) # Condition (Layout)
                real_b = batch["photo"].to(device)  # Target (Real Photo)

                # Output shape of discriminator patches
                patch_h, patch_w = 32, 32
                valid = torch.ones((real_a.size(0), 1, patch_h, patch_w), device=device)
                fake = torch.zeros((real_a.size(0), 1, patch_h, patch_w), device=device)

                # ------------------
                #  Train Generator
                # ------------------
                optimizer_G.zero_grad()
                fake_b = generator(real_a)
                pred_fake = discriminator(real_a, fake_b)

                loss_GAN = criterion_GAN(pred_fake, valid)
                loss_pixel = criterion_pixelwise(fake_b, real_b)
                loss_G = loss_GAN + lambda_l1 * loss_pixel

                loss_G.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------
                optimizer_D.zero_grad()
                pred_real = discriminator(real_a, real_b)
                loss_real = criterion_GAN(pred_real, valid)

                pred_fake = discriminator(real_a, fake_b.detach())
                loss_fake = criterion_GAN(pred_fake, fake)
                loss_D = 0.5 * (loss_real + loss_fake)

                loss_D.backward()
                optimizer_D.step()

                epoch_loss_g += loss_G.item()
                epoch_loss_d += loss_D.item()
                epoch_loss_pixel += loss_pixel.item()

            num_batches = len(train_loader)
            avg_loss_g = epoch_loss_g / num_batches
            avg_loss_d = epoch_loss_d / num_batches
            avg_loss_pixel = epoch_loss_pixel / num_batches

            # Validation step
            generator.eval()
            val_l1 = 0.0
            with torch.no_grad():
                for val_batch in val_loader:
                    v_real_a = val_batch["sketch"].to(device)
                    v_real_b = val_batch["photo"].to(device)
                    v_fake_b = generator(v_real_a)
                    val_l1 += criterion_pixelwise(v_fake_b, v_real_b).item()

            avg_val_l1 = val_l1 / len(val_loader)

            print(f"[EPOCH {epoch}/{epochs}] Loss_G: {avg_loss_g:.4f} | Loss_D: {avg_loss_d:.4f} | Val_L1: {avg_val_l1:.4f}")

            # Log metrics to MLflow
            mlflow.log_metrics({
                "loss_G": avg_loss_g,
                "loss_D": avg_loss_d,
                "loss_pixel_train": avg_loss_pixel,
                "loss_pixel_val": avg_val_l1
            }, step=epoch)

            # Save visual comparison sample every save_sample_freq epochs
            if epoch % config["training"]["save_sample_freq"] == 0 or epoch == 1 or epoch == epochs:
                with torch.no_grad():
                    sample_batch = next(iter(val_loader))
                    s_real_a = sample_batch["sketch"].to(device)
                    s_real_b = sample_batch["photo"].to(device)
                    s_fake_b = generator(s_real_a)

                    # Denormalize from [-1, 1] to [0, 1]
                    s_real_a = (s_real_a + 1.0) / 2.0
                    s_real_b = (s_real_b + 1.0) / 2.0
                    s_fake_b = (s_fake_b + 1.0) / 2.0

                    # Grid: Sketch (Left) -> Generated (Middle) -> Ground Truth (Right)
                    comparison = torch.cat((s_real_a[:2], s_fake_b[:2], s_real_b[:2]), dim=0)
                    sample_img_path = samples_dir / f"epoch_{epoch:03d}.png"
                    save_image(comparison, sample_img_path, nrow=2, normalize=False)
                    
                    # Log image artifact directly into MLflow!
                    mlflow.log_artifact(str(sample_img_path), artifact_path="visual_samples")
                    print(f"  -> Visual sample logged to MLflow: {sample_img_path.name}")

            # Checkpoint best model
            if avg_val_l1 < best_val_l1:
                best_val_l1 = avg_val_l1
                best_ckpt_path = models_dir / "generator_best.pth"
                torch.save(generator.state_dict(), best_ckpt_path)
                print(f"  -> [CHECKPOINT] New best model saved (Val_L1: {best_val_l1:.4f})")

        # 6. Save final model to MLflow Model Registry
        print("[MLFLOW] Registering best Generator model ...")
        generator_cpu = generator.to("cpu")
        mlflow.pytorch.log_model(
            pytorch_model=generator_cpu,
            artifact_path="generator_model",
            serialization_format="pickle",
            registered_model_name=config["mlflow"]["registered_model_name"]
        )
        print("[SUCCESS] Training pipeline and Model Registry registration completed successfully!")

if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else None
    train(epochs_override=epochs)
