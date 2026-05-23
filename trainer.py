import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from metrics import MAE, RMSE_MAE_MAPE


class HybridMultiScaleLoss(nn.Module):
    """Masked MAE with log-space and Huber auxiliary terms."""

    def __init__(self, rho: float = 0.5):
        super().__init__()
        self.rho = rho

    def forward(self, preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = labels != null_val

        mask = mask.float()
        mask = mask / (torch.mean(mask) + 1e-6)
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

        loss_mae = torch.mean(torch.abs(preds - labels) * mask)

        preds_log = torch.log1p(F.relu(preds))
        labels_log = torch.log1p(F.relu(labels))
        loss_log = torch.mean(torch.abs(preds_log - labels_log) * mask)

        loss_huber = torch.mean(F.smooth_l1_loss(preds, labels, reduction="none") * mask)
        return loss_mae + 5.0 * self.rho * loss_log + 0.1 * loss_huber


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        scaler,
        loaders,
        optimizer: torch.optim.Optimizer,
        logger: Optional[SummaryWriter] = None,
        scheduler=None,
        device=None,
        checkpoint: str = "best.pth",
        checkpoint_dir: str = "./checkpoints",
        save_every_epoch: bool = False,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.scaler = scaler
        self.train_loader, self.val_loader, self.test_loader = loaders
        self.optimizer = optimizer
        self.logger = logger
        self.scheduler = scheduler
        self.checkpoint = checkpoint
        self.checkpoint_dir = checkpoint_dir
        self.save_every_epoch = save_every_epoch

        self.criterion = HybridMultiScaleLoss(rho=0.5)
        self.scaler_amp = GradScaler(enabled=self._is_cuda())

        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.eff_train_times = []
        self.eff_train_mems = []

        print(f"[INFO] Trainer initialized for {self.model.__class__.__name__}.")
        print(f"[INFO] Trainable parameters: {self.num_params / 1e6:.3f}M")

    def _cuda_device(self) -> torch.device:
        return self.device if isinstance(self.device, torch.device) else torch.device(self.device)

    def _is_cuda(self) -> bool:
        return torch.cuda.is_available() and self._cuda_device().type == "cuda"

    def _cuda_synchronize(self) -> None:
        if self._is_cuda():
            torch.cuda.synchronize(self._cuda_device())

    def _reset_peak_memory(self) -> None:
        if self._is_cuda():
            torch.cuda.reset_peak_memory_stats(self._cuda_device())

    def _peak_memory_mib(self) -> float:
        if self._is_cuda():
            return torch.cuda.max_memory_allocated(self._cuda_device()) / 1024 / 1024
        return 0.0

    def _checkpoint_path(self, filename: str) -> str:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        return os.path.join(self.checkpoint_dir, filename)

    def _print_training_efficiency_summary(self) -> None:
        if not self.eff_train_times:
            return

        avg_all = float(np.mean(self.eff_train_times))
        avg_wo_first = float(np.mean(self.eff_train_times[1:])) if len(self.eff_train_times) > 1 else avg_all
        max_mem = float(max(self.eff_train_mems)) if self.eff_train_mems else 0.0

        print("\n" + "=" * 50)
        print("TRAINING EFFICIENCY SUMMARY")
        print("=" * 50)
        print(f"Params                 : {self.num_params / 1e6:.3f}M")
        print(f"Avg train time/epoch   : {avg_all:.2f}s")
        print(f"Avg time w/o 1st epoch : {avg_wo_first:.2f}s")
        print(f"Peak train memory      : {max_mem:.2f}MiB")
        print(f"Per-epoch train time   : {', '.join(f'{t:.2f}s' for t in self.eff_train_times)}")
        print("=" * 50 + "\n")

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batch = len(self.train_loader)

        for step, batch in enumerate(self.train_loader):
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)[..., :1]

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=self._is_cuda()):
                out = self.model(x)
                out = self.scaler.inverse_transform(out)
                loss = self.criterion(out, y, null_val=0.0)

            if torch.isnan(loss):
                print(f"[WARN] NaN loss at epoch {epoch}, step {step}; skipped.")
                continue

            self.scaler_amp.scale(loss).backward()
            self.scaler_amp.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler_amp.step(self.optimizer)
            self.scaler_amp.update()

            total_loss += loss.item()

            if self.logger is not None:
                global_step = epoch * num_batch + step
                self.logger.add_scalar("Step/Loss(train)", loss.item(), global_step)
                self.logger.add_scalar("Step/LR", self.optimizer.param_groups[0]["lr"], global_step)

        epoch_loss = total_loss / max(num_batch, 1)
        if self.logger is not None:
            self.logger.add_scalar("Epoch/Loss(train)", epoch_loss, epoch)
        return epoch_loss

    @torch.no_grad()
    def evaluate(self, epoch: int, split: str = "val") -> float:
        self.model.eval()
        total_loss = 0.0
        ys, outs = [], []

        loader = self.test_loader if split == "test" else self.val_loader
        num_batch = len(loader)

        for step, batch in enumerate(loader):
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)[..., :1]

            with torch.amp.autocast(device_type="cuda", enabled=self._is_cuda()):
                out = self.model(x)
                out = self.scaler.inverse_transform(out)
                loss = self.criterion(out, y, null_val=0.0)

            total_loss += loss.item()
            ys.append(y)
            outs.append(out)

            if self.logger is not None:
                self.logger.add_scalar(f"Step/Loss({split})", loss.item(), epoch * num_batch + step)

        ys_cat = torch.cat(ys, dim=0)
        outs_cat = torch.cat(outs, dim=0)
        mae = MAE(ys_cat.float().cpu().numpy(), outs_cat.float().cpu().numpy())

        epoch_loss = total_loss / max(num_batch, 1)
        if self.logger is not None:
            self.logger.add_scalar(f"Epoch/Loss({split})", epoch_loss, epoch)
            self.logger.add_scalar(f"Epoch/MAE({split})", mae, epoch)
        return epoch_loss

    def save_checkpoint(self, checkpoint: dict, filename: str) -> None:
        torch.save(checkpoint, self._checkpoint_path(filename))

    def load_checkpoint(self, filename: str):
        filepath = self._checkpoint_path(filename)
        if not os.path.exists(filepath):
            print(f"[WARN] Checkpoint not found: {filepath}.")
            return 0, float("inf")

        print(f"[INFO] Loading checkpoint: {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_amp") is not None:
            self.scaler_amp.load_state_dict(checkpoint["scaler_amp"])

        return int(checkpoint.get("epoch", 0)) + 1, float(checkpoint.get("loss", float("inf")))

    def train(
        self,
        epochs: int = 200,
        early_stopping: int = 20,
        verbose: bool = True,
        resume_from: Optional[str] = None,
    ) -> None:
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        start_epoch = 0

        if resume_from is not None:
            start_epoch, best_val_loss = self.load_checkpoint(resume_from)
            best_epoch = max(start_epoch - 1, 0)

        print("\n[INFO] Start training.")

        for epoch in range(start_epoch, epochs):
            self._reset_peak_memory()
            self._cuda_synchronize()
            train_start = time.perf_counter()

            train_loss = self.train_epoch(epoch)

            self._cuda_synchronize()
            train_time = time.perf_counter() - train_start
            train_memory = self._peak_memory_mib()

            self.eff_train_times.append(train_time)
            self.eff_train_mems.append(train_memory)

            val_loss = self.evaluate(epoch, "val")
            test_loss = self.evaluate(epoch, "test")

            if self.scheduler is not None:
                self.scheduler.step()

            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "scaler_amp": self.scaler_amp.state_dict(),
                "epoch": epoch,
                "loss": val_loss,
            }

            if self.save_every_epoch:
                self.save_checkpoint(checkpoint, f"epoch_{epoch:03d}.pth")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                self.save_checkpoint(checkpoint, self.checkpoint)
            else:
                patience_counter += 1

            if verbose:
                print(
                    f"Epoch {epoch:03d} | "
                    f"Train {train_loss:.4f} | "
                    f"Val {val_loss:.4f} | "
                    f"Test {test_loss:.4f} | "
                    f"Best {best_val_loss:.4f} @ {best_epoch:03d} | "
                    f"Time {train_time:.2f}s | "
                    f"Mem {train_memory:.2f}MiB"
                )

            if patience_counter >= early_stopping:
                print(f"[INFO] Early stopping at epoch {epoch}. Best validation loss: {best_val_loss:.4f}.")
                break

        self._print_training_efficiency_summary()

    @torch.no_grad()
    def test(self, loader):
        print("\n[INFO] Loading best checkpoint for testing.")
        self.load_checkpoint(self.checkpoint)
        self.model.eval()

        ys, outs = [], []
        num_batches = 0
        num_samples = 0
        infer_time = 0.0

        self._reset_peak_memory()

        for batch in loader:
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)[..., :1]

            num_batches += 1
            num_samples += x.shape[0]

            self._cuda_synchronize()
            infer_start = time.perf_counter()

            with torch.amp.autocast(device_type="cuda", enabled=self._is_cuda()):
                out = self.model(x)
                out = self.scaler.inverse_transform(out)

            self._cuda_synchronize()
            infer_time += time.perf_counter() - infer_start

            ys.append(y.detach().cpu())
            outs.append(out.detach().cpu())

        infer_memory = self._peak_memory_mib()
        latency_ms_per_batch = infer_time / max(num_batches, 1) * 1000
        throughput = num_samples / max(infer_time, 1e-12)

        ys = torch.cat(ys, dim=0).float().numpy()
        outs = torch.cat(outs, dim=0).float().numpy()

        print("\n" + "=" * 40)
        print("FINAL TEST RESULTS")
        print("=" * 40)
        print(f"Params              : {self.num_params / 1e6:.3f}M")
        print(f"Inference time      : {infer_time:.2f}s / full test set")
        print(f"Inference latency   : {latency_ms_per_batch:.3f}ms/batch")
        print(f"Throughput          : {throughput:.2f} samples/s")
        print(f"Peak infer memory   : {infer_memory:.2f}MiB")
        print("-" * 40)

        for i in range(ys.shape[1]):
            rmse, mae, mape = RMSE_MAE_MAPE(ys[:, i, :, :], outs[:, i, :, :])
            print(f"Horizon {i + 1:02d} | MAE {mae:.4f} | RMSE {rmse:.4f} | MAPE {mape:.4f}")

        print("-" * 40)
        for horizon in [12, 24, 36, 48]:
            idx = horizon - 1
            if ys.shape[1] > idx:
                rmse, mae, mape = RMSE_MAE_MAPE(ys[:, idx, :, :], outs[:, idx, :, :])
                print(f"Exact step {horizon:02d} | MAE {mae:.4f} | RMSE {rmse:.4f} | MAPE {mape:.4f}")

        rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(ys, outs)
        print("-" * 40)
        print(f"Global average | MAE {mae_all:.4f} | RMSE {rmse_all:.4f} | MAPE {mape_all:.4f}")
        print("=" * 40 + "\n")

        return ys, outs
