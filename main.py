import argparse
import os
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from prepare import PrepareDataset
from trainer import Trainer
from PriSM import PriSM, PriSMConfig


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


DATASET_CONFIGS = {
    "PEMS03": {"steps_per_day": 288, "causal_file": "PEMS03_pcmci_fast_L3_alpha0.01_k10.npz"},
    "PEMS04": {"steps_per_day": 288, "causal_file": "PEMS04_pcmci_fast_L3_alpha0.01_k10.npz"},
    "PEMS08": {"steps_per_day": 288, "causal_file": "PEMS08_pcmci_fast_L3_alpha0.01_k10.npz"},
    "HZMETRO": {"steps_per_day": 66, "causal_file": "HZMetro_pcmci_fast_L3_alpha0.01_k10.npz"},
    "SHMETRO": {"steps_per_day": 66, "causal_file": "SHMetro_pcmci_fast_L3_alpha0.01_k10.npz"},
    "LUGU": {"steps_per_day": 288, "causal_file": "Lugu_pcmci_fast_L2_alpha0.01_k10.npz"},
    "NYCTAXI": {"steps_per_day": 48, "causal_file": None},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train PriSM for traffic forecasting.")

    parser.add_argument("--dataset", "-dataset", type=str, default="HZMetro")
    parser.add_argument("--model", "-model", type=str, default="PriSM")
    parser.add_argument("--data_root", type=str, default="./datasets")
    parser.add_argument("--graph_path", type=str, default=None)
    parser.add_argument("--causal_path", type=str, default=None)

    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--log_tag", "-log_tag", type=str, default="train")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_every_epoch", action="store_true")

    parser.add_argument("--seq_len", "-seq_len", type=int, default=12)
    parser.add_argument("--pred_len", "-pred_len", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early_stopping", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", "-seed", type=int, default=42)

    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True


def dataset_key(dataset: str) -> str:
    upper = dataset.upper()
    if upper in DATASET_CONFIGS:
        return upper
    return dataset


def default_causal_path(dataset: str, data_root: str) -> Optional[str]:
    key = dataset_key(dataset)
    cfg = DATASET_CONFIGS.get(key, {"causal_file": None})
    causal_file = cfg.get("causal_file")
    if causal_file is None:
        return None
    return os.path.join(data_root, dataset, causal_file)


def build_causal_context_support(B_tau: Optional[torch.Tensor], eps: float = 1e-6):
    if B_tau is None:
        return None

    B_agg = (B_tau.max(dim=0).values > 0).float()
    B_agg.fill_diagonal_(0.0)

    A_c = B_agg.transpose(0, 1).contiguous()
    row_sum = A_c.sum(dim=1, keepdim=True)

    A_c = A_c / (row_sum + eps)
    A_c = torch.where(row_sum > 0, A_c, torch.zeros_like(A_c))
    return A_c


def load_B_tau_from_pcmci(npz_path: str, alpha: Optional[float] = None, max_lag: Optional[int] = None):
    if npz_path is None or not os.path.exists(npz_path):
        return None

    data = np.load(npz_path, allow_pickle=True)

    if "B_tau" in data.files:
        B = (data["B_tau"].astype(np.float32) > 0).astype(np.float32)
        for k in range(B.shape[0]):
            np.fill_diagonal(B[k], 0.0)
        return torch.from_numpy(B)

    if "p_matrix" not in data.files:
        return None

    p_matrix = data["p_matrix"]
    file_max_lag = int(data["max_lag"]) if "max_lag" in data.files else p_matrix.shape[2] - 1
    max_lag = file_max_lag if max_lag is None else min(max_lag, file_max_lag)
    alpha = float(data["alpha_level"]) if alpha is None and "alpha_level" in data.files else (0.01 if alpha is None else alpha)

    B_list = []
    for tau in range(1, min(max_lag, p_matrix.shape[2] - 1) + 1):
        A = (p_matrix[:, :, tau] < alpha).astype(np.float32)
        np.fill_diagonal(A, 0.0)
        B_list.append(A)

    if not B_list:
        return None

    return torch.from_numpy(np.stack(B_list, axis=0).astype(np.float32))


def main():
    args = parse_args()
    seed_everything(args.seed)

    dataset_upper = args.dataset.upper()
    if dataset_upper in {"HZMETRO", "SHMETRO"}:
        args.seq_len = 4

    train_loader, val_loader, test_loader, graph_data, scaler = PrepareDataset(
        dataset=args.dataset,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        train_propotion=0.6,
        valid_propotion=0.2,
        data_root=args.data_root,
        graph_path=args.graph_path,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sample_x = next(iter(train_loader))[0]
    print(f"\n[INFO] Device: {device} | input_dim={sample_x.shape[-1]}")

    if args.model != "PriSM":
        raise ValueError(f"Unsupported model: {args.model}")

    cfg_key = dataset_key(args.dataset)
    cfg = DATASET_CONFIGS.get(cfg_key, {"steps_per_day": 288, "causal_file": None})
    steps_per_day = cfg["steps_per_day"]

    causal_path = args.causal_path or default_causal_path(args.dataset, args.data_root)

    B_tau = None
    if causal_path is not None and os.path.exists(causal_path):
        try:
            B_tau = load_B_tau_from_pcmci(causal_path)
            if B_tau is not None:
                B_tau = B_tau.float().to(device)
                print(f"[INFO] Loaded lag-aware causal prior: {causal_path}")
            else:
                print(f"[WARN] No valid B_tau/p_matrix found in: {causal_path}")
        except Exception as exc:
            print(f"[WARN] Failed to load causal prior from {causal_path}: {exc}")
    else:
        print(f"[WARN] Causal prior file not found: {causal_path}")

    if graph_data.adj is None:
        raise ValueError(f"Physical adjacency is missing for {args.dataset}.")

    physical_adj = torch.as_tensor(graph_data.adj, dtype=torch.float32, device=device)
    physical_adj.fill_diagonal_(0.0)

    A_c = build_causal_context_support(B_tau).to(device) if B_tau is not None else None

    config = PriSMConfig(
        num_nodes=graph_data.num_nodes,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        in_dim=24,
        d_model=72,
        n_layers=1,
        ada_emb_d=72,
        steps_per_day=steps_per_day,
        tod_emb_d=24,
        days_per_week=7,
        dow_emb_d=24,
        cau_emb_d=8,
        deg_emb_d=4,
        d_state=4,
        d_conv=3,
        dropout=0.1,
        adj=physical_adj,
        B_tau=B_tau,
        A_c=A_c,
    )

    model = PriSM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 60], gamma=0.5)

    log_dir = os.path.join(args.log_dir, args.dataset, f"{args.model}_{args.log_tag}")
    logger = SummaryWriter(log_dir)

    trainer = Trainer(
        model=model,
        scaler=scaler,
        loaders=(train_loader, val_loader, test_loader),
        optimizer=optimizer,
        logger=logger,
        scheduler=scheduler,
        device=device,
        checkpoint=f"{args.model}_{args.dataset}.pth",
        checkpoint_dir=args.checkpoint_dir,
        save_every_epoch=args.save_every_epoch,
    )

    trainer.train(epochs=args.epochs, early_stopping=args.early_stopping)
    trainer.test(test_loader)


if __name__ == "__main__":
    main()
