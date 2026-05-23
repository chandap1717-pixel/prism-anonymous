import os
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.utils.data as utils


@dataclass
class GraphData:
    num_nodes: int
    adj: torch.Tensor
    L: torch.Tensor
    eigenvectors: torch.Tensor
    eigenvalues: torch.Tensor


class StandardScaler:
    """Standardize selected leading feature dimensions."""

    def __init__(self, mean=None, std=None, fit_dim: int = 1):
        self.mean = mean
        self.std = std
        self.fit_dim = fit_dim
        self.mean_t = None
        self.std_t = None

    def fit(self, data: np.ndarray) -> None:
        target = data[..., : self.fit_dim]
        self.mean = target.mean()
        self.std = target.std()
        print(f"[Scaler] fit_dim={self.fit_dim}, mean={self.mean:.4f}, std={self.std:.4f}")

    def transform(self, data):
        if isinstance(data, torch.Tensor):
            mean, std = self._torch_stats(data)
            out = data.clone()
            if out.shape[-1] >= self.fit_dim:
                out[..., : self.fit_dim] = (out[..., : self.fit_dim] - mean) / (std + 1e-6)
                return out
            return (data - mean) / (std + 1e-6)

        out = data.copy()
        if out.shape[-1] >= self.fit_dim:
            out[..., : self.fit_dim] = (out[..., : self.fit_dim] - self.mean) / (self.std + 1e-6)
            return out
        return (out - self.mean) / (self.std + 1e-6)

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor):
            mean, std = self._torch_stats(data)
            out = data.clone()
            if out.shape[-1] >= self.fit_dim:
                out[..., : self.fit_dim] = out[..., : self.fit_dim] * std + mean
                return out
            return data * std + mean

        out = data.copy()
        if out.shape[-1] >= self.fit_dim:
            out[..., : self.fit_dim] = out[..., : self.fit_dim] * self.std + self.mean
            return out
        return out * self.std + self.mean

    def _torch_stats(self, data: torch.Tensor):
        if self.mean_t is None or self.mean_t.device != data.device or self.mean_t.dtype != data.dtype:
            self.mean_t = torch.tensor(self.mean, device=data.device, dtype=data.dtype)
            self.std_t = torch.tensor(self.std, device=data.device, dtype=data.dtype)
        return self.mean_t, self.std_t


def _as_numpy_adj(adj, num_nodes: int) -> np.ndarray:
    if isinstance(adj, torch.Tensor):
        adj = adj.detach().cpu().numpy()
    elif hasattr(adj, "toarray"):
        adj = adj.toarray()
    else:
        adj = np.asarray(adj)

    if adj.shape != (num_nodes, num_nodes):
        raise ValueError(f"Graph shape mismatch: expected {(num_nodes, num_nodes)}, got {adj.shape}.")
    return adj.astype(np.float32)


def load_pickle_graph(path: str, num_nodes: int) -> np.ndarray:
    with open(path, "rb") as f:
        adj = pickle.load(f)
    return _as_numpy_adj(adj, num_nodes)


def load_distance_matrix_to_adj(path: Optional[str], num_nodes: int) -> np.ndarray:
    if path is None or not os.path.exists(path):
        print("[WARN] Physical graph file not found; using identity support.")
        return np.eye(num_nodes, dtype=np.float32)

    if path.endswith(".pkl"):
        try:
            print(f"[INFO] Loading physical graph from {path}")
            return load_pickle_graph(path, num_nodes)
        except Exception as exc:
            print(f"[WARN] Failed to load graph pickle ({exc}); using identity support.")
            return np.eye(num_nodes, dtype=np.float32)

    try:
        df = pd.read_csv(path, header=None)
        if df.shape[1] < 3:
            df = pd.read_csv(path)
        df.columns = ["from", "to", "cost"] + list(df.columns[3:])

        dist = np.full((num_nodes, num_nodes), np.inf, dtype=np.float32)
        for _, row in df.iterrows():
            i, j = int(row["from"]), int(row["to"])
            if i < num_nodes and j < num_nodes:
                dist[i, j] = dist[j, i] = float(row["cost"])

        np.fill_diagonal(dist, 0.0)
        valid = dist[np.isfinite(dist)]
        std = valid.std() if valid.size > 0 else 1.0
        adj = np.exp(-np.square(dist / (std + 1e-6)))
        adj[~np.isfinite(adj)] = 0.0
        adj[adj < 0.1] = 0.0
        return adj.astype(np.float32)
    except Exception as exc:
        print(f"[WARN] Failed to build graph from {path} ({exc}); using identity support.")
        return np.eye(num_nodes, dtype=np.float32)


def _default_graph_path(dataset_dir: str, dataset: str) -> Optional[str]:
    dataset_upper = dataset.upper()
    if dataset_upper == "HZMETRO":
        return os.path.join(dataset_dir, "graph_hz_conn.pkl")
    if dataset_upper == "SHMETRO":
        return os.path.join(dataset_dir, "graph_sh_conn.pkl")

    csv_path = os.path.join(dataset_dir, f"{dataset}.csv")
    return csv_path if os.path.exists(csv_path) else None


def _add_time_features(x: np.ndarray, offset: int, steps_per_day: int) -> np.ndarray:
    num_samples, seq_len, num_nodes, _ = x.shape
    base_idx = np.arange(num_samples) + offset
    abs_time = base_idx[:, None] + np.arange(seq_len)[None, :]

    tod = (abs_time % steps_per_day) / float(steps_per_day)
    tod = np.tile(tod[:, :, None, None], (1, 1, num_nodes, 1)).astype(np.float32)

    dow = ((abs_time // steps_per_day) % 7) / 7.0
    dow = np.tile(dow[:, :, None, None], (1, 1, num_nodes, 1)).astype(np.float32)

    return np.concatenate([x.astype(np.float32), tod, dow], axis=-1)


def PrepareDataset(
    dataset: str = "HZMetro",
    batch_size: int = 64,
    seq_len: int = 12,
    pred_len: int = 12,
    train_propotion: float = 0.7,
    valid_propotion: float = 0.1,
    data_root: str = "./datasets",
    graph_path: Optional[str] = None,
    num_workers: int = 4,
    pin_memory: Optional[bool] = None,
):
    dataset_dir = os.path.join(data_root, dataset)
    dataset_upper = dataset.upper()

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if dataset_upper in {"HZMETRO", "SHMETRO"}:
        print(f"\n[INFO] {dataset_upper}: pre-split PKL mode.")

        steps_per_day = 66

        def load_pkl(split: str, offset: int):
            fpath = os.path.join(dataset_dir, f"{split}.pkl")
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Missing data split: {fpath}")

            with open(fpath, "rb") as f:
                data = pickle.load(f)

            x, y = data["x"].astype(np.float32), data["y"].astype(np.float32)
            x = _add_time_features(x, offset=offset, steps_per_day=steps_per_day)
            return torch.from_numpy(x), torch.from_numpy(y), x.shape[0]

        x_tr, y_tr, n_train = load_pkl("train", 0)
        x_val, y_val, n_val = load_pkl("val", n_train)
        x_te, y_te, _ = load_pkl("test", n_train + n_val)

        scaler = StandardScaler(fit_dim=2)
        scaler.fit(x_tr.numpy())

        x_tr = torch.from_numpy(scaler.transform(x_tr.numpy())).float()
        x_val = torch.from_numpy(scaler.transform(x_val.numpy())).float()
        x_te = torch.from_numpy(scaler.transform(x_te.numpy())).float()

        num_nodes = x_tr.shape[2]
        graph_file = graph_path or _default_graph_path(dataset_dir, dataset)
        adj_np = load_distance_matrix_to_adj(graph_file, num_nodes)

    elif dataset_upper == "NYCTAXI":
        print(f"\n[INFO] {dataset_upper}: pre-split NPZ mode.")

        def load_npz(split: str):
            fpath = os.path.join(dataset_dir, f"{split}.npz")
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Missing data split: {fpath}")
            data = np.load(fpath)
            return data["x"].astype(np.float32), data["y"].astype(np.float32)

        x_tr_np, y_tr_np = load_npz("train")
        x_val_np, y_val_np = load_npz("val")
        x_te_np, y_te_np = load_npz("test")

        num_nodes = x_tr_np.shape[2]
        scaler = StandardScaler(fit_dim=2)
        scaler.fit(x_tr_np)

        x_tr = torch.from_numpy(scaler.transform(x_tr_np)).float()
        x_val = torch.from_numpy(scaler.transform(x_val_np)).float()
        x_te = torch.from_numpy(scaler.transform(x_te_np)).float()

        y_tr = torch.from_numpy(y_tr_np).float()
        y_val = torch.from_numpy(y_val_np).float()
        y_te = torch.from_numpy(y_te_np).float()

        graph_file = graph_path or os.path.join(dataset_dir, "adj_mx.npz")
        if os.path.exists(graph_file) and graph_file.endswith(".npz"):
            try:
                adj_data = np.load(graph_file)
                key = "adj_mx" if "adj_mx" in adj_data.files else adj_data.files[0]
                adj_np = _as_numpy_adj(adj_data[key], num_nodes)
            except Exception as exc:
                print(f"[WARN] Failed to load {graph_file} ({exc}); using identity support.")
                adj_np = np.eye(num_nodes, dtype=np.float32)
        else:
            adj_np = np.eye(num_nodes, dtype=np.float32)

    else:
        print(f"\n[INFO] {dataset_upper}: NPZ sliding-window mode.")

        npz_path = os.path.join(dataset_dir, f"{dataset}.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Dataset file not found: {npz_path}")

        data_dict = np.load(npz_path)
        key = "data" if "data" in data_dict.files else data_dict.files[0]
        data = data_dict[key].astype(np.float32)

        num_nodes = data.shape[1]
        inputs, labels = [], []
        for t in range(data.shape[0] - seq_len - pred_len + 1):
            inputs.append(data[t : t + seq_len])
            labels.append(data[t + seq_len : t + seq_len + pred_len])
        inputs = np.asarray(inputs, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)

        if dataset_upper == "LUGU":
            current_train_prop = 0.7
            current_valid_prop = 0.1
        else:
            current_train_prop = train_propotion
            current_valid_prop = valid_propotion

        s1 = int(len(inputs) * current_train_prop)
        s2 = int(len(inputs) * (current_train_prop + current_valid_prop))
        print(f"[INFO] Split sizes: train={s1}, val={s2 - s1}, test={len(inputs) - s2}")

        scaler = StandardScaler(fit_dim=1)
        scaler.fit(inputs[:s1])
        inputs = scaler.transform(inputs)

        x_tr, y_tr = torch.tensor(inputs[:s1]), torch.tensor(labels[:s1])
        x_val, y_val = torch.tensor(inputs[s1:s2]), torch.tensor(labels[s1:s2])
        x_te, y_te = torch.tensor(inputs[s2:]), torch.tensor(labels[s2:])

        graph_file = graph_path or _default_graph_path(dataset_dir, dataset)
        adj_np = load_distance_matrix_to_adj(graph_file, num_nodes)

    train_loader = utils.DataLoader(utils.TensorDataset(x_tr, y_tr), shuffle=True, **loader_kwargs)
    valid_loader = utils.DataLoader(utils.TensorDataset(x_val, y_val), shuffle=False, **loader_kwargs)
    test_loader = utils.DataLoader(utils.TensorDataset(x_te, y_te), shuffle=False, **loader_kwargs)

    adj = torch.from_numpy(adj_np).float()
    adj_loop = adj + torch.eye(num_nodes)
    degree_inv_sqrt = torch.diag(torch.pow(adj_loop.sum(1), -0.5).nan_to_num(0.0))
    norm_lap = torch.eye(num_nodes) - degree_inv_sqrt @ adj_loop @ degree_inv_sqrt

    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(norm_lap)
    except RuntimeError:
        jitter = 1e-5 * torch.eye(num_nodes)
        eigenvalues, eigenvectors = torch.linalg.eigh(norm_lap + jitter)

    graph_data = GraphData(num_nodes, adj_loop, norm_lap, eigenvectors, eigenvalues)
    print("[INFO] Dataset preparation complete.")
    return train_loader, valid_loader, test_loader, graph_data, scaler
