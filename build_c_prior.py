import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI


def load_flow(dataset_dir: str, dataset: str) -> np.ndarray:
    npz_path = os.path.join(dataset_dir, f"{dataset}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Dataset file not found: {npz_path}")

    data_file = np.load(npz_path)
    key = "data" if "data" in data_file.files else data_file.files[0]
    data = data_file[key].astype(np.float32)

    if data.ndim != 3 or data.shape[2] < 1:
        raise ValueError(f"Expected data shape [T, N, C>=1], got {data.shape}.")

    flow = data[..., 0]
    print(f"[INFO] Loaded flow data: {flow.shape}")
    return flow


def load_topk_neighbors_from_csv(csv_path: str, num_nodes: int, topk: int = 10):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Physical graph CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if not {"from", "to", "cost"}.issubset(df.columns):
        df = pd.read_csv(csv_path, header=None)
        df.columns = ["from", "to", "cost"] + list(df.columns[3:])

    dist = np.full((num_nodes, num_nodes), np.inf, dtype=np.float32)
    for _, row in df.iterrows():
        i, j, d = int(row["from"]), int(row["to"]), float(row["cost"])
        if i < num_nodes and j < num_nodes:
            dist[i, j] = min(dist[i, j], d)
            dist[j, i] = min(dist[j, i], d)

    neighbors = {}
    for i in range(num_nodes):
        valid = np.isfinite(dist[i])
        valid[i] = False
        idx = np.where(valid)[0]
        if idx.size == 0:
            neighbors[i] = []
            continue

        idx = idx[np.argsort(dist[i, idx])]
        neighbors[i] = idx[:topk].tolist() if topk is not None else idx.tolist()

    print(f"[INFO] Built top-{topk} physical neighbors for {num_nodes} nodes.")
    return neighbors


def build_link_assumptions(num_nodes: int, neighbors: dict, max_lag: int):
    link_assumptions = {}
    for target in range(num_nodes):
        assumptions = {}
        for tau in range(1, max_lag + 1):
            assumptions[(target, -tau)] = "-?>"
            for source in neighbors.get(target, []):
                assumptions[(source, -tau)] = "-?>"
        link_assumptions[target] = assumptions
    return link_assumptions


def build_pcmci_causal_graph(
    flow: np.ndarray,
    csv_path: str,
    max_lag: int = 3,
    alpha_level: float = 0.01,
    pc_alpha: float = 0.05,
    selected_nodes: Optional[list[int]] = None,
    downsample_step: int = 1,
    topk_neighbors: int = 10,
    verbosity: int = 1,
):
    total_steps, total_nodes = flow.shape

    if selected_nodes is not None:
        node_indices = np.array(selected_nodes, dtype=int)
        flow_used = flow[:, node_indices]
        print(f"[INFO] Using node subset: {len(node_indices)} / {total_nodes}")
    else:
        node_indices = np.arange(total_nodes)
        flow_used = flow

    if downsample_step > 1:
        flow_used = flow_used[::downsample_step]
        print(f"[INFO] Downsampled time axis by {downsample_step}: {total_steps} -> {flow_used.shape[0]}")

    steps, num_nodes = flow_used.shape
    print(f"[INFO] Running PCMCI on shape [T={steps}, N={num_nodes}], max_lag={max_lag}")

    neighbors_full = load_topk_neighbors_from_csv(csv_path, num_nodes=total_nodes, topk=topk_neighbors)
    index_map = {orig: idx for idx, orig in enumerate(node_indices)}

    neighbors_sub = {}
    for orig_target in node_indices:
        target = index_map[orig_target]
        neighbors_sub[target] = [
            index_map[source] for source in neighbors_full.get(orig_target, []) if source in index_map
        ]

    flow_std = StandardScaler().fit_transform(flow_used)
    dataframe = pp.DataFrame(flow_std)

    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=ParCorr(significance="analytic"),
        verbosity=verbosity,
    )

    link_assumptions = build_link_assumptions(num_nodes, neighbors_sub, max_lag)

    results = pcmci.run_pcmci(
        tau_min=1,
        tau_max=max_lag,
        pc_alpha=pc_alpha,
        link_assumptions=link_assumptions,
    )

    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]

    B_tau = []
    adj_causal = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for tau in range(1, max_lag + 1):
        A_tau = (p_matrix[:, :, tau] < alpha_level).astype(np.float32)
        np.fill_diagonal(A_tau, 0.0)
        B_tau.append(A_tau)
        adj_causal = np.maximum(adj_causal, A_tau)

    B_tau = np.stack(B_tau, axis=0).astype(np.float32)

    print(f"[INFO] Directed causal edges: {int(adj_causal.sum())}")
    return adj_causal, B_tau, p_matrix, val_matrix, node_indices


def parse_selected_nodes(value: Optional[str]):
    if value is None or value.strip() == "":
        return None
    return [int(x) for x in value.split(",")]


def parse_args():
    parser = argparse.ArgumentParser(description="Build lag-aware PCMCI causal prior.")
    parser.add_argument("--dataset", type=str, default="Lugu")
    parser.add_argument("--data_root", type=str, default="./datasets")
    parser.add_argument("--graph_csv", type=str, default=None)
    parser.add_argument("--max_lag", type=int, default=2)
    parser.add_argument("--alpha_level", type=float, default=0.01)
    parser.add_argument("--pc_alpha", type=float, default=0.05)
    parser.add_argument("--topk_neighbors", type=int, default=10)
    parser.add_argument("--downsample_step", type=int, default=1)
    parser.add_argument("--selected_nodes", type=str, default=None)
    parser.add_argument("--verbosity", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_dir = os.path.join(args.data_root, args.dataset)
    graph_csv = args.graph_csv or os.path.join(dataset_dir, f"{args.dataset}.csv")
    selected_nodes = parse_selected_nodes(args.selected_nodes)

    flow = load_flow(dataset_dir, args.dataset)

    adj_causal, B_tau, p_matrix, val_matrix, node_indices = build_pcmci_causal_graph(
        flow=flow,
        csv_path=graph_csv,
        max_lag=args.max_lag,
        alpha_level=args.alpha_level,
        pc_alpha=args.pc_alpha,
        selected_nodes=selected_nodes,
        downsample_step=args.downsample_step,
        topk_neighbors=args.topk_neighbors,
        verbosity=args.verbosity,
    )

    out_name = (
        f"{args.dataset}_pcmci_fast_L{args.max_lag}_"
        f"alpha{args.alpha_level}_k{args.topk_neighbors}.npz"
    )
    out_path = os.path.join(dataset_dir, out_name)

    np.savez(
        out_path,
        adj=adj_causal,
        B_tau=B_tau,
        p_matrix=p_matrix,
        val_matrix=val_matrix,
        node_indices=node_indices,
        max_lag=args.max_lag,
        alpha_level=args.alpha_level,
        pc_alpha=args.pc_alpha,
        topk_neighbors=args.topk_neighbors,
        downsample_step=args.downsample_step,
    )

    print(f"[INFO] Causal prior saved to: {out_path}")


if __name__ == "__main__":
    main()
