from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DATASET_HANDLE = "namandixit07/multi-system-chaotic-dynamics-dataset"
SUPPORTED_SUFFIXES = {".csv", ".tsv", ".parquet", ".npy", ".npz", ".mat"}


@dataclass
class PreparedData:
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    scaler: StandardScaler
    state_cols: list[str]
    group_col: str
    time_col: str
    frame: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/load the Kaggle chaotic dynamics dataset and train a GRU next-step forecaster."
    )
    parser.add_argument("--data-dir", type=Path, help="Dataset folder. If omitted, kagglehub downloads the dataset.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"), help="Directory for plots, metrics, predictions.")
    parser.add_argument("--file-glob", default="**/*", help="Glob used under --data-dir. Default: **/*")
    parser.add_argument("--system-col", help="Column identifying the system/trajectory. Auto-detected if omitted.")
    parser.add_argument("--time-col", help="Time/step column. Auto-detected if omitted.")
    parser.add_argument("--state-cols", help="Comma-separated state columns, e.g. x,y,z. Auto-detected if omitted.")
    parser.add_argument("--max-rows", type=int, default=500_000, help="Max rows loaded per tabular file.")
    parser.add_argument("--max-windows", type=int, default=200_000, help="Max sliding windows per split.")
    parser.add_argument("--seq-len", type=int, default=64, help="Input sequence length.")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon. 1 means next time step.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-train", action="store_true", help="Only load, summarize, and plot the dataset.")
    return parser.parse_args()


def resolve_data_dir(data_dir: Path | None) -> Path:
    if data_dir is not None:
        if not data_dir.exists():
            raise SystemExit(f"--data-dir does not exist: {data_dir}")
        return data_dir

    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "kagglehub is not installed. Run:\n"
            "  pip install kagglehub\n\n"
            "Or download the Kaggle dataset manually and pass:\n"
            "  python chaotic_forecast.py --data-dir /path/to/dataset"
        ) from exc

    print(f"[+] downloading Kaggle dataset: {DATASET_HANDLE}")
    return Path(kagglehub.dataset_download(DATASET_HANDLE))


def find_data_files(data_dir: Path, file_glob: str) -> list[Path]:
    files = [
        p
        for p in sorted(data_dir.glob(file_glob))
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES and not p.name.startswith(".")
    ]
    if not files:
        suffixes = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise SystemExit(f"No supported data files found under {data_dir}. Expected one of: {suffixes}")
    return files


def ndarray_to_frame(arr: np.ndarray, source: str, key: str | None = None) -> pd.DataFrame:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        df = pd.DataFrame({"value": arr})
    elif arr.ndim == 2:
        df = pd.DataFrame(arr, columns=[f"x{i}" for i in range(arr.shape[1])])
    elif arr.ndim == 3:
        traj, steps, dims = arr.shape
        flat = arr.reshape(traj * steps, dims)
        df = pd.DataFrame(flat, columns=[f"x{i}" for i in range(dims)])
        df.insert(0, "step", np.tile(np.arange(steps), traj))
        df.insert(0, "trajectory_id", np.repeat(np.arange(traj), steps))
    else:
        raise ValueError(f"Unsupported array shape {arr.shape} in {source}")

    df["source_file"] = source
    if key is not None:
        df["source_array"] = key
    return df


def read_one_file(path: Path, max_rows: int) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, nrows=max_rows)
    elif suffix == ".tsv":
        df = pd.read_csv(path, sep="\t", nrows=max_rows)
    elif suffix == ".parquet":
        df = pd.read_parquet(path).head(max_rows)
    elif suffix == ".npy":
        return ndarray_to_frame(np.load(path, allow_pickle=False), path.name)
    elif suffix == ".npz":
        loaded = np.load(path, allow_pickle=False)
        frames = []
        for key in loaded.files:
            arr = loaded[key]
            if np.issubdtype(arr.dtype, np.number) and arr.ndim in {1, 2, 3}:
                frames.append(ndarray_to_frame(arr, path.name, key))
        if not frames:
            raise ValueError(f"No numeric arrays found in {path}")
        return pd.concat(frames, ignore_index=True)
    elif suffix == ".mat":
        try:
            from scipy.io import loadmat
        except ImportError as exc:
            raise SystemExit("Reading .mat files requires scipy. Run: pip install scipy") from exc

        data = loadmat(path)
        frames = []
        for key, arr in data.items():
            if key.startswith("__"):
                continue
            arr = np.asarray(arr)
            if np.issubdtype(arr.dtype, np.number) and arr.ndim in {1, 2, 3}:
                frames.append(ndarray_to_frame(arr, path.name, key))
        if not frames:
            raise ValueError(f"No numeric arrays found in {path}")
        return pd.concat(frames, ignore_index=True)
    else:
        raise ValueError(f"Unsupported file: {path}")

    df["source_file"] = path.name
    return df


def load_dataset(files: Iterable[Path], max_rows: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            df = read_one_file(path, max_rows=max_rows)
        except Exception as exc:
            warnings.warn(f"Skipping {path.name}: {exc}")
            continue
        if len(df):
            frames.append(df)
            print(f"[+] loaded {path.name}: {df.shape[0]:,} rows x {df.shape[1]:,} cols")

    if not frames:
        raise SystemExit("No readable data files were loaded.")

    return pd.concat(frames, ignore_index=True, sort=False)


def first_matching_column(columns: list[str], names: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for name in names:
        if name in lowered:
            return lowered[name]
    return None


def parse_column_list(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def infer_columns(df: pd.DataFrame, args: argparse.Namespace) -> tuple[str, str, list[str]]:
    columns = list(df.columns)
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)

    system_col = args.system_col or first_matching_column(
        columns,
        [
            "trajectory_id",
            "trajectory",
            "traj",
            "system",
            "system_name",
            "attractor",
            "model",
            "series",
            "source_file",
        ],
    )
    if system_col is None:
        system_col = "__series"
        df[system_col] = "all"

    time_col = args.time_col or first_matching_column(
        columns,
        ["time", "t", "step", "timestamp", "frame", "index", "sample"],
    )
    if time_col is None:
        time_col = "__step"
        df[time_col] = df.groupby(system_col, sort=False).cumcount()

    state_cols = parse_column_list(args.state_cols)
    if state_cols is None:
        candidates = [c for c in numeric_cols if c != time_col]
        preferred = [
            c
            for c in candidates
            if re.fullmatch(r"(x|y|z|u|v|w|x\d+|y\d+|z\d+|state_?\d+|coord_?\d+)", c.lower())
        ]
        state_cols = preferred if len(preferred) >= 1 else candidates

    missing = [c for c in state_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing --state-cols in dataframe: {missing}")
    if not state_cols:
        raise SystemExit("Could not infer state columns. Pass --state-cols x,y,z")

    print(f"[+] group column: {system_col}")
    print(f"[+] time column : {time_col}")
    print(f"[+] state cols  : {', '.join(state_cols)}")
    return system_col, time_col, state_cols


def summarize(df: pd.DataFrame, state_cols: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = df[state_cols].describe().T
    summary["missing"] = df[state_cols].isna().sum()
    summary.to_csv(out_dir / "eda_summary.csv")
    print(f"[+] wrote {out_dir / 'eda_summary.csv'}")


def sorted_groups(df: pd.DataFrame, group_col: str, time_col: str) -> list[pd.DataFrame]:
    groups = []
    for _, group in df.groupby(group_col, sort=False):
        group = group.sort_values(time_col).reset_index(drop=True)
        groups.append(group)
    return groups


def split_train_test(
    groups: list[pd.DataFrame],
    train_frac: float,
    seq_len: int,
    horizon: int,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    train_groups: list[pd.DataFrame] = []
    test_groups: list[pd.DataFrame] = []
    min_len = seq_len + horizon + 2

    for group in groups:
        if len(group) < min_len * 2:
            continue
        cut = max(min_len, min(len(group) - min_len, int(len(group) * train_frac)))
        train_groups.append(group.iloc[:cut].copy())
        test_groups.append(group.iloc[cut - seq_len - horizon + 1 :].copy())

    if not train_groups or not test_groups:
        raise SystemExit(
            "Not enough rows to create train/test windows. Try smaller --seq-len or --horizon."
        )
    return train_groups, test_groups


def make_windows(
    groups: list[pd.DataFrame],
    state_cols: list[str],
    scaler: StandardScaler,
    seq_len: int,
    horizon: int,
    max_windows: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for group in groups:
        values = scaler.transform(group[state_cols].to_numpy(dtype=np.float32))
        last_start = len(values) - seq_len - horizon + 1
        for start in range(max(0, last_start)):
            target_at = start + seq_len + horizon - 1
            xs.append(values[start : start + seq_len])
            ys.append(values[target_at])

    if not xs:
        raise SystemExit("No sliding windows were created.")

    x = np.stack(xs).astype(np.float32)
    y = np.stack(ys).astype(np.float32)

    if len(x) > max_windows:
        idx = np.linspace(0, len(x) - 1, max_windows).astype(int)
        x = x[idx]
        y = y[idx]
    return x, y


def prepare_data(df: pd.DataFrame, args: argparse.Namespace) -> PreparedData:
    group_col, time_col, state_cols = infer_columns(df, args)
    df = df.dropna(subset=state_cols).copy()
    groups = sorted_groups(df, group_col, time_col)
    train_groups, test_groups = split_train_test(groups, args.train_frac, args.seq_len, args.horizon)

    scaler = StandardScaler()
    scaler.fit(pd.concat(train_groups, ignore_index=True)[state_cols].to_numpy(dtype=np.float32))

    train_x, train_y = make_windows(
        train_groups, state_cols, scaler, args.seq_len, args.horizon, args.max_windows
    )
    test_x, test_y = make_windows(
        test_groups, state_cols, scaler, args.seq_len, args.horizon, args.max_windows
    )

    print(f"[+] train windows: {train_x.shape}")
    print(f"[+] test windows : {test_x.shape}")

    return PreparedData(
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        scaler=scaler,
        state_cols=state_cols,
        group_col=group_col,
        time_col=time_col,
        frame=df,
    )


def plot_phase(df: pd.DataFrame, data: PreparedData, out_dir: Path, max_points: int = 8_000) -> None:
    group = next(iter(sorted_groups(df, data.group_col, data.time_col)))
    group = group.iloc[:max_points]
    cols = data.state_cols[:3]

    fig = plt.figure(figsize=(8, 6))
    if len(cols) >= 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(group[cols[0]], group[cols[1]], group[cols[2]], linewidth=0.7)
        ax.set_xlabel(cols[0])
        ax.set_ylabel(cols[1])
        ax.set_zlabel(cols[2])
    elif len(cols) == 2:
        ax = fig.add_subplot(111)
        ax.plot(group[cols[0]], group[cols[1]], linewidth=0.7)
        ax.set_xlabel(cols[0])
        ax.set_ylabel(cols[1])
    else:
        ax = fig.add_subplot(111)
        ax.plot(group[data.time_col], group[cols[0]], linewidth=0.7)
        ax.set_xlabel(data.time_col)
        ax.set_ylabel(cols[0])
    ax.set_title("Chaotic trajectory preview")
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_preview.png", dpi=180)
    plt.close(fig)


def train_gru(data: PreparedData, args: argparse.Namespace, out_dir: Path) -> tuple[np.ndarray, list[float]]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit("Training requires PyTorch. Run: pip install torch") from exc

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[+] device: {device}")

    class GRUForecaster(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, num_layers: int):
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.1 if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, input_size),
            )

        def forward(self, x):
            out, _ = self.gru(x)
            return self.head(out[:, -1, :])

    train_ds = TensorDataset(torch.from_numpy(data.train_x), torch.from_numpy(data.train_y))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = GRUForecaster(
        input_size=len(data.state_cols),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    losses: list[float] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(xb)
            seen += len(xb)
        epoch_loss = total / max(seen, 1)
        losses.append(epoch_loss)
        print(f"[+] epoch {epoch:03d}/{args.epochs} mse={epoch_loss:.6f}")

    model.eval()
    preds = []
    test_tensor = torch.from_numpy(data.test_x)
    with torch.no_grad():
        for start in range(0, len(test_tensor), args.batch_size):
            xb = test_tensor[start : start + args.batch_size].to(device)
            preds.append(model(xb).cpu().numpy())
    pred_scaled = np.concatenate(preds, axis=0)

    torch.save(
        {
            "model_state": model.state_dict(),
            "state_cols": data.state_cols,
            "seq_len": args.seq_len,
            "horizon": args.horizon,
            "scaler_mean": data.scaler.mean_,
            "scaler_scale": data.scaler.scale_,
        },
        out_dir / "gru_forecaster.pt",
    )
    return pred_scaled, losses


def write_metrics_and_plots(
    data: PreparedData,
    pred_scaled: np.ndarray,
    losses: list[float],
    out_dir: Path,
) -> None:
    y_true = data.scaler.inverse_transform(data.test_y)
    y_pred = data.scaler.inverse_transform(pred_scaled)
    errors = y_pred - y_true
    per_col_rmse = {
        col: float(math.sqrt(np.mean(errors[:, i] ** 2))) for i, col in enumerate(data.state_cols)
    }
    overall_rmse = float(math.sqrt(np.mean(errors**2)))

    metrics = {
        "overall_rmse": overall_rmse,
        "per_column_rmse": per_col_rmse,
        "state_cols": data.state_cols,
        "train_windows": int(len(data.train_x)),
        "test_windows": int(len(data.test_x)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pred_df = pd.DataFrame(y_true, columns=[f"true_{c}" for c in data.state_cols])
    for i, col in enumerate(data.state_cols):
        pred_df[f"pred_{col}"] = y_pred[:, i]
    pred_df.head(5_000).to_csv(out_dir / "predictions_head.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses)
    ax.set_title("Training loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    fig.tight_layout()
    fig.savefig(out_dir / "training_loss.png", dpi=180)
    plt.close(fig)

    first = data.state_cols[0]
    n = min(500, len(y_true))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_true[:n, 0], label=f"true {first}", linewidth=1.2)
    ax.plot(y_pred[:n, 0], label=f"pred {first}", linewidth=1.2)
    ax.set_title(f"Forecast preview: {first}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "forecast_preview.png", dpi=180)
    plt.close(fig)

    print(f"[+] overall RMSE: {overall_rmse:.6f}")
    print(f"[+] wrote {out_dir / 'metrics.json'}")
    print(f"[+] wrote {out_dir / 'predictions_head.csv'}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = resolve_data_dir(args.data_dir)
    print(f"[+] dataset dir: {data_dir}")

    files = find_data_files(data_dir, args.file_glob)
    print("[+] candidate files:")
    for path in files[:20]:
        print(f"    - {path.relative_to(data_dir)}")
    if len(files) > 20:
        print(f"    ... {len(files) - 20} more")

    df = load_dataset(files, max_rows=args.max_rows)
    prepared = prepare_data(df, args)
    summarize(prepared.frame, prepared.state_cols, args.out_dir)
    plot_phase(prepared.frame, prepared, args.out_dir)

    if args.no_train:
        print("[+] --no-train set; stopping after EDA.")
        return

    pred_scaled, losses = train_gru(prepared, args, args.out_dir)
    write_metrics_and_plots(prepared, pred_scaled, losses, args.out_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")
