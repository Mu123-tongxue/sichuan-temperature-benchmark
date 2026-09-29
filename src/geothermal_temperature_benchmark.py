import os
import time
import re
import sys
import subprocess
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from scipy.stats import pearsonr, spearmanr

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Optional XGBoost dependency
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# Optional PyTorch / PINN dependency
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# Optional SHAP dependency
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False


# ----------------------------
# Global parameters
# ----------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

LAM_BOUNDS = (0.60, 1.70)
A_BOUNDS = (0.60, 1.70)

N_INIT = 60
N_ITER = 25
N_CAND = 2000
TOP_K = 6

W_EXCEED = 1.0
REG_ALPHA = 0.02

OUTPUT_BOTTOM_MAPS = True
OUTPUT_TOP_MAPS = True
N_CONTOURS = 20

TARGET_DEPTHS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
DEFAULT_LAYER_COUNT = 14

# Measured-temperature cleaning parameters
DUP_TOL = 3.0
MAX_DOWNJUMP = 3.0

# PINN parameters
PINN_EPOCHS = 1200
PINN_BATCH_SIZE = 256
PINN_LR = 1e-3
PINN_DATA_W = 1.0
PINN_PHYS_W = 0.5
PINN_MONO_W = 0.2
PINN_BOUNDARY_W = 0.3
PINN_WD = 1e-5

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

if _HAS_TORCH:
    torch.manual_seed(RANDOM_SEED)


def safe_name(s: str) -> str:
    s = str(s)
    s = re.sub(r'[\\/:*?"<>|]', '_', s)
    s = s.strip()
    return s if s else "unnamed"


def ensure_dir(path: str):
    if not path:
        raise ValueError("Please select an output directory first.")
    os.makedirs(path, exist_ok=True)


def split_sample_indices(n_samples: int, evaluation_ratio: float = 0.2, random_seed: int = RANDOM_SEED):
    """Return the manuscript's deterministic sample-level training/evaluation split."""
    idx_all = np.arange(n_samples)
    return train_test_split(
        idx_all,
        test_size=evaluation_ratio,
        random_state=random_seed,
        shuffle=True,
    )


def open_folder_cross_platform(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory does not exist: {path}")
    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


if _HAS_TORCH:
    class PINNNet(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 256),
                nn.Tanh(),
                nn.Linear(256, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )

        def forward(self, x):
            return self.net(x)


class GeoThermalDoctor:
    def __init__(self, root):
        self.root = root
        self.root.title("Deep Formation Temperature Benchmark V9.0 (PINN-XGBoost + SHAP)")
        self.root.geometry("1360x1020")

        self.q_path = tk.StringVar()
        self.m_path = tk.StringVar()
        self.target_dir = tk.StringVar(value=os.getcwd())

        self.layers = []
        self.setup_ui()

    # ============================
    # UI
    # ============================
    def setup_ui(self):
        path_frame = tk.LabelFrame(self.root, text="Output directory (figures and CSV results)", fg="red")
        path_frame.pack(fill='x', padx=15, pady=5)

        tk.Button(
            path_frame,
            text="Select output directory",
            width=15,
            command=self.choose_output_dir
        ).pack(side=tk.LEFT, padx=8, pady=5)

        tk.Entry(path_frame, textvariable=self.target_dir, width=132).pack(side=tk.LEFT, padx=8, pady=5)

        input_frame = tk.LabelFrame(self.root, text="Input data", pady=10)
        input_frame.pack(fill='x', padx=15)

        tk.Button(input_frame, text="Terrestrial heat flow (.dat)", width=15,
                  command=lambda: self.q_path.set(filedialog.askopenfilename(
                      title="Select terrestrial heat-flow file",
                      filetypes=[("DAT files", "*.dat"), ("All files", "*.*")]
                  ))).grid(row=0, column=0, padx=5)
        tk.Entry(input_frame, textvariable=self.q_path, width=110).grid(row=0, column=1)

        tk.Button(input_frame, text="Measured temperature (.csv)", width=15,
                  command=lambda: self.m_path.set(filedialog.askopenfilename(
                      title="Select measured-temperature CSV",
                      filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
                  ))).grid(row=1, column=0, padx=5)
        tk.Entry(input_frame, textvariable=self.m_path, width=110).grid(row=1, column=1)

        mid = tk.LabelFrame(self.root, text="Stratigraphic interface depths + initial thermophysical properties (14+ layers supported)")
        mid.pack(fill='both', expand=True, padx=15, pady=5)

        top_tools = tk.Frame(mid)
        top_tools.pack(fill='x', pady=4)

        tk.Button(top_tools, text="Add layer", bg="#2E86C1", fg="white",
                  command=self.add_layer_row).pack(side=tk.LEFT, padx=5)
        tk.Button(top_tools, text="Remove last layer", bg="#CB4335", fg="white",
                  command=self.remove_last_layer).pack(side=tk.LEFT, padx=5)

        self.layer_count_label = tk.Label(top_tools, text="Current layer count: 0")
        self.layer_count_label.pack(side=tk.LEFT, padx=15)

        self.canvas = tk.Canvas(mid)
        self.sb = ttk.Scrollbar(mid, orient="vertical", command=self.canvas.yview)
        self.sf = tk.Frame(self.canvas)
        self.sf.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.canvas.create_window((0, 0), window=self.sf, anchor="nw")
        self.canvas.configure(yscrollcommand=self.sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.sb.pack(side="right", fill="y")

        for _ in range(DEFAULT_LAYER_COUNT):
            self.add_layer_row()

        bot = tk.Frame(self.root, pady=10, bg="#ecf0f1")
        bot.pack(fill='x')

        tk.Label(bot, text="Default surface temperature Ts (°C):", bg="#ecf0f1").pack(side=tk.LEFT, padx=10)
        self.ts = tk.Entry(bot, width=7)
        self.ts.insert(0, "18.0")
        self.ts.pack(side=tk.LEFT)

        tk.Label(bot, text="Evaluation ratio:", bg="#ecf0f1").pack(side=tk.LEFT, padx=10)
        self.test_ratio = tk.Entry(bot, width=6)
        self.test_ratio.insert(0, "0.2")
        self.test_ratio.pack(side=tk.LEFT)

        tk.Label(bot, text="0-m temperature is used when available; otherwise the default Ts is used.", bg="#ecf0f1", fg="blue").pack(side=tk.LEFT, padx=10)
        tk.Label(bot, text=f"PINN: {'available' if _HAS_TORCH else 'not installed; skipped'}",
                 bg="#ecf0f1", fg=("green" if _HAS_TORCH else "red")).pack(side=tk.LEFT, padx=8)
        tk.Label(bot, text=f"XGBoost: {'available' if _HAS_XGB else 'not installed; skipped'}",
                 bg="#ecf0f1", fg=("green" if _HAS_XGB else "red")).pack(side=tk.LEFT, padx=8)
        tk.Label(bot, text=f"SHAP: {'available' if _HAS_SHAP else 'not installed; skipped'}",
                 bg="#ecf0f1", fg=("green" if _HAS_SHAP else "red")).pack(side=tk.LEFT, padx=8)

        self.progress_label = tk.Label(bot, text="Ready", bg="#ecf0f1")
        self.progress_label.pack(side=tk.LEFT, padx=15)

        self.pb = ttk.Progressbar(bot, orient="horizontal", length=250, mode="determinate")
        self.pb.pack(side=tk.LEFT)

        tk.Button(bot, text="Open output directory", bg="#5D6D7E", fg="white",
                  command=self.open_output_dir).pack(side=tk.RIGHT, padx=10)
        tk.Button(bot, text="Run full benchmark", bg="#1E8449", fg="white",
                  font=('Arial', 11, 'bold'), command=self.engine_start).pack(side=tk.RIGHT, padx=20)

    def choose_output_dir(self):
        chosen = filedialog.askdirectory(title="Select result directory")
        if chosen:
            self.target_dir.set(chosen)

    def open_output_dir(self):
        try:
            open_folder_cross_platform(self.target_dir.get())
        except Exception as e:
            messagebox.showwarning("Notice", str(e))

    def add_layer_row(self):
        idx = len(self.layers) + 1
        f = tk.Frame(self.sf, pady=2)
        f.pack(fill='x')

        lbl_idx = tk.Label(f, text=f"{idx:02d}.", width=4)
        lbl_idx.pack(side=tk.LEFT)

        name = tk.Entry(f, width=12)
        name.insert(0, f"Layer {idx}")
        name.pack(side=tk.LEFT, padx=2)

        p = tk.StringVar()
        tk.Entry(f, textvariable=p, width=40).pack(side=tk.LEFT)
        tk.Button(f, text="Select interface Z file",
                  command=lambda v=p: v.set(filedialog.askopenfilename(
                      title="Select stratigraphic-interface Z file",
                      filetypes=[("DAT files", "*.dat"), ("TXT files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
                  ))).pack(side=tk.LEFT, padx=3)

        l_val, a_val = tk.Entry(f, width=7), tk.Entry(f, width=7)
        l_val.insert(0, "2.45")
        a_val.insert(0, "2.31")
        tk.Label(f, text="λ:").pack(side=tk.LEFT)
        l_val.pack(side=tk.LEFT, padx=2)
        tk.Label(f, text="A:").pack(side=tk.LEFT)
        a_val.pack(side=tk.LEFT, padx=2)

        self.layers.append({
            'frame': f,
            'idx_label': lbl_idx,
            'name': name,
            'path': p,
            'lam': l_val,
            'a': a_val
        })
        self.refresh_layer_labels()

    def remove_last_layer(self):
        if len(self.layers) <= 1:
            messagebox.showwarning("Notice", "At least one layer must be retained.")
            return
        obj = self.layers.pop()
        obj['frame'].destroy()
        self.refresh_layer_labels()

    def refresh_layer_labels(self):
        for i, layer in enumerate(self.layers, start=1):
            layer['idx_label'].config(text=f"{i:02d}.")
        self.layer_count_label.config(text=f"Current layer count: {len(self.layers)}")

    def update_pb(self, val, text):
        self.pb['value'] = val
        self.progress_label.config(text=text)
        self.root.update()

    # ============================
    # Plotting and tabular-output helpers
    # ============================
    def save_scatter_plot_and_csv(self, real, pred, out_dir, base_name, title):
        df_plot = pd.DataFrame({
            'RealT': np.asarray(real, dtype=float),
            'PredT': np.asarray(pred, dtype=float),
            'Error': np.asarray(pred, dtype=float) - np.asarray(real, dtype=float),
            'AbsError': np.abs(np.asarray(pred, dtype=float) - np.asarray(real, dtype=float))
        })
        df_plot.to_csv(os.path.join(out_dir, f"{base_name}_data.csv"), index=False, encoding='utf-8')

        plt.figure(figsize=(6, 5))
        plt.scatter(real, pred, alpha=0.7, edgecolors='w')
        mn = min(np.min(real), np.min(pred))
        mx = max(np.max(real), np.max(pred))
        plt.plot([mn, mx], [mn, mx], 'r--')
        plt.title(title)
        plt.xlabel("Observed temperature (°C)")
        plt.ylabel("Predicted temperature (°C)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{base_name}.png"), dpi=300)
        plt.close()

    def save_dual_scatter_plot_and_csv(self, real_tr, pred_tr, real_te, pred_te, out_dir, base_name, title_left, title_right):
        pd.DataFrame({
            'RealT': np.asarray(real_tr, dtype=float),
            'PredT': np.asarray(pred_tr, dtype=float),
            'Error': np.asarray(pred_tr, dtype=float) - np.asarray(real_tr, dtype=float),
            'AbsError': np.abs(np.asarray(pred_tr, dtype=float) - np.asarray(real_tr, dtype=float))
        }).to_csv(os.path.join(out_dir, f"{base_name}_TRAIN_data.csv"), index=False, encoding='utf-8')

        pd.DataFrame({
            'RealT': np.asarray(real_te, dtype=float),
            'PredT': np.asarray(pred_te, dtype=float),
            'Error': np.asarray(pred_te, dtype=float) - np.asarray(real_te, dtype=float),
            'AbsError': np.abs(np.asarray(pred_te, dtype=float) - np.asarray(real_te, dtype=float))
        }).to_csv(os.path.join(out_dir, f"{base_name}_EVALUATION_data.csv"), index=False, encoding='utf-8')

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(real_tr, pred_tr, alpha=0.7, edgecolors='w')
        mn = min(np.min(real_tr), np.min(pred_tr))
        mx = max(np.max(real_tr), np.max(pred_tr))
        plt.plot([mn, mx], [mn, mx], 'r--')
        plt.title(title_left)
        plt.xlabel("Observed temperature (°C)")
        plt.ylabel("Predicted temperature (°C)")

        plt.subplot(1, 2, 2)
        plt.scatter(real_te, pred_te, alpha=0.7, edgecolors='w')
        mn = min(np.min(real_te), np.min(pred_te))
        mx = max(np.max(real_te), np.max(pred_te))
        plt.plot([mn, mx], [mn, mx], 'r--')
        plt.title(title_right)
        plt.xlabel("Observed temperature (°C)")
        plt.ylabel("Predicted temperature (°C)")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{base_name}.png"), dpi=300)
        plt.close()

    def save_curve_plot_and_csv(self, df_curve, x_col, y_cols, out_dir, base_name, title, xlabel, ylabel):
        df_curve.to_csv(os.path.join(out_dir, f"{base_name}_data.csv"), index=False, encoding='utf-8')
        plt.figure(figsize=(7.2, 4.5))
        for c in y_cols:
            plt.plot(df_curve[x_col], df_curve[c], label=c)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if len(y_cols) > 1:
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{base_name}.png"), dpi=300)
        plt.close()

    def save_map_plot_and_csv(self, x, y, z, out_dir, base_name, title):
        df_map = pd.DataFrame({
            'X': np.asarray(x, dtype=float),
            'Y': np.asarray(y, dtype=float),
            'Value': np.asarray(z, dtype=float)
        })
        df_map.to_csv(os.path.join(out_dir, f"{base_name}_data.csv"), index=False, encoding='utf-8')

        plt.figure(figsize=(8.2, 7))
        plt.tricontourf(x, y, z, N_CONTOURS, cmap='Spectral_r')
        plt.colorbar(label='Temperature (°C)')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{base_name}.png"), dpi=300)
        plt.close()

    def create_method_comparison_plots(self, summary_df, out_dir):
        metric_pairs = [
            ('Evaluation_RMSE', 'method_comparison_Evaluation_RMSE', 'Evaluation RMSE'),
            ('Evaluation_MAE', 'method_comparison_Evaluation_MAE', 'Evaluation MAE'),
            ('Evaluation_R2', 'method_comparison_Evaluation_R2', 'Evaluation R2'),
            ('Evaluation_Within5Ratio', 'method_comparison_Evaluation_Within5Ratio', 'Evaluation Within5Ratio'),
        ]
        usable = summary_df.copy()
        usable = usable[usable['Kind'].isin(['Physics-Base', 'Physics-Optimized', 'PINN', 'Hybrid'])].copy()

        for metric, base_name, ylabel in metric_pairs:
            if metric not in usable.columns:
                continue
            dfp = usable[['Method', metric]].dropna().copy()
            if len(dfp) == 0:
                continue
            dfp.to_csv(os.path.join(out_dir, f"{base_name}_data.csv"), index=False, encoding='utf-8')

            plt.figure(figsize=(10, 5))
            plt.bar(dfp['Method'], dfp[metric])
            plt.xticks(rotation=30, ha='right')
            plt.ylabel(ylabel)
            plt.title(base_name)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{base_name}.png"), dpi=300)
            plt.close()

    # ============================
    # Data input
    # ============================
    def read_xyz_dat(self, path, names=('X', 'Y', 'Z'), bad_export_name=None):
        df = pd.read_csv(path, sep=None, engine='python', header=None,
                         names=list(names), dtype=str)

        for c in names:
            df[c] = df[c].astype(str).str.strip().str.replace(',', '', regex=False)
            df[c] = pd.to_numeric(df[c], errors='coerce')

        bad = df[df[list(names)].isna().any(axis=1)]
        good = df.dropna(subset=list(names))

        if bad_export_name and len(bad) > 0:
            outp = os.path.join(self.target_dir.get(), bad_export_name)
            try:
                bad.to_csv(outp, index=False, encoding='utf-8')
            except Exception:
                bad.to_csv(outp, index=False, encoding='utf-8')

        return good

    def read_measure_csv(self, path):
        try:
            df = pd.read_csv(path, encoding='utf-8', dtype=str)
        except Exception:
            df = pd.read_csv(path, encoding='utf-8', dtype=str)

        colmap = {c.lower(): c for c in df.columns}
        required = ['x', 'y', 'depth', 'temp']
        for r in required:
            if r not in colmap:
                raise ValueError(f"Measured-temperature CSV is missing required column: {r} (required: X, Y, Depth, temp)")

        rename_dict = {
            colmap['x']: 'X',
            colmap['y']: 'Y',
            colmap['depth']: 'Depth',
            colmap['temp']: 'temp'
        }
        if 'wellid' in colmap:
            rename_dict[colmap['wellid']] = 'WellID'

        df = df.rename(columns=rename_dict)

        for c in ['X', 'Y', 'Depth', 'temp']:
            df[c] = df[c].astype(str).str.strip().str.replace(',', '', regex=False)
            df[c] = pd.to_numeric(df[c], errors='coerce')

        bad = df[df[['X', 'Y', 'Depth', 'temp']].isna().any(axis=1)]
        if len(bad) > 0:
            outp = os.path.join(self.target_dir.get(), "measured_temperature_invalid_rows_removed.csv")
            try:
                bad.to_csv(outp, index=False, encoding='utf-8')
            except Exception:
                bad.to_csv(outp, index=False, encoding='utf-8')

        df = df.dropna(subset=['X', 'Y', 'Depth', 'temp'])
        if len(df) == 0:
            raise ValueError("No measured-temperature data remain after cleaning.")

        if 'WellID' not in df.columns:
            df['WellID'] = df.apply(lambda r: f"WELL_{round(float(r['X']), 2)}_{round(float(r['Y']), 2)}", axis=1)

        return df

    def clean_measurements(self, df, default_ts=18.0, dup_tol=3.0, max_downjump=3.0):
        df = df.copy()

        merged_rows = []
        dup_bad_rows = []

        for (wid, dep), g in df.groupby(['WellID', 'Depth']):
            vals = g['temp'].values.astype(float)
            x0 = float(g['X'].iloc[0])
            y0 = float(g['Y'].iloc[0])

            if len(vals) == 1:
                final_temp = float(vals[0])
            else:
                med = float(np.median(vals))
                keep = vals[np.abs(vals - med) <= dup_tol]
                if len(keep) > 0:
                    final_temp = float(np.mean(keep))
                else:
                    final_temp = med

                for v in vals:
                    if abs(v - final_temp) > dup_tol:
                        dup_bad_rows.append({
                            'WellID': wid,
                            'X': x0,
                            'Y': y0,
                            'Depth': float(dep),
                            'RawTemp': float(v),
                            'MergedTemp': float(final_temp),
                            'Flag': 'Outlier among duplicate-depth observations'
                        })

            merged_rows.append({
                'WellID': wid,
                'X': x0,
                'Y': y0,
                'Depth': float(dep),
                'temp': float(final_temp)
            })

        df2 = pd.DataFrame(merged_rows)

        ts_map = {}
        for wid, g in df2.groupby('WellID'):
            g0 = g[np.isclose(g['Depth'].values, 0.0)]
            if len(g0) > 0:
                ts_map[wid] = float(g0['temp'].iloc[0])
            else:
                ts_map[wid] = float(default_ts)

        df2['Ts'] = df2['WellID'].map(ts_map)

        profile_bad_rows = []
        cleaned_rows = []

        for wid, g in df2.groupby('WellID'):
            g = g.sort_values('Depth').reset_index(drop=True)
            keep_flag = np.ones(len(g), dtype=bool)

            for i in range(1, len(g)):
                d1, t1 = float(g.loc[i - 1, 'Depth']), float(g.loc[i - 1, 'temp'])
                d2, t2 = float(g.loc[i, 'Depth']), float(g.loc[i, 'temp'])

                if d2 > d1 and (t2 - t1) < -max_downjump:
                    keep_flag[i] = False
                    profile_bad_rows.append({
                        'WellID': wid,
                        'X': float(g.loc[i, 'X']),
                        'Y': float(g.loc[i, 'Y']),
                        'Depth': d2,
                        'Temp': t2,
                        'PrevDepth': d1,
                        'PrevTemp': t1,
                        'Flag': 'Anomalous downhole temperature decrease'
                    })

            cleaned_rows.append(g.loc[keep_flag])

        df_clean = pd.concat(cleaned_rows, ignore_index=True)

        if len(dup_bad_rows) > 0:
            pd.DataFrame(dup_bad_rows).to_csv(
                os.path.join(self.target_dir.get(), "measured_temperature_duplicate_depth_outliers.csv"),
                index=False, encoding='utf-8'
            )

        if len(profile_bad_rows) > 0:
            pd.DataFrame(profile_bad_rows).to_csv(
                os.path.join(self.target_dir.get(), "measured_temperature_downhole_anomalies.csv"),
                index=False, encoding='utf-8'
            )

        df_clean.to_csv(
            os.path.join(self.target_dir.get(), "measured_temperature_cleaned.csv"),
            index=False, encoding='utf-8'
        )

        ts_df = pd.DataFrame({
            'WellID': list(ts_map.keys()),
            'Ts': list(ts_map.values()),
            'TsSource': ['0-m observation preferred; otherwise default Ts'] * len(ts_map)
        })
        ts_df.to_csv(
            os.path.join(self.target_dir.get(), "well_surface_temperature_Ts_summary.csv"),
            index=False, encoding='utf-8'
        )

        return df_clean, ts_map

    def spatial_join_z(self, grid_xy, layer_df, layer_name_for_debug="layer"):
        layer_xy = layer_df[['X', 'Y']].values
        if not np.isfinite(layer_xy).all():
            raise ValueError(f"{layer_name_for_debug}: stratigraphic-interface X/Y contains non-numeric or infinite values.")
        tree = KDTree(layer_xy)
        _, idx = tree.query(grid_xy)
        return layer_df['Z'].values[idx]

    # ============================
    # Physical model
    # ============================
    def physics_vec(self, q_arr, d_matrix, l_list, a_list, target_z, ts):
        q_arr = np.asarray(q_arr).astype(float)
        d_matrix = np.asarray(d_matrix).astype(float)
        target_z = np.asarray(target_z).astype(float)

        N = q_arr.shape[0]

        ts = np.asarray(ts, dtype=float)
        if ts.ndim == 0:
            temp = np.full(N, float(ts), dtype=float)
        else:
            temp = ts.astype(float).copy()

        q_z = q_arr * 1e-3
        last_z = np.zeros(N, dtype=float)

        d_abs = np.abs(d_matrix)
        d_abs = np.maximum.accumulate(d_abs, axis=1)

        for i in range(len(l_list)):
            layer_depth = d_abs[:, i]
            dz = np.clip(np.minimum(layer_depth, target_z) - last_z, 0, None)

            lam = float(l_list[i])
            A = float(a_list[i]) * 1e-6

            temp += (q_z / lam) * dz - (A / (2 * lam)) * (dz ** 2)
            q_z -= A * dz
            last_z = layer_depth

        lam_last = float(l_list[-1])
        dz_rest = np.clip(target_z - last_z, 0, None)
        temp += (q_z / lam_last) * dz_rest
        return temp

    # ============================
    # Evaluation metrics
    # ============================
    def calc_metrics(self, real, pred):
        real = np.asarray(real, dtype=float)
        pred = np.asarray(pred, dtype=float)
        err = pred - real

        mse = mean_squared_error(real, pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(real, pred))
        r2 = float(r2_score(real, pred))

        try:
            pr = float(pearsonr(real, pred)[0])
        except Exception:
            pr = np.nan
        try:
            sr = float(spearmanr(real, pred).correlation)
        except Exception:
            sr = np.nan

        within5 = float(np.mean(np.abs(err) <= 5.0))
        exceed = float(np.mean(np.clip(np.abs(err) - 5.0, 0, None)))

        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'PearsonR': pr,
            'SpearmanR': sr,
            'Within5Ratio': within5,
            'MeanExceedOver5': exceed
        }

    def build_objective(self, base_lams, base_As, q_train, d_train, z_train, t_train, ts_train):
        L = len(base_lams)

        def obj(theta):
            theta = np.asarray(theta, dtype=float)
            lam_mult = theta[:L]
            a_mult = theta[L:]

            lams = base_lams * lam_mult
            As = base_As * a_mult

            pred = self.physics_vec(q_train, d_train, lams, As, z_train, ts_train)
            metrics = self.calc_metrics(t_train, pred)

            reg = np.mean((np.log(lam_mult)) ** 2) + np.mean((np.log(a_mult)) ** 2)
            loss = metrics['RMSE'] + W_EXCEED * metrics['MeanExceedOver5'] + REG_ALPHA * reg
            return float(loss)

        return obj

    # ============================
    # Surrogate-assisted optimization
    # ============================
    def smbo_optimize(self, method_name, surrogate, obj_func, bounds,
                      n_init, n_iter, n_cand, top_k, out_dir):
        rng = np.random.default_rng(RANDOM_SEED)

        dim = len(bounds)
        low = np.array([b[0] for b in bounds], dtype=float)
        high = np.array([b[1] for b in bounds], dtype=float)

        def sample(n):
            return rng.random((n, dim)) * (high - low) + low

        X = sample(n_init)
        y = np.array([obj_func(x) for x in X], dtype=float)

        history = [{'eval': i + 1, 'best_loss': float(np.min(y[:i + 1]))} for i in range(len(y))]

        for _ in range(n_iter):
            surrogate.fit(X, y)

            C = sample(n_cand)
            y_hat = surrogate.predict(C)
            pick = np.argsort(y_hat)[:top_k]
            X_new = C[pick]
            y_new = np.array([obj_func(x) for x in X_new], dtype=float)

            X = np.vstack([X, X_new])
            y = np.concatenate([y, y_new])

            history.append({'eval': len(y), 'best_loss': float(np.min(y))})

        hist_df = pd.DataFrame(history)
        self.save_curve_plot_and_csv(
            hist_df, 'eval', ['best_loss'], out_dir,
            f"{safe_name(method_name)}_convergence",
            f"{method_name} layer-wise parameter-calibration convergence",
            "True forward-model evaluations", "Best objective value"
        )

        best_idx = int(np.argmin(y))
        return X[best_idx].copy(), float(y[best_idx]), hist_df

    # ============================
    # PINN
    # ============================
    def build_pinn_features(self, xy, q_arr, d_matrix, depth_arr, phys_temp, ts_arr):
        xy = np.asarray(xy, dtype=float)
        q_arr = np.asarray(q_arr, dtype=float).reshape(-1, 1)
        depth_arr = np.asarray(depth_arr, dtype=float).reshape(-1, 1)
        phys_temp = np.asarray(phys_temp, dtype=float).reshape(-1, 1)
        ts_arr = np.asarray(ts_arr, dtype=float).reshape(-1, 1)
        d_matrix = np.asarray(d_matrix, dtype=float)
        return np.hstack([xy, q_arr, depth_arr, phys_temp, ts_arr, d_matrix])

    def build_hybrid_features(self, xy, q_arr, d_matrix, depth_arr, phys_temp, ts_arr, pinn_temp):
        xy = np.asarray(xy, dtype=float)
        q_arr = np.asarray(q_arr, dtype=float).reshape(-1, 1)
        depth_arr = np.asarray(depth_arr, dtype=float).reshape(-1, 1)
        phys_temp = np.asarray(phys_temp, dtype=float).reshape(-1, 1)
        ts_arr = np.asarray(ts_arr, dtype=float).reshape(-1, 1)
        pinn_temp = np.asarray(pinn_temp, dtype=float).reshape(-1, 1)
        d_matrix = np.asarray(d_matrix, dtype=float)
        return np.hstack([xy, q_arr, depth_arr, phys_temp, ts_arr, pinn_temp, d_matrix])

    def standardize_train_test(self, X_tr, X_te):
        mu = X_tr.mean(axis=0, keepdims=True)
        sd = X_tr.std(axis=0, keepdims=True)
        sd[sd < 1e-12] = 1.0
        return (X_tr - mu) / sd, (X_te - mu) / sd, mu, sd

    def pinn_predict(self, model, X_raw, mu, sd, device):
        Xn = (X_raw - mu) / sd
        with torch.no_grad():
            xt = torch.tensor(Xn, dtype=torch.float32, device=device)
            pred = model(xt).cpu().numpy().reshape(-1)
        return pred

    def train_pinn(self, X_tr_raw, y_tr, X_te_raw, y_te, out_dir):
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is not installed; PINN cannot be run.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_tr, X_te, mu, sd = self.standardize_train_test(X_tr_raw, X_te_raw)
        y_tr = y_tr.reshape(-1, 1).astype(np.float32)
        y_te = y_te.reshape(-1, 1).astype(np.float32)

        ds = TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32)
        )
        dl = DataLoader(ds, batch_size=min(PINN_BATCH_SIZE, len(ds)), shuffle=True)

        model = PINNNet(X_tr.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=PINN_LR, weight_decay=PINN_WD)
        loss_mse = nn.MSELoss()

        mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
        sd_t = torch.tensor(sd, dtype=torch.float32, device=device)

        # Feature order: [X, Y, Q, Depth, PhysT0, Ts, Z1, Z2, ...]
        depth_col = 3
        phys_col = 4
        ts_col = 5

        Xte_tensor = torch.tensor(X_te, dtype=torch.float32, device=device)
        yte_tensor = torch.tensor(y_te, dtype=torch.float32, device=device)

        hist = []
        best_state = None
        best_val = np.inf

        for ep in range(PINN_EPOCHS):
            model.train()
            epoch_loss = 0.0

            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)
                xb.requires_grad_(True)
                optimizer.zero_grad()

                pred = model(xb)

                data_loss = loss_mse(pred, yb)

                phys_prior_raw = xb[:, phys_col:phys_col + 1] * sd_t[:, phys_col:phys_col + 1] + mu_t[:, phys_col:phys_col + 1]
                phys_loss = loss_mse(pred, phys_prior_raw)

                grad = torch.autograd.grad(
                    outputs=pred,
                    inputs=xb,
                    grad_outputs=torch.ones_like(pred),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0][:, depth_col:depth_col + 1]
                mono_loss = torch.mean(torch.relu(-grad))

                xb_surface = xb.clone()
                xb_surface[:, depth_col] = (0.0 - mu_t[0, depth_col]) / sd_t[0, depth_col]
                pred_surface = model(xb_surface)

                ts_raw = xb[:, ts_col:ts_col + 1] * sd_t[:, ts_col:ts_col + 1] + mu_t[:, ts_col:ts_col + 1]
                boundary_loss = loss_mse(pred_surface, ts_raw)

                loss = (PINN_DATA_W * data_loss +
                        PINN_PHYS_W * phys_loss +
                        PINN_MONO_W * mono_loss +
                        PINN_BOUNDARY_W * boundary_loss)

                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item()) * len(xb)

            epoch_loss /= len(ds)

            model.eval()
            with torch.no_grad():
                pred_te = model(Xte_tensor)
                val_loss = float(loss_mse(pred_te, yte_tensor).item())

            hist.append({
                'epoch': ep + 1,
                'train_loss': epoch_loss,
                'val_mse': val_loss
            })

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        hist_df = pd.DataFrame(hist)
        self.save_curve_plot_and_csv(
            hist_df, 'epoch', ['train_loss', 'val_mse'], out_dir,
            "PINN_training_curve", "PINN training curve", "Epoch", "Loss"
        )

        pred_tr = self.pinn_predict(model, X_tr_raw, mu, sd, device)
        pred_te = self.pinn_predict(model, X_te_raw, mu, sd, device)

        return model, mu, sd, pred_tr, pred_te, hist_df, device

    # ============================
    # SHAP
    # ============================
    def run_shap_analysis(self, model, X_df, out_dir, model_name="XGBoost", top_n=10, max_samples=2000):
        if not _HAS_SHAP:
            pd.DataFrame([{
                'Info': 'SHAP is not installed; SHAP analysis was skipped.'
            }]).to_csv(os.path.join(out_dir, f"{safe_name(model_name)}_SHAP_note.csv"),
                        index=False, encoding='utf-8')
            return

        if not isinstance(X_df, pd.DataFrame):
            X_df = pd.DataFrame(X_df)

        if len(X_df) > max_samples:
            X_use = X_df.sample(n=max_samples, random_state=RANDOM_SEED).reset_index(drop=True)
        else:
            X_use = X_df.reset_index(drop=True)

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_use)
        except Exception:
            explainer = shap.Explainer(model, X_use)
            shap_exp = explainer(X_use)
            shap_values = shap_exp.values

        shap_values = np.asarray(shap_values, dtype=float)

        shap_df = pd.DataFrame(shap_values, columns=X_use.columns)
        shap_df.to_csv(os.path.join(out_dir, f"{safe_name(model_name)}_SHAP_values.csv"),
                       index=False, encoding='utf-8')

        importance = np.mean(np.abs(shap_values), axis=0)
        imp_df = pd.DataFrame({
            'Feature': X_use.columns,
            'MeanAbsSHAP': importance
        }).sort_values('MeanAbsSHAP', ascending=False)
        imp_df.to_csv(os.path.join(out_dir, f"{safe_name(model_name)}_SHAP_feature_importance.csv"),
                      index=False, encoding='utf-8')

        plt.figure()
        shap.summary_plot(shap_values, X_use, show=False, max_display=top_n)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{safe_name(model_name)}_SHAP_summary_beeswarm.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure()
        shap.summary_plot(shap_values, X_use, plot_type="bar", show=False, max_display=top_n)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{safe_name(model_name)}_SHAP_summary_bar.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

        top_features = imp_df['Feature'].head(min(top_n, len(imp_df))).tolist()
        dep_rows = []
        for feat in top_features[:5]:
            dep_rows.append({'Feature': feat})
            plt.figure()
            shap.dependence_plot(feat, shap_values, X_use, show=False, interaction_index='auto')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{safe_name(model_name)}_SHAP_dependence_{safe_name(feat)}.png"),
                        dpi=300, bbox_inches='tight')
            plt.close()

        pd.DataFrame(dep_rows).to_csv(os.path.join(out_dir, f"{safe_name(model_name)}_SHAP_dependence_index.csv"),
                                      index=False, encoding='utf-8')


    # ============================
    # Main workflow
    # ============================
    def engine_start(self):
        try:
            ensure_dir(self.target_dir.get())
            start_time = time.time()
            self.update_pb(3, "Loading data...")

            best_hybrid_model = None
            best_pinn_model = None
            best_pinn_mu = None
            best_pinn_sd = None
            best_pinn_device = None

            if not self.q_path.get() or not self.m_path.get():
                raise ValueError("Please select both a terrestrial heat-flow file (.dat) and a measured-temperature file (.csv).")

            # 1) Heat flow
            df_q = self.read_xyz_dat(self.q_path.get(), names=('X', 'Y', 'Q'),
                                     bad_export_name="heat_flow_invalid_rows_removed.csv")
            if len(df_q) == 0:
                raise ValueError("No heat-flow data remain after cleaning.")

            grid = df_q.copy()
            grid_xy = grid[['X', 'Y']].values
            if not np.isfinite(grid_xy).all():
                raise ValueError("Heat-flow grid X/Y contains non-numeric or infinite values.")

            # 2) Measured temperature
            default_ts = float(self.ts.get())
            test_ratio = float(self.test_ratio.get())
            test_ratio = min(max(test_ratio, 0.05), 0.5)

            df_m_raw = self.read_measure_csv(self.m_path.get())
            df_m, ts_map = self.clean_measurements(
                df_m_raw,
                default_ts=default_ts,
                dup_tol=DUP_TOL,
                max_downjump=MAX_DOWNJUMP
            )

            # 3) Stratigraphic interfaces
            self.update_pb(12, "Reading and matching stratigraphic-interface depths (KDTree nearest neighbor)...")

            valid_configs = []
            for layer in self.layers:
                path = layer['path'].get().strip()
                if not path:
                    continue

                lname = layer['name'].get().strip()
                lam0 = float(layer['lam'].get())
                a0 = float(layer['a'].get())

                bad_name = f"interface_Z_{safe_name(lname)}_invalid_rows_removed.csv"
                ldf = self.read_xyz_dat(path, names=('X', 'Y', 'Z'), bad_export_name=bad_name)
                if len(ldf) == 0:
                    raise ValueError(f"No interface-Z data remain after cleaning for {lname}.")

                z_on_grid = self.spatial_join_z(grid_xy, ldf, layer_name_for_debug=lname)
                col = f"Z_{lname}"
                grid[col] = z_on_grid

                valid_configs.append({'name': lname, 'lam0': lam0, 'a0': a0, 'col': col})

            if len(valid_configs) < 2:
                raise ValueError("At least two valid stratigraphic-interface files are required.")

            d_matrix = grid[[c['col'] for c in valid_configs]].values
            q_arr = grid['Q'].values.astype(float)

            if not np.isfinite(d_matrix).all():
                raise ValueError("The stratigraphic-interface depth matrix contains non-numeric or infinite values.")
            if not np.isfinite(q_arr).all():
                raise ValueError("Heat-flow Q contains non-numeric or infinite values.")

            grid_tree = KDTree(grid_xy)

            # 4) Observation-to-grid matching
            self.update_pb(20, "Matching measured-temperature samples to grid nodes...")

            m_xy = df_m[['X', 'Y']].values
            _, m_idx = grid_tree.query(m_xy)

            q_m = q_arr[m_idx]
            d_m = d_matrix[m_idx, :]
            z_m = df_m['Depth'].values.astype(float)
            t_m = df_m['temp'].values.astype(float)
            ts_m = df_m['Ts'].values.astype(float)

            base_lams = np.array([c['lam0'] for c in valid_configs], dtype=float)
            base_As = np.array([c['a0'] for c in valid_configs], dtype=float)
            L = len(base_lams)

            t_phys0 = self.physics_vec(q_m, d_m, base_lams, base_As, z_m, ts_m)

            # 5) Training/evaluation split
            idx_tr, idx_te = split_sample_indices(
                len(df_m), evaluation_ratio=test_ratio, random_seed=RANDOM_SEED
            )

            q_tr, d_tr, z_tr, t_tr, ts_tr = q_m[idx_tr], d_m[idx_tr], z_m[idx_tr], t_m[idx_tr], ts_m[idx_tr]
            q_te, d_te, z_te, t_te, ts_te = q_m[idx_te], d_m[idx_te], z_m[idx_te], t_m[idx_te], ts_m[idx_te]

            xy_tr = m_xy[idx_tr]
            xy_te = m_xy[idx_te]

            bounds = [LAM_BOUNDS] * L + [A_BOUNDS] * L

            self.update_pb(30, "Calibrating layer-wise thermophysical parameters (RF/XGBoost/MLP)...")

            out_root = self.target_dir.get()
            ensure_dir(out_root)

            methods = []

            rf_sur = RandomForestRegressor(
                n_estimators=500, random_state=RANDOM_SEED, n_jobs=-1, min_samples_leaf=1
            )
            methods.append(("RF-assisted calibration", rf_sur))

            if _HAS_XGB:
                xgb_sur = XGBRegressor(
                    n_estimators=800,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="reg:squarederror",
                    random_state=RANDOM_SEED
                )
                methods.append(("XGBoost-assisted calibration", xgb_sur))
            else:
                methods.append(("XGBoost-assisted calibration [xgboost not installed; skipped]", None))

            mlp_sur = Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPRegressor(
                    hidden_layer_sizes=(160, 160),
                    max_iter=1200,
                    random_state=RANDOM_SEED,
                    early_stopping=True,
                    n_iter_no_change=25
                ))
            ])
            methods.append(("MLP-assisted calibration", mlp_sur))

            summary_rows = []
            best_method = None
            best_evaluation_rmse = np.inf
            best_surrogate = None
            best_kind = None

            # baseline
            base_tr_pred = self.physics_vec(q_tr, d_tr, base_lams, base_As, z_tr, ts_tr)
            base_te_pred = self.physics_vec(q_te, d_te, base_lams, base_As, z_te, ts_te)

            base_tr_metrics = self.calc_metrics(t_tr, base_tr_pred)
            base_te_metrics = self.calc_metrics(t_te, base_te_pred)

            base_dir = os.path.join(out_root, "Initial_conductive_model")
            ensure_dir(base_dir)

            base_detail = []
            for tag, idxs, tx, px, tsx in [
                ("TRAIN", idx_tr, t_tr, base_tr_pred, ts_tr),
                ("EVALUATION", idx_te, t_te, base_te_pred, ts_te)
            ]:
                err = px - tx
                for j in range(len(idxs)):
                    rid = int(idxs[j])
                    base_detail.append({
                        'Set': tag,
                        'WellIndex': rid,
                        'WellID': str(df_m.iloc[rid]['WellID']),
                        'X': float(df_m.iloc[rid]['X']),
                        'Y': float(df_m.iloc[rid]['Y']),
                        'Depth': float(df_m.iloc[rid]['Depth']),
                        'Ts': float(df_m.iloc[rid]['Ts']),
                        'RealT': float(tx[j]),
                        'PredT_Final': float(px[j]),
                        'Error': float(err[j]),
                        'AbsError': float(abs(err[j])),
                        'Within5C': int(abs(err[j]) <= 5.0)
                    })
            pd.DataFrame(base_detail).to_csv(os.path.join(base_dir, "sample_predictions_training_evaluation.csv"),
                                             index=False, encoding='utf-8')

            self.save_dual_scatter_plot_and_csv(
                t_tr, base_tr_pred, t_te, base_te_pred,
                base_dir, "observed_vs_predicted_training_evaluation",
                f"Initial conductive model TRAIN\nR2={base_tr_metrics['R2']:.3f}, RMSE={base_tr_metrics['RMSE']:.2f}, ≤5℃={base_tr_metrics['Within5Ratio']:.2%}",
                f"Initial conductive model EVALUATION\nR2={base_te_metrics['R2']:.3f}, RMSE={base_te_metrics['RMSE']:.2f}, ≤5℃={base_te_metrics['Within5Ratio']:.2%}"
            )

            base_row = {'Method': 'Initial conductive model', 'Kind': 'Physics-Base'}
            for k, v in base_tr_metrics.items():
                base_row[f"Train_{k}"] = v
            for k, v in base_te_metrics.items():
                base_row[f"Evaluation_{k}"] = v
            summary_rows.append(base_row)

            progress_base = 35
            progress_span = 20

            # 6) RF/XGBoost/MLP parameter calibration
            for mi, (mname, surrogate) in enumerate(methods):
                if surrogate is None:
                    summary_rows.append({'Method': mname, 'Kind': 'Skipped'})
                    continue

                method_dir = os.path.join(out_root, safe_name(mname))
                ensure_dir(method_dir)

                self.update_pb(progress_base + (mi / max(1, len(methods) - 1)) * progress_span,
                               f"{mname}: calibrating layer-wise parameters...")

                obj_tr = self.build_objective(base_lams, base_As, q_tr, d_tr, z_tr, t_tr, ts_tr)

                theta_best, loss_best, hist_df = self.smbo_optimize(
                    method_name=mname,
                    surrogate=surrogate,
                    obj_func=obj_tr,
                    bounds=bounds,
                    n_init=N_INIT,
                    n_iter=N_ITER,
                    n_cand=N_CAND,
                    top_k=TOP_K,
                    out_dir=method_dir
                )

                lam_mult = theta_best[:L]
                a_mult = theta_best[L:]
                lams_opt = base_lams * lam_mult
                As_opt = base_As * a_mult

                pred_tr = self.physics_vec(q_tr, d_tr, lams_opt, As_opt, z_tr, ts_tr)
                pred_te = self.physics_vec(q_te, d_te, lams_opt, As_opt, z_te, ts_te)

                tr_metrics = self.calc_metrics(t_tr, pred_tr)
                te_metrics = self.calc_metrics(t_te, pred_te)

                detail = []
                for tag, idxs, qx, dx, zx, tx, tsx in [
                    ("TRAIN", idx_tr, q_tr, d_tr, z_tr, t_tr, ts_tr),
                    ("EVALUATION", idx_te, q_te, d_te, z_te, t_te, ts_te)
                ]:
                    p0 = self.physics_vec(qx, dx, base_lams, base_As, zx, tsx)
                    p1 = self.physics_vec(qx, dx, lams_opt, As_opt, zx, tsx)
                    err = p1 - tx
                    for j in range(len(idxs)):
                        rid = int(idxs[j])
                        detail.append({
                            'Set': tag,
                            'WellIndex': rid,
                            'WellID': str(df_m.iloc[rid]['WellID']),
                            'X': float(df_m.iloc[rid]['X']),
                            'Y': float(df_m.iloc[rid]['Y']),
                            'Depth': float(df_m.iloc[rid]['Depth']),
                            'Ts': float(df_m.iloc[rid]['Ts']),
                            'RealT': float(tx[j]),
                            'PhysT_Initial': float(p0[j]),
                            'PredT_Final': float(p1[j]),
                            'Error': float(err[j]),
                            'AbsError': float(abs(err[j])),
                            'Within5C': int(abs(err[j]) <= 5.0)
                        })
                pd.DataFrame(detail).to_csv(os.path.join(method_dir, "sample_predictions_training_evaluation.csv"),
                                            index=False, encoding='utf-8')

                prop = []
                for iL in range(L):
                    prop.append({
                        'Layer': valid_configs[iL]['name'],
                        'lambda_init': float(base_lams[iL]),
                        'A_init_uWm3': float(base_As[iL]),
                        'lambda_mult': float(lam_mult[iL]),
                        'A_mult': float(a_mult[iL]),
                        'lambda_opt': float(lams_opt[iL]),
                        'A_opt_uWm3': float(As_opt[iL]),
                    })
                pd.DataFrame(prop).to_csv(os.path.join(method_dir, "calibrated_layer_thermophysical_parameters.csv"),
                                          index=False, encoding='utf-8')

                self.save_dual_scatter_plot_and_csv(
                    t_tr, pred_tr, t_te, pred_te,
                    method_dir, "observed_vs_predicted_training_evaluation",
                    f"{mname} TRAIN\nR2={tr_metrics['R2']:.3f}, RMSE={tr_metrics['RMSE']:.2f}, ≤5℃={tr_metrics['Within5Ratio']:.2%}",
                    f"{mname} EVALUATION\nR2={te_metrics['R2']:.3f}, RMSE={te_metrics['RMSE']:.2f}, ≤5℃={te_metrics['Within5Ratio']:.2%}"
                )

                row = {'Method': mname, 'Kind': 'Physics-Optimized', 'BestTrainObjective': float(loss_best)}
                for k, v in tr_metrics.items():
                    row[f"Train_{k}"] = v
                for k, v in te_metrics.items():
                    row[f"Evaluation_{k}"] = v
                summary_rows.append(row)

                if te_metrics['RMSE'] < best_evaluation_rmse:
                    best_evaluation_rmse = te_metrics['RMSE']
                    best_method = mname
                    best_surrogate = surrogate
                    best_kind = 'Physics-Optimized'

            # 7) PINN
            self.update_pb(58, "Training PINN...")

            pinn_dir = os.path.join(out_root, "PINN_physics_informed_neural_network")
            pred_tr_pinn, pred_te_pinn = None, None
            model_pinn, mu_pinn, sd_pinn, pinn_device = None, None, None, None

            if _HAS_TORCH:
                ensure_dir(pinn_dir)

                Xtr_pinn = self.build_pinn_features(
                    xy_tr, q_tr, d_tr, z_tr,
                    self.physics_vec(q_tr, d_tr, base_lams, base_As, z_tr, ts_tr),
                    ts_tr
                )
                Xte_pinn = self.build_pinn_features(
                    xy_te, q_te, d_te, z_te,
                    self.physics_vec(q_te, d_te, base_lams, base_As, z_te, ts_te),
                    ts_te
                )

                model_pinn, mu_pinn, sd_pinn, pred_tr_pinn, pred_te_pinn, hist_pinn, pinn_device = self.train_pinn(
                    Xtr_pinn, t_tr.astype(np.float32),
                    Xte_pinn, t_te.astype(np.float32),
                    pinn_dir
                )

                tr_metrics_pinn = self.calc_metrics(t_tr, pred_tr_pinn)
                te_metrics_pinn = self.calc_metrics(t_te, pred_te_pinn)

                detail = []
                for tag, idxs, xx, qq, dd, zz, tt, tss in [
                    ("TRAIN", idx_tr, xy_tr, q_tr, d_tr, z_tr, t_tr, ts_tr),
                    ("EVALUATION", idx_te, xy_te, q_te, d_te, z_te, t_te, ts_te)
                ]:
                    phys0 = self.physics_vec(qq, dd, base_lams, base_As, zz, tss)
                    Xp = self.build_pinn_features(xx, qq, dd, zz, phys0, tss)
                    pp = self.pinn_predict(model_pinn, Xp, mu_pinn, sd_pinn, pinn_device)
                    err = pp - tt
                    for j in range(len(idxs)):
                        rid = int(idxs[j])
                        detail.append({
                            'Set': tag,
                            'WellIndex': rid,
                            'WellID': str(df_m.iloc[rid]['WellID']),
                            'X': float(df_m.iloc[rid]['X']),
                            'Y': float(df_m.iloc[rid]['Y']),
                            'Depth': float(df_m.iloc[rid]['Depth']),
                            'Ts': float(df_m.iloc[rid]['Ts']),
                            'RealT': float(tt[j]),
                            'PhysT_Initial': float(phys0[j]),
                            'PredT_Final': float(pp[j]),
                            'Error': float(err[j]),
                            'AbsError': float(abs(err[j])),
                            'Within5C': int(abs(err[j]) <= 5.0)
                        })
                pd.DataFrame(detail).to_csv(os.path.join(pinn_dir, "sample_predictions_training_evaluation.csv"),
                                            index=False, encoding='utf-8')

                self.save_dual_scatter_plot_and_csv(
                    t_tr, pred_tr_pinn, t_te, pred_te_pinn,
                    pinn_dir, "observed_vs_predicted_training_evaluation",
                    f"PINN TRAIN\nR2={tr_metrics_pinn['R2']:.3f}, RMSE={tr_metrics_pinn['RMSE']:.2f}, ≤5℃={tr_metrics_pinn['Within5Ratio']:.2%}",
                    f"PINN EVALUATION\nR2={te_metrics_pinn['R2']:.3f}, RMSE={te_metrics_pinn['RMSE']:.2f}, ≤5℃={te_metrics_pinn['Within5Ratio']:.2%}"
                )

                row = {'Method': 'Physics-informed neural network (PINN)', 'Kind': 'PINN'}
                for k, v in tr_metrics_pinn.items():
                    row[f"Train_{k}"] = v
                for k, v in te_metrics_pinn.items():
                    row[f"Evaluation_{k}"] = v
                summary_rows.append(row)

                if te_metrics_pinn['RMSE'] < best_evaluation_rmse:
                    best_evaluation_rmse = te_metrics_pinn['RMSE']
                    best_method = 'Physics-informed neural network (PINN)'
                    best_kind = 'PINN'
            else:
                summary_rows.append({
                    'Method': 'Physics-informed neural network (PINN) [PyTorch not installed; skipped]',
                    'Kind': 'Skipped'
                })

            # 8) PINN + XGBoost residual correction
            self.update_pb(68, "Training PINN-XGBoost residual correction...")

            if _HAS_TORCH and _HAS_XGB and (pred_tr_pinn is not None) and (pred_te_pinn is not None):
                hybrid_dir = os.path.join(out_root, "PINN_XGBoost_residual_correction")
                ensure_dir(hybrid_dir)

                phys_tr = self.physics_vec(q_tr, d_tr, base_lams, base_As, z_tr, ts_tr)
                phys_te = self.physics_vec(q_te, d_te, base_lams, base_As, z_te, ts_te)

                Xtr_hybrid = self.build_hybrid_features(
                    xy_tr, q_tr, d_tr, z_tr, phys_tr, ts_tr, pred_tr_pinn
                )
                Xte_hybrid = self.build_hybrid_features(
                    xy_te, q_te, d_te, z_te, phys_te, ts_te, pred_te_pinn
                )

                res_tr = t_tr - pred_tr_pinn
                res_te = t_te - pred_te_pinn

                xgb_res = XGBRegressor(
                    n_estimators=1000,
                    learning_rate=0.03,
                    max_depth=5,
                    min_child_weight=3,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.0,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    random_state=RANDOM_SEED
                )
                xgb_res.fit(Xtr_hybrid, res_tr)

                res_pred_tr = xgb_res.predict(Xtr_hybrid)
                res_pred_te = xgb_res.predict(Xte_hybrid)

                pred_tr_hybrid = pred_tr_pinn + res_pred_tr
                pred_te_hybrid = pred_te_pinn + res_pred_te

                tr_metrics_hybrid = self.calc_metrics(t_tr, pred_tr_hybrid)
                te_metrics_hybrid = self.calc_metrics(t_te, pred_te_hybrid)

                detail = []
                for tag, idxs, xx, qq, dd, zz, tt, tss, pp in [
                    ("TRAIN", idx_tr, xy_tr, q_tr, d_tr, z_tr, t_tr, ts_tr, pred_tr_pinn),
                    ("EVALUATION", idx_te, xy_te, q_te, d_te, z_te, t_te, ts_te, pred_te_pinn)
                ]:
                    phys0 = self.physics_vec(qq, dd, base_lams, base_As, zz, tss)
                    Xh = self.build_hybrid_features(xx, qq, dd, zz, phys0, tss, pp)
                    rr = xgb_res.predict(Xh)
                    pf = pp + rr
                    err = pf - tt

                    for j in range(len(idxs)):
                        rid = int(idxs[j])
                        detail.append({
                            'Set': tag,
                            'WellIndex': rid,
                            'WellID': str(df_m.iloc[rid]['WellID']),
                            'X': float(df_m.iloc[rid]['X']),
                            'Y': float(df_m.iloc[rid]['Y']),
                            'Depth': float(df_m.iloc[rid]['Depth']),
                            'Ts': float(df_m.iloc[rid]['Ts']),
                            'RealT': float(tt[j]),
                            'PhysT_Initial': float(phys0[j]),
                            'PINN_T': float(pp[j]),
                            'XGB_Residual_Correction': float(rr[j]),
                            'PredT_Final': float(pf[j]),
                            'Error': float(err[j]),
                            'AbsError': float(abs(err[j])),
                            'Within5C': int(abs(err[j]) <= 5.0)
                        })
                pd.DataFrame(detail).to_csv(
                    os.path.join(hybrid_dir, "sample_predictions_training_evaluation.csv"),
                    index=False, encoding='utf-8'
                )

                self.save_dual_scatter_plot_and_csv(
                    t_tr, pred_tr_hybrid, t_te, pred_te_hybrid,
                    hybrid_dir, "observed_vs_predicted_training_evaluation",
                    f"PINN-XGB TRAIN\nR2={tr_metrics_hybrid['R2']:.3f}, RMSE={tr_metrics_hybrid['RMSE']:.2f}, ≤5℃={tr_metrics_hybrid['Within5Ratio']:.2%}",
                    f"PINN-XGB EVALUATION\nR2={te_metrics_hybrid['R2']:.3f}, RMSE={te_metrics_hybrid['RMSE']:.2f}, ≤5℃={te_metrics_hybrid['Within5Ratio']:.2%}"
                )

                res_df = pd.DataFrame({
                    'Set': ['TRAIN'] * len(res_tr) + ['EVALUATION'] * len(res_te),
                    'Residual_RealMinusPINN': np.concatenate([res_tr, res_te]),
                    'Residual_PredictedByXGB': np.concatenate([res_pred_tr, res_pred_te])
                })
                res_df.to_csv(os.path.join(hybrid_dir, "PINN_residual_XGBoost_learning.csv"),
                              index=False, encoding='utf-8')

                # SHAP
                feature_names_hybrid = (
                    ['X', 'Y', 'Q', 'Depth', 'PhysT', 'Ts', 'PINN_T'] +
                    [f'Z_{cfg["name"]}' for cfg in valid_configs]
                )
                Xte_hybrid_df = pd.DataFrame(Xte_hybrid, columns=feature_names_hybrid)

                self.run_shap_analysis(
                    model=xgb_res,
                    X_df=Xte_hybrid_df,
                    out_dir=hybrid_dir,
                    model_name="PINN_XGBoost_residual_correction",
                    top_n=10,
                    max_samples=2000
                )

                row = {'Method': 'Hybrid PINN-XGBoost residual correction', 'Kind': 'Hybrid'}
                for k, v in tr_metrics_hybrid.items():
                    row[f"Train_{k}"] = v
                for k, v in te_metrics_hybrid.items():
                    row[f"Evaluation_{k}"] = v
                summary_rows.append(row)

                if te_metrics_hybrid['RMSE'] < best_evaluation_rmse:
                    best_evaluation_rmse = te_metrics_hybrid['RMSE']
                    best_method = 'Hybrid PINN-XGBoost residual correction'
                    best_kind = 'Hybrid'
                    best_hybrid_model = xgb_res
                    best_pinn_model = model_pinn
                    best_pinn_mu = mu_pinn
                    best_pinn_sd = sd_pinn
                    best_pinn_device = pinn_device
            else:
                summary_rows.append({
                    'Method': 'Hybrid PINN-XGBoost residual correction [PyTorch or XGBoost unavailable; skipped]',
                    'Kind': 'Skipped'
                })

            # 9) Summary and ranking
            self.update_pb(76, "Writing multi-method summary, ranking, and comparison plots...")

            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(os.path.join(out_root, "method_comparison_summary.csv"),
                              index=False, encoding='utf-8')

            rank_df = summary_df.copy()
            if 'Evaluation_RMSE' in rank_df.columns:
                rank_df = rank_df.sort_values(by='Evaluation_RMSE', ascending=True, na_position='last')
            rank_df.to_csv(os.path.join(out_root, "method_ranking_by_Evaluation_RMSE.csv"),
                           index=False, encoding='utf-8')

            self.create_method_comparison_plots(rank_df, out_root)


            if best_method is None:
                raise RuntimeError("No modeling method completed successfully.")

            # 10) Full-dataset refinement
            self.update_pb(84, f"Selected method: {best_method}. Refining on the full dataset...")

            best_dir = os.path.join(out_root, "BEST_" + safe_name(best_method))
            ensure_dir(best_dir)

            if best_kind == 'Physics-Optimized':
                obj_all = self.build_objective(base_lams, base_As, q_m, d_m, z_m, t_m, ts_m)

                theta_best_all, _, hist_all = self.smbo_optimize(
                    method_name=best_method + "_ALL",
                    surrogate=best_surrogate,
                    obj_func=obj_all,
                    bounds=[LAM_BOUNDS] * L + [A_BOUNDS] * L,
                    n_init=max(40, N_INIT // 2),
                    n_iter=max(18, N_ITER // 2),
                    n_cand=N_CAND,
                    top_k=TOP_K,
                    out_dir=best_dir
                )

                lam_mult_all = theta_best_all[:L]
                a_mult_all = theta_best_all[L:]
                lams_all = base_lams * lam_mult_all
                As_all = base_As * a_mult_all

                pred_all = self.physics_vec(q_m, d_m, lams_all, As_all, z_m, ts_m)
                all_metrics = self.calc_metrics(t_m, pred_all)

                all_detail = pd.DataFrame({
                    'WellID': df_m['WellID'].values,
                    'X': df_m['X'].values,
                    'Y': df_m['Y'].values,
                    'Depth': df_m['Depth'].values,
                    'Ts': df_m['Ts'].values,
                    'RealT': t_m,
                    'PhysT_Initial': t_phys0,
                    'PredT_Final': pred_all,
                    'Error': pred_all - t_m,
                    'AbsError': np.abs(pred_all - t_m),
                    'Within5C': (np.abs(pred_all - t_m) <= 5.0).astype(int)
                })
                all_detail.to_csv(os.path.join(best_dir, "sample_predictions_full_dataset.csv"),
                                  index=False, encoding='utf-8')

                final_prop = []
                for iL in range(L):
                    final_prop.append({
                        'Layer': valid_configs[iL]['name'],
                        'lambda_init': float(base_lams[iL]),
                        'A_init_uWm3': float(base_As[iL]),
                        'lambda_mult': float(lam_mult_all[iL]),
                        'A_mult': float(a_mult_all[iL]),
                        'lambda_opt': float(lams_all[iL]),
                        'A_opt_uWm3': float(As_all[iL]),
                    })
                pd.DataFrame(final_prop).to_csv(os.path.join(best_dir, "final_calibrated_layer_thermophysical_parameters.csv"),
                                                index=False, encoding='utf-8')

                best_predict_func = lambda qq, dd, zz, xy: self.physics_vec(qq, dd, lams_all, As_all, zz, default_ts)

            elif best_kind == 'PINN':
                Xall_pinn = self.build_pinn_features(
                    m_xy, q_m, d_m, z_m,
                    self.physics_vec(q_m, d_m, base_lams, base_As, z_m, ts_m),
                    ts_m
                )

                model_pinn_all, mu_all, sd_all, _, pred_all, hist_all, device_all = self.train_pinn(
                    Xall_pinn, t_m.astype(np.float32),
                    Xall_pinn, t_m.astype(np.float32),
                    best_dir
                )

                all_metrics = self.calc_metrics(t_m, pred_all)

                all_detail = pd.DataFrame({
                    'WellID': df_m['WellID'].values,
                    'X': df_m['X'].values,
                    'Y': df_m['Y'].values,
                    'Depth': df_m['Depth'].values,
                    'Ts': df_m['Ts'].values,
                    'RealT': t_m,
                    'PhysT_Initial': t_phys0,
                    'PredT_Final': pred_all,
                    'Error': pred_all - t_m,
                    'AbsError': np.abs(pred_all - t_m),
                    'Within5C': (np.abs(pred_all - t_m) <= 5.0).astype(int)
                })
                all_detail.to_csv(os.path.join(best_dir, "sample_predictions_full_dataset.csv"),
                                  index=False, encoding='utf-8')

                pd.DataFrame([{
                    'Info': 'PINN does not output layer-wise lambda/A calibration results; temperature is learned through network weights and physical guidance.'
                }]).to_csv(os.path.join(best_dir, "final_calibrated_layer_thermophysical_parameters.csv"),
                            index=False, encoding='utf-8')

                ts_grid = np.full(len(q_arr), default_ts, dtype=float)
                best_predict_func = lambda qq, dd, zz, xy: self.pinn_predict(
                    model_pinn_all,
                    self.build_pinn_features(
                        xy, qq, dd, zz,
                        self.physics_vec(qq, dd, base_lams, base_As, zz, ts_grid),
                        ts_grid
                    ),
                    mu_all, sd_all, device_all
                )

            elif best_kind == 'Hybrid':
                Xall_pinn = self.build_pinn_features(
                    m_xy, q_m, d_m, z_m,
                    self.physics_vec(q_m, d_m, base_lams, base_As, z_m, ts_m),
                    ts_m
                )

                model_pinn_all, mu_all, sd_all, _, pred_all_pinn, hist_all, device_all = self.train_pinn(
                    Xall_pinn, t_m.astype(np.float32),
                    Xall_pinn, t_m.astype(np.float32),
                    best_dir
                )

                phys_all = self.physics_vec(q_m, d_m, base_lams, base_As, z_m, ts_m)
                Xall_hybrid = self.build_hybrid_features(
                    m_xy, q_m, d_m, z_m, phys_all, ts_m, pred_all_pinn
                )
                res_all = t_m - pred_all_pinn

                xgb_res_all = XGBRegressor(
                    n_estimators=1000,
                    learning_rate=0.03,
                    max_depth=5,
                    min_child_weight=3,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.0,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    random_state=RANDOM_SEED
                )
                xgb_res_all.fit(Xall_hybrid, res_all)

                res_pred_all = xgb_res_all.predict(Xall_hybrid)
                pred_all = pred_all_pinn + res_pred_all
                all_metrics = self.calc_metrics(t_m, pred_all)

                all_detail = pd.DataFrame({
                    'WellID': df_m['WellID'].values,
                    'X': df_m['X'].values,
                    'Y': df_m['Y'].values,
                    'Depth': df_m['Depth'].values,
                    'Ts': df_m['Ts'].values,
                    'RealT': t_m,
                    'PhysT_Initial': phys_all,
                    'PINN_T': pred_all_pinn,
                    'XGB_Residual_Correction': res_pred_all,
                    'PredT_Final': pred_all,
                    'Error': pred_all - t_m,
                    'AbsError': np.abs(pred_all - t_m),
                    'Within5C': (np.abs(pred_all - t_m) <= 5.0).astype(int)
                })
                all_detail.to_csv(os.path.join(best_dir, "sample_predictions_full_dataset.csv"),
                                  index=False, encoding='utf-8')

                pd.DataFrame([{
                    'Info': 'The hybrid model combines the PINN main prediction with XGBoost residual correction and does not output layer-wise lambda/A calibration parameters.'
                }]).to_csv(os.path.join(best_dir, "final_calibrated_layer_thermophysical_parameters.csv"),
                            index=False, encoding='utf-8')

                feature_names_hybrid = (
                    ['X', 'Y', 'Q', 'Depth', 'PhysT', 'Ts', 'PINN_T'] +
                    [f'Z_{cfg["name"]}' for cfg in valid_configs]
                )
                Xall_hybrid_df = pd.DataFrame(Xall_hybrid, columns=feature_names_hybrid)
                self.run_shap_analysis(
                    model=xgb_res_all,
                    X_df=Xall_hybrid_df,
                    out_dir=best_dir,
                    model_name="BEST_PINN_XGBoost_residual_correction",
                    top_n=10,
                    max_samples=3000
                )

                ts_grid = np.full(len(q_arr), default_ts, dtype=float)

                def best_predict_func(qq, dd, zz, xy):
                    phys_grid = self.physics_vec(qq, dd, base_lams, base_As, zz, ts_grid)

                    Xgrid_pinn = self.build_pinn_features(
                        xy, qq, dd, zz, phys_grid, ts_grid
                    )
                    pinn_grid = self.pinn_predict(model_pinn_all, Xgrid_pinn, mu_all, sd_all, device_all)

                    Xgrid_hybrid = self.build_hybrid_features(
                        xy, qq, dd, zz, phys_grid, ts_grid, pinn_grid
                    )
                    res_grid = xgb_res_all.predict(Xgrid_hybrid)

                    return pinn_grid + res_grid
            else:
                raise RuntimeError("Unrecognized selected-method type.")

            report_df = pd.DataFrame([{'Metric': k, 'Value': v} for k, v in all_metrics.items()])
            report_df.to_csv(os.path.join(best_dir, "selected_method_full_dataset_metrics.csv"),
                             index=False, encoding='utf-8')

            self.save_scatter_plot_and_csv(
                t_m, pred_all, best_dir, "full_dataset_observed_vs_predicted",
                f"{best_method} full dataset\nR2={all_metrics['R2']:.3f}, RMSE={all_metrics['RMSE']:.2f}, ≤5℃={all_metrics['Within5Ratio']:.2%}"
            )

            # 11) Temperature maps at 1-8 km
            self.update_pb(92, "Calculating and exporting 1-8 km temperature surfaces...")

            grid_res = grid.copy()
            for di, d in enumerate(TARGET_DEPTHS):
                self.update_pb(92 + (di / len(TARGET_DEPTHS)) * 4,
                               f"Calculating the {d} m temperature surface...")

                t_map = best_predict_func(
                    q_arr,
                    d_matrix,
                    np.full(len(q_arr), d, dtype=float),
                    grid_xy
                )
                grid_res[f"T_{d}m"] = np.round(t_map, 2)

                self.save_map_plot_and_csv(
                    grid_res['X'], grid_res['Y'], grid_res[f"T_{d}m"],
                    best_dir, f"temperature_map_{d}m",
                    f"Sichuan Basin predicted temperature at {d} m ({best_method})"
                )

            # 12) Formation top/bottom and surface temperatures
            self.update_pb(97, "Calculating and saving formation top/bottom and surface temperatures...")

            surface_temp_df = grid_res[['X', 'Y']].copy()

            for i, c in enumerate(valid_configs):
                z_bottom = np.abs(grid_res[c['col']].values.astype(float))
                if i == 0:
                    z_top = np.zeros_like(z_bottom, dtype=float)
                else:
                    z_top = np.abs(grid_res[valid_configs[i - 1]['col']].values.astype(float))

                t_top = best_predict_func(q_arr, d_matrix, z_top, grid_xy)
                t_bottom = best_predict_func(q_arr, d_matrix, z_bottom, grid_xy)

                col_top = f"T_Top_{c['name']}"
                col_bottom = f"T_Bottom_{c['name']}"
                grid_res[col_top] = np.round(t_top, 2)
                grid_res[col_bottom] = np.round(t_bottom, 2)

                surface_temp_df[f"T_Surface_{c['name']}"] = np.round(default_ts, 2)

                if OUTPUT_TOP_MAPS:
                    self.save_map_plot_and_csv(
                        grid_res['X'], grid_res['Y'], grid_res[col_top],
                        best_dir, f"top_temperature_map_{safe_name(c['name'])}",
                        f"{c['name']} top-interface temperature ({best_method})"
                    )

                if OUTPUT_BOTTOM_MAPS:
                    self.save_map_plot_and_csv(
                        grid_res['X'], grid_res['Y'], grid_res[col_bottom],
                        best_dir, f"bottom_temperature_map_{safe_name(c['name'])}",
                        f"{c['name']} bottom-interface temperature ({best_method})"
                    )

            cols_keep = ['X', 'Y']
            for c in valid_configs:
                cols_keep.append(f"T_Top_{c['name']}")
                cols_keep.append(f"T_Bottom_{c['name']}")

            grid_res[cols_keep].to_csv(
                os.path.join(best_dir, "formation_top_bottom_temperature_summary.csv"),
                index=False, encoding='utf-8'
            )

            surface_temp_df.to_csv(
                os.path.join(best_dir, "formation_surface_temperature_summary.csv"),
                index=False, encoding='utf-8'
            )

            grid_res.to_csv(os.path.join(best_dir, "Sichuan_Basin_selected_method_grid_temperature_results.csv"),
                            index=False, encoding='utf-8')

            self.update_pb(100, "Completed")

            cost = time.strftime("%M:%S", time.gmtime(time.time() - start_time))
            messagebox.showinfo(
                "Completed",
                f"Elapsed time: {cost}\n"
                f"Selected method: {best_method}\n"
                f"Full-dataset RMSE: {all_metrics['RMSE']:.2f} ℃\n"
                f"Full-dataset within-±5°C ratio: {all_metrics['Within5Ratio']:.2%}\n"
                f"Results saved to: {best_dir}\n\n"
                f"Key outputs:\n"
                f"1) method_comparison_summary.csv\n"
                f"2) method_ranking_by_Evaluation_RMSE.csv\n"
                f"3) Method-comparison figures and CSV files (RMSE/MAE/R2/Within5)\n"
                f"4) Training/evaluation scatter plots and CSV files\n"
                f"5) Selected-method full-dataset metrics, scatter plot, and CSV\n"
                f"6) 1-8 km temperature maps and CSV files\n"
                f"7) Formation top/bottom temperature maps and CSV files\n"
                f"8) PINN-XGBoost SHAP figures and SHAP-value tables"
            )

        except Exception:
            import traceback
            messagebox.showerror("Runtime error", f"Error details:\n{traceback.format_exc()}")


if __name__ == "__main__":
    root = tk.Tk()
    app = GeoThermalDoctor(root)
    root.mainloop()
