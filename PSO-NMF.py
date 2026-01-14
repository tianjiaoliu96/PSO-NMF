# -*- coding: utf-8 -*-
"""
Advanced Geochemical Unmixing - Geo-Mathematical Fusion (Final Split-Figure Version)
[2025-02-16 布局优化版：图A与图B独立绘制，坐标轴防重叠]

核心特征：
1. [数学模型] 包含平滑性约束项(μ)，Alpha范围 [0.98, 1.02]。
2. [绘图优化] 图A（K值优选）与图B（收敛性）拆分为独立图表，布局更宽松。
3. [输出格式] 生成长格式 Excel 贡献度表。
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from tqdm import tqdm
import seaborn as sns
import warnings
import os

# === 扩展库 ===
import holoviews as hv
from pyswarm import pso
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score

# ==========================================
# 0. 全局配置
# ==========================================
hv.extension('bokeh')
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.5})


# ==========================================
# 1. 数据预处理
# ==========================================
def load_and_preprocess(filepath):
   print(f"正在读取数据: {filepath}...")
   try:
       df_raw = pd.read_excel(filepath)
       # 【关键】按照 地区 -> 井号 排序，这是平滑性约束生效的前提
       if 'Region' in df_raw.columns and 'Well' in df_raw.columns:
           df_raw = df_raw.sort_values(by=['Region', 'Well']).reset_index(drop=True)
   except FileNotFoundError:
       print("未找到文件，生成模拟数据...")
       data = {
           'Region': ['R1'] * 10 + ['R2'] * 10,
           'Well': [f'W{i:02d}' for i in range(20)],
           'd13C1': np.random.normal(-35, 2, 20), 'd13C2': np.random.normal(-25, 2, 20),
           'd13C3': np.random.normal(-22, 1, 20), 'd13C4': np.random.normal(-20, 1, 20),
           'C1': np.random.rand(20), 'C2': np.random.rand(20), 'C3': np.random.rand(20),
           'N2': np.random.rand(20), 'CO2': np.random.rand(20)
       }
       df_raw = pd.DataFrame(data)

   meta_data = df_raw.iloc[:, 0:2]
   meta_data.columns = ['Region', 'Well']
   raw_features = df_raw.iloc[:, 2:12]
   feature_names = raw_features.columns.tolist()

   imputer = KNNImputer(n_neighbors=3)
   data_imputed = pd.DataFrame(imputer.fit_transform(raw_features), columns=feature_names)

   df_clipped = data_imputed.copy()
   for col in df_clipped.columns:
       lower = df_clipped[col].quantile(0.025)
       upper = df_clipped[col].quantile(0.975)
       df_clipped[col] = df_clipped[col].clip(lower, upper)

   if df_clipped.shape[1] >= 2:
       d13c1 = df_clipped.iloc[:, 0]
       d13c2 = df_clipped.iloc[:, 1]
       delta_c12 = d13c1 - d13c2
       f_delta = (delta_c12 - delta_c12.mean()) / (delta_c12.std() + 1e-9)
   else:
       f_delta = np.zeros(df_clipped.shape[0])

   min_vals = df_clipped.min()
   max_vals = df_clipped.max()
   ranges = max_vals - min_vals
   ranges[ranges == 0] = 1.0
   V_norm = (df_clipped - min_vals) / ranges

   info = {
       'min': min_vals, 'range': ranges,
       'f_delta': f_delta.values if hasattr(f_delta, 'values') else f_delta,
       'feature_names': feature_names
   }
   return V_norm, meta_data, info


# ==========================================
# 2. 核心算法: MOPSO-NMF (含平滑性约束)
# ==========================================
class GeochemicalNMF:
   def __init__(self, V, n_components, preprocessing_info,
                reg_lambda=0.01, reg_beta=0.01,
                weight_gamma=5.0, weight_eta=1.0,
                weight_mu=1.0,  # 平滑性权重
                kappa=0.1,
                init_strategy='random'):
       self.V = V.values if hasattr(V, 'values') else V
       self.m, self.n = self.V.shape
       self.K = n_components
       self.info = preprocessing_info

       self.lam = reg_lambda
       self.bet = reg_beta
       self.gam = weight_gamma
       self.eta = weight_eta
       self.mu = weight_mu
       self.kappa = kappa
       self.init_strategy = init_strategy

   def _initialize(self):
       np.random.seed(None)
       self.W = np.random.uniform(0.1, 1.0, (self.m, self.K))
       self.H = np.random.uniform(0.1, 1.0, (self.K, self.n))
       self.W = self.W / (self.W.sum(axis=1, keepdims=True) + 1e-9)
       f_delta = self.info['f_delta'].reshape(-1, 1)
       self.target_alpha = 1.0 + self.kappa * f_delta
       self.alpha = self.target_alpha + np.random.normal(0, 0.01, (self.m, self.K))

   def calculate_loss(self):
       """
       计算总损失函数，包含平滑性项
       """
       f_delta = self.info['f_delta'].reshape(-1, 1)
       Psi = 1.0 + (self.alpha - 1.0)
       W_eff = self.W * Psi
       Recon = W_eff @ self.H

       # 基础各项
       term_fit = np.linalg.norm(self.V - Recon, 'fro') ** 2
       term_l2 = self.lam * np.linalg.norm(self.W, 'fro') ** 2
       term_l1 = self.bet * np.sum(np.abs(self.W))
       term_unity = self.gam * np.sum((np.sum(self.W, axis=1) - 1) ** 2)

       # 同位素约束
       target = np.tile(self.target_alpha, (1, self.K))
       term_iso = self.eta * np.sum((self.alpha - target) ** 2)

       # 【新增】平滑性约束 (Smoothness)
       term_smooth = self.mu * np.sum((self.alpha[:-1, :] - self.alpha[1:, :]) ** 2)

       return term_fit + term_l2 + term_l1 + term_unity + term_iso + term_smooth

   def fit(self, max_iter=100, tol=1e-5):
       self._initialize()
       loss_history = []
       eye_K = np.eye(self.K) * np.sqrt(self.lam)
       zeros_K = np.zeros(self.K)
       ones_row = np.ones((1, self.K)) * np.sqrt(self.gam)
       unity_target = np.array([np.sqrt(self.gam)])

       for it in range(max_iter):
           # --- 1. 更新 H ---
           Psi = 1.0 + (self.alpha - 1.0)
           W_eff = self.W * Psi
           for j in range(self.n):
               self.H[:, j], _ = nnls(W_eff, self.V[:, j])

           # --- 2. 更新 W ---
           for i in range(self.m):
               H_eff = (self.H.T * Psi[i, :]).T
               A_aug = np.vstack([H_eff.T, eye_K, ones_row])
               b_aug = np.concatenate([self.V[i, :], zeros_K, unity_target])
               self.W[i, :], _ = nnls(A_aug, b_aug)

           if self.bet > 0:
               self.W = np.maximum(0, self.W - self.bet * 0.001)

           # --- 3. 更新 Alpha (梯度下降) ---
           R = (self.W * Psi) @ self.H - self.V

           # 基础梯度
           grad_alpha = 2 * (R @ self.H.T) * self.W
           grad_alpha += 2 * self.eta * (self.alpha - np.tile(self.target_alpha, (1, self.K)))

           # 平滑性梯度
           grad_smooth = np.zeros_like(self.alpha)
           if self.m > 1:
               grad_smooth[1:-1, :] = 2 * self.mu * (2 * self.alpha[1:-1, :] - self.alpha[:-2, :] - self.alpha[2:, :])
               grad_smooth[0, :] = 2 * self.mu * (self.alpha[0, :] - self.alpha[1, :])
               grad_smooth[-1, :] = 2 * self.mu * (self.alpha[-1, :] - self.alpha[-2, :])
           grad_alpha += grad_smooth

           # 更新
           self.alpha -= 0.001 * grad_alpha
           # 【重要】Alpha 范围限制在 [0.98, 1.02]
           self.alpha = np.clip(self.alpha, 0.98, 1.02)

           loss = self.calculate_loss()
           loss_history.append(loss)
           if it > 10 and abs(loss_history[-2] - loss) < tol:
               break
       return loss_history


# ==========================================
# 3. 绘图逻辑 (图A和图B 独立绘制)
# ==========================================
def plot_separated_figures(V_norm, info):
   """
   修改后的绘图函数：将图A和图B分开绘制，避免坐标轴拥挤
   """

   # === 1. 计算图A数据 (K值扫描) ===
   print("\n[Step 1] 计算 K 值选择指标 (RMSE/AIC)...")
   k_range = list(range(2, 9))
   rmse_means, rmse_stds, aic_means = [], [], []

   for k in tqdm(k_range, desc="Scanning K"):
       k_errors, k_aics = [], []
       for _ in range(8):
           noise = np.random.normal(0, 0.02, V_norm.shape)
           model = GeochemicalNMF(pd.DataFrame(V_norm.values + noise, columns=V_norm.columns),
                                  k, info, reg_lambda=0.01, reg_beta=0.01)
           model.fit(max_iter=50)
           Psi = 1.0 + (model.alpha - 1.0)
           Recon = (model.W * Psi) @ model.H
           rss = np.sum((V_norm.values - Recon) ** 2)
           mse = rss / (V_norm.shape[0] * V_norm.shape[1])
           k_errors.append(np.sqrt(mse))
           n_samples = V_norm.shape[0] * V_norm.shape[1]
           n_params = k * (V_norm.shape[0] + V_norm.shape[1])
           aic = n_samples * np.log(rss / n_samples) + 2 * n_params
           k_aics.append(aic)
       rmse_means.append(np.mean(k_errors))
       rmse_stds.append(np.std(k_errors))
       aic_means.append(np.mean(k_aics))

   rmse_arr = np.array(rmse_means)
   rmse_std_arr = np.array(rmse_stds)
   second_deriv = []
   k_deriv = []
   for i in range(1, len(rmse_arr) - 1):
       d2 = (rmse_arr[i + 1] + rmse_arr[i - 1] - 2 * rmse_arr[i])
       second_deriv.append(d2)
       k_deriv.append(k_range[i])

   # === 绘制图 A (独立画板，宽布局) ===
   print("  -> 绘制图 A (Multi-Criteria)...")
   plt.figure(figsize=(12, 7))  # 宽布局
   ax_a = plt.gca()

   # 1. RMSE (左轴)
   p1, = ax_a.plot(k_range, rmse_means, 'o-', color='navy', linewidth=2.5, label='RMSE (Left)')
   ax_a.fill_between(k_range,
                     rmse_arr - 2 * rmse_std_arr,
                     rmse_arr + 2 * rmse_std_arr,
                     color='navy', alpha=0.20, label='95% CI')
   ax_a.set_xlabel('Number of End-members (K)', fontsize=14, fontweight='bold')
   ax_a.set_ylabel('RMSE', fontsize=14, fontweight='bold', color='navy')
   ax_a.tick_params(axis='y', labelcolor='navy')

   # 2. Curvature (右轴1)
   ax_a2 = ax_a.twinx()
   p2, = ax_a2.plot(k_deriv, second_deriv, color='red', marker='D', linestyle='--', linewidth=2,
                    label='Curvature (Right-1)')
   ax_a2.set_ylabel('Curvature', fontsize=14, fontweight='bold', color='red')
   ax_a2.tick_params(axis='y', labelcolor='red')

   # 3. AIC (右轴2 - 偏移位置)
   ax_a3 = ax_a.twinx()
   # 将第三个坐标轴向右推，避免重叠
   ax_a3.spines["right"].set_position(("axes", 1.2))
   p3, = ax_a3.plot(k_range, aic_means, color='#33a02c', marker='s', linestyle=':', linewidth=2.5,
                    label='AIC Trend (Right-2)')
   ax_a3.set_ylabel('AIC Criterion', fontsize=14, fontweight='bold', color='#33a02c')
   ax_a3.tick_params(axis='y', labelcolor='#33a02c')

   # 合并图例
   lines = [p1, p2, p3]
   ax_a.legend(lines, [l.get_label() for l in lines], loc='upper center', fontsize=11, frameon=True)
   plt.title('(a) Multi-Criteria for K Selection', fontsize=18, pad=20)
   plt.tight_layout()  # 自动调整布局，防止被切掉
   plt.savefig('Figure_A_K_Selection.png', dpi=300, bbox_inches='tight')
   plt.show()

   # === 绘制图 B (独立画板) ===
   print("  -> 绘制图 B (Convergence)...")
   max_iter = 80
   all_histories = []
   for i in range(50):
       model = GeochemicalNMF(V_norm, 5, info, init_strategy='random')
       hist = model.fit(max_iter=max_iter)
       if len(hist) < max_iter:
           hist += [hist[-1]] * (max_iter - len(hist))
       all_histories.append(hist)
   hist_arr = np.array(all_histories)
   mean_hist = np.mean(hist_arr, axis=0)
   std_hist = np.std(hist_arr, axis=0)

   plt.figure(figsize=(10, 7))  # 标准大方布局
   ax_b = plt.gca()
   ax_b.plot(mean_hist, color='blue', alpha=0.3, linewidth=1)
   idx = np.arange(0, max_iter, 8)
   ax_b.errorbar(idx, mean_hist[idx], yerr=std_hist[idx], fmt='o', color='blue', ecolor='red', elinewidth=2.5,
                 capsize=5, markersize=7, label='Loss Sample')
   ax_b.set_xlabel('Iterations', fontsize=14, fontweight='bold')
   ax_b.set_ylabel('Loss Function', fontsize=14, fontweight='bold', color='blue')
   ax_b.set_title('(b) Optimization Convergence', fontsize=18, pad=15)
   ax_b.legend(fontsize=12)
   plt.tight_layout()
   plt.savefig('Figure_B_Convergence.png', dpi=300, bbox_inches='tight')
   plt.show()


# ==========================================
# 4. 图C - 保持不变
# ==========================================
def plot_figure_C(mean_W, std_W, meta, k_val):
   print(f"\n[Step] 正在绘制 K={k_val} 的贡献图...")
   wells = meta['Well'].unique()
   display_wells = wells[:15] if len(wells) > 15 else wells
   idx_map = [meta[meta['Well'] == w].index[0] for w in display_wells]

   plt.figure(figsize=(14, 7))
   colors = sns.color_palette("bright", k_val)
   x = np.arange(len(display_wells))

   for j in range(k_val):
       vals = mean_W[idx_map, j]
       errs = std_W[idx_map, j] * 1.96  # 95% CI
       plt.errorbar(x + (j - k_val / 2) * 0.15, vals, yerr=errs, fmt='o',
                    color=colors[j], ecolor='red', elinewidth=2.0,
                    capsize=4, label=f'EM{j + 1}')

   plt.xlabel('Well ID', fontsize=14, fontweight='bold')
   plt.ylabel('Contribution', fontsize=14, fontweight='bold')
   plt.title(f'(c) End-Member Contributions (95% CI) - K={k_val}', fontsize=16)
   plt.xticks(x, display_wells, rotation=45)
   plt.legend()
   plt.tight_layout()
   plt.savefig(f'Result_Figure_C_K{k_val}.png', dpi=300)
   plt.show()


def plot_region_rc_bootstrap_ci(boot_W, meta, k_val, out_png=None):
    """
    Region尺度：RC by region (mean ± 95% CI), bootstrap
    boot_W: ndarray, shape = (n_boot, n_samples, k)
    meta: DataFrame with ['Region','Well']
    """
    if meta is None or len(meta) == 0 or ('Region' not in meta.columns):
        print("     [警告] meta 缺少 Region，无法生成 Figure C1")
        return
    if boot_W is None or len(boot_W) == 0:
        print("     [警告] boot_W 为空，无法生成 Figure C1")
        return

    boot_W = np.array(boot_W, dtype=float)
    n_boot, n_samples, kk = boot_W.shape
    if kk != k_val:
        print("     [警告] boot_W 的 K 与 k_val 不一致，无法生成 Figure C1")
        return

    # Region顺序（按出现顺序，也可以改成排序）
    regions = meta['Region'].astype(str).values
    uniq_regions = pd.unique(regions)

    # 每次bootstrap：先在region内取样本均值 -> 得到 (n_region, k)
    region_boot = []  # list of (n_region, k)
    for b in range(n_boot):
        Wb = boot_W[b]  # (n_samples, k)
        rows = []
        for rg in uniq_regions:
            idx = np.where(regions == rg)[0]
            rows.append(np.mean(Wb[idx, :], axis=0))
        region_boot.append(np.vstack(rows))
    region_boot = np.stack(region_boot, axis=0)  # (n_boot, n_region, k)

    # 均值 + 95%CI（分位数法）
    mean_rc = np.mean(region_boot, axis=0)                     # (n_region, k)
    low_rc  = np.quantile(region_boot, 0.025, axis=0)          # (n_region, k)
    high_rc = np.quantile(region_boot, 0.975, axis=0)          # (n_region, k)

    # 画图：k个子图
    fig, axes = plt.subplots(k_val, 1, figsize=(16, 4*k_val), sharex=True)
    if k_val == 1:
        axes = [axes]

    x = np.arange(len(uniq_regions))
    for j in range(k_val):
        ax = axes[j]
        y = mean_rc[:, j]
        yerr = np.vstack([y - low_rc[:, j], high_rc[:, j] - y])
        ax.bar(x, y)
        ax.errorbar(x, y, yerr=yerr, fmt='none', ecolor='black', capsize=3)
        ax.set_ylabel(f'EM{j+1} RC')
        ax.grid(True, alpha=0.2)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(uniq_regions, rotation=30, ha='right')
    axes[-1].set_xlabel('Region')

    fig.suptitle(f'Figure C1 (region). RC by region (mean ± 95% CI, bootstrap n={n_boot}, K={k_val})', y=0.995)
    plt.tight_layout()

    if out_png is None:
        out_png = f'Figure_C1_region_RC_bootstrap_n{n_boot}_K{k_val}.png'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  -> [保存成功] {out_png}")


# ==========================================
# 5. 辅助功能函数
# ==========================================
def optimize_with_pso_wrapped(data_matrix_imputed, n_components, info):
   print(f"  -> [PSO] 寻找 K={n_components} 的最佳参数...")

   def objective_function(params):
       reg_lambda, reg_beta = params
       model = GeochemicalNMF(data_matrix_imputed, n_components, info,
                              reg_lambda=reg_lambda, reg_beta=reg_beta)
       hist = model.fit(max_iter=40)
       return hist[-1]

   lb = [0.001, 0.001]
   ub = [0.5, 0.1]
   best_params, _ = pso(objective_function, lb, ub, swarmsize=10, maxiter=80, minfunc=1e-4)
   print(f"     PSO 最佳参数: Lambda={best_params[0]:.4f}, Beta={best_params[1]:.4f}")
   return best_params


# ==========================================
# 5.1 弦图（仅在这里做修改：稳定 + 按Region排序 + Region|Well 标签）
# ==========================================
def plot_chord_diagram_transplanted(W_matrix, meta_data, k_val, threshold=15.0):
   """
   稳定版弦图（Chord Diagram）——只改绘图，不改模型结果。

   改进点（不改变你现有模型/流程）：
   1) Well 节点按 Region -> Well 排序（更符合地质分区阅读）
   2) Well 节点显示为 "Region|Well"（一眼识别地区）
   3) nodes 使用整数 index（HoloViews Chord 最稳的数据模式，避免 DataError）
   4) 边颜色按端元 source_em；节点颜色按 group（EndMember vs Region）
   5) 保留原逻辑：按行归一化为百分比，阈值过滤（默认 >15%）
   """
   print(f"  -> 生成 K={k_val} 的弦图 (HTML, threshold>{threshold}%)...")

   # ---------- 0) 基础检查 ----------
   if meta_data is None or len(meta_data) == 0:
       print("     [警告] meta_data 为空，无法生成弦图。")
       return

   if 'Region' not in meta_data.columns or 'Well' not in meta_data.columns:
       print("     [警告] meta_data 缺少 Region/Well 列，无法生成弦图。")
       return

   if W_matrix is None:
       print("     [警告] W_matrix 为空，无法生成弦图。")
       return

   W_matrix = np.array(W_matrix, dtype=float)
   if W_matrix.shape[0] != meta_data.shape[0]:
       print("     [警告] W_matrix 行数与 meta_data 行数不一致，无法生成弦图。")
       return
   if W_matrix.shape[1] != k_val:
       print("     [警告] W_matrix 列数与 k_val 不一致，无法生成弦图。")
       return

   # ---------- 1) 排序：Region -> Well（只影响展示顺序，不改变结果数值）----------
   meta_sorted = meta_data.copy()
   meta_sorted['Region'] = meta_sorted['Region'].astype(str)
   meta_sorted['Well'] = meta_sorted['Well'].astype(str)

   if 'Region_Clean' in meta_sorted.columns:
       meta_sorted['Region_Clean'] = meta_sorted['Region_Clean'].astype(str)
       meta_sorted = meta_sorted.sort_values(by=['Region_Clean', 'Well']).reset_index(drop=True)
   else:
       meta_sorted = meta_sorted.sort_values(by=['Region', 'Well']).reset_index(drop=True)

   # W 也按同样排序重排（对齐 labels）
   # 用原始行号建立 order，避免“meta_sorted reset_index”后无法追溯
   idx_df = meta_data.copy()
   idx_df['__idx__'] = np.arange(len(idx_df))
   idx_df['Region'] = idx_df['Region'].astype(str)
   idx_df['Well'] = idx_df['Well'].astype(str)
   if 'Region_Clean' in idx_df.columns:
       idx_df['Region_Clean'] = idx_df['Region_Clean'].astype(str)
       idx_df = idx_df.sort_values(by=['Region_Clean', 'Well']).reset_index(drop=True)
   else:
       idx_df = idx_df.sort_values(by=['Region', 'Well']).reset_index(drop=True)
   order = idx_df['__idx__'].values
   W_sorted = W_matrix[order, :]

   # ---------- 2) 归一化为百分比 ----------
   row_sum = np.sum(W_sorted, axis=1, keepdims=True) + 1e-9
   W_norm = W_sorted / row_sum * 100.0

   # 标签：Well 用 Region|Well
   well_labels = (meta_sorted['Region'].astype(str) + "|" + meta_sorted['Well'].astype(str)).values
   region_labels = meta_sorted['Region'].astype(str).values
   em_labels = [f'EM{i + 1}' for i in range(k_val)]

   # ---------- 3) 构建 nodes：用整数 index（最稳）----------
   nodes = []
   name_to_idx = {}
   idx = 0

   # EM 节点
   for em in em_labels:
       name_to_idx[em] = idx
       nodes.append({'index': idx, 'name': em, 'group': 'EndMember'})
       idx += 1

   # Well 节点
   for wl, rg in zip(well_labels, region_labels):
       wl = str(wl)
       if wl not in name_to_idx:
           name_to_idx[wl] = idx
           nodes.append({'index': idx, 'name': wl, 'group': str(rg)})
           idx += 1

   df_nodes = pd.DataFrame(nodes)

   # ---------- 4) 构建 links：source/target 用 index ----------
   links = []
   for i in range(len(well_labels)):
       target_name = str(well_labels[i])
       t_idx = name_to_idx[target_name]

       for j in range(k_val):
           val = float(W_norm[i, j])
           if val > float(threshold):
               source_name = em_labels[j]
               s_idx = name_to_idx[source_name]
               links.append({
                   'source': s_idx,
                   'target': t_idx,
                   'value': val,
                   'source_em': source_name,
                   'target_well': target_name,
                   'region': str(region_labels[i])
               })

   df_links = pd.DataFrame(links)
   if df_links.empty:
       print(f"     [警告] 没有足够显著的连接 (>{threshold}%) 生成弦图。")
       return

   # ---------- 5) 生成 chord ----------
   nodes_ds = hv.Dataset(df_nodes, kdims=['index'], vdims=['name', 'group'])
   chord = hv.Chord((df_links, nodes_ds))

   chord = chord.opts(
       cmap='Category20',
       edge_color='source_em',                 # 边按端元着色
       edge_line_width=hv.dim('value') * 0.03, # 线宽随贡献变化
       node_color='group',                     # 节点按 Region/EndMember 分组着色
       labels='name',
       label_text_font_size='9pt',
       width=950,
       height=950,
       title=f'Geochemical Connectivity (K={k_val}, >{threshold:.0f}% links)',
       tools=['hover'],
       inspection_policy='edges'
   )

   output_file = f'Result_Chord_K{k_val}.html'
   hv.save(chord, output_file)
   print(f"     [保存成功] {output_file}")


class SimpleModel(BaseEstimator):
   def fit(self, X, y=None):
       return self

   def predict(self, X):
       return np.sum(X, axis=1)

   def score(self, X, y):
       return r2_score(y, self.predict(X))


def pfi_analysis_transplanted(H_real, feature_names, k_val):
   print(f"  -> 生成 K={k_val} 的 PFI 特征重要性图...")
   model = SimpleModel()
   y = model.predict(H_real)
   pfi = permutation_importance(model, H_real, y, n_repeats=30, random_state=42)
   plt.figure(figsize=(10, 6))
   plt.bar(range(len(pfi.importances_mean)), pfi.importances_mean, yerr=pfi.importances_std, color='teal', capsize=5)
   plt.xticks(range(len(pfi.importances_mean)), feature_names, rotation=45)
   plt.title(f'Feature Importance (PFI) - K={k_val}')
   plt.tight_layout()
   plt.savefig(f'Result_PFI_K{k_val}.png', dpi=300)
   plt.show()


# ==========================================
# 6. 主程序
# ==========================================
if __name__ == "__main__":
   data_file = '天然气四端元2.xlsx'

   # 1. 加载数据
   V_norm, meta, info = load_and_preprocess(data_file)

   if V_norm is not None:
       # 2. 生成图A和图B (现在是独立绘制，不会重叠)
       plot_separated_figures(V_norm, info)

       # 准备两个 Excel Writer
       writer_contrib = pd.ExcelWriter('Result_Contributions.xlsx', engine='openpyxl')
       writer_em = pd.ExcelWriter('Result_EndMembers.xlsx', engine='openpyxl')

       # 3. 循环计算 K=4, 5, 6
       target_ks = [4, 5, 6]

       for k in target_ks:
           print(f"\n{'=' * 40}")
           print(f"正在处理 K = {k}")
           print(f"{'=' * 40}")

           # (A) PSO 优化
           best_params = optimize_with_pso_wrapped(V_norm, k, info)

           # (B) Bootstrap
           print(f"  -> 正在进行 Bootstrap (n=50) 计算误差...")
           n_boot = 50
           boot_W = []

           base_model = GeochemicalNMF(V_norm, k, info, reg_lambda=best_params[0], reg_beta=best_params[1])
           base_model.fit(max_iter=50)
           base_H_order = base_model.H[:, 0].argsort()

           for _ in tqdm(range(n_boot), desc="Bootstrap"):
               noise = np.random.normal(0, 0.02, V_norm.shape)
               model = GeochemicalNMF(pd.DataFrame(V_norm.values + noise, columns=V_norm.columns),
                                      k, info, reg_lambda=best_params[0], reg_beta=best_params[1])
               model.fit(max_iter=50)
               sorted_idx = np.argsort(model.H[:, 0])
               boot_W.append(model.W[:, sorted_idx])

           boot_W = np.array(boot_W)
           mean_W = np.mean(boot_W, axis=0)
           std_W = np.std(boot_W, axis=0)
           plot_region_rc_bootstrap_ci(boot_W, meta, k)

           # (C) 绘制图 C
           plot_figure_C(mean_W, std_W, meta, k)

           # (D) 导出 Excel
           print(f"  -> 写入 Excel Sheet (K={k})...")
           contrib_rows = []
           wells = meta['Well'].values
           for i in range(len(wells)):
               well_name = wells[i]
               for j in range(k):
                   mean_val = mean_W[i, j]
                   std_val = std_W[i, j]
                   lower_val = max(0, mean_val - 1.96 * std_val)
                   upper_val = min(1, mean_val + 1.96 * std_val)
                   contrib_rows.append({
                       'Well': well_name,
                       'EndMember': f'EM{j + 1}',
                       'Mean_Ratio': mean_val,
                       'Lower_95CI': lower_val,
                       'Upper_95CI': upper_val
                   })
           df_contrib_sheet = pd.DataFrame(contrib_rows)
           df_contrib_sheet.to_excel(writer_contrib, sheet_name=f'K={k}', index=False)

           H_real = np.zeros((k, len(info['feature_names'])))
           H_sorted = base_model.H[base_H_order, :]
           for r in range(k):
               H_real[r, :] = H_sorted[r, :] * info['range'].values + info['min'].values
           df_em = pd.DataFrame(H_real, columns=info['feature_names'])
           df_em.insert(0, 'EndMember', [f'EM{i + 1}' for i in range(k)])
           df_em.to_excel(writer_em, sheet_name=f'K={k}', index=False)

           # (E) 弦图（函数名不变，主程序不改；需要更强阈值可改 threshold=20.0）
           plot_chord_diagram_transplanted(mean_W, meta, k)

           # (F) PFI 分析
           pfi_analysis_transplanted(H_real, info['feature_names'], k)

       writer_contrib.close()
       writer_em.close()

       print("\n所有任务 (K=4,5,6) 已全部完成！")
