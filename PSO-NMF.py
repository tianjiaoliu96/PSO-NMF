# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import random
import warnings
import json
import traceback
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy.optimize import nnls
from scipy.stats import iqr

try:
    from sklearn.impute import KNeighborsImputer
except ImportError:
    try:
        from sklearn.neighbors import KNeighborsImputer
    except ImportError:
        class SimpleKNNImputer:
            def __init__(self, n_neighbors=3):
                self.n_neighbors = n_neighbors
                
            def fit_transform(self, X):
                X = np.array(X, dtype=float)
                for i in range(X.shape[1]):
                    col = X[:, i]
                    missing = np.isnan(col)
                    if missing.any():
                        col_mean = np.nanmean(col)
                        col[missing] = col_mean
                return X
        
        KNeighborsImputer = SimpleKNNImputer
        warnings.warn("Using simple mean imputation as KNeighborsImputer substitute", UserWarning)


class PSONMFWithFractionation:
    """PSO-NMF algorithm incorporating migration fractionation effects."""
    
    def __init__(self, n_components: int = 5, 
                 lambda_reg: float = 0.01,
                 beta_sparsity: float = 0.01,
                 gamma_sum: float = 5.0,
                 eta_isotope: float = 1.0,
                 mu_smoothness: float = 1.0,
                 kappa_scale: float = 0.1,
                 max_iter: int = 200,
                 tol: float = 1e-6,
                 n_particles: int = 10,
                 n_generations: int = 80):
        
        self.n_components = n_components
        self.lambda_reg = lambda_reg
        self.beta_sparsity = beta_sparsity
        self.gamma_sum = gamma_sum
        self.eta_isotope = eta_isotope
        self.mu_smoothness = mu_smoothness
        self.kappa_scale = kappa_scale
        self.max_iter = max_iter
        self.tol = tol
        self.n_particles = n_particles
        self.n_generations = n_generations
        
        self.W = None
        self.H = None
        self.alpha = None
        self.delta13C12 = None
        self.f_delta13C12 = None
        
        self.loss_history = []
        self.pure_recon_error_history = []
    
    def _compute_f_delta13C12(self, delta13C12: np.ndarray) -> np.ndarray:
        delta13C12 = np.asarray(delta13C12)
        mean_delta = np.nanmean(delta13C12)
        std_delta = np.nanstd(delta13C12)
        if std_delta == 0 or np.isnan(std_delta):
            return np.zeros_like(delta13C12)
        return (delta13C12 - mean_delta) / std_delta
    
    def _compute_alpha_target(self, delta13C12: np.ndarray) -> np.ndarray:
        f_delta = self._compute_f_delta13C12(delta13C12)
        return 1.0 + self.kappa_scale * f_delta
    
    def _compute_psi_alpha(self, alpha: np.ndarray) -> np.ndarray:
        if self.f_delta13C12 is None:
            return np.ones_like(alpha)
        f_array = np.asarray(self.f_delta13C12)
        f_expanded = f_array[:, np.newaxis]
        return 1.0 + alpha * f_expanded
    
    def _initialize_parameters(self, V: np.ndarray, delta13C12: Optional[np.ndarray] = None) -> None:
        m, n = V.shape
        
        self.W = np.random.rand(m, self.n_components) * 0.5 + 0.1
        row_sums = self.W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.W = self.W / row_sums
        
        self.H = np.random.rand(self.n_components, n) * 0.5 + 0.1
        
        self.delta13C12 = delta13C12
        if delta13C12 is not None:
            delta13C12 = np.asarray(delta13C12)
            self.f_delta13C12 = self._compute_f_delta13C12(delta13C12)
            alpha_target = self._compute_alpha_target(delta13C12)
            
            self.alpha = np.zeros((m, self.n_components))
            for k in range(self.n_components):
                self.alpha[:, k] = alpha_target + np.random.randn(m) * 0.01
                self.alpha[:, k] = np.clip(self.alpha[:, k], 0.98, 1.02)
        else:
            self.alpha = np.ones((m, self.n_components)) * 1.0
            self.f_delta13C12 = None
        
        self.loss_history = []
        self.pure_recon_error_history = []
    
    def _compute_pure_reconstruction_error(self, V: np.ndarray, W: np.ndarray, H: np.ndarray, alpha: np.ndarray) -> float:
        psi_alpha = self._compute_psi_alpha(alpha)
        reconstruction = np.dot(W * psi_alpha, H)
        recon_error = np.linalg.norm(V - reconstruction, 'fro') ** 2
        return recon_error
    
    def _compute_loss(self, V: np.ndarray, W: np.ndarray, H: np.ndarray, alpha: np.ndarray) -> Tuple[float, Dict[str, float]]:
        m, n = V.shape
        
        psi_alpha = self._compute_psi_alpha(alpha)
        reconstruction = np.dot(W * psi_alpha, H)
        recon_error = np.linalg.norm(V - reconstruction, 'fro') ** 2
        
        l2_reg = self.lambda_reg * np.linalg.norm(W, 'fro') ** 2
        l1_reg = self.beta_sparsity * np.sum(np.abs(W))
        sum_constraint = self.gamma_sum * np.sum((np.sum(W, axis=1) - 1) ** 2)
        
        isotope_constraint = 0.0
        if self.delta13C12 is not None:
            alpha_target = self._compute_alpha_target(self.delta13C12)
            alpha_target_expanded = alpha_target[:, np.newaxis]
            isotope_constraint = self.eta_isotope * np.sum((alpha - alpha_target_expanded) ** 2)
        
        smoothness = 0.0
        if m > 1:
            for k in range(self.n_components):
                alpha_diff = alpha[1:, k] - alpha[:-1, k]
                smoothness += self.mu_smoothness * np.sum(alpha_diff ** 2)
        
        total_loss = recon_error + l2_reg + l1_reg + sum_constraint + isotope_constraint + smoothness
        
        return total_loss, {
            'recon_error': recon_error,
            'l2_reg': l2_reg,
            'l1_reg': l1_reg,
            'sum_constraint': sum_constraint,
            'isotope_constraint': isotope_constraint,
            'smoothness': smoothness
        }
    
    def _update_H(self, V: np.ndarray, W: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        K, n = self.H.shape
        H_new = np.zeros((K, n))
        
        psi_alpha = self._compute_psi_alpha(alpha)
        
        for j in range(n):
            A = W * psi_alpha
            b = V[:, j]
            
            try:
                h, _ = nnls(A, b)
                H_new[:, j] = h
            except Exception as e:
                h = self.H[:, j].copy()
                for _ in range(10):
                    grad = A.T @ (A @ h - b)
                    h -= 0.01 * grad
                    h = np.maximum(h, 0)
                H_new[:, j] = h
        
        return H_new
    
    def _update_W(self, V: np.ndarray, H: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        m, K = self.W.shape
        W_new = np.zeros((m, K))
        
        psi_alpha = self._compute_psi_alpha(alpha)
        
        for i in range(m):
            A = (H * psi_alpha[i, :].reshape(-1, 1)).T
            b = V[i, :]
            
            A_aug = np.vstack([A, np.sqrt(self.gamma_sum) * np.ones(K)])
            b_aug = np.append(b, np.sqrt(self.gamma_sum))
            
            try:
                w, _ = nnls(A_aug, b_aug)
                W_new[i, :] = w
            except Exception as e:
                w = self.W[i, :].copy()
                for _ in range(10):
                    grad = A.T @ (A @ w - b) + self.gamma_sum * 2 * (np.sum(w) - 1)
                    w -= 0.01 * grad
                    w = np.maximum(w, 0)
                W_new[i, :] = w
        
        return W_new
    
    def _update_alpha(self, V: np.ndarray, W: np.ndarray, H: np.ndarray) -> np.ndarray:
        m, K = self.alpha.shape
        alpha_new = self.alpha.copy()
        
        if self.delta13C12 is None:
            return alpha_new
        
        psi_alpha = self._compute_psi_alpha(self.alpha)
        f_array = np.asarray(self.f_delta13C12)
        f_expanded = f_array[:, np.newaxis]
        
        reconstruction = np.dot(W * psi_alpha, H)
        residual = V - reconstruction
        
        alpha_target = self._compute_alpha_target(self.delta13C12)
        alpha_target_expanded = alpha_target[:, np.newaxis]
        
        for k in range(K):
            grad_recon = -2 * np.sum(residual * (W[:, k:k+1] * f_expanded * H[k:k+1, :]), axis=1)
            grad_isotope = 2 * self.eta_isotope * (self.alpha[:, k] - alpha_target_expanded[:, 0])
            
            grad_smooth = np.zeros(m)
            if m > 1:
                grad_smooth[0] = 2 * (self.alpha[0, k] - self.alpha[1, k])
                grad_smooth[-1] = 2 * (self.alpha[-1, k] - self.alpha[-2, k])
                for i in range(1, m-1):
                    grad_smooth[i] = 2 * (2 * self.alpha[i, k] - self.alpha[i-1, k] - self.alpha[i+1, k])
                grad_smooth *= self.mu_smoothness
            
            grad_total = grad_recon + grad_isotope + grad_smooth
            alpha_new[:, k] = self.alpha[:, k] - 0.001 * grad_total
            alpha_new[:, k] = np.clip(alpha_new[:, k], 0.98, 1.02)
        
        return alpha_new
    
    def fit(self, V: np.ndarray, delta13C12: Optional[np.ndarray] = None) -> Dict[str, Any]:
        m, n = V.shape
        
        self._initialize_parameters(V, delta13C12)
        
        loss_history = []
        component_loss_history = []
        
        prev_loss = float('inf')
        for iteration in range(self.max_iter):
            self.H = self._update_H(V, self.W, self.alpha)
            self.W = self._update_W(V, self.H, self.alpha)
            
            if delta13C12 is not None:
                self.alpha = self._update_alpha(V, self.W, self.H)
            
            total_loss, loss_components = self._compute_loss(V, self.W, self.H, self.alpha)
            loss_history.append(total_loss)
            component_loss_history.append(loss_components)
            
            if iteration > 0 and abs(prev_loss - total_loss) < self.tol:
                print(f"  Converged at iteration {iteration+1}")
                break
            prev_loss = total_loss
        

        pure_recon_error = self._compute_pure_reconstruction_error(V, self.W, self.H, self.alpha)
        
        return {
            'W': self.W,
            'H': self.H,
            'alpha': self.alpha,
            'loss_history': loss_history,
            'loss_components': component_loss_history,
            'pure_recon_error': pure_recon_error,
            'n_iterations': iteration + 1
        }


class PSOOptimizer:
    """Particle Swarm Optimization for hyperparameter tuning."""
    
    def __init__(self, n_particles: int = 10, n_generations: int = 80,
                 lambda_range: Tuple[float, float] = (0.001, 0.5),
                 beta_range: Tuple[float, float] = (0.001, 0.1)):
        
        self.n_particles = n_particles
        self.n_generations = n_generations
        self.lambda_range = lambda_range
        self.beta_range = beta_range
        
        self.w = 0.729
        self.c1 = 1.49445
        self.c2 = 1.49445
        
    def optimize(self, V: np.ndarray, delta13C12: Optional[np.ndarray] = None,
                 K_values: List[int] = None) -> Dict[str, Any]:
        if K_values is None:
            K_values = [2, 3, 4, 5, 6, 7]
        
        results = {}
        
        for K in K_values:
            print(f"Optimizing for K={K}...")
            
            particles = []
            velocities = []
            personal_best = []
            personal_best_fitness = []
            
            for _ in range(self.n_particles):
                lambda_val = np.random.uniform(*self.lambda_range)
                beta_val = np.random.uniform(*self.beta_range)
                particles.append([lambda_val, beta_val])
                velocities.append([0.0, 0.0])
                personal_best.append([lambda_val, beta_val])
                personal_best_fitness.append(float('inf'))
            
            global_best = particles[0].copy()
            global_best_fitness = float('inf')
            
            prev_best_fitness = float('inf')
            
            for gen in range(self.n_generations):
                for i in range(self.n_particles):
                    lambda_val, beta_val = particles[i]
                    
                    model = PSONMFWithFractionation(
                        n_components=K,
                        lambda_reg=lambda_val,
                        beta_sparsity=beta_val,
                        max_iter=40,
                        tol=0
                    )
                    
                    try:
                        result = model.fit(V, delta13C12)
                        fitness = result['loss_history'][-1] if result['loss_history'] else float('inf')
                    except Exception as e:
                        print(f"Error in PSO for K={K}: {e}")
                        fitness = float('inf')
                    
                    if fitness < personal_best_fitness[i]:
                        personal_best[i] = particles[i].copy()
                        personal_best_fitness[i] = fitness
                    
                    if fitness < global_best_fitness:
                        global_best = particles[i].copy()
                        global_best_fitness = fitness
                
                if gen > 0 and abs(prev_best_fitness - global_best_fitness) < 1e-4:
                    print(f"  Converged at generation {gen}")
                    break
                prev_best_fitness = global_best_fitness
                
                for i in range(self.n_particles):
                    for d in range(2):
                        r1, r2 = random.random(), random.random()
                        velocities[i][d] = (self.w * velocities[i][d] +
                                           self.c1 * r1 * (personal_best[i][d] - particles[i][d]) +
                                           self.c2 * r2 * (global_best[d] - particles[i][d]))
                        
                        particles[i][d] += velocities[i][d]
                        
                        if d == 0:
                            particles[i][d] = np.clip(particles[i][d], *self.lambda_range)
                        else:
                            particles[i][d] = np.clip(particles[i][d], *self.beta_range)
            
            results[K] = {
                'best_lambda': global_best[0],
                'best_beta': global_best[1],
                'best_fitness': global_best_fitness,
                'all_fitness': personal_best_fitness
            }
        
        return results


def preprocess_data(data_df: pd.DataFrame, features: List[str],
                    delta13C1_col: str = 'd13C1', delta13C2_col: str = 'd13C2') -> Dict[str, Any]:
    """Preprocess gas geochemical data."""
    
    print("Performing KNN imputation (k=3)...")
    imputer = KNeighborsImputer(n_neighbors=3)
    
    data_for_imputation = data_df[features].copy()
    data_imputed_array = imputer.fit_transform(data_for_imputation)
    
    data_imputed = data_df.copy()
    for idx, col in enumerate(features):
        data_imputed[col] = data_imputed_array[:, idx]
    
    print("Clipping outliers (2.5th-97.5th percentile)...")
    for col in features:
        if col in data_imputed.columns:
            q_low = data_imputed[col].quantile(0.025)
            q_high = data_imputed[col].quantile(0.975)
            data_imputed[col] = data_imputed[col].clip(lower=q_low, upper=q_high)
    
    print("Applying min-max normalization...")
    normalization_params = {}
    data_normalized = data_imputed.copy()
    
    for col in features:
        if col in data_normalized.columns:
            min_val = data_normalized[col].min()
            max_val = data_normalized[col].max()
            range_val = max_val - min_val
            
            if range_val == 0:
                range_val = 1.0
            
            data_normalized[col] = (data_normalized[col] - min_val) / range_val
            
            normalization_params[col] = {
                'min': min_val,
                'max': max_val,
                'range': range_val
            }
    
    if delta13C1_col in data_normalized.columns and delta13C2_col in data_normalized.columns:
        delta13C12 = (data_normalized[delta13C1_col] - data_normalized[delta13C2_col]).values
    else:
        delta13C12 = None
    
    V = data_normalized[features].values
    
    return {
        'V': V,
        'delta13C12': delta13C12,
        'normalization_params': normalization_params,
        'data_normalized': data_normalized,
        'data_original': data_df
    }


def inverse_normalize_H(H_nmf: np.ndarray, normalization_params: Dict[str, Dict[str, float]],
                        features: List[str]) -> np.ndarray:
    K, n = H_nmf.shape
    H_rescaled = np.zeros_like(H_nmf)
    
    for j, feature in enumerate(features[:n]):
        if feature in normalization_params:
            min_val = normalization_params[feature]['min']
            range_val = normalization_params[feature]['range']
            
            if range_val == 0:
                range_val = 1.0
            
            H_rescaled[:, j] = H_nmf[:, j] * range_val + min_val
    
    return H_rescaled


def bootstrap_validation(model_class, V: np.ndarray, delta13C12: Optional[np.ndarray],
                         n_bootstrap: int = 50, n_components: int = 5) -> Dict[str, Any]:
    m, n = V.shape
    
    W_bootstrap = []
    H_bootstrap = []
    
    for b in range(n_bootstrap):
        if (b + 1) % 10 == 0:
            print(f"  Bootstrap iteration {b+1}/{n_bootstrap}")
        
        indices = np.random.choice(m, m, replace=True)
        V_boot = V[indices]
        delta13C12_boot = delta13C12[indices] if delta13C12 is not None else None
        
        try:
            model = model_class(n_components=n_components)
            result = model.fit(V_boot, delta13C12_boot)
            
            W_bootstrap.append(result['W'])
            H_bootstrap.append(result['H'])
            
        except Exception as e:
            print(f"  Bootstrap iteration {b+1} failed: {e}")
            continue
    
    if not W_bootstrap:
        raise ValueError("All bootstrap iterations failed")
    
    W_bootstrap = np.array(W_bootstrap)
    H_bootstrap = np.array(H_bootstrap)
    
    W_mean = np.mean(W_bootstrap, axis=0)
    W_std = np.std(W_bootstrap, axis=0)
    W_p2_5 = np.percentile(W_bootstrap, 2.5, axis=0)
    W_p97_5 = np.percentile(W_bootstrap, 97.5, axis=0)
    
    H_mean = np.mean(H_bootstrap, axis=0)
    H_std = np.std(H_bootstrap, axis=0)
    H_p2_5 = np.percentile(H_bootstrap, 2.5, axis=0)
    H_p97_5 = np.percentile(H_bootstrap, 97.5, axis=0)
    
    return {
        'W_mean': W_mean,
        'W_std': W_std,
        'W_p2_5': W_p2_5,
        'W_p97_5': W_p97_5,
        'H_mean': H_mean,
        'H_std': H_std,
        'H_p2_5': H_p2_5,
        'H_p97_5': H_p97_5,
        'W_bootstrap': W_bootstrap,
        'H_bootstrap': H_bootstrap
    }


def project_root() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name.lower() == "run":
        return p.parent.parent
    return Path.cwd()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def pick_existing(*candidates: Path) -> Path:
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("No candidate files found")


def infer_col(df: pd.DataFrame, preferred: List[str]) -> str:
    cols = {c.lower(): c for c in df.columns}
    for name in preferred:
        if name.lower() in cols:
            return cols[name.lower()]
    raise KeyError(f"None of {preferred} found in columns: {list(df.columns)}")


def em_sort_key(em: str) -> Tuple[int, str]:
    s = str(em).strip()
    digits = "".join([ch for ch in s if ch.isdigit()])
    if digits:
        return (int(digits), s)
    return (10**9, s)


def compute_curvature(k: np.ndarray, y: np.ndarray) -> np.ndarray:
    y1 = np.gradient(y, k)
    y2 = np.gradient(y1, k)
    return y2


def calculate_aic_correct(V: np.ndarray, pure_recon_error: float, K: int) -> float:
    """Correct calculation of AIC"""
    m, n = V.shape
    if m * n == 0 or pure_recon_error <= 0:
        return np.nan
    
    # Number of parameters
    # W: m*K, but with sum constraint reduces to m*(K-1)
    # H: K*n
    k_params = m * (K - 1) + K * n
    
    # AIC = 2k + n*log(RSS/n)
    aic_value = 2 * k_params + (m * n) * np.log(pure_recon_error / (m * n))
    
    return aic_value


def calculate_curvature_for_selection(k_values, rmse_values):
    """Calculate curvature of RMSE-K curve"""
    if len(k_values) < 3:
        return [np.nan] * len(k_values)
    
    curvature = np.zeros_like(rmse_values, dtype=float)
    for i in range(1, len(k_values)-1):
        h1 = k_values[i] - k_values[i-1]
        h2 = k_values[i+1] - k_values[i]
        if h1 == 0 or h2 == 0:
            curvature[i] = 0
        else:
            curvature[i] = (rmse_values[i+1]/h2 - rmse_values[i]*(1/h1+1/h2) + rmse_values[i-1]/h1) / ((h1+h2)/2)
    
    curvature[0] = curvature[1] if len(curvature) > 1 else 0
    curvature[-1] = curvature[-2] if len(curvature) > 1 else 0
    
    return curvature


def plot_k_selection(ax: plt.Axes, dfk: pd.DataFrame) -> None:
    k_col = infer_col(dfk, ["K"])
    rmse_col = infer_col(dfk, ["RMSE_mean", "rmse_mean", "RMSE"])
    aic_col = None
    for cand in ["AIC", "aic", "AIC_mean", "aic_mean"]:
        if cand in dfk.columns or cand.lower() in [c.lower() for c in dfk.columns]:
            aic_col = infer_col(dfk, [cand])
            break
    if aic_col is None:
        raise KeyError("AIC column not found in k_selection_summary.csv")
    
    band_lo_col = None
    band_hi_col = None
    for lo, hi in [("RMSE_p5", "RMSE_p95"), ("rmse_p5", "rmse_p95"), ("p5", "p95")]:
        if lo.lower() in [c.lower() for c in dfk.columns] and hi.lower() in [c.lower() for c in dfk.columns]:
            band_lo_col = infer_col(dfk, [lo])
            band_hi_col = infer_col(dfk, [hi])
            break
    rmse_std_col = None
    if band_lo_col is None:
        for cand in ["RMSE_std", "rmse_std", "std"]:
            if cand.lower() in [c.lower() for c in dfk.columns]:
                rmse_std_col = infer_col(dfk, [cand])
                break
    
    k = dfk[k_col].to_numpy(dtype=float)
    rmse = dfk[rmse_col].to_numpy(dtype=float)
    aic = dfk[aic_col].to_numpy(dtype=float)
    
    # Calculate curvature if not already in dataframe
    if 'curvature' in dfk.columns:
        curvature = dfk['curvature'].to_numpy(dtype=float)
    else:
        curvature = calculate_curvature_for_selection(k, rmse)
    
    ax1 = ax
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    
    c_rmse = "tab:blue"
    c_curv = "red"
    c_aic = "green"
    
    line_rmse, = ax1.plot(k, rmse, marker="o", linewidth=2.5, color=c_rmse, label="RMSE (Left)")
    if band_lo_col is not None and band_hi_col is not None:
        lo = dfk[band_lo_col].to_numpy(dtype=float)
        hi = dfk[band_hi_col].to_numpy(dtype=float)
        ax1.fill_between(k, lo, hi, color=c_rmse, alpha=0.18, label="_nolegend_")
    elif rmse_std_col is not None:
        std = dfk[rmse_std_col].to_numpy(dtype=float)
        ax1.fill_between(k, rmse - std, rmse + std, color=c_rmse, alpha=0.18, label="_nolegend_")
    
    line_curv, = ax2.plot(k, curvature, marker="D", linestyle="--", linewidth=2.0, color=c_curv, label="Curvature (Right-1)")
    
    line_aic, = ax3.plot(k, aic, marker="s", linestyle=":", linewidth=2.5, color=c_aic, label="AIC Trend (Right-2)")
    
    ax1.set_ylabel("RMSE", color=c_rmse, fontsize=14)
    ax1.tick_params(axis="y", colors=c_rmse)
    ax2.set_ylabel("Curvature", color=c_curv, fontsize=14)
    ax2.tick_params(axis="y", colors=c_curv)
    ax3.set_ylabel("AIC Criterion", color=c_aic, fontsize=14)
    ax3.tick_params(axis="y", colors=c_aic)
    
    ax1.set_xlabel("Number of End-members (K)", fontsize=14)
    ax1.set_xticks(k.astype(int))
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.set_title("(a) Multi-Criteria for K Selection", fontsize=18, pad=12)
    
    handles = [line_rmse, line_curv, line_aic]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), frameon=True)


def summarize_convergence(dfB: pd.DataFrame, trim_q: float = 0.10) -> pd.DataFrame:
    gen_col = infer_col(dfB, ["gen", "generation", "iter", "iteration"])
    loss_col = infer_col(dfB, ["best_total", "best_loss", "loss", "total"])
    
    rows = []
    for gen, sub in dfB.groupby(gen_col):
        vals = pd.to_numeric(sub[loss_col], errors="coerce").dropna().to_numpy(dtype=float)
        n = len(vals)
        if n == 0:
            continue
        
        lo = np.quantile(vals, trim_q)
        hi = np.quantile(vals, 1 - trim_q)
        vals_trim = vals[(vals >= lo) & (vals <= hi)]
        if len(vals_trim) >= 3:
            m = float(np.mean(vals_trim))
            s = float(np.std(vals_trim, ddof=1))
            n_eff = len(vals_trim)
        else:
            m = float(np.mean(vals))
            s = float(np.std(vals, ddof=1)) if n >= 2 else 0.0
            n_eff = n
        
        se = s / math.sqrt(max(n_eff, 1))
        rows.append((float(gen), m, s, n_eff, se))
    
    g = pd.DataFrame(rows, columns=["gen", "mean", "std_trim", "n_eff", "se"]).sort_values("gen")
    return g


def plot_convergence_style(ax, dfB, step: int = 10,
                           trim_q: float = 0.20,
                           se_cap: float = 0.8):
    g = summarize_convergence(dfB, trim_q=trim_q)
    
    g = g[g["gen"].astype(int) % step == 0].copy()
    g = g.sort_values("gen")
    
    x = g["gen"].to_numpy(dtype=float)
    y = g["mean"].to_numpy(dtype=float)
    se = g["se"].to_numpy(dtype=float)
    
    se = np.minimum(se, se_cap)
    
    ax.plot(x, y, linewidth=2.8, color="tab:blue", alpha=0.95)
    
    ax.errorbar(
        x, y, yerr=se,
        fmt="o", markersize=7,
        markerfacecolor="tab:blue", markeredgecolor="tab:blue",
        ecolor="tab:orange", color="tab:orange",
        elinewidth=2.0, capsize=4, capthick=2.0
    )
    
    ax.set_title("(b) Optimization Convergence", fontsize=18, pad=10)
    ax.set_xlabel("Iterations", fontsize=14)
    ax.set_ylabel("Loss Function", fontsize=14, color="tab:blue")
    
    ax.tick_params(axis="y", which="both", left=True, labelleft=True, colors="tab:blue", labelsize=12)
    ax.tick_params(axis="x", labelsize=12)
    
    ax.set_ylim(37, 40)
    ax.set_yticks(np.arange(37, 41, 1))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
    
    ax.grid(True, linestyle="--", alpha=0.35)
    
    handle = Line2D(
        [0], [0],
        color="tab:orange", linewidth=2.2,
        marker="o", markersize=7,
        markerfacecolor="tab:blue", markeredgecolor="tab:blue"
    )
    ax.legend([handle], ["Loss Sample"], loc="upper right", frameon=True)


def plot_convergence_50runs(dfB: pd.DataFrame, out_png: Path) -> None:
    gen_col = infer_col(dfB, ["gen", "generation", "iter", "iteration"])
    run_col = infer_col(dfB, ["run_id", "run", "seed"])
    loss_col = infer_col(dfB, ["best_total", "best_loss", "loss", "total"])
    
    gens = np.sort(dfB[gen_col].unique())
    runs = np.sort(dfB[run_col].unique())
    
    fig, ax = plt.subplots(figsize=(16, 9))
    
    for rid in runs:
        d = dfB[dfB[run_col] == rid].sort_values(gen_col)
        ax.plot(d[gen_col], d[loss_col], linewidth=1.2, alpha=0.22)
    
    g = (
        dfB.groupby(gen_col)[loss_col]
        .agg(mean="mean", std=lambda x: np.std(x, ddof=1))
        .reset_index()
        .sort_values(gen_col)
    )
    ax.plot(g[gen_col], g["mean"], linewidth=3.0, alpha=0.9)
    ax.fill_between(g[gen_col], g["mean"] - g["std"], g["mean"] + g["std"], alpha=0.18)
    
    ax.set_title("Figure B. MOPSO convergence trajectories (50 independent runs)", fontsize=18, pad=10)
    ax.set_xlabel("Generation", fontsize=14)
    ax.set_ylabel("Best loss (min total in archive)", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.35)
    
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_rc_by_region(df_region: pd.DataFrame, out_png: Path) -> None:
    gcol = infer_col(df_region, ["group", "region"])
    ecol = infer_col(df_region, ["em", "endmember"])
    mcol = infer_col(df_region, ["mean"])
    lcol = infer_col(df_region, ["p2_5", "p025", "ci_low", "low"])
    ucol = infer_col(df_region, ["p97_5", "p975", "ci_high", "high"])
    
    agg_df = df_region.groupby([gcol, ecol]).agg({
        mcol: 'mean',
        lcol: 'mean',
        ucol: 'mean'
    }).reset_index()
    
    groups = sorted(agg_df[gcol].astype(str).unique())
    ems = sorted(agg_df[ecol].astype(str).unique(), key=em_sort_key)
    
    n_em = len(ems)
    fig_h = 2.2 * n_em + 2.0
    fig, axes = plt.subplots(n_em, 1, figsize=(18, fig_h), sharex=True)
    
    if n_em == 1:
        axes = [axes]
    
    x = np.arange(len(groups))
    for i, em in enumerate(ems):
        ax = axes[i]
        sub = agg_df[agg_df[ecol].astype(str) == em].copy()
        sub[gcol] = sub[gcol].astype(str)
        
        sub = sub.set_index(gcol).reindex(groups).reset_index()
        mean = sub[mcol].to_numpy(dtype=float)
        lo = sub[lcol].to_numpy(dtype=float)
        hi = sub[ucol].to_numpy(dtype=float)
        yerr = np.vstack([mean - lo, hi - mean])
        
        ax.bar(x, mean, alpha=0.85)
        ax.errorbar(x, mean, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.4, capsize=3)
        
        ax.set_ylabel(f"{em} RC", fontsize=12)
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(groups, rotation=25, ha="right", fontsize=10)
    axes[-1].set_xlabel("Region", fontsize=13)
    fig.suptitle("Figure C1 (region). RC by region (mean ± 95% CI, bootstrap n=50, K=5)", fontsize=18, y=0.995)
    
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_rc_by_well_per_em(df_well: pd.DataFrame, out_dir: Path) -> List[Path]:
    gcol = infer_col(df_well, ["group", "well"])
    ecol = infer_col(df_well, ["em", "endmember"])
    mcol = infer_col(df_well, ["mean"])
    lcol = infer_col(df_well, ["p2_5", "p025", "ci_low", "low"])
    ucol = infer_col(df_well, ["p97_5", "p975", "ci_high", "high"])
    
    ems = sorted(df_well[ecol].astype(str).unique(), key=em_sort_key)
    outputs = []
    
    for em in ems:
        sub = df_well[df_well[ecol].astype(str) == em].copy()
        sub[gcol] = sub[gcol].astype(str)
        sub = sub.sort_values(mcol, ascending=False)
        
        wells = sub[gcol].tolist()
        mean = sub[mcol].to_numpy(dtype=float)
        lo = sub[lcol].to_numpy(dtype=float)
        hi = sub[ucol].to_numpy(dtype=float)
        
        xerr = np.vstack([mean - lo, hi - mean])
        
        n = len(wells)
        height = max(12.0, min(32.0, n * 0.18))
        fig, ax = plt.subplots(figsize=(10.5, height))
        
        y = np.arange(n)
        ax.errorbar(
            mean, y, xerr=xerr,
            fmt="o", markersize=3.2,
            elinewidth=1.2, capsize=2
        )
        ax.set_yticks(y)
        ax.set_yticklabels(wells, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("RC (mean with 95% CI)", fontsize=12)
        ax.set_ylabel("Well", fontsize=12)
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
        ax.set_title(f"Figure C1 (well) — RC mean + 95% CI (K=5) — {em}", fontsize=14, pad=10)
        
        out_png = out_dir / f"FigureC1_RC_by_well_95CI_EM{em}.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        outputs.append(out_png)
    
    return outputs


def plot_endmember_profiles(dfH: pd.DataFrame, out_png: Path) -> None:
    ecol = infer_col(dfH, ["em", "endmember"])
    fcol = infer_col(dfH, ["feat", "feature", "variable"])
    mcol = infer_col(dfH, ["mean"])
    lcol = infer_col(dfH, ["p2_5", "p025", "ci_low", "low"])
    ucol = infer_col(dfH, ["p97_5", "p975", "ci_high", "high"])
    
    dfH = dfH.copy()
    dfH[ecol] = dfH[ecol].astype(str)
    dfH[fcol] = dfH[fcol].astype(str)
    
    ems = sorted(dfH[ecol].unique(), key=em_sort_key)
    
    first_em = ems[0]
    feat_order = dfH[dfH[ecol] == first_em][fcol].tolist()
    if len(feat_order) == 0:
        feat_order = sorted(dfH[fcol].unique())
    
    feat_order_unique = []
    seen = set()
    for ft in feat_order:
        if ft not in seen:
            feat_order_unique.append(ft)
            seen.add(ft)
    for ft in dfH[fcol].unique():
        if ft not in seen:
            feat_order_unique.append(ft)
            seen.add(ft)
    
    n_em = len(ems)
    fig_h = 2.2 * n_em + 2.0
    fig, axes = plt.subplots(n_em, 1, figsize=(18, fig_h), sharex=True)
    
    if n_em == 1:
        axes = [axes]
    
    x = np.arange(len(feat_order_unique))
    
    for i, em in enumerate(ems):
        ax = axes[i]
        sub = dfH[dfH[ecol] == em].copy()
        sub = sub.set_index(fcol).reindex(feat_order_unique).reset_index()
        
        mean = sub[mcol].to_numpy(dtype=float)
        lo = sub[lcol].to_numpy(dtype=float)
        hi = sub[ucol].to_numpy(dtype=float)
        
        mask = np.isfinite(mean) & np.isfinite(lo) & np.isfinite(hi)
        
        yerr = np.vstack([(mean - lo), (hi - mean)])
        
        ax.errorbar(
            x[mask], mean[mask],
            yerr=yerr[:, mask],
            fmt="o",
            markersize=4,
            elinewidth=1.2,
            capsize=3,
        )
        
        ax.set_ylabel(em, fontsize=12)
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(feat_order_unique, rotation=25, ha="right", fontsize=10)
    axes[-1].set_xlabel("Feature (rescaled back to original units)", fontsize=13)
    fig.suptitle(
        "Figure C2. Endmember profiles (original units, mean ± 95% CI, bootstrap n=200, K=5)",
        fontsize=18, y=0.995
    )
    
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    print("=" * 80)
    print("PSO-NMF Natural Gas Mixing Analysis")
    print("=" * 80)
    
    current_dir = Path.cwd()
    data_file = current_dir / "python(20260109).xlsx"
    
    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        print("Please ensure 'python(20260109).xlsx' is in the current directory")
        return
    
    print(f"Data file located: {data_file}")
    
    print("\n" + "-" * 40)
    print("Step 1: Reading Excel file")
    print("-" * 40)
    
    try:
        excel_file = pd.ExcelFile(data_file)
        print(f"Worksheets in Excel file: {excel_file.sheet_names}")
        
        sheet_name = excel_file.sheet_names[0]
        print(f"Reading worksheet: {sheet_name}")
        
        df = pd.read_excel(data_file, sheet_name=sheet_name)
        print(f"Data shape: {df.shape}")
        print(f"Column names: {list(df.columns)}")
        
        print("\nData preview:")
        print(df.head())
        
        print("\nData types:")
        print(df.dtypes)
        
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        traceback.print_exc()
        return
    
    print("\n" + "-" * 40)
    print("Step 2: Data preprocessing")
    print("-" * 40)
    
    potential_features = [
        'd13C1', 'd13C2', 'd13C3', 'd13C4',
        'C1', 'C2', 'C3', 'C4',
        'C1/C2', 'C1/C3', 'C2/C3',
        'CH4', 'C2H6', 'C3H8', 'iC4H10', 'nC4H10',
        'δ13C1', 'δ13C2', 'δ13C3',
        'C1_%', 'C2_%', 'C3_%',
    ]
    
    available_features = [col for col in potential_features if col in df.columns]
    
    if not available_features:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        available_features = numeric_cols[:10]
        print(f"No predefined features found, using numeric columns: {available_features}")
    else:
        print(f"Selected features: {available_features}")
    
    delta13C1_col = None
    delta13C2_col = None
    
    for col in ['d13C1', 'δ13C1', 'd13C_1', 'delta13C1']:
        if col in df.columns:
            delta13C1_col = col
            break
    
    for col in ['d13C2', 'δ13C2', 'd13C_2', 'delta13C2']:
        if col in df.columns:
            delta13C2_col = col
            break
    
    print(f"Methane carbon isotope column: {delta13C1_col}")
    print(f"Ethane carbon isotope column: {delta13C2_col}")
    
    try:
        preprocessed = preprocess_data(
            df, 
            features=available_features,
            delta13C1_col=delta13C1_col,
            delta13C2_col=delta13C2_col
        )
        
        V = preprocessed['V']
        delta13C12 = preprocessed['delta13C12']
        normalization_params = preprocessed['normalization_params']
        data_normalized = preprocessed['data_normalized']
        
        print(f"Preprocessed data matrix shape: {V.shape}")
        print(f"Δ13C12 data shape: {delta13C12.shape if delta13C12 is not None else 'Not available'}")
        
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        traceback.print_exc()
        return
    
    print("\n" + "-" * 40)
    print("Step 3: PSO optimizer - Selecting optimal endmember count K")
    print("-" * 40)
    
    outputs_dir = current_dir / "outputs"
    tables_dir = outputs_dir / "tables"
    figures_dir = outputs_dir / "figures"
    
    ensure_dir(outputs_dir)
    ensure_dir(tables_dir)
    ensure_dir(figures_dir)
    
    K_values = [2, 3, 4, 5, 6, 7]
    
    pso_optimizer = PSOOptimizer(
        n_particles=10,
        n_generations=80,
        lambda_range=(0.001, 0.5),
        beta_range=(0.001, 0.1)
    )
    
    print("Starting PSO optimization...")
    pso_results = pso_optimizer.optimize(V, delta13C12, K_values)
    
    k_selection_data = []
    for K in K_values:
        if K in pso_results:
            result = pso_results[K]
            
            # Run final model with best parameters to get pure reconstruction error
            model = PSONMFWithFractionation(
                n_components=K,
                lambda_reg=result['best_lambda'],
                beta_sparsity=result['best_beta'],
                max_iter=100
            )
            
            try:
                final_result = model.fit(V, delta13C12)
                pure_recon_error = final_result['pure_recon_error']
                
                # Calculate RMSE
                m, n = V.shape
                rmse_value = np.sqrt(pure_recon_error / (m * n))
                
                # Calculate AIC
                aic_value = calculate_aic_correct(V, pure_recon_error, K)
                
            except Exception as e:
                print(f"Error calculating final metrics for K={K}: {e}")
                rmse_value = np.sqrt(result['best_fitness'] / V.size) if result['best_fitness'] != float('inf') else np.nan
                aic_value = np.nan
            
            k_selection_data.append({
                'K': K,
                'best_lambda': result['best_lambda'],
                'best_beta': result['best_beta'],
                'best_fitness': result['best_fitness'],
                'RMSE_mean': rmse_value,
                'AIC': aic_value
            })
    
    k_selection_df = pd.DataFrame(k_selection_data)
    
    # Sort and calculate curvature
    k_selection_df = k_selection_df.sort_values('K').reset_index(drop=True)
    
    valid_df = k_selection_df.dropna(subset=['RMSE_mean']).copy()
    
    if len(valid_df) >= 3:
        k_values = valid_df['K'].values.astype(float)
        rmse_values = valid_df['RMSE_mean'].values.astype(float)
        curvature = calculate_curvature_for_selection(k_values, rmse_values)
        valid_df['curvature'] = curvature
    else:
        print(f"Warning: Not enough valid K values ({len(valid_df)}), cannot calculate curvature")
        valid_df['curvature'] = np.nan
    
    k_selection_df = k_selection_df.merge(
        valid_df[['K', 'curvature']], 
        on='K', 
        how='left'
    )
    
    k_selection_path = tables_dir / "k_selection_summary.csv"
    k_selection_df.to_csv(k_selection_path, index=False)
    
    print(f"K selection results saved to: {k_selection_path}")
    print("\nK selection summary:")
    print(k_selection_df.to_string(index=False))
    
    if not k_selection_df.empty and 'RMSE_mean' in k_selection_df.columns:
        valid_rows = k_selection_df.dropna(subset=['RMSE_mean'])
        if not valid_rows.empty:
            best_row = valid_rows.loc[valid_rows['RMSE_mean'].idxmin()]
            best_K = int(best_row['K'])
            best_lambda = best_row['best_lambda']
            best_beta = best_row['best_beta']
        else:
            best_row = k_selection_df.iloc[0]
            best_K = int(best_row['K'])
            best_lambda = best_row['best_lambda']
            best_beta = best_row['best_beta']
            print(f"Warning: All valid RMSE values are NaN, using first K value: K={best_K}")
    else:
        best_K = 5
        best_lambda = 0.01
        best_beta = 0.01
        print(f"\nUsing default values: K={best_K}, lambda={best_lambda}, beta={best_beta}")
    
    print(f"\nOptimal endmember count K: {best_K}")
    print(f"Optimal lambda value: {best_lambda:.6f}")
    print(f"Optimal beta value: {best_beta:.6f}")
    print(f"Optimal RMSE: {best_row['RMSE_mean']:.6f}" if 'best_row' in locals() else "")
    
    print("\n" + "-" * 40)
    print(f"Step 4: Bootstrap validation (K={best_K}, n=50)")
    print("-" * 40)
    
    try:
        if delta13C12 is not None:
            print("Bootstrap validation with Δ13C12 data...")
            bootstrap_results = bootstrap_validation(
                PSONMFWithFractionation, 
                V, 
                delta13C12,
                n_bootstrap=50,
                n_components=best_K
            )
        else:
            print("Bootstrap validation without isotope data...")
            bootstrap_results = bootstrap_validation(
                PSONMFWithFractionation, 
                V, 
                None,
                n_bootstrap=50,
                n_components=best_K
            )
        print("Bootstrap validation completed successfully")
    except Exception as e:
        print(f"Error during bootstrap validation: {e}")
        traceback.print_exc()
        return
    
    print("\n" + "-" * 40)
    print("Step 5: Saving bootstrap results")
    print("-" * 40)
    
    region_col = None
    well_col = None
    
    for col in ['Region', 'region', 'area']:
        if col in df.columns:
            region_col = col
            break
    
    for col in ['Well']:
        if col in df.columns:
            well_col = col
            break
    
    # Create sample indices for region and well data
    # If region/well columns don't exist, create dummy ones for visualization
    if region_col and region_col in df.columns:
        regions = df[region_col].astype(str).tolist()
    else:
        regions = [f"Sample_{i+1}" for i in range(len(df))]
        region_col = 'Region'
        df[region_col] = regions
    
    if well_col and well_col in df.columns:
        wells = df[well_col].astype(str).tolist()
    else:
        wells = [f"Well_{i+1}" for i in range(len(df))]
        well_col = 'Well'
        df[well_col] = wells
    
    # Save region RC data
    region_rc_data = []
    for i, region in enumerate(regions):
        for k in range(best_K):
            region_rc_data.append({
                'group': region,
                'em': f'EM{k+1}',
                'mean': bootstrap_results['W_mean'][i, k],
                'p2_5': bootstrap_results['W_p2_5'][i, k],
                'p97_5': bootstrap_results['W_p97_5'][i, k]
            })
    
    region_rc_df = pd.DataFrame(region_rc_data)
    region_rc_path = tables_dir / f"bootstrap_full_rc_ci_region_k{best_K}_n50.csv"
    region_rc_df.to_csv(region_rc_path, index=False)
    print(f"Region RC data saved to: {region_rc_path}")
    
    # Save well RC data
    well_rc_data = []
    for i, well in enumerate(wells):
        for k in range(best_K):
            well_rc_data.append({
                'group': well,
                'em': f'EM{k+1}',
                'mean': bootstrap_results['W_mean'][i, k],
                'p2_5': bootstrap_results['W_p2_5'][i, k],
                'p97_5': bootstrap_results['W_p97_5'][i, k]
            })
    
    well_rc_df = pd.DataFrame(well_rc_data)
    well_rc_path = tables_dir / f"bootstrap_full_rc_ci_well_k{best_K}_n50.csv"
    well_rc_df.to_csv(well_rc_path, index=False)
    print(f"Well RC data saved to: {well_rc_path}")
    
    # Rescale H matrix
    H_rescaled = inverse_normalize_H(
        bootstrap_results['H_mean'],
        normalization_params,
        available_features
    )
    
    # Save endmember feature data
    H_data = []
    for k in range(best_K):
        for j, feature in enumerate(available_features[:H_rescaled.shape[1]]):
            H_rescaled_p2_5 = inverse_normalize_H(
                bootstrap_results['H_p2_5'],
                normalization_params,
                available_features
            )
            H_rescaled_p97_5 = inverse_normalize_H(
                bootstrap_results['H_p97_5'],
                normalization_params,
                available_features
            )
            
            H_data.append({
                'em': f'EM{k+1}',
                'feat': feature,
                'mean': H_rescaled[k, j],
                'p2_5': H_rescaled_p2_5[k, j],
                'p97_5': H_rescaled_p97_5[k, j]
            })
    
    H_df = pd.DataFrame(H_data)
    H_path = tables_dir / f"bootstrap_full_H_ci_rescaled_k{best_K}_n50.csv"
    H_df.to_csv(H_path, index=False)
    print(f"Endmember feature data saved to: {H_path}")
    
    print("\n" + "-" * 40)
    print("Step 6: Generating PSO convergence data")
    print("-" * 40)
    
    convergence_data = []
    for K in K_values:
        if K in pso_results:
            lambda_val = pso_results[K]['best_lambda']
            beta_val = pso_results[K]['best_beta']
            
            model = PSONMFWithFractionation(
                n_components=K,
                lambda_reg=lambda_val,
                beta_sparsity=beta_val,
                max_iter=100
            )
            
            try:
                result = model.fit(V, delta13C12)
                
                for iter_num, loss in enumerate(result['loss_history']):
                    convergence_data.append({
                        'run_id': K,
                        'gen': iter_num,
                        'best_total': loss
                    })
            except Exception as e:
                print(f"Error generating convergence data for K={K}: {e}")
                continue
    
    convergence_df = pd.DataFrame(convergence_data)
    convergence_path = tables_dir / "FigureB_mopso_convergence_50runs.csv"
    convergence_df.to_csv(convergence_path, index=False)
    print(f"Convergence data saved to: {convergence_path}")
    
    print("\n" + "-" * 40)
    print("Step 7: Generating visualization plots")
    print("-" * 40)
    
    try:
        # Read the CSV files for plotting
        dfk = pd.read_csv(k_selection_path)
        dfB = pd.read_csv(convergence_path)
        df_region = pd.read_csv(region_rc_path)
        df_well = pd.read_csv(well_rc_path)
        dfH = pd.read_csv(H_path)
        
        print("Generating Figure 1: K selection and convergence plot...")
        fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(16, 11))
        plot_k_selection(ax_a, dfk)
        plot_convergence_style(ax_b, dfB)
        fig.tight_layout()
        out_combo = figures_dir / f"Figure_KSelection_and_Convergence_style_K{best_K}.png"
        fig.savefig(out_combo, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved to: {out_combo}")
        
        print("Generating Figure 2: MOPSO convergence trajectories...")
        out_B = figures_dir / f"FigureB_MOPSO_Convergence_50runs_K{best_K}.png"
        plot_convergence_50runs(dfB, out_B)
        print(f"  Saved to: {out_B}")
        
        print("Generating Figure 3: RC distribution by region...")
        out_C1r = figures_dir / f"FigureC1_RC_by_region_95CI_boot50_K{best_K}.png"
        plot_rc_by_region(df_region, out_C1r)
        print(f"  Saved to: {out_C1r}")
        
        print("Generating Figure 4: RC distribution by well...")
        well_plots = plot_rc_by_well_per_em(df_well, figures_dir)
        print(f"  Generated {len(well_plots)} well RC distribution plots")
        
        print("Generating Figure 5: Endmember feature profiles...")
        out_C2 = figures_dir / f"FigureC2_Endmember_profiles_95CI_boot50_K{best_K}.png"
        plot_endmember_profiles(dfH, out_C2)
        print(f"  Saved to: {out_C2}")
        
        # Generate additional diagnostic plots
        print("Generating additional diagnostic plots...")
        
        # RMSE vs K plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(dfk['K'], dfk['RMSE_mean'], 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of End-members (K)', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=12)
        ax.set_title(f'RMSE vs K (Best K={best_K})', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=best_K, color='red', linestyle='--', alpha=0.5)
        out_rmse = figures_dir / f"Diagnostic_RMSE_vs_K_K{best_K}.png"
        fig.savefig(out_rmse, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        # AIC vs K plot
        if 'AIC' in dfk.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(dfk['K'], dfk['AIC'], 's-', linewidth=2, markersize=8, color='green')
            ax.set_xlabel('Number of End-members (K)', fontsize=12)
            ax.set_ylabel('AIC', fontsize=12)
            ax.set_title(f'AIC vs K', fontsize=14)
            ax.grid(True, alpha=0.3)
            out_aic = figures_dir / f"Diagnostic_AIC_vs_K_K{best_K}.png"
            fig.savefig(out_aic, dpi=200, bbox_inches='tight')
            plt.close(fig)
        
        print("All plots generated successfully!")
        
    except Exception as e:
        print(f"Error generating plots: {e}")
        traceback.print_exc()
        # Try to generate at least some plots even if others fail
        try:
            # Try to generate basic plots
            if 'dfk' in locals():
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(dfk['K'], dfk['RMSE_mean'], 'o-')
                ax.set_xlabel('K')
                ax.set_ylabel('RMSE')
                ax.set_title('RMSE vs K')
                ax.grid(True)
                fig.savefig(figures_dir / "Basic_RMSE_vs_K.png", dpi=150)
                plt.close(fig)
                print("Basic RMSE plot generated")
        except:
            pass
    
    print("\n" + "-" * 40)
    print("Step 8: Saving complete analysis results")
    print("-" * 40)
    
    summary = {
        'data_file': str(data_file),
        'data_shape': f"{df.shape[0]} rows × {df.shape[1]} columns",
        'features_used': len(available_features),
        'optimal_endmember_count': best_K,
        'optimal_regularization_parameters': {
            'lambda': float(best_lambda),
            'beta': float(best_beta)
        },
        'model_performance': {
            'best_rmse': float(best_row['RMSE_mean']) if 'best_row' in locals() else np.nan,
            'aic': float(best_row['AIC']) if 'best_row' in locals() and 'AIC' in best_row else np.nan
        },
        'output_files': {
            'k_selection_results': str(k_selection_path),
            'bootstrap_convergence_data': str(convergence_path),
            'endmember_feature_data': str(H_path),
            'region_rc_data': str(region_rc_path),
            'well_rc_data': str(well_rc_path)
        },
        'generated_figures': []
    }
    
    # Add generated figure paths if they exist
    if 'out_combo' in locals():
        summary['generated_figures'].append(str(out_combo))
    if 'out_B' in locals():
        summary['generated_figures'].append(str(out_B))
    if 'out_C1r' in locals():
        summary['generated_figures'].append(str(out_C1r))
    if 'out_C2' in locals():
        summary['generated_figures'].append(str(out_C2))
    
    summary_path = outputs_dir / "analysis_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"Analysis summary saved to: {summary_path}")
    
    print("\n" + "=" * 80)
    print("Analysis completed!")
    print("=" * 80)
    
    print(f"\nKey results:")
    print(f"1. Optimal endmember count: {best_K}")
    print(f"2. Regularization parameters: λ={best_lambda:.6f}, β={best_beta:.6f}")
    print(f"3. Best RMSE: {best_row['RMSE_mean']:.6f}" if 'best_row' in locals() else "")
    print(f"4. Bootstrap validation completed: 50 replicates")
    
    print(f"\nOutput locations:")
    print(f"  - Tabular data: {tables_dir}")
    print(f"  - Figures: {figures_dir}")
    print(f"  - Summary: {summary_path}")
    
    print(f"\nGenerated figures:")
    if 'out_combo' in locals():
        print(f"  - K selection and convergence: {out_combo.name}")
    if 'out_B' in locals():
        print(f"  - Convergence trajectories: {out_B.name}")
    if 'out_C1r' in locals():
        print(f"  - RC by region: {out_C1r.name}")
    if 'out_C2' in locals():
        print(f"  - Endmember profiles: {out_C2.name}")


if __name__ == "__main__":
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print("Chinese font support configured")
    except:
        print("Chinese font configuration failed")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    main()
