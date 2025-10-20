import torch
from torch import Tensor
import matplotlib.pyplot as plt
import pandas as pd
from typing import Any, Optional
import math as mt

EULER_MASCHERONI = 0.57721566490153286060

class NotFittedError(ValueError, AttributeError):
    """Raised when the GLMCC model is used before fitting."""


class _BaseGLM:
    def __init__(
        self,
        bin_width,
        window,
        delay,
        tau,
        beta,
        theta,
        dtype = torch.float64,
        device = None,
    ):
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype

        self.delta = float(bin_width)
        self.w = float(window)
        self.delay = float(delay)
        self.tau = float(tau)
        self.beta = float(beta)

        if self.delta <= 0:
            raise ValueError("bin_width must be positive.")
        if self.w <= 0:
            raise ValueError("window must be positive.")
        if self.tau <= 0:
            raise ValueError("tau must be positive.")
        if self.beta <= 0:
            raise ValueError("beta must be positive.")

        self.m = int(round(2 * self.w / self.delta))
        if self.m <= 0:
            raise ValueError("Computed number of bins 'm' must be positive. Check bin_width and window.")

        self.k = torch.arange(1, self.m + 1, dtype=self.dtype, device=self.device)
        self.xk = self.k * self.delta - self.w
        self.xk_n1 = self.xk - self.delta
        self.areas = (
            self.xk <= -self.delay,
            (self.xk > -self.delay) & (self.xk <= self.delay),
            self.xk > self.delay,
        )

        self.theta = self._to_tensor(theta) if theta is not None else torch.zeros(self.m + 2, dtype=self.dtype, device=self.device)
        if self.theta.shape != (self.m + 2,):
            raise ValueError(f"theta must have shape ({self.m + 2},), got {self.theta.shape}.")
        self.theta = self.theta.clone()

    def spike_time(self, t_i, t_j, dt):
        t_i_tensor = self._to_tensor(t_i).flatten()
        t_j_tensor = self._to_tensor(t_j).flatten()
        mask_i = (t_i_tensor == 1)
        mask_j = (t_j_tensor == 1)
        spike_times_i = Tensor([i * dt for i in range(len(t_i_tensor))])
        spike_times_j = Tensor([j * dt for j in range(len(t_j_tensor))])

        return spike_times_i[mask_i], spike_times_j[mask_j]

    def spiketime_relative(self, spiketime_tar, spiketime_ref, window_size=50.0):
        t_sp = []
        min_i, max_i = 0, 0

        for j, tsp_j in enumerate(spiketime_ref):
            # reuse min_i and max_i values for next iteration to decrease the amount of elements to scan
            min_i = self._search_max_idx(lst=spiketime_tar, upper=tsp_j - window_size, start_idx=min_i)
            max_i = self._search_max_idx(lst=spiketime_tar, upper=tsp_j + window_size, start_idx=max_i)

            # a list of relative spike time
            t_sp.extend([(spiketime_tar[i] - spiketime_ref[j]) for i in range(min_i, max_i)])

        return t_sp

    def _search_max_idx(self, lst, upper, start_idx=0):

        idx = start_idx
        while len(lst) > idx and lst[idx] <= upper:
            idx += 1
        return idx

    def _to_tensor(self, value):
        if isinstance(value, torch.Tensor):
            return value.to(device=self.device, dtype=self.dtype)
        return torch.as_tensor(value, device=self.device, dtype=self.dtype)

    def func_f(self, s):
        if not isinstance(s, torch.Tensor):
            s = torch.as_tensor(s, dtype=self.dtype, device=self.device)
        else:
            s = s.to(device=self.device, dtype=self.dtype)

        fs = torch.zeros_like(s)
        mask = s >= self.delay
        if mask.any():
            fs[mask] = torch.exp(-(s[mask] - self.delay) / self.tau)
        return fs

    def log_lambda_cc(self, s):
        return self.theta[: self.m] + self.theta[-2] * self.func_f(s) + self.theta[-1] * self.func_f(-s)


class _Gk(_BaseGLM):
    def __init__(
        self,
        bin_width,
        window,
        delay,
        tau,
        beta,
        theta,
        dtype = torch.float64,
        device = None,
    ):
        super().__init__(bin_width, window, delay, tau, beta, theta, dtype=dtype, device=device)

    @staticmethod
    def _expi_torch(x, max_terms = 60, tol = None):
        if tol is None:
            tol = float(torch.finfo(x.dtype).eps)
        gamma = torch.tensor(EULER_MASCHERONI, dtype=x.dtype, device=x.device)
        abs_x = torch.abs(x)
        tiny = torch.finfo(x.dtype).tiny
        log_term = torch.log(abs_x.clamp_min(tiny))

        term = x.clone()
        series = term.clone()
        for k in range(2, max_terms + 1):
            term = term * x / k
            increment = term / k
            series = series + increment
            if torch.all(torch.abs(increment) <= tol):
                break

        result = gamma + log_term + series
        result = torch.where(x == 0, torch.full_like(x, -torch.inf), result)
        return result
    
    @torch.no_grad()
    def gk(self):
        gk = torch.zeros(self.m, dtype=self.dtype, device=self.device)
        theta_main = self.theta[: self.m]
        ak_n1 = torch.cat([torch.zeros(1, dtype=self.dtype, device=self.device), theta_main[:-1]])

        center_idx = torch.nonzero(self.areas[1], as_tuple=False).squeeze(-1)
        if center_idx.numel() > 0:
            gk[center_idx] = self.delta * torch.exp(theta_main[center_idx])

        left_idx = torch.nonzero(self.areas[0], as_tuple=False).squeeze(-1)
        if left_idx.numel() > 0:
            func_vals = self.func_f(-self.xk[left_idx])
            theta_factor = self.theta[-1] * func_vals
            approx_mask = torch.abs(theta_factor) < 1.0e-06
            if approx_mask.any():
                gk[left_idx[approx_mask]] = self.delta * torch.exp(theta_main[left_idx[approx_mask]])
            not_approx_mask = ~approx_mask
            if not_approx_mask.any():
                idx = left_idx[not_approx_mask]
                gk[idx] = self.tau * torch.exp(theta_main[idx]) * (
                    self._expi_torch(self.theta[-1] * func_vals[not_approx_mask])
                    - self._expi_torch(self.theta[-1] * self.func_f(-self.xk_n1[idx]))
                )

        right_idx = torch.nonzero(self.areas[2], as_tuple=False).squeeze(-1)
        if right_idx.numel() > 0:
            func_vals = self.func_f(self.xk_n1[right_idx])
            theta_factor = self.theta[-2] * func_vals
            approx_mask = torch.abs(theta_factor) < 1.0e-06
            if approx_mask.any():
                idx = right_idx[approx_mask]
                gk[idx] = self.delta * torch.exp(ak_n1[idx])
            not_approx_mask = ~approx_mask
            if not_approx_mask.any():
                idx = right_idx[not_approx_mask]
                gk[idx] = self.tau * torch.exp(theta_main[idx]) * (
                    self._expi_torch(self.theta[-2] * self.func_f(self.xk_n1[idx]))
                    - self._expi_torch(self.theta[-2] * self.func_f(self.xk[idx]))
                )

        return gk

    @torch.no_grad()
    def gk_first_derivative(self):
        dgk_dj_ij = torch.zeros(self.m, dtype=self.dtype, device=self.device)
        dgk_dj_ji = torch.zeros(self.m, dtype=self.dtype, device=self.device)
        theta_main = self.theta[: self.m]

        right_idx = torch.nonzero(self.areas[2], as_tuple=False).squeeze(-1)
        theta_ij = self.theta[-2]
        if right_idx.numel() > 0:
            if torch.abs(theta_ij).item() < 1.0e-03:
                vals = (
                    self.tau
                    * torch.exp(theta_main[right_idx])
                    * self.func_f(self.xk_n1[right_idx])
                    * (1.0 - torch.exp(-self.delta / self.tau))
                )
                dgk_dj_ij[right_idx] = vals
            else:
                numerator = torch.exp(theta_ij * self.func_f(self.xk_n1[right_idx])) - torch.exp(
                    theta_ij * self.func_f(self.xk[right_idx])
                )
                dgk_dj_ij[right_idx] = (self.tau * torch.exp(theta_main[right_idx]) / theta_ij) * numerator

        left_idx = torch.nonzero(self.areas[0], as_tuple=False).squeeze(-1)
        theta_ji = self.theta[-1]
        if left_idx.numel() > 0:
            if torch.abs(theta_ji).item() < 1.0e-03:
                vals = (
                    self.tau
                    * torch.exp(theta_main[left_idx])
                    * self.func_f(-self.xk[left_idx])
                    * (1.0 - mt.exp(-self.delta / self.tau))
                )
                dgk_dj_ji[left_idx] = vals
            else:
                numerator = torch.exp(theta_ji * self.func_f(-self.xk[left_idx])) - torch.exp(
                    theta_ji * self.func_f(-self.xk_n1[left_idx])
                )
                dgk_dj_ji[left_idx] = (self.tau * torch.exp(theta_main[left_idx]) / theta_ji) * numerator

        return dgk_dj_ij, dgk_dj_ji

    @torch.no_grad()
    def gk_second_derivative(self):
        d2gk_dj_ij2 = torch.zeros(self.m, dtype=self.dtype, device=self.device)
        d2gk_dj_ji2 = torch.zeros(self.m, dtype=self.dtype, device=self.device)
        theta_main = self.theta[: self.m]

        right_idx = torch.nonzero(self.areas[2], as_tuple=False).squeeze(-1)
        theta_ij = self.theta[-2]
        if right_idx.numel() > 0:
            if torch.abs(theta_ij).item() < 1.0e-03:
                vals = (
                    0.5
                    * self.tau
                    * torch.exp(theta_main[right_idx])
                    * self.func_f(self.xk_n1[right_idx]).pow(2)
                    * (1.0 - torch.exp(-2.0 * self.delta / self.tau))
                )
                d2gk_dj_ij2[right_idx] = vals
            else:
                diff = self._func_h(theta_ij * self.func_f(self.xk_n1[right_idx])) - self._func_h(
                    theta_ij * self.func_f(self.xk[right_idx])
                )
                coef = self.tau * torch.exp(theta_main[right_idx]) / (theta_ij ** 2)
                d2gk_dj_ij2[right_idx] = coef * diff

        left_idx = torch.nonzero(self.areas[0], as_tuple=False).squeeze(-1)
        theta_ji = self.theta[-1]
        if left_idx.numel() > 0:
            if torch.abs(theta_ji).item() < 1.0e-03:
                vals = (
                    0.5
                    * self.tau
                    * torch.exp(theta_main[left_idx])
                    * self.func_f(-self.xk[left_idx]).pow(2)
                    * (1.0 - mt.exp(-2.0 * self.delta / self.tau))
                )
                d2gk_dj_ji2[left_idx] = vals
            else:
                diff = self._func_h(theta_ji * self.func_f(-self.xk[left_idx])) - self._func_h(
                    theta_ji * self.func_f(-self.xk_n1[left_idx])
                )
                coef = self.tau * torch.exp(theta_main[left_idx]) / (theta_ji ** 2)
                d2gk_dj_ji2[left_idx] = coef * diff

        return d2gk_dj_ij2, d2gk_dj_ji2

    @staticmethod
    def _func_h(x):
        return (x - 1.0) * torch.exp(x)


class GLMCC(_Gk):
    def __init__(
        self,
        bin_width = 1.0,
        window = 50.0,
        delay = 3.0,
        tau = 4.0,
        beta = 4000.0,
        theta = None,
        dtype = torch.float64,
        device = None,
    ):
        super().__init__(bin_width, window, delay, tau, beta, theta, dtype=dtype, device=device)
        self.max_log_posterior: Optional[float] = None
        self.j_thresholds: Optional[Tensor] = None
        self.__is_fitted = False

    def make_cc(self, t_sp):
        t_sp_tensor = self._to_tensor(t_sp).flatten()
        if t_sp_tensor.numel() == 0:
            return torch.zeros(self.m, dtype=self.dtype, device=self.device)

        bin_edges = torch.linspace(-self.w, self.w, steps=self.m + 1, dtype=self.dtype, device=self.device)
        bin_indices = torch.bucketize(t_sp_tensor, bin_edges, right=True) - 1
        valid_mask = (bin_indices >= 0) & (bin_indices < self.m)
        if not valid_mask.any():
            return torch.zeros(self.m, dtype=self.dtype, device=self.device)

        bin_indices = bin_indices[valid_mask].to(dtype=torch.int64)
        counts = torch.bincount(bin_indices, minlength=self.m).to(self.dtype)
        return counts

    def fit(
        self,
        t_sp: Any,
        clm = 0.01,
        eta = 0.1,
        max_iter: int = 1000,
        j_min = -3.0,
        j_max = 5.0,
        verbose = True,
    ):
        with torch.no_grad():
            t_sp_vec = self._to_tensor(t_sp).flatten()
            cc = self.make_cc(t_sp_vec)

            baseline_rate = torch.log((1.0 + cc.sum()) / (2.0 * self.w))
            self.theta[: self.m].fill_(baseline_rate)
            self.theta[-2:].fill_(0.1)

            iter_count = 0

            for _ in range(max_iter):
                grad = self._gradient(t_sp_vec, cc)
                hess = self._hessian()

                tmp_log_posterior = self._log_posterior(t_sp_vec, cc)
                tmp_theta = self.theta.clone()

                reg_matrix = hess + clm * torch.diag(torch.diag(hess))
                try:
                    delta_theta = torch.linalg.solve(reg_matrix, grad)
                except (RuntimeError, torch.linalg.LinAlgError) as exc:
                    print(f"Hessian solve failed: {exc}")
                    return False

                self.theta -= delta_theta
                self.theta[-2:] = torch.clamp(self.theta[-2:], j_min, j_max)

                new_log_posterior = self._log_posterior(t_sp_vec, cc)
                iter_count += 1

                if iter_count == max_iter and verbose:
                    print(f"Values did not converge within {max_iter} iterations: breaking the loop.")

                if new_log_posterior >= tmp_log_posterior:
                    if torch.abs(new_log_posterior - tmp_log_posterior) < 1.0e-4:
                        break
                    clm *= eta
                else:
                    self.theta = tmp_theta
                    clm *= 1.0 / eta

            if verbose:
                print(f"iterations until convergence: {iter_count}")

            self.max_log_posterior = float(self._log_posterior(t_sp_vec, cc).item())
            self.__statistical_test()
            self.__is_fitted = True
            return True

    @torch.no_grad()
    def __statistical_test(self, z_alpha = 3.29):
        theta_main = self.theta[: self.m]
        exp_theta = torch.exp(theta_main)

        mask_ij = (self.xk >= self.delay) & (self.xk <= self.delay + self.tau)
        mask_ji = (self.xk <= -self.delay) & (self.xk >= -(self.delay + self.tau))

        cc_0 = torch.zeros(2, dtype=self.dtype, device=self.device)
        cc_0[0] = exp_theta[mask_ij].mean() if mask_ij.any() else torch.tensor(1.0, dtype=self.dtype, device=self.device)
        cc_0[1] = exp_theta[mask_ji].mean() if mask_ji.any() else torch.tensor(1.0, dtype=self.dtype, device=self.device)

        eps = torch.finfo(self.dtype).eps
        self.j_thresholds = 1.57 * z_alpha / torch.sqrt(self.tau * torch.clamp(cc_0, min=eps))

    @torch.no_grad()
    def _log_posterior(self, t_sp: Tensor, cc: Tensor) -> Tensor:
        theta_main = self.theta[: self.m]
        smooth_penalty = (self.beta / (2.0 * self.delta)) * torch.sum((theta_main[1:] - theta_main[:-1]).pow(2))
        log_posterior = (
            torch.dot(cc, theta_main)
            + torch.sum(self.theta[-2] * self.func_f(t_sp))
            + torch.sum(self.theta[-1] * self.func_f(-t_sp))
            - torch.sum(self.gk())
            - smooth_penalty
        )
        return log_posterior

    @torch.no_grad()
    def _gradient(self, t_sp: Tensor, cc: Tensor) -> Tensor:
        gradient = torch.zeros_like(self.theta)
        theta_main = self.theta[: self.m]

        ak_n1 = torch.cat([torch.zeros(1, dtype=self.dtype, device=self.device), theta_main[:-1]])
        ak_p1 = torch.cat([theta_main[1:], torch.zeros(1, dtype=self.dtype, device=self.device)])

        gk_vals = self.gk()
        prefactor = self.beta / self.delta
        gradient[: self.m] = cc - gk_vals + prefactor * (
            (self._k_delta(1) - 1.0) * (theta_main - ak_n1) + (self._k_delta(self.m) - 1.0) * (theta_main - ak_p1)
        )

        dgk_dj_ij, dgk_dj_ji = self.gk_first_derivative()
        mask_ij = t_sp > self.delay
        mask_ji = t_sp < -self.delay

        gradient[-2] = torch.sum(self.func_f(t_sp[mask_ij])) - torch.sum(dgk_dj_ij)
        gradient[-1] = torch.sum(self.func_f(-t_sp[mask_ji])) - torch.sum(dgk_dj_ji)
        return gradient

    @torch.no_grad()
    def _hessian(self) -> Tensor:
        size = self.theta.shape[0]
        hessian = torch.zeros((size, size), dtype=self.dtype, device=self.device)

        theta_main = self.theta[: self.m]
        gk_vals = self.gk()

        prefactor = self.beta / self.delta
        hessian_main = torch.zeros((self.m, self.m), dtype=self.dtype, device=self.device)
        if self.m > 1:
            idx = torch.arange(self.m - 1, device=self.device)
            hessian_main[idx + 1, idx] = prefactor
            hessian_main[idx, idx + 1] = prefactor

        diag_indices = torch.arange(self.m, device=self.device)
        diag_values = -gk_vals + prefactor * (self._k_delta(1) + self._k_delta(self.m) - 2.0)
        hessian_main[diag_indices, diag_indices] = diag_values
        hessian[: self.m, : self.m] = hessian_main

        dgk_dj_ij, dgk_dj_ji = self.gk_first_derivative()
        hessian[: self.m, -2] = -dgk_dj_ij
        hessian[: self.m, -1] = -dgk_dj_ji
        hessian[-2, : self.m] = -dgk_dj_ij
        hessian[-1, : self.m] = -dgk_dj_ji

        d2gk_dj_ij2, d2gk_dj_ji2 = self.gk_second_derivative()
        hessian[-2, -2] = -torch.sum(d2gk_dj_ij2)
        hessian[-1, -1] = -torch.sum(d2gk_dj_ji2)

        return hessian

    def _k_delta(self, l: int) -> Tensor:
        return (self.k == float(l)).to(self.dtype)

    def plot(self, ax: plt.Axes, t_sp: Any, colors: dict[int, str] = None, verbose = True) -> plt.Axes:
        if colors is None:
            colors = {0: "gray", 1: "cyan", 2: "magenta"}

        t_sp_tensor = self._to_tensor(t_sp).flatten()
        cc_tensor = self.make_cc(t_sp_tensor)

        x_values = self.xk.cpu().numpy()
        cc = cc_tensor.cpu().numpy()
        ax.bar(x_values, cc, color="black", width=self.delta)

        if self.__is_fitted and self.j_thresholds is not None:
            at = torch.exp(self.theta[: self.m])
            j_ij = torch.exp(self.theta[-2] * self.func_f(self.xk) + self.theta[: self.m])
            j_ji = torch.exp(self.theta[-1] * self.func_f(-self.xk) + self.theta[: self.m])

            ax.plot(x_values, j_ij.cpu().numpy(), linewidth=3.0, color=colors[self._color_index(self.theta[-2], self.j_thresholds[0])])
            ax.plot(x_values, j_ji.cpu().numpy(), linewidth=3.0, color=colors[self._color_index(self.theta[-1], self.j_thresholds[1])])
            ax.plot(x_values, at.cpu().numpy(), linewidth=3.0, color="lime")
        else:
            if verbose:
                print("GLM not fitted yet, plotting only cross-correlogram.")

        ax.set_xticks([-self.w, 0.0, self.w])
        ax.set_xlim(-self.w, self.w)

        if self.__is_fitted:
            at_vals = torch.exp(self.theta[: self.m])
            j_ij_vals = torch.exp(self.theta[-2] * self.func_f(self.xk) + self.theta[: self.m])
            j_ji_vals = torch.exp(self.theta[-1] * self.func_f(-self.xk) + self.theta[: self.m])
            y_max = max(
                float(cc_tensor.max().item()),
                float(at_vals.max().item()),
                float(j_ij_vals.max().item()),
                float(j_ji_vals.max().item()),
            )
        else:
            y_max = float(cc_tensor.max().item()) if cc_tensor.numel() > 0 else 1.0
        ax.set_ylim(0.0, y_max * 1.1 if y_max > 0 else 1.0)
        return ax

    @staticmethod
    def _color_index(theta_value: Tensor, threshold: Tensor) -> int:
        theta_val = float(theta_value.item())
        threshold_val = float(threshold.item())
        if abs(theta_val) < threshold_val:
            return 0
        return 2 if theta_val > 0 else 1

    def summary(self):
        if not self.__is_fitted:
            raise NotFittedError("GLMCC is not fitted yet.")
        if self.j_thresholds is None:
            raise RuntimeError("Statistical test has not been computed.")

        thresholds = self.j_thresholds.cpu().tolist()
        theta_ij = float(self.theta[-2].item())
        theta_ji = float(self.theta[-1].item())

        summary_lines = [
            "=" * 15 + " GLMCC summary " + "=" * 15,
            "connectivity from neuron j to neuron i:",
            f"\t estimated J_ij, J_ji: {round(theta_ij, 2)}, {round(theta_ji, 2)}",
            f"\t threshold J_ij, J_ji: {round(thresholds[0], 2)}, {round(thresholds[1], 2)}",
            f"max log posterior: {int(self.max_log_posterior) if self.max_log_posterior is not None else 'N/A'}",
            "=" * 15 + "===============" + "=" * 15,
        ]
        print("\n".join(summary_lines))


if __name__ == "__main__":
    df = pd.read_csv("dataAnalysis/example/sample_data.csv")
    tensor_data = torch.as_tensor(df.to_numpy(dtype="float64"), dtype=torch.float64)
    glm = GLMCC(delay=4.0)  # set synaptic delay to initialize GLMC

    fig, ax = plt.subplots(figsize=(3, 3))
    idx_i, idx_j = 1, 5

    # relative spiketime (target neuron - reference neuron)
    t_sp = glm.spiketime_relative(spiketime_tar=list(df.query('neuron==@idx_i').spiketime), 
                            spiketime_ref=list(df.query('neuron==@idx_j').spiketime), window_size=50.0)
    glm.plot(ax=ax, t_sp=t_sp)

    ax.set_title(f'cross-correlogram: neuron {idx_i} to {idx_j}')
    plt.show()

    def fit_and_plot(ax, idx_i, idx_j, delay, window_size=50.0, verbose=True):
        # prepare relative spiketime (target neuron - reference neuron)
        glm = GLMCC(delay=delay)  # tune synaptic delay
        t_sp = glm.spiketime_relative(spiketime_tar=list(df.query('neuron==@idx_i').spiketime), 
        spiketime_ref=list(df.query('neuron==@idx_j').spiketime), window_size=window_size)

        # model settings
        glm.fit(t_sp, verbose=verbose)
        glm.plot(ax=ax, t_sp=t_sp)

        # recommended plot layouts
        ax.set_xlabel(r'$\tau [ms]$', fontsize=16)
        ax.set_ylabel(r'$C(\tau)$', fontsize=16)
        ax.set_title(f'neuron {idx_i} to {idx_j}', fontsize=18)
        ax.tick_params(direction='in', which='major', labelsize=12)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        return glm
    
    fig, ax = plt.subplots(figsize=(4, 4))
    glm = fit_and_plot(ax=ax, idx_i=4, idx_j=7, delay=4.0)
    glm.summary()
    plt.show()