"""
====================================================================
DISCRETE NAVIER–STOKES PROOF / AUDIT FRAMEWORK
====================================================================

STATUS
------
This file is an experimental/numerical audit framework.

The computations below can:
    * test scaling identities,
    * detect discretization/aliasing artifacts,
    * measure shell and triadic interactions,
    * search for numerical counterexamples,
    * test candidate Schur-type bounds.

They DO NOT constitute a proof of
    sup_u C3(u) < infinity
and do not establish global regularity of 3-D Navier–Stokes.

CORE QUOTIENT
-------------
    C3(u) = |<grad p, grad q>| / (||u||_3 D3(u))

where

    -Delta p = div((u . grad)u)
    -Delta q = div(|u|u)

and

    D3(u) = integral |u| |grad u|^2 dx.

IMPORTANT SPECTRAL DISTINCTION
------------------------------
Different OUTPUT Fourier shells of grad p and grad q are exactly
orthogonal when the shell projectors have disjoint support.

That fact alone does NOT establish decay of nonlinear cross-scale
velocity interactions.

The more informative object is therefore the TRIADIC SOURCE audit:

    (u_m,u_n) -> Delta_j(u_m tensor u_n) -> grad p_j

and

    (u_l,u_r) -> Delta_j(|u_l|u_r) -> grad q_j.

All numerical conclusions must be interpreted as experimental.

====================================================================
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# ====================================================================
# RESULT CONTAINERS
# ====================================================================

@dataclass
class ScalingResult:
    amplitude: float
    c3: float
    r3: float
    l3: float
    d3: float
    pressure_work: float


@dataclass
class ResolutionResult:
    N: int
    c3: float
    l3: float
    d3: float
    pressure_work: float


@dataclass
class ConcentrationResult:
    sigma: float
    delta: float
    c3: float
    l3: float
    d3: float
    pressure_work: float


# ====================================================================
# DISCRETE PROOF AUDIT
# ====================================================================

class DiscreteProofAudit:
    """
    Unified shell / triadic / concentration / variational audit.

    The class expects a StructurePreservingLatticeFluid3D instance
    exposing the following existing operations:

        N
        h
        Kx, Ky, Kz
        K_tilde_x, K_tilde_y, K_tilde_z
        inv_K_sq
        skew_convection()
        inner_product()
        norm_l2()
        norm_l3()
        leray_project()
        random_divergence_free_field()

    This deliberately keeps the entire audit in ONE file.
    """

    def __init__(self, fluid):
        self.fl = fluid

        self.Kmag = np.sqrt(
            self.fl.Kx ** 2 +
            self.fl.Ky ** 2 +
            self.fl.Kz ** 2
        )

    # ================================================================
    # BASIC OPERATORS
    # ================================================================

    def velocity_magnitude(self, u):
        return np.sqrt(np.sum(u ** 2, axis=0) + 1e-30)

    def gradient_squared(self, u):
        """
        Spectral gradient norm:

            |grad u|^2
        """
        u_hat = np.fft.fftn(u, axes=(1, 2, 3))

        grad_sq = np.zeros_like(u[0], dtype=float)

        for a in range(3):
            gx = np.real(np.fft.ifftn(
                1j * self.fl.K_tilde_x * u_hat[a],
                axes=(0, 1, 2)
            ))

            gy = np.real(np.fft.ifftn(
                1j * self.fl.K_tilde_y * u_hat[a],
                axes=(0, 1, 2)
            ))

            gz = np.real(np.fft.ifftn(
                1j * self.fl.K_tilde_z * u_hat[a],
                axes=(0, 1, 2)
            ))

            grad_sq += gx ** 2 + gy ** 2 + gz ** 2

        return grad_sq

    def critical_dissipation(self, u):
        u_mag = self.velocity_magnitude(u)
        grad_sq = self.gradient_squared(u)

        return np.sum(
            u_mag * grad_sq
        ) * self.fl.h ** 3

    # ================================================================
    # PRESSURE OPERATORS
    # ================================================================

    def pressure_gradients(self, u):
        """
        Compute grad p and grad q.

        Physical pressure:
            -Delta p = div((u . grad)u)

        Auxiliary pressure:
            -Delta q = div(|u|u)
        """

        # ------------------------------------------------------------
        # Physical pressure
        # ------------------------------------------------------------

        B = self.fl.skew_convection(u, u)

        B_hat = np.fft.fftn(
            B,
            axes=(1, 2, 3)
        )

        divB_hat = (
            1j * self.fl.K_tilde_x * B_hat[0] +
            1j * self.fl.K_tilde_y * B_hat[1] +
            1j * self.fl.K_tilde_z * B_hat[2]
        )

        p_hat = -divB_hat * self.fl.inv_K_sq

        grad_p_hat = np.stack([
            1j * self.fl.K_tilde_x * p_hat,
            1j * self.fl.K_tilde_y * p_hat,
            1j * self.fl.K_tilde_z * p_hat
        ], axis=0)

        # ------------------------------------------------------------
        # Auxiliary pressure
        # ------------------------------------------------------------

        u_mag = self.velocity_magnitude(u)
        abs_u_u = u_mag * u

        q_source_hat = np.fft.fftn(
            abs_u_u,
            axes=(1, 2, 3)
        )

        divq_hat = (
            1j * self.fl.K_tilde_x * q_source_hat[0] +
            1j * self.fl.K_tilde_y * q_source_hat[1] +
            1j * self.fl.K_tilde_z * q_source_hat[2]
        )

        q_hat = -divq_hat * self.fl.inv_K_sq

        grad_q_hat = np.stack([
            1j * self.fl.K_tilde_x * q_hat,
            1j * self.fl.K_tilde_y * q_hat,
            1j * self.fl.K_tilde_z * q_hat
        ], axis=0)

        return grad_p_hat, grad_q_hat

    def pressure_work(self, u):
        grad_p_hat, grad_q_hat = self.pressure_gradients(u)

        gp = np.real(
            np.fft.ifftn(
                grad_p_hat,
                axes=(1, 2, 3)
            )
        )

        gq = np.real(
            np.fft.ifftn(
                grad_q_hat,
                axes=(1, 2, 3)
            )
        )

        return abs(
            self.fl.inner_product(gp, gq)
        )

    # ================================================================
    # CRITICAL QUOTIENT
    # ================================================================

    def critical_quotient(self, u):
        l3 = self.fl.norm_l3(u)
        d3 = self.critical_dissipation(u)
        wp = self.pressure_work(u)

        return wp / (l3 * d3 + 1e-30)

    # ================================================================
    # CHAIN-RULE DEFECT
    # ================================================================

    def chain_rule_defect(self, u):
        """
        R_h =
          |<B_h(u,u), |u|u>|
          --------------------------------
          ||u||_3 D3(u)

        This is a diagnostic of the discrete Leibniz defect.
        """

        u_mag = self.velocity_magnitude(u)
        B = self.fl.skew_convection(u, u)

        numerator = abs(
            self.fl.inner_product(
                B,
                u_mag * u
            )
        )

        denominator = (
            self.fl.norm_l3(u) *
            self.critical_dissipation(u)
        )

        return numerator / (denominator + 1e-30)

    # ================================================================
    # LINEAR SHELLS
    # ================================================================

    def linear_shell_masks(self, jmax=None):
        if jmax is None:
            jmax = self.fl.N // 3

        masks = {}

        for j in range(1, jmax + 1):
            masks[j] = (
                (self.Kmag >= j - 1.0) &
                (self.Kmag < j)
            )

        return masks

    # ================================================================
    # DYADIC SHELLS
    # ================================================================

    def dyadic_shell_masks(self, oct_max=None):
        if oct_max is None:
            oct_max = max(
                1,
                int(np.log2(max(2, self.fl.N // 2)))
            )

        masks = {}

        for j in range(1, oct_max + 1):
            low = 0.0 if j == 1 else 2.0 ** (j - 1)
            high = 2.0 ** j

            masks[j] = (
                (self.Kmag >= low) &
                (self.Kmag < high)
            )

        return masks

    # ================================================================
    # SHELL FILTER
    # ================================================================

    def filter_hat(self, field_hat, mask):
        out = np.zeros_like(field_hat)
        out[:, mask] = field_hat[:, mask]
        return out

    def shell_field(self, u, mask):
        u_hat = np.fft.fftn(
            u,
            axes=(1, 2, 3)
        )

        filtered = np.zeros_like(u_hat)
        filtered[:, mask] = u_hat[:, mask]

        return np.real(
            np.fft.ifftn(
                filtered,
                axes=(1, 2, 3)
            )
        )

    # ================================================================
    # SHELL DISSIPATION
    # ================================================================

    def shell_dissipation(self, u, mask):
        u_mag = self.velocity_magnitude(u)
        uj = self.shell_field(u, mask)

        grad_sq = self.gradient_squared(uj)

        return np.sum(
            u_mag * grad_sq
        ) * self.fl.h ** 3

    # ================================================================
    # OUTPUT-SHELL PRESSURE MATRIX
    # ================================================================

    def shell_pressure_matrix(
        self,
        u,
        shell_type="dyadic"
    ):
        """
        Computes

            M_jk =
              |< Delta_j grad p,
                 Delta_k grad q >|.

        For disjoint Fourier output shells, j != k should vanish
        up to numerical precision.

        This is OUTPUT orthogonality, not nonlinear source
        interaction decay.
        """

        if shell_type == "dyadic":
            masks = self.dyadic_shell_masks()
        else:
            masks = self.linear_shell_masks()

        grad_p_hat, grad_q_hat = self.pressure_gradients(u)

        shells = [
            j for j, mask in masks.items()
            if np.count_nonzero(mask) > 0
        ]

        n = len(shells)

        M = np.zeros((n, n))
        D = np.zeros(n)

        parseval = (
            self.fl.h ** 3 /
            self.fl.N ** 3
        )

        for a, j in enumerate(shells):
            mask = masks[j]

            gp = np.zeros_like(grad_p_hat)
            gq = np.zeros_like(grad_q_hat)

            gp[:, mask] = grad_p_hat[:, mask]
            gq[:, mask] = grad_q_hat[:, mask]

            for b, k in enumerate(shells):

                gp_k = np.zeros_like(grad_p_hat)
                gq_k = np.zeros_like(grad_q_hat)

                gp_k[:, masks[j]] = grad_p_hat[:, masks[j]]
                gq_k[:, masks[k]] = grad_q_hat[:, masks[k]]

                # Fourier-space Parseval inner product.
                inner = np.sum(
                    gp_k * np.conj(gq_k)
                )

                M[a, b] = abs(inner) * parseval

            D[a] = self.shell_dissipation(
                u,
                mask
            )

        l3 = self.fl.norm_l3(u)

        Gamma = np.zeros_like(M)

        for a in range(n):
            for b in range(n):
                denominator = (
                    l3 *
                    np.sqrt(
                        D[a] * D[b] +
                        1e-30
                    )
                )

                Gamma[a, b] = (
                    M[a, b] /
                    (denominator + 1e-30)
                )

        return shells, M, Gamma, D

    # ================================================================
    # SCHUR ROW SUM
    # ================================================================

    def schur_audit(self, u, shell_type="dyadic"):
        shells, M, Gamma, D = self.shell_pressure_matrix(
            u,
            shell_type=shell_type
        )

        row_sums = np.sum(
            Gamma,
            axis=1
        )

        return {
            "shells": shells,
            "M": M,
            "Gamma": Gamma,
            "D": D,
            "row_sums": row_sums,
            "sup_row_sum": float(np.max(row_sums))
            if len(row_sums)
            else 0.0
        }

    # ================================================================
    # FOURIER SOURCE DECOMPOSITION
    # ================================================================

    def nonlinear_sources(self, u):
        """
        Return Fourier representations of

            B(u,u)

        and

            |u|u.
        """

        B = self.fl.skew_convection(
            u,
            u
        )

        abs_u_u = (
            self.velocity_magnitude(u) *
            u
        )

        B_hat = np.fft.fftn(
            B,
            axes=(1, 2, 3)
        )

        Q_hat = np.fft.fftn(
            abs_u_u,
            axes=(1, 2, 3)
        )

        return B_hat, Q_hat

    # ================================================================
    # TRIADIC SOURCE AUDIT
    # ================================================================

    def triadic_source_audit(self, u, shell_type="dyadic"):
        """
        Measures nonlinear source energy by output shell.

        This deliberately does NOT claim that output-shell
        orthogonality implies velocity-shell decoupling.

        The source fields contain convolution sums:

            B_hat(k) =
                sum_{m+n=k} ...
            
            (|u|u)_hat(k) =
                sum_{m+n=k} ...

        The diagnostic therefore tracks the shell-localized
        nonlinear source magnitudes.
        """

        if shell_type == "dyadic":
            masks = self.dyadic_shell_masks()
        else:
            masks = self.linear_shell_masks()

        B_hat, Q_hat = self.nonlinear_sources(u)

        result = {}

        for j, mask in masks.items():
            if np.count_nonzero(mask) == 0:
                continue

            B_energy = np.sum(
                np.abs(B_hat[:, mask]) ** 2
            )

            Q_energy = np.sum(
                np.abs(Q_hat[:, mask]) ** 2
            )

            result[j] = {
                "B_source_energy": float(B_energy),
                "Q_source_energy": float(Q_energy),
                "B_source_norm": float(np.sqrt(B_energy)),
                "Q_source_norm": float(np.sqrt(Q_energy))
            }

        return result

    # ================================================================
    # DE-ALIASING MASK
    # ================================================================

    def two_thirds_mask(self):
        """
        Standard Cartesian 2/3-rule mask.

        This is a stress-test diagnostic rather than a claim that
        the MAC discretization itself is a pseudospectral method.
        """

        cutoff = self.fl.N / 3.0

        return (
            np.abs(self.fl.Kx) <= cutoff
        ) & (
            np.abs(self.fl.Ky) <= cutoff
        ) & (
            np.abs(self.fl.Kz) <= cutoff
        )

    def dealias_field(self, u):
        u_hat = np.fft.fftn(
            u,
            axes=(1, 2, 3)
        )

        mask = self.two_thirds_mask()

        out = np.zeros_like(u_hat)
        out[:, mask] = u_hat[:, mask]

        return np.real(
            np.fft.ifftn(
                out,
                axes=(1, 2, 3)
            )
        )

    # ================================================================
    # ALIASING AUDIT
    # ================================================================

    def aliasing_audit(self, u):
        """
        Compare the quotient before and after spectral truncation.
        """

        c_raw = self.critical_quotient(u)

        u_filtered = self.dealias_field(u)

        c_dealiased = self.critical_quotient(
            u_filtered
        )

        return {
            "C3_raw": float(c_raw),
            "C3_dealiased": float(c_dealiased),
            "absolute_difference":
                float(abs(c_raw - c_dealiased)),
            "relative_difference":
                float(
                    abs(c_raw - c_dealiased) /
                    (abs(c_raw) + 1e-30)
                )
        }

    # ================================================================
    # AMPLITUDE INVARIANCE
    # ================================================================

    def amplitude_invariance_test(
        self,
        u,
        amplitudes=(1.0, 2.0, 4.0, 8.0)
    ):
        """
        Tests

            C3(Au) = C3(u)

        and the corresponding scale invariance of R_h.
        """

        base_norm = self.fl.norm_l2(u)

        if base_norm <= 0:
            raise ValueError(
                "Input field has zero L2 norm."
            )

        u0 = u / base_norm

        results = []

        for A in amplitudes:
            ua = A * u0

            l3 = self.fl.norm_l3(ua)
            d3 = self.critical_dissipation(ua)
            wp = self.pressure_work(ua)

            c3 = wp / (
                l3 * d3 + 1e-30
            )

            r3 = self.chain_rule_defect(
                ua
            )

            results.append(
                ScalingResult(
                    amplitude=A,
                    c3=c3,
                    r3=r3,
                    l3=l3,
                    d3=d3,
                    pressure_work=wp
                )
            )

        return results

    # ================================================================
    # CONCENTRATION FAMILY
    # ================================================================

    def concentration_family(
        self,
        u,
        sigma,
        delta=1.0
    ):
        """
        Construct an anisotropically concentrated field by Fourier
        rescaling.

        This is a numerical family generator. Because a finite
        periodic lattice cannot represent arbitrary continuum
        concentration limits, results must be checked for resolution
        convergence.
        """

        u_hat = np.fft.fftn(
            u,
            axes=(1, 2, 3)
        )

        kx = self.fl.Kx
        ky = self.fl.Ky
        kz = self.fl.Kz

        # Anisotropic effective frequency.
        effective_k = np.sqrt(
            (sigma * kx) ** 2 +
            (sigma * ky) ** 2 +
            (sigma ** delta * kz) ** 2
        )

        # Smooth spectral concentration envelope.
        envelope = np.exp(
            -0.5 * effective_k ** 2
        )

        concentrated_hat = (
            u_hat * envelope
        )

        concentrated = np.real(
            np.fft.ifftn(
                concentrated_hat,
                axes=(1, 2, 3)
            )
        )

        # Restore divergence-free constraint.
        concentrated = self.fl.leray_project(
            concentrated
        )

        return concentrated

    def concentration_sweep(
        self,
        u,
        sigmas=(1.0, 0.75, 0.5, 0.25, 0.125),
        deltas=(1.0, 0.5, 2.0)
    ):
        results = []

        for delta in deltas:
            for sigma in sigmas:

                us = self.concentration_family(
                    u,
                    sigma=sigma,
                    delta=delta
                )

                l3 = self.fl.norm_l3(us)
                d3 = self.critical_dissipation(us)
                wp = self.pressure_work(us)

                c3 = wp / (
                    l3 * d3 + 1e-30
                )

                results.append(
                    ConcentrationResult(
                        sigma=sigma,
                        delta=delta,
                        c3=c3,
                        l3=l3,
                        d3=d3,
                        pressure_work=wp
                    )
                )

        return results

    # ================================================================
    # RESOLUTION TEST
    # ================================================================

    def resolution_measurement(self, u):
        l3 = self.fl.norm_l3(u)
        d3 = self.critical_dissipation(u)
        wp = self.pressure_work(u)

        c3 = wp / (
            l3 * d3 + 1e-30
        )

        return ResolutionResult(
            N=self.fl.N,
            c3=c3,
            l3=l3,
            d3=d3,
            pressure_work=wp
        )

    # ================================================================
    # PHASE RANDOMIZATION
    # ================================================================

    def randomize_fourier_phases(
        self,
        u,
        seed=None
    ):
        rng = np.random.default_rng(seed)

        u_hat = np.fft.fftn(
            u,
            axes=(1, 2, 3)
        )

        phase = rng.uniform(
            0.0,
            2.0 * np.pi,
            size=u_hat.shape[1:]
        )

        # Preserve magnitudes while changing phases.
        randomized = (
            np.abs(u_hat) *
            np.exp(
                1j * phase[None, ...]
            )
        )

        ur = np.real(
            np.fft.ifftn(
                randomized,
                axes=(1, 2, 3)
            )
        )

        return self.fl.leray_project(
            ur
        )

    # ================================================================
    # ADVERSARIAL MULTI-START SEARCH
    # ================================================================

    def adversarial_search(
        self,
        starts=20,
        steps=30,
        lr=0.01,
        seed=101,
        normalize="L3"
    ):
        """
        Multi-start numerical search.

        IMPORTANT:
            This is NOT a proof of a supremum.

        The search only finds values within the chosen finite
        parameterization.
        """

        rng = np.random.default_rng(seed)

        best_c3 = -np.inf
        best_field = None
        history = []

        for s in range(starts):

            field_seed = int(
                rng.integers(0, 2**31 - 1)
            )

            u = self.fl.random_divergence_free_field(
                seed=field_seed,
                amplitude=1.0
            )

            for step in range(steps):

                gp_hat, gq_hat = (
                    self.pressure_gradients(u)
                )

                gp = np.real(
                    np.fft.ifftn(
                        gp_hat,
                        axes=(1, 2, 3)
                    )
                )

                gq = np.real(
                    np.fft.ifftn(
                        gq_hat,
                        axes=(1, 2, 3)
                    )
                )

                direction = gp + gq

                # Remove compressible component.
                direction = self.fl.leray_project(
                    direction
                )

                u = u + lr * direction

                if normalize.upper() == "L3":
                    norm = self.fl.norm_l3(u)
                else:
                    norm = self.fl.norm_l2(u)

                if norm > 1e-30:
                    u = u / norm

                c3 = self.critical_quotient(u)

                history.append({
                    "start": s,
                    "step": step,
                    "C3": float(c3)
                })

                if c3 > best_c3:
                    best_c3 = c3
                    best_field = u.copy()

        return {
            "best_C3": float(best_c3),
            "best_field": best_field,
            "history": history
        }

    # ================================================================
    # SHELL AMPLITUDE ADVERSARY
    # ================================================================

    def shell_amplitude_search(
        self,
        u,
        iterations=50,
        learning_rate=0.05,
        shell_type="dyadic"
    ):
        """
        Reweights Fourier shells and searches for concentration of
        critical mass in the shell hierarchy.

        This attacks the assumption that a prescribed
        a_j ~ 2^{-j/3} profile is worst-case.
        """

        if shell_type == "dyadic":
            masks = self.dyadic_shell_masks()
        else:
            masks = self.linear_shell_masks()

        shell_ids = [
            j for j, m in masks.items()
            if np.count_nonzero(m)
        ]

        u_hat = np.fft.fftn(
            u,
            axes=(1, 2, 3)
        )

        amplitudes = np.ones(
            len(shell_ids),
            dtype=float
        )

        history = []

        for _ in range(iterations):

            weighted_hat = np.zeros_like(
                u_hat
            )

            for i, j in enumerate(shell_ids):
                weighted_hat[
                    :, masks[j]
                ] = (
                    amplitudes[i] *
                    u_hat[:, masks[j]]
                )

            candidate = np.real(
                np.fft.ifftn(
                    weighted_hat,
                    axes=(1, 2, 3)
                )
            )

            candidate = self.fl.leray_project(
                candidate
            )

            n = self.fl.norm_l3(candidate)

            if n > 1e-30:
                candidate /= n

            c3 = self.critical_quotient(
                candidate
            )

            history.append(
                float(c3)
            )

            # Finite-difference shell gradient.
            gradients = np.zeros_like(
                amplitudes
            )

            eps = 1e-3

            for i in range(
                len(amplitudes)
            ):
                old = amplitudes[i]

                amplitudes[i] = old + eps

                plus_hat = np.zeros_like(
                    u_hat
                )

                for a, j in enumerate(
                    shell_ids
                ):
                    plus_hat[
                        :, masks[j]
                    ] = (
                        amplitudes[a] *
                        u_hat[:, masks[j]]
                    )

                plus = np.real(
                    np.fft.ifftn(
                        plus_hat,
                        axes=(1, 2, 3)
                    )
                )

                plus = self.fl.leray_project(
                    plus
                )

                pn = self.fl.norm_l3(plus)

                if pn > 1e-30:
                    plus /= pn

                cp = self.critical_quotient(
                    plus
                )

                amplitudes[i] = old - eps

                minus_hat = np.zeros_like(
                    u_hat
                )

                for a, j in enumerate(
                    shell_ids
                ):
                    minus_hat[
                        :, masks[j]
                    ] = (
                        amplitudes[a] *
                        u_hat[:, masks[j]]
                    )

                minus = np.real(
                    np.fft.ifftn(
                        minus_hat,
                        axes=(1, 2, 3)
                    )
                )

                minus = self.fl.leray_project(
                    minus
                )

                mn = self.fl.norm_l3(minus)

                if mn > 1e-30:
                    minus /= mn

                cm = self.critical_quotient(
                    minus
                )

                gradients[i] = (
                    cp - cm
                ) / (2.0 * eps)

                amplitudes[i] = old

            amplitudes += (
                learning_rate *
                gradients
            )

            amplitudes = np.maximum(
                amplitudes,
                0.0
            )

        return {
            "shells": shell_ids,
            "amplitudes": amplitudes,
            "history": history,
            "maximum_C3": max(history)
            if history
            else 0.0
        }

    # ================================================================
    # POWER-LAW FIT
    # ================================================================

    @staticmethod
    def power_law_fit(
        x,
        y
    ):
        """
        Fit y ~ A x^alpha.

        Returns A, alpha, R^2.
        """

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        valid = (
            (x > 0) &
            (y > 0) &
            np.isfinite(x) &
            np.isfinite(y)
        )

        x = x[valid]
        y = y[valid]

        if len(x) < 2:
            return np.nan, np.nan, np.nan

        X = np.log(x)
        Y = np.log(y)

        slope, intercept = np.polyfit(
            X,
            Y,
            1
        )

        prediction = (
            slope * X +
            intercept
        )

        ss_res = np.sum(
            (Y - prediction) ** 2
        )

        ss_tot = np.sum(
            (Y - np.mean(Y)) ** 2
        )

        r2 = (
            1.0 - ss_res / ss_tot
            if ss_tot > 0
            else 1.0
        )

        return (
            float(np.exp(intercept)),
            float(slope),
            float(r2)
        )

    # ================================================================
    # ZETA / SHELL SUM AUDIT
    # ================================================================

    def shell_decay_audit(
        self,
        shell_values
    ):
        """
        Given a dictionary

            shell -> positive quantity,

        estimate a power-law exponent and compare partial sums.
        """

        shells = np.array(
            sorted(shell_values.keys()),
            dtype=float
        )

        values = np.array([
            shell_values[int(j)]
            for j in shells
        ])

        A, alpha, r2 = (
            self.power_law_fit(
                shells,
                values
            )
        )

        partial_sums = np.cumsum(
            values
        )

        return {
            "A": A,
            "alpha": alpha,
            "R2": r2,
            "partial_sums": partial_sums,
            "shells": shells,
            "values": values
        }

    # ================================================================
    # VARIATIONAL FINITE-DIFFERENCE CHECK
    # ================================================================

    def variational_gradient_check(
        self,
        u,
        seed=123,
        epsilon=1e-5
    ):
        """
        Check a directional derivative numerically.

        This does not derive an analytic gradient; it detects whether
        an implemented gradient direction is at least consistent
        with finite differences.
        """

        rng = np.random.default_rng(
            seed
        )

        direction = rng.normal(
            size=u.shape
        )

        direction = self.fl.leray_project(
            direction
        )

        dn = self.fl.norm_l2(
            direction
        )

        if dn > 1e-30:
            direction /= dn

        plus = (
            u +
            epsilon * direction
        )

        minus = (
            u -
            epsilon * direction
        )

        fp = self.critical_quotient(
            plus
        )

        fm = self.critical_quotient(
            minus
        )

        numerical_derivative = (
            fp - fm
        ) / (
            2.0 * epsilon
        )

        return {
            "finite_difference_directional_derivative":
                float(numerical_derivative),
            "epsilon": epsilon
        }


# ====================================================================
# COMPLETE TEST HARNESS
# ====================================================================

def run_discrete_proof_audit(
    resolutions=(16, 32, 64),
    seed=42
):
    """
    Run the complete one-file audit.

    Requires StructurePreservingLatticeFluid3D to be defined/imported
    in the same execution environment.
    """

    print("=" * 78)
    print("DISCRETE NAVIER-STOKES PROOF / AUDIT")
    print("=" * 78)
    print()
    print("STATUS: EXPERIMENTAL — NOT A CONTINUUM PROOF")
    print()

    resolution_results = []

    # ---------------------------------------------------------------
    # RESOLUTION AUDIT
    # ---------------------------------------------------------------

    for N in resolutions:

        print("-" * 78)
        print(f"GRID RESOLUTION N = {N}")
        print("-" * 78)

        fluid = StructurePreservingLatticeFluid3D(
            N=N,
            L=2.0 * np.pi,
            nu=0.01
        )

        audit = DiscreteProofAudit(
            fluid
        )

        u = (
            fluid.random_divergence_free_field(
                seed=seed,
                amplitude=1.0
            )
        )

        # -----------------------------------------------------------
        # BASIC QUOTIENT
        # -----------------------------------------------------------

        l3 = fluid.norm_l3(u)
        d3 = audit.critical_dissipation(u)
        wp = audit.pressure_work(u)
        c3 = audit.critical_quotient(u)
        r3 = audit.chain_rule_defect(u)

        print(
            f"C3                       = {c3:.8e}"
        )
        print(
            f"R_h                      = {r3:.8e}"
        )
        print(
            f"||u||_3                  = {l3:.8e}"
        )
        print(
            f"D3                       = {d3:.8e}"
        )
        print(
            f"|<grad p, grad q>|       = {wp:.8e}"
        )

        resolution_results.append(
            audit.resolution_measurement(u)
        )

        # -----------------------------------------------------------
        # AMPLITUDE INVARIANCE
        # -----------------------------------------------------------

        print()
        print("AMPLITUDE INVARIANCE")

        amp_results = (
            audit.amplitude_invariance_test(
                u
            )
        )

        for result in amp_results:
            print(
                f"A={result.amplitude:6.2f}  "
                f"C3={result.c3:.8e}  "
                f"R_h={result.r3:.8e}"
            )

        # -----------------------------------------------------------
        # SHELL / SCHUR AUDIT
        # -----------------------------------------------------------

        print()
        print("DYADIC SCHUR AUDIT")

        schur = audit.schur_audit(
            u,
            shell_type="dyadic"
        )

        print(
            f"Active shells              : "
            f"{len(schur['shells'])}"
        )

        print(
            f"sup_j row sum              : "
            f"{schur['sup_row_sum']:.8e}"
        )

        Gamma = schur["Gamma"]

        if Gamma.size:
            print(
                f"Diagonal max Gamma        : "
                f"{np.max(np.diag(Gamma)):.8e}"
            )

            if Gamma.shape[0] > 1:
                offdiag = (
                    Gamma -
                    np.diag(
                        np.diag(Gamma)
                    )
                )

                print(
                    f"Off-diagonal max Gamma   : "
                    f"{np.max(offdiag):.8e}"
                )

        # -----------------------------------------------------------
        # TRIADIC SOURCE AUDIT
        # -----------------------------------------------------------

        print()
        print("NONLINEAR SOURCE / TRIADIC AUDIT")

        triadic = audit.triadic_source_audit(
            u,
            shell_type="dyadic"
        )

        for j, data in triadic.items():
            print(
                f"shell {j:2d}: "
                f"B={data['B_source_norm']:.6e}  "
                f"Q={data['Q_source_norm']:.6e}"
            )

        # -----------------------------------------------------------
        # ALIASING
        # -----------------------------------------------------------

        print()
        print("ALIASING STRESS TEST")

        alias = audit.aliasing_audit(
            u
        )

        print(
            f"C3 raw                    : "
            f"{alias['C3_raw']:.8e}"
        )

        print(
            f"C3 dealiased              : "
            f"{alias['C3_dealiased']:.8e}"
        )

        print(
            f"Relative difference       : "
            f"{alias['relative_difference']:.8e}"
        )

        # -----------------------------------------------------------
        # ADVERSARIAL SEARCH
        # -----------------------------------------------------------

        print()
        print("MULTI-START ADVERSARIAL SEARCH")

        adversary = (
            audit.adversarial_search(
                starts=5,
                steps=10,
                lr=0.01,
                seed=seed + N,
                normalize="L3"
            )
        )

        print(
            f"Best discovered C3       : "
            f"{adversary['best_C3']:.8e}"
        )

        # -----------------------------------------------------------
        # VARIATIONAL CHECK
        # -----------------------------------------------------------

        print()
        print("VARIATIONAL FINITE-DIFFERENCE CHECK")

        check = (
            audit.variational_gradient_check(
                u
            )
        )

        print(
            f"Directional derivative   : "
            f"{check['finite_difference_directional_derivative']:.8e}"
        )

    # =================================================================
    # FINAL SUMMARY
    # =================================================================

    print()
    print("=" * 78)
    print("RESOLUTION SUMMARY")
    print("=" * 78)

    for result in resolution_results:
        print(
            f"N={result.N:4d}   "
            f"C3={result.c3:.8e}   "
            f"D3={result.d3:.8e}"
        )

    print()
    print("=" * 78)
    print("MATHEMATICAL STATUS")
    print("=" * 78)

    print(
        """
The audit can test the following conjectural structure:

    |<grad p, grad q>|
        <= C ||u||_3 D3(u)

using shell, triadic, concentration, aliasing, and adversarial
numerical experiments.

The following implication is NOT made:

    finite numerical sup(C3)
        =>
    uniform continuum bound.

In particular, output-shell orthogonality

    <Delta_j grad p, Delta_k grad q> = 0,  j != k

does not by itself prove a nonlinear cross-scale estimate.

The principal unresolved analytical target is a uniform estimate on
the TRIADIC SOURCE INTERACTIONS, valid for every smooth
divergence-free velocity field and stable under the continuum limit.

====================================================================
"""
    )

    return resolution_results


# ====================================================================
# OPTIONAL SELF-TEST
# ====================================================================

if __name__ == "__main__":

    try:
        results = run_discrete_proof_audit(
            resolutions=(16, 32, 64),
            seed=42
        )

    except NameError:
        print(
            "ERROR: StructurePreservingLatticeFluid3D is not defined."
        )
        print(
            "Place the existing MAC fluid-engine class above this "
            "audit class, or import it into this file."
        )
