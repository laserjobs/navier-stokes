"""
==============================================================================
DISCRETE NAVIER-STOKES PROOF / AUDIT
==============================================================================

STATUS
------
EXPERIMENTAL NUMERICAL AUDIT — NOT A CONTINUUM PROOF.

This single file contains both:

    1. The structure-preserving periodic lattice Navier-Stokes engine.
    2. The numerical audit framework.

The computations test finite-dimensional discrete identities and
candidate estimates. They do NOT establish global regularity of the
three-dimensional continuum Navier-Stokes equations.

CORE QUOTIENT
-------------
    C3(u) = |<grad p, grad q>| / (||u||_3 D3(u))

where

    -Delta p = div(B_h(u,u))
    -Delta q = div(|u|u)

and

    D3(u) = integral |u| |grad u|^2 dx.

IMPORTANT
---------
The central-difference operator has additional Nyquist null modes.
The discrete Leray projection explicitly handles those modes.

Output Fourier shells with disjoint support are orthogonal. This is
an exact projection-space identity and must not be confused with a
proof of nonlinear cross-scale decay.

Numerical experiments are used here for diagnostics and falsification.
==============================================================================

Dependencies:
    Python >= 3.10
    NumPy

Run:
    python discrete_proof.py

Optional:
    python discrete_proof.py --quick
    python discrete_proof.py --full
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


# ============================================================================
# RESULT CONTAINERS
# ============================================================================

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


# ============================================================================
# STRUCTURE-PRESERVING LATTICE FLUID
# ============================================================================

class StructurePreservingLatticeFluid3D:
    """
    Structure-preserving semidiscrete 3-D Navier-Stokes system on a
    periodic cubic lattice.

    Properties:

      * D0 is the periodic central difference operator.
      * D0* = -D0.
      * The Leray projection uses the same central-difference symbol.
      * The scalar Laplacian is the standard 7-point Laplacian.
      * The convection term is the Morinishi skew form.
      * Nyquist modes are handled explicitly by the projection.
    """

    def __init__(
        self,
        N: int = 16,
        L: float = 2.0 * np.pi,
        nu: float = 0.01,
    ):
        if N < 4:
            raise ValueError("N must be at least 4.")
        if L <= 0:
            raise ValueError("L must be positive.")
        if nu < 0:
            raise ValueError("nu must be nonnegative.")

        self.N = int(N)
        self.L = float(L)
        self.h = self.L / self.N
        self.nu = float(nu)

        # ------------------------------------------------------------------
        # Physical grid
        # ------------------------------------------------------------------

        coord = np.arange(self.N) * self.h

        self.X, self.Y, self.Z = np.meshgrid(
            coord,
            coord,
            coord,
            indexing="ij",
        )

        # ------------------------------------------------------------------
        # Fourier wave numbers
        # ------------------------------------------------------------------

        k = 2.0 * np.pi * np.fft.fftfreq(
            self.N,
            d=self.h,
        )

        self.Kx, self.Ky, self.Kz = np.meshgrid(
            k,
            k,
            k,
            indexing="ij",
        )

        # ------------------------------------------------------------------
        # Central-difference Fourier symbol
        #
        # D0 -> i sin(kh)/h
        # ------------------------------------------------------------------

        self.K_tilde_x = np.sin(self.Kx * self.h) / self.h
        self.K_tilde_y = np.sin(self.Ky * self.h) / self.h
        self.K_tilde_z = np.sin(self.Kz * self.h) / self.h

        self.K_sq = (
            self.K_tilde_x ** 2
            + self.K_tilde_y ** 2
            + self.K_tilde_z ** 2
        )

        # Central differences vanish at zero and at Nyquist.
        self.projectable_modes = self.K_sq > 1e-14

        self.inv_K_sq = np.zeros_like(self.K_sq)

        self.inv_K_sq[self.projectable_modes] = (
            1.0 / self.K_sq[self.projectable_modes]
        )

    # ======================================================================
    # FINITE DIFFERENCES
    # ======================================================================

    def D0(self, f, axis: int):
        """Periodic central difference."""
        return (
            np.roll(f, -1, axis=axis)
            - np.roll(f, 1, axis=axis)
        ) / (2.0 * self.h)

    def D_plus(self, f, axis: int):
        """Periodic forward difference."""
        return (
            np.roll(f, -1, axis=axis) - f
        ) / self.h

    def D_minus(self, f, axis: int):
        """Periodic backward difference."""
        return (
            f - np.roll(f, 1, axis=axis)
        ) / self.h

    # ======================================================================
    # DIVERGENCE / GRADIENT
    # ======================================================================

    def divergence(self, u):
        return (
            self.D0(u[0], 0)
            + self.D0(u[1], 1)
            + self.D0(u[2], 2)
        )

    def divergence_l2(self, u):
        d = self.divergence(u)
        return np.sqrt(
            np.sum(d * d) * self.h ** 3
        )

    def divergence_linf(self, u):
        return float(
            np.max(np.abs(self.divergence(u)))
        )

    def gradient(self, p):
        return np.stack(
            [
                self.D0(p, 0),
                self.D0(p, 1),
                self.D0(p, 2),
            ],
            axis=0,
        )

    # ======================================================================
    # LAPLACIAN
    # ======================================================================

    def laplacian(self, f):
        lap = np.zeros_like(f)

        for axis in range(3):
            lap += (
                np.roll(f, -1, axis=axis)
                - 2.0 * f
                + np.roll(f, 1, axis=axis)
            ) / self.h ** 2

        return lap

    # ======================================================================
    # LERAY PROJECTION
    # ======================================================================

    def leray_project(self, u):
        """
        Orthogonal projection onto the kernel of the central-difference
        divergence.

        Modes where the central derivative symbol vanishes are left
        unchanged.
        """

        u_hat = np.fft.fftn(
            u,
            axes=(1, 2, 3),
        )

        kdotu = (
            self.K_tilde_x * u_hat[0]
            + self.K_tilde_y * u_hat[1]
            + self.K_tilde_z * u_hat[2]
        )

        correction = (
            kdotu * self.inv_K_sq
        )

        projected_hat = np.empty_like(u_hat)

        projected_hat[0] = (
            u_hat[0]
            - self.K_tilde_x * correction
        )

        projected_hat[1] = (
            u_hat[1]
            - self.K_tilde_y * correction
        )

        projected_hat[2] = (
            u_hat[2]
            - self.K_tilde_z * correction
        )

        return np.real(
            np.fft.ifftn(
                projected_hat,
                axes=(1, 2, 3),
            )
        )

    # ======================================================================
    # SKEW CONVECTION
    # ======================================================================

    def skew_convection(self, u, v):
        """
        Morinishi skew-symmetric convection:

            B_i(u,v)
              = 1/2 sum_j [
                    u_j D_j v_i
                    + D_j(u_j v_i)
                ]

        For divergence-free u this has the expected discrete
        skew-symmetry.
        """

        B = np.zeros_like(v)

        for i in range(3):
            for j in range(3):
                term1 = (
                    u[j] * self.D0(v[i], j)
                )

                term2 = self.D0(
                    u[j] * v[i],
                    j,
                )

                B[i] += 0.5 * (
                    term1 + term2
                )

        return B

    # ======================================================================
    # NAVIER-STOKES RHS
    # ======================================================================

    def rhs(self, u):
        convection = self.skew_convection(
            u,
            u,
        )

        projected_convection = (
            self.leray_project(convection)
        )

        diffusion = np.stack(
            [
                self.laplacian(u[0]),
                self.laplacian(u[1]),
                self.laplacian(u[2]),
            ],
            axis=0,
        )

        return (
            -projected_convection
            + self.nu * diffusion
        )

    # ======================================================================
    # INNER PRODUCTS / NORMS
    # ======================================================================

    def inner_product(self, u, v):
        return float(
            np.sum(u * v) * self.h ** 3
        )

    def norm_l2(self, u):
        return float(
            np.sqrt(
                max(
                    self.inner_product(u, u),
                    0.0,
                )
            )
        )

    def norm_l3(self, u):
        magnitude = np.sqrt(
            np.sum(u ** 2, axis=0)
        )

        return float(
            np.sum(magnitude ** 3)
            * self.h ** 3
        ) ** (1.0 / 3.0)

    def energy(self, u):
        return 0.5 * self.inner_product(u, u)

    # ======================================================================
    # ENSTROPHY / ENERGY
    # ======================================================================

    def enstrophy(self, u):
        result = 0.0

        for i in range(3):
            for j in range(3):
                d = self.D_plus(
                    u[i],
                    j,
                )

                result += self.inner_product(
                    d,
                    d,
                )

        return float(result)

    def energy_derivative(self, u):
        return float(
            -self.nu * self.enstrophy(u)
        )

    # ======================================================================
    # RANDOM DIVERGENCE-FREE FIELD
    # ======================================================================

    def random_divergence_free_field(
        self,
        seed: int = 1234,
        amplitude: float = 1.0,
    ):
        rng = np.random.default_rng(seed)

        u = rng.standard_normal(
            (3, self.N, self.N, self.N)
        )

        u = self.leray_project(u)

        current = self.norm_l2(u)

        if current > 0:
            u *= amplitude / current

        return u

    # ======================================================================
    # TAYLOR-GREEN VORTEX
    # ======================================================================

    def taylor_green(self, amplitude: float = 1.0):
        u = np.zeros(
            (3, self.N, self.N, self.N),
            dtype=float,
        )

        u[0] = (
            amplitude
            * np.sin(self.X)
            * np.cos(self.Y)
            * np.cos(self.Z)
        )

        u[1] = (
            -amplitude
            * np.cos(self.X)
            * np.sin(self.Y)
            * np.cos(self.Z)
        )

        return self.leray_project(u)

    # ======================================================================
    # TIME INTEGRATION
    # ======================================================================

    def rk4_step(self, u, dt):
        k1 = self.rhs(u)
        k2 = self.rhs(u + 0.5 * dt * k1)
        k3 = self.rhs(u + 0.5 * dt * k2)
        k4 = self.rhs(u + dt * k3)

        return u + (
            dt / 6.0
        ) * (
            k1
            + 2.0 * k2
            + 2.0 * k3
            + k4
        )


# ============================================================================
# DISCRETE AUDIT
# ============================================================================

class DiscreteProofAudit:
    """
    Numerical audit layer attached directly to the structure-preserving
    lattice fluid.

    Everything required by the audit is implemented here rather than
    relying on a second source file.
    """

    def __init__(self, fluid):
        self.fl = fluid

        self.Kmag = np.sqrt(
            self.fl.Kx ** 2
            + self.fl.Ky ** 2
            + self.fl.Kz ** 2
        )

    # ======================================================================
    # BASIC QUANTITIES
    # ======================================================================

    def velocity_magnitude(self, u):
        return np.sqrt(
            np.sum(u ** 2, axis=0)
            + 1e-30
        )

    def spectral_gradient_squared(self, u):
        u_hat = np.fft.fftn(
            u,
            axes=(1, 2, 3),
        )

        grad_sq = np.zeros(
            u.shape[1:],
            dtype=float,
        )

        for component in range(3):
            gx = np.real(
                np.fft.ifftn(
                    1j
                    * self.fl.K_tilde_x
                    * u_hat[component],
                    axes=(0, 1, 2),
                )
            )

            gy = np.real(
                np.fft.ifftn(
                    1j
                    * self.fl.K_tilde_y
                    * u_hat[component],
                    axes=(0, 1, 2),
                )
            )

            gz = np.real(
                np.fft.ifftn(
                    1j
                    * self.fl.K_tilde_z
                    * u_hat[component],
                    axes=(0, 1, 2),
                )
            )

            grad_sq += (
                gx ** 2
                + gy ** 2
                + gz ** 2
            )

        return grad_sq

    def critical_dissipation(self, u):
        magnitude = self.velocity_magnitude(u)
        grad_sq = self.spectral_gradient_squared(u)

        return float(
            np.sum(
                magnitude * grad_sq
            ) * self.fl.h ** 3
        )

    # ======================================================================
    # PRESSURE GRADIENTS
    # ======================================================================

    def pressure_gradients(self, u):
        """
        Computes the discrete pressure gradients.

        The same central-difference Fourier symbol used by the
        structure-preserving projection is used throughout.
        """

        # ------------------------------------------------------------------
        # Physical pressure
        # ------------------------------------------------------------------

        B = self.fl.skew_convection(
            u,
            u,
        )

        B_hat = np.fft.fftn(
            B,
            axes=(1, 2, 3),
        )

        div_B_hat = (
            1j * self.fl.K_tilde_x * B_hat[0]
            + 1j * self.fl.K_tilde_y * B_hat[1]
            + 1j * self.fl.K_tilde_z * B_hat[2]
        )

        p_hat = (
            -div_B_hat
            * self.fl.inv_K_sq
        )

        grad_p_hat = np.stack(
            [
                1j * self.fl.K_tilde_x * p_hat,
                1j * self.fl.K_tilde_y * p_hat,
                1j * self.fl.K_tilde_z * p_hat,
            ],
            axis=0,
        )

        # ------------------------------------------------------------------
        # Auxiliary pressure
        # ------------------------------------------------------------------

        magnitude = self.velocity_magnitude(u)
        abs_u_u = magnitude * u

        Q_hat = np.fft.fftn(
            abs_u_u,
            axes=(1, 2, 3),
        )

        div_Q_hat = (
            1j * self.fl.K_tilde_x * Q_hat[0]
            + 1j * self.fl.K_tilde_y * Q_hat[1]
            + 1j * self.fl.K_tilde_z * Q_hat[2]
        )

        q_hat = (
            -div_Q_hat
            * self.fl.inv_K_sq
        )

        grad_q_hat = np.stack(
            [
                1j * self.fl.K_tilde_x * q_hat,
                1j * self.fl.K_tilde_y * q_hat,
                1j * self.fl.K_tilde_z * q_hat,
            ],
            axis=0,
        )

        return grad_p_hat, grad_q_hat

    def pressure_work(self, u):
        gp_hat, gq_hat = (
            self.pressure_gradients(u)
        )

        gp = np.real(
            np.fft.ifftn(
                gp_hat,
                axes=(1, 2, 3),
            )
        )

        gq = np.real(
            np.fft.ifftn(
                gq_hat,
                axes=(1, 2, 3),
            )
        )

        return float(
            abs(
                self.fl.inner_product(
                    gp,
                    gq,
                )
            )
        )

    # ======================================================================
    # CRITICAL QUOTIENT
    # ======================================================================

    def critical_quotient(self, u):
        l3 = self.fl.norm_l3(u)
        d3 = self.critical_dissipation(u)
        wp = self.pressure_work(u)

        denominator = l3 * d3

        if denominator <= 1e-30:
            return 0.0

        return float(
            wp / denominator
        )

    # ======================================================================
    # CHAIN-RULE DEFECT
    # ======================================================================

    def chain_rule_defect(self, u):
        magnitude = self.velocity_magnitude(u)

        B = self.fl.skew_convection(
            u,
            u,
        )

        numerator = abs(
            self.fl.inner_product(
                B,
                magnitude * u,
            )
        )

        denominator = (
            self.fl.norm_l3(u)
            * self.critical_dissipation(u)
        )

        if denominator <= 1e-30:
            return 0.0

        return float(
            numerator / denominator
        )

    # ======================================================================
    # DYADIC SHELLS
    # ======================================================================

    def dyadic_shell_masks(self):
        maximum = max(
            1,
            int(
                np.log2(
                    max(
                        2,
                        self.fl.N // 2,
                    )
                )
            ),
        )

        masks = {}

        for j in range(1, maximum + 1):
            low = (
                0.0
                if j == 1
                else 2.0 ** (j - 1)
            )

            high = 2.0 ** j

            masks[j] = (
                (self.Kmag >= low)
                & (self.Kmag < high)
            )

        return masks

    # ======================================================================
    # SHELL FILTER
    # ======================================================================

    def shell_field(self, u, mask):
        u_hat = np.fft.fftn(
            u,
            axes=(1, 2, 3),
        )

        filtered = np.zeros_like(
            u_hat
        )

        filtered[:, mask] = (
            u_hat[:, mask]
        )

        return np.real(
            np.fft.ifftn(
                filtered,
                axes=(1, 2, 3),
            )
        )

    def shell_dissipation(self, u, mask):
        uj = self.shell_field(
            u,
            mask,
        )

        grad_sq = (
            self.spectral_gradient_squared(
                uj
            )
        )

        magnitude = (
            self.velocity_magnitude(u)
        )

        return float(
            np.sum(
                magnitude * grad_sq
            ) * self.fl.h ** 3
        )

    # ======================================================================
    # OUTPUT-SHELL MATRIX
    # ======================================================================

    def shell_pressure_matrix(self, u):
        masks = self.dyadic_shell_masks()

        gp_hat, gq_hat = (
            self.pressure_gradients(u)
        )

        shells = [
            j
            for j, mask in masks.items()
            if np.count_nonzero(mask) > 0
        ]

        n = len(shells)

        M = np.zeros(
            (n, n),
            dtype=float,
        )

        D = np.zeros(
            n,
            dtype=float,
        )

        parseval = (
            self.fl.h ** 3
            / self.fl.N ** 3
        )

        # Shell dissipation
        for a, j in enumerate(shells):
            D[a] = self.shell_dissipation(
                u,
                masks[j],
            )

        # Matrix
        for a, j in enumerate(shells):
            for b, k in enumerate(shells):

                gp_j = np.zeros_like(
                    gp_hat
                )

                gq_k = np.zeros_like(
                    gq_hat
                )

                gp_j[:, masks[j]] = (
                    gp_hat[:, masks[j]]
                )

                gq_k[:, masks[k]] = (
                    gq_hat[:, masks[k]]
                )

                inner = np.sum(
                    gp_j
                    * np.conj(gq_k)
                )

                M[a, b] = (
                    abs(inner)
                    * parseval
                )

        l3 = self.fl.norm_l3(u)

        Gamma = np.zeros_like(M)

        for a in range(n):
            for b in range(n):
                denominator = (
                    l3
                    * np.sqrt(
                        max(
                            D[a] * D[b],
                            0.0,
                        )
                    )
                )

                if denominator > 1e-30:
                    Gamma[a, b] = (
                        M[a, b]
                        / denominator
                    )

        return (
            shells,
            M,
            Gamma,
            D,
        )

    def schur_audit(self, u):
        shells, M, Gamma, D = (
            self.shell_pressure_matrix(u)
        )

        row_sums = (
            np.sum(Gamma, axis=1)
            if Gamma.size
            else np.zeros(0)
        )

        return {
            "shells": shells,
            "M": M,
            "Gamma": Gamma,
            "D": D,
            "row_sums": row_sums,
            "sup_row_sum": float(
                np.max(row_sums)
            )
            if row_sums.size
            else 0.0,
        }

    # ======================================================================
    # NONLINEAR SOURCE AUDIT
    # ======================================================================

    def triadic_source_audit(self, u):
        masks = self.dyadic_shell_masks()

        B = self.fl.skew_convection(
            u,
            u,
        )

        magnitude = (
            self.velocity_magnitude(u)
        )

        Q = magnitude * u

        B_hat = np.fft.fftn(
            B,
            axes=(1, 2, 3),
        )

        Q_hat = np.fft.fftn(
            Q,
            axes=(1, 2, 3),
        )

        result = {}

        for j, mask in masks.items():

            if np.count_nonzero(mask) == 0:
                continue

            B_energy = np.sum(
                np.abs(
                    B_hat[:, mask]
                ) ** 2
            )

            Q_energy = np.sum(
                np.abs(
                    Q_hat[:, mask]
                ) ** 2
            )

            result[j] = {
                "B_source_energy":
                    float(B_energy),
                "Q_source_energy":
                    float(Q_energy),
                "B_source_norm":
                    float(np.sqrt(B_energy)),
                "Q_source_norm":
                    float(np.sqrt(Q_energy)),
            }

        return result

    # ======================================================================
    # ALIASING AUDIT
    # ======================================================================

    def two_thirds_mask(self):
        cutoff = self.fl.N / 3.0

        return (
            (np.abs(self.fl.Kx) <= cutoff)
            & (np.abs(self.fl.Ky) <= cutoff)
            & (np.abs(self.fl.Kz) <= cutoff)
        )

    def dealias_field(self, u):
        u_hat = np.fft.fftn(
            u,
            axes=(1, 2, 3),
        )

        mask = self.two_thirds_mask()

        filtered = np.zeros_like(
            u_hat
        )

        filtered[:, mask] = (
            u_hat[:, mask]
        )

        result = np.real(
            np.fft.ifftn(
                filtered,
                axes=(1, 2, 3),
            )
        )

        return self.fl.leray_project(
            result
        )

    def aliasing_audit(self, u):
        raw = self.critical_quotient(u)

        dealiased_field = (
            self.dealias_field(u)
        )

        dealiased = (
            self.critical_quotient(
                dealiased_field
            )
        )

        difference = abs(
            raw - dealiased
        )

        return {
            "C3_raw": float(raw),
            "C3_dealiased":
                float(dealiased),
            "absolute_difference":
                float(difference),
            "relative_difference":
                float(
                    difference
                    / (abs(raw) + 1e-30)
                ),
        }

    # ======================================================================
    # AMPLITUDE INVARIANCE
    # ======================================================================

    def amplitude_invariance_test(
        self,
        u,
        amplitudes=(1.0, 2.0, 4.0, 8.0),
    ):
        base = self.fl.norm_l2(u)

        if base <= 1e-30:
            raise ValueError(
                "Cannot test amplitude scaling "
                "of a zero field."
            )

        u0 = u / base

        results = []

        for amplitude in amplitudes:
            ua = amplitude * u0

            l3 = self.fl.norm_l3(ua)
            d3 = self.critical_dissipation(ua)
            wp = self.pressure_work(ua)

            c3 = (
                wp / (l3 * d3)
                if l3 * d3 > 1e-30
                else 0.0
            )

            r3 = (
                self.chain_rule_defect(ua)
            )

            results.append(
                ScalingResult(
                    amplitude=float(amplitude),
                    c3=float(c3),
                    r3=float(r3),
                    l3=float(l3),
                    d3=float(d3),
                    pressure_work=float(wp),
                )
            )

        return results

    # ======================================================================
    # RESOLUTION MEASUREMENT
    # ======================================================================

    def resolution_measurement(self, u):
        l3 = self.fl.norm_l3(u)
        d3 = self.critical_dissipation(u)
        wp = self.pressure_work(u)

        c3 = (
            wp / (l3 * d3)
            if l3 * d3 > 1e-30
            else 0.0
        )

        return ResolutionResult(
            N=self.fl.N,
            c3=float(c3),
            l3=float(l3),
            d3=float(d3),
            pressure_work=float(wp),
        )

    # ======================================================================
    # ADVERSARIAL SEARCH
    # ======================================================================

    def adversarial_search(
        self,
        starts=3,
        steps=5,
        lr=0.01,
        seed=101,
    ):
        """
        Small deterministic projected search.

        This is deliberately described as a search, not as a supremum
        calculation or proof.
        """

        rng = np.random.default_rng(
            seed
        )

        best_c3 = -np.inf
        best_field = None

        for _ in range(starts):

            field_seed = int(
                rng.integers(
                    0,
                    2**31 - 1,
                )
            )

            u = (
                self.fl
                .random_divergence_free_field(
                    seed=field_seed,
                    amplitude=1.0,
                )
            )

            for _ in range(steps):

                gp_hat, gq_hat = (
                    self.pressure_gradients(u)
                )

                gp = np.real(
                    np.fft.ifftn(
                        gp_hat,
                        axes=(1, 2, 3),
                    )
                )

                gq = np.real(
                    np.fft.ifftn(
                        gq_hat,
                        axes=(1, 2, 3),
                    )
                )

                direction = (
                    self.fl.leray_project(
                        gp + gq
                    )
                )

                u = (
                    u
                    + lr * direction
                )

                norm = (
                    self.fl.norm_l3(u)
                )

                if norm > 1e-30:
                    u /= norm

                c3 = (
                    self.critical_quotient(u)
                )

                if c3 > best_c3:
                    best_c3 = c3
                    best_field = u.copy()

        return {
            "best_C3":
                float(best_c3),
            "best_field":
                best_field,
        }


# ============================================================================
# STRUCTURAL OPERATOR TESTS
# ============================================================================

def run_operator_tests():
    print("=" * 78)
    print("STRUCTURE-PRESERVING LATTICE NAVIER-STOKES TEST SUITE")
    print("=" * 78)

    N = 16
    fluid = StructurePreservingLatticeFluid3D(
        N=N,
        L=2.0 * np.pi,
        nu=0.01,
    )

    rng = np.random.default_rng(42)

    # ----------------------------------------------------------------------
    # Test 1: D0 skew-adjointness
    # ----------------------------------------------------------------------

    f = rng.standard_normal(
        (N, N, N)
    )

    g = rng.standard_normal(
        (N, N, N)
    )

    Df = fluid.D0(
        f,
        axis=0,
    )

    Dg = fluid.D0(
        g,
        axis=0,
    )

    lhs = fluid.inner_product(
        Df,
        g,
    )

    rhs = -fluid.inner_product(
        f,
        Dg,
    )

    skew_defect = (
        abs(lhs - rhs)
        / max(
            1.0,
            abs(lhs),
            abs(rhs),
        )
    )

    print(
        f"Test 1: D0 skew-adjoint defect = "
        f"{skew_defect:.3e}"
    )

    # ----------------------------------------------------------------------
    # Test 2: Divergence-free projection
    # ----------------------------------------------------------------------

    u = fluid.random_divergence_free_field(
        seed=123,
        amplitude=1.0,
    )

    div_inf = (
        fluid.divergence_linf(u)
    )

    print(
        f"Test 2: ||div_h u||_inf = "
        f"{div_inf:.3e}"
    )

    # ----------------------------------------------------------------------
    # Test 3: Projection idempotence
    # ----------------------------------------------------------------------

    u_random = rng.standard_normal(
        (3, N, N, N)
    )

    Pu = fluid.leray_project(
        u_random
    )

    P2u = fluid.leray_project(
        Pu
    )

    projection_defect = (
        fluid.norm_l2(
            P2u - Pu
        )
        / max(
            1.0,
            fluid.norm_l2(Pu),
        )
    )

    print(
        f"Test 3: ||P^2u-Pu||/||Pu|| = "
        f"{projection_defect:.3e}"
    )

    # ----------------------------------------------------------------------
    # Test 4: Convective energy cancellation
    # ----------------------------------------------------------------------

    B = fluid.skew_convection(
        u,
        u,
    )

    nonlinear_energy = (
        fluid.inner_product(
            B,
            u,
        )
    )

    nonlinear_relative = (
        abs(nonlinear_energy)
        / max(
            1.0,
            fluid.norm_l2(B)
            * fluid.norm_l2(u),
        )
    )

    print(
        f"Test 4: |<B(u,u),u>| normalized = "
        f"{nonlinear_relative:.3e}"
    )

    # ----------------------------------------------------------------------
    # Test 5: RHS energy identity
    # ----------------------------------------------------------------------

    rhs_u = fluid.rhs(u)

    numerical_dE = (
        fluid.inner_product(
            rhs_u,
            u,
        )
    )

    theoretical_dE = (
        -fluid.nu
        * fluid.enstrophy(u)
    )

    energy_defect = (
        abs(
            numerical_dE
            - theoretical_dE
        )
        / max(
            1.0,
            abs(theoretical_dE),
        )
    )

    print(
        f"Test 5: Energy identity relative defect = "
        f"{energy_defect:.3e}"
    )

    # ----------------------------------------------------------------------
    # Test 6: Taylor-Green divergence
    # ----------------------------------------------------------------------

    tg = fluid.taylor_green()

    tg_div = (
        fluid.divergence_linf(tg)
    )

    print(
        f"Test 6: Taylor-Green ||div_h u||_inf = "
        f"{tg_div:.3e}"
    )

    # ----------------------------------------------------------------------
    # PASS/FAIL
    # ----------------------------------------------------------------------

    tolerances = {
        "skew": 1e-12,
        "div": 1e-12,
        "projection": 1e-12,
        "energy": 1e-12,
        "tg": 1e-12,
    }

    passed = (
        skew_defect < tolerances["skew"]
        and div_inf < tolerances["div"]
        and projection_defect < tolerances["projection"]
        and nonlinear_relative < 1e-12
        and energy_defect < tolerances["energy"]
        and tg_div < tolerances["tg"]
    )

    print(
        "Operator tests: "
        + ("PASS" if passed else "FAIL")
    )

    if not passed:
        raise RuntimeError(
            "Structure-preserving operator "
            "tests failed."
        )

    return fluid


# ============================================================================
# NUMERICAL AUDIT
# ============================================================================

def run_discrete_proof_audit(
    resolutions: Sequence[int] = (16, 32, 64),
    seed: int = 42,
    adversarial: bool = True,
):
    print("=" * 78)
    print("DISCRETE NAVIER-STOKES PROOF / AUDIT")
    print("=" * 78)
    print()
    print(
        "STATUS: EXPERIMENTAL — NOT A CONTINUUM PROOF"
    )

    resolution_results = []

    for N in resolutions:

        print("-" * 78)
        print(
            f"GRID RESOLUTION N = {N}"
        )
        print("-" * 78)

        fluid = StructurePreservingLatticeFluid3D(
            N=N,
            L=2.0 * np.pi,
            nu=0.01,
        )

        audit = DiscreteProofAudit(
            fluid
        )

        u = (
            fluid.random_divergence_free_field(
                seed=seed,
                amplitude=1.0,
            )
        )

        # ------------------------------------------------------------------
        # Basic quotient
        # ------------------------------------------------------------------

        l3 = fluid.norm_l3(u)
        d3 = audit.critical_dissipation(u)
        wp = audit.pressure_work(u)
        c3 = audit.critical_quotient(u)
        r3 = audit.chain_rule_defect(u)

        print(
            f"N={N:4d}   "
            f"C3={c3:.8e}   "
            f"D3={d3:.8e}"
        )

        print(
            f"||u||_3={l3:.8e}   "
            f"|<grad p,grad q>|={wp:.8e}"
        )

        print(
            f"R_h={r3:.8e}"
        )

        resolution_results.append(
            audit.resolution_measurement(u)
        )

        # ------------------------------------------------------------------
        # Amplitude invariance
        # ------------------------------------------------------------------

        print()
        print("AMPLITUDE INVARIANCE")

        for result in (
            audit.amplitude_invariance_test(u)
        ):
            print(
                f"A={result.amplitude:6.2f}  "
                f"C3={result.c3:.8e}  "
                f"R_h={result.r3:.8e}"
            )

        # ------------------------------------------------------------------
        # Schur audit
        # ------------------------------------------------------------------

        print()
        print("DYADIC SCHUR AUDIT")

        schur = audit.schur_audit(u)

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
            diagonal = np.diag(Gamma)

            print(
                f"Diagonal max Gamma        : "
                f"{np.max(diagonal):.8e}"
            )

            if Gamma.shape[0] > 1:
                offdiag = (
                    Gamma
                    - np.diag(diagonal)
                )

                print(
                    f"Off-diagonal max Gamma   : "
                    f"{np.max(np.abs(offdiag)):.8e}"
                )

        # ------------------------------------------------------------------
        # Triadic source audit
        # ------------------------------------------------------------------

        print()
        print(
            "NONLINEAR SOURCE / TRIADIC AUDIT"
        )

        triadic = (
            audit.triadic_source_audit(u)
        )

        for j, data in triadic.items():
            print(
                f"shell {j:2d}: "
                f"B={data['B_source_norm']:.6e}  "
                f"Q={data['Q_source_norm']:.6e}"
            )

        # ------------------------------------------------------------------
        # Aliasing
        # ------------------------------------------------------------------

        print()
        print("ALIASING STRESS TEST")

        alias = audit.aliasing_audit(u)

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

        # ------------------------------------------------------------------
        # Adversarial search
        # ------------------------------------------------------------------

        if adversarial:
            print()
            print(
                "MULTI-START ADVERSARIAL SEARCH"
            )

            adversary = (
                audit.adversarial_search(
                    starts=3,
                    steps=5,
                    lr=0.01,
                    seed=seed + N,
                )
            )

            print(
                f"Best discovered C3       : "
                f"{adversary['best_C3']:.8e}"
            )

    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------

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

    print("=" * 78)
    print("MATHEMATICAL STATUS")
    print("=" * 78)

    print(
        """
This computation tests finite-dimensional discrete quantities.

It does NOT establish:

    sup_u C3(u) < infinity

uniformly over continuum divergence-free fields.

Likewise,

    <Delta_j grad p, Delta_k grad q> = 0,   j != k

is an output-shell orthogonality identity for disjoint Fourier
supports. It is NOT a proof of nonlinear cross-scale decay.

The nonlinear source terms contain convolution/triadic interactions.
Those interactions must be controlled separately if one seeks a
continuum-compatible estimate.

Numerical boundedness, resolution stability, successful optimization,
or failure to find a counterexample are all experimental evidence,
not a continuum theorem.
"""
    )

    print("=" * 78)
    print("AUDIT COMPLETE")
    print("=" * 78)

    return resolution_results


# ============================================================================
# COMMAND-LINE ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Unified structure-preserving discrete "
            "Navier-Stokes proof/audit."
        )
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only N=16 and skip adversarial search.",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Run N=16,32,64 with adversarial search.",
    )

    args = parser.parse_args()

    print(
        "Python:",
        __import__("sys").version.split()[0],
    )

    print(
        "NumPy:",
        np.__version__,
    )

    print()

    # Always run structural tests first.
    run_operator_tests()

    print()

    if args.quick:
        resolutions = (16,)
        use_adversarial = False
    elif args.full:
        resolutions = (16, 32, 64)
        use_adversarial = True
    else:
        resolutions = (16, 32, 64)
        use_adversarial = True

    run_discrete_proof_audit(
        resolutions=resolutions,
        seed=42,
        adversarial=use_adversarial,
    )


if __name__ == "__main__":
    main()
