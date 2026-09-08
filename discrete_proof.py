"""
====================================================================
DISCRETE NAVIER-STOKES PROOF / AUDIT FRAMEWORK
====================================================================

STATUS
------
EXPERIMENTAL NUMERICAL AUDIT — NOT A CONTINUUM PROOF.

This single file contains:
    1. The structure-preserving periodic lattice fluid engine.
    2. Discrete operator tests.
    3. The critical C3 quotient.
    4. Shell / Schur diagnostics.
    5. Nonlinear source-shell diagnostics.
    6. Aliasing stress tests.
    7. Amplitude-scaling tests.
    8. Concentration tests.
    9. Multi-start adversarial searches.
   10. Reproducible resolution summaries.

The computations do NOT establish:
    sup_u C3(u) < infinity
or global regularity of the continuum 3-D Navier-Stokes equations.

CORE QUOTIENT
-------------
    C3(u) = |<grad p, grad q>| / (||u||_3 D3(u))

with

    -Delta_h p = div_h B_h(u,u)
    -Delta_h q = div_h(|u|u)

and

    D3(u) = integral |u| |grad_h u|^2 dx.

SPECTRAL DISTINCTION
--------------------
Disjoint OUTPUT Fourier shells of grad p and grad q are orthogonal
under the discrete Parseval inner product. This is an exact property
of the chosen Fourier shell projection, up to floating-point error.

It does NOT imply that nonlinear velocity shells decouple.

The nonlinear source terms contain convolution/triadic interactions.
Those are separately measured by the source-shell audit.

SCIENTIFIC RULE
---------------
Numerical boundedness, optimizer failure, shell orthogonality, or
resolution stability is evidence about the finite computational
experiment only. None of these observations is a continuum theorem.

====================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import sys
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


# ====================================================================
# STRUCTURE-PRESERVING LATTICE FLUID
# ====================================================================

class StructurePreservingLatticeFluid3D:
    """
    Structure-preserving semidiscrete 3-D Navier-Stokes system on a
    periodic cubic lattice.

    Operators:
      - D0: central difference, with D0* = -D0.
      - Leray projection: uses the same central-difference symbol.
      - Laplacian: standard 7-point periodic Laplacian.
      - Convection: skew-symmetric Morinishi form.

    Central differences have additional Nyquist null modes. Those
    modes are retained by the Leray projection because the discrete
    divergence imposes no constraint there.
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

        coord = np.arange(self.N, dtype=float) * self.h

        self.X, self.Y, self.Z = np.meshgrid(
            coord,
            coord,
            coord,
            indexing="ij",
        )

        k1d = 2.0 * np.pi * np.fft.fftfreq(
            self.N,
            d=self.h,
        )

        self.Kx, self.Ky, self.Kz = np.meshgrid(
            k1d,
            k1d,
            k1d,
            indexing="ij",
        )

        # Central-difference Fourier symbol:
        # D0 -> i sin(kh)/h.
        self.K_tilde_x = np.sin(self.Kx * self.h) / self.h
        self.K_tilde_y = np.sin(self.Ky * self.h) / self.h
        self.K_tilde_z = np.sin(self.Kz * self.h) / self.h

        self.K_sq = (
            self.K_tilde_x ** 2
            + self.K_tilde_y ** 2
            + self.K_tilde_z ** 2
        )

        self.projectable_modes = self.K_sq > 1e-14

        self.inv_K_sq = np.zeros_like(self.K_sq)

        self.inv_K_sq[self.projectable_modes] = (
            1.0 / self.K_sq[self.projectable_modes]
        )

    # ----------------------------------------------------------------
    # Difference operators
    # ----------------------------------------------------------------

    def D0(self, f: np.ndarray, axis: int) -> np.ndarray:
        return (
            np.roll(f, -1, axis=axis)
            - np.roll(f, 1, axis=axis)
        ) / (2.0 * self.h)

    def D_plus(self, f: np.ndarray, axis: int) -> np.ndarray:
        return (
            np.roll(f, -1, axis=axis) - f
        ) / self.h

    def D_minus(self, f: np.ndarray, axis: int) -> np.ndarray:
        return (
            f - np.roll(f, 1, axis=axis)
        ) / self.h

    # ----------------------------------------------------------------
    # Differential operators
    # ----------------------------------------------------------------

    def divergence(self, u: np.ndarray) -> np.ndarray:
        return (
            self.D0(u[0], 0)
            + self.D0(u[1], 1)
            + self.D0(u[2], 2)
        )

    def gradient(self, p: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                self.D0(p, 0),
                self.D0(p, 1),
                self.D0(p, 2),
            ],
            axis=0,
        )

    def laplacian(self, f: np.ndarray) -> np.ndarray:
        lap = np.zeros_like(f)

        for axis in range(3):
            lap += (
                np.roll(f, -1, axis=axis)
                - 2.0 * f
                + np.roll(f, 1, axis=axis)
            ) / self.h ** 2

        return lap

    # ----------------------------------------------------------------
    # Leray projection
    # ----------------------------------------------------------------

    def leray_project(self, u: np.ndarray) -> np.ndarray:
        u_hat = np.fft.fftn(
            u,
            axes=(1, 2, 3),
        )

        kdotu = (
            self.K_tilde_x * u_hat[0]
            + self.K_tilde_y * u_hat[1]
            + self.K_tilde_z * u_hat[2]
        )

        correction = kdotu * self.inv_K_sq

        projected = np.empty_like(u_hat)

        projected[0] = (
            u_hat[0]
            - self.K_tilde_x * correction
        )

        projected[1] = (
            u_hat[1]
            - self.K_tilde_y * correction
        )

        projected[2] = (
            u_hat[2]
            - self.K_tilde_z * correction
        )

        return np.real(
            np.fft.ifftn(
                projected,
                axes=(1, 2, 3),
            )
        )

    # ----------------------------------------------------------------
    # Skew convection
    # ----------------------------------------------------------------

    def skew_convection(
        self,
        u: np.ndarray,
        v: np.ndarray,
    ) -> np.ndarray:
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

    # ----------------------------------------------------------------
    # Navier-Stokes RHS
    # ----------------------------------------------------------------

    def rhs(self, u: np.ndarray) -> np.ndarray:
        conv = self.skew_convection(u, u)
        projected_conv = self.leray_project(conv)

        diff = np.stack(
            [
                self.laplacian(u[0]),
                self.laplacian(u[1]),
                self.laplacian(u[2]),
            ],
            axis=0,
        )

        return (
            -projected_conv
            + self.nu * diff
        )

    # ----------------------------------------------------------------
    # Inner products / norms
    # ----------------------------------------------------------------

    def inner_product(
        self,
        u: np.ndarray,
        v: np.ndarray,
    ) -> float:
        return float(
            np.sum(u * v) * self.h ** 3
        )

    def norm_l2(self, u: np.ndarray) -> float:
        value = self.inner_product(u, u)
        return float(np.sqrt(max(value, 0.0)))

    def norm_l3(self, u: np.ndarray) -> float:
        magnitude = np.sqrt(
            np.sum(u ** 2, axis=0)
        )

        return float(
            np.sum(magnitude ** 3) * self.h ** 3
        ) ** (1.0 / 3.0)

    def energy(self, u: np.ndarray) -> float:
        return 0.5 * self.inner_product(u, u)

    # ----------------------------------------------------------------
    # Enstrophy / divergence
    # ----------------------------------------------------------------

    def enstrophy(self, u: np.ndarray) -> float:
        value = 0.0

        for i in range(3):
            for j in range(3):
                d = self.D_plus(u[i], j)
                value += self.inner_product(d, d)

        return float(value)

    def divergence_l2(self, u: np.ndarray) -> float:
        d = self.divergence(u)

        return float(
            np.sqrt(
                np.sum(d ** 2) * self.h ** 3
            )
        )

    def divergence_linf(self, u: np.ndarray) -> float:
        return float(
            np.max(
                np.abs(
                    self.divergence(u)
                )
            )
        )

    def energy_derivative(self, u: np.ndarray) -> float:
        return -self.nu * self.enstrophy(u)

    # ----------------------------------------------------------------
    # Random divergence-free field
    # ----------------------------------------------------------------

    def random_divergence_free_field(
        self,
        seed: int = 1234,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)

        u = rng.standard_normal(
            (3, self.N, self.N, self.N)
        )

        u = self.leray_project(u)

        current = self.norm_l2(u)

        if current > 0.0:
            u *= amplitude / current

        return u

    # ----------------------------------------------------------------
    # Taylor-Green vortex
    # ----------------------------------------------------------------

    def taylor_green(
        self,
        amplitude: float = 1.0,
    ) -> np.ndarray:
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

    # ----------------------------------------------------------------
    # RK4
    # ----------------------------------------------------------------

    def rk4_step(
        self,
        u: np.ndarray,
        dt: float,
    ) -> np.ndarray:
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

    def evolve(
        self,
        u0: np.ndarray,
        dt: float,
        steps: int,
    ) -> list[np.ndarray]:
        u = np.array(
            u0,
            dtype=float,
            copy=True,
        )

        trajectory = [u.copy()]

        for _ in range(steps):
            u = self.rk4_step(u, dt)
            trajectory.append(u.copy())

        return trajectory


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
    Unified numerical audit attached directly to the lattice engine.
    """

    def __init__(
        self,
        fluid: StructurePreservingLatticeFluid3D,
    ):
        self.fl = fluid

        self.Kmag = np.sqrt(
            self.fl.Kx ** 2
            + self.fl.Ky ** 2
            + self.fl.Kz ** 2
        )

    # ----------------------------------------------------------------
    # Basic quantities
    # ----------------------------------------------------------------

    def velocity_magnitude(
        self,
        u: np.ndarray,
    ) -> np.ndarray:
        return np.sqrt(
            np.sum(u ** 2, axis=0)
            + 1e-30
        )

    def gradient_squared(
        self,
        u: np.ndarray,
    ) -> np.ndarray:
        """
        Spectral gradient norm using the SAME central-difference
        symbol used by the lattice projection.
        """
        u_hat = np.fft.fftn(
            u,
            axes=(1, 2, 3),
        )

        grad_sq = np.zeros_like(
            u[0],
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

    def critical_dissipation(
        self,
        u: np.ndarray,
    ) -> float:
        u_mag = self.velocity_magnitude(u)
        grad_sq = self.gradient_squared(u)

        return float(
            np.sum(
                u_mag * grad_sq
            ) * self.fl.h ** 3
        )

    # ----------------------------------------------------------------
    # Pressure gradients
    # ----------------------------------------------------------------

    def pressure_gradients(
        self,
        u: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the two pressure gradients using the same central
        derivative symbol as the discrete divergence.

            -Delta_c p = div_c B(u,u)
            -Delta_c q = div_c(|u|u)

        where

            Delta_c = div_c grad_c.

        This is intentionally consistent with the Leray projection.
        """

        # Physical pressure.
        B = self.fl.skew_convection(u, u)

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
                1j
                * self.fl.K_tilde_x
                * p_hat,
                1j
                * self.fl.K_tilde_y
                * p_hat,
                1j
                * self.fl.K_tilde_z
                * p_hat,
            ],
            axis=0,
        )

        # Auxiliary pressure.
        u_mag = self.velocity_magnitude(u)
        abs_u_u = u_mag * u

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
                1j
                * self.fl.K_tilde_x
                * q_hat,
                1j
                * self.fl.K_tilde_y
                * q_hat,
                1j
                * self.fl.K_tilde_z
                * q_hat,
            ],
            axis=0,
        )

        return grad_p_hat, grad_q_hat

    def pressure_work(
        self,
        u: np.ndarray,
    ) -> float:
        gp_hat, gq_hat = (
            self.pressure_gradients(u)
        )

        # Parseval is both faster and less susceptible to an
        # unnecessary pair of inverse transforms.
        parseval = (
            self.fl.h ** 3
            / self.fl.N ** 3
        )

        inner = np.sum(
            gp_hat
            * np.conj(gq_hat)
        )

        return float(
            abs(inner)
            * parseval
        )

    # ----------------------------------------------------------------
    # Critical quotient
    # ----------------------------------------------------------------

    def critical_quotient(
        self,
        u: np.ndarray,
    ) -> float:
        l3 = self.fl.norm_l3(u)
        d3 = self.critical_dissipation(u)
        wp = self.pressure_work(u)

        denominator = l3 * d3

        if denominator <= 1e-30:
            return 0.0

        return float(
            wp / denominator
        )

    # ----------------------------------------------------------------
    # Chain-rule defect
    # ----------------------------------------------------------------

    def chain_rule_defect(
        self,
        u: np.ndarray,
    ) -> float:
        u_mag = self.velocity_magnitude(u)

        B = self.fl.skew_convection(
            u,
            u,
        )

        numerator = abs(
            self.fl.inner_product(
                B,
                u_mag * u,
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

    # ----------------------------------------------------------------
    # Shell definitions
    # ----------------------------------------------------------------

    def linear_shell_masks(
        self,
        jmax: Optional[int] = None,
    ) -> Dict[int, np.ndarray]:
        if jmax is None:
            jmax = max(
                1,
                self.fl.N // 3,
            )

        masks = {}

        for j in range(1, jmax + 1):
            masks[j] = (
                (self.Kmag >= j - 1.0)
                & (self.Kmag < float(j))
            )

        return masks

    def dyadic_shell_masks(
        self,
        oct_max: Optional[int] = None,
    ) -> Dict[int, np.ndarray]:
        if oct_max is None:
            oct_max = max(
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

        for j in range(1, oct_max + 1):
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

    # ----------------------------------------------------------------
    # Shell filtering
    # ----------------------------------------------------------------

    def shell_field(
        self,
        u: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        u_hat = np.fft.fftn(
            u,
            axes=(1, 2, 3),
        )

        filtered = np.zeros_like(u_hat)

        filtered[:, mask] = (
            u_hat[:, mask]
        )

        return np.real(
            np.fft.ifftn(
                filtered,
                axes=(1, 2, 3),
            )
        )

    def shell_dissipation(
        self,
        u: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        u_mag = self.velocity_magnitude(u)
        u_shell = self.shell_field(
            u,
            mask,
        )

        grad_sq = self.gradient_squared(
            u_shell
        )

        return float(
            np.sum(
                u_mag * grad_sq
            ) * self.fl.h ** 3
        )

    # ----------------------------------------------------------------
    # Output-shell pressure matrix
    # ----------------------------------------------------------------

    def shell_pressure_matrix(
        self,
        u: np.ndarray,
        shell_type: str = "dyadic",
    ):
        """
        M_jk =
            |<Delta_j grad p, Delta_k grad q>|.

        Since the Fourier masks are disjoint for j != k, those
        entries should be zero up to floating-point roundoff.

        IMPORTANT:
        This is output-shell orthogonality, not a nonlinear
        cross-scale estimate.
        """

        if shell_type == "dyadic":
            masks = self.dyadic_shell_masks()
        elif shell_type == "linear":
            masks = self.linear_shell_masks()
        else:
            raise ValueError(
                "shell_type must be 'dyadic' or 'linear'."
            )

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

        # Pre-filter once.
        gp_shell = {}
        gq_shell = {}

        for j in shells:
            mask = masks[j]

            gp = np.zeros_like(gp_hat)
            gq = np.zeros_like(gq_hat)

            gp[:, mask] = gp_hat[:, mask]
            gq[:, mask] = gq_hat[:, mask]

            gp_shell[j] = gp
            gq_shell[j] = gq

            D[shells.index(j)] = (
                self.shell_dissipation(
                    u,
                    mask,
                )
            )

        for a, j in enumerate(shells):
            for b, k in enumerate(shells):
                inner = np.sum(
                    gp_shell[j]
                    * np.conj(gq_shell[k])
                )

                M[a, b] = float(
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

        return shells, M, Gamma, D

    # ----------------------------------------------------------------
    # Schur audit
    # ----------------------------------------------------------------

    def schur_audit(
        self,
        u: np.ndarray,
        shell_type: str = "dyadic",
    ):
        shells, M, Gamma, D = (
            self.shell_pressure_matrix(
                u,
                shell_type,
            )
        )

        row_sums = np.sum(
            Gamma,
            axis=1,
        )

        return {
            "shells": shells,
            "M": M,
            "Gamma": Gamma,
            "D": D,
            "row_sums": row_sums,
            "sup_row_sum": (
                float(np.max(row_sums))
                if len(row_sums)
                else 0.0
            ),
        }

    # ----------------------------------------------------------------
    # Nonlinear source audit
    # ----------------------------------------------------------------

    def nonlinear_sources(
        self,
        u: np.ndarray,
    ):
        B = self.fl.skew_convection(
            u,
            u,
        )

        Q = (
            self.velocity_magnitude(u)
            * u
        )

        return (
            np.fft.fftn(
                B,
                axes=(1, 2, 3),
            ),
            np.fft.fftn(
                Q,
                axes=(1, 2, 3),
            ),
        )

    def triadic_source_audit(
        self,
        u: np.ndarray,
        shell_type: str = "dyadic",
    ):
        if shell_type == "dyadic":
            masks = self.dyadic_shell_masks()
        elif shell_type == "linear":
            masks = self.linear_shell_masks()
        else:
            raise ValueError(
                "shell_type must be 'dyadic' or 'linear'."
            )

        B_hat, Q_hat = (
            self.nonlinear_sources(u)
        )

        result = {}

        for j, mask in masks.items():
            if np.count_nonzero(mask) == 0:
                continue

            B_energy = float(
                np.sum(
                    np.abs(
                        B_hat[:, mask]
                    ) ** 2
                )
            )

            Q_energy = float(
                np.sum(
                    np.abs(
                        Q_hat[:, mask]
                    ) ** 2
                )
            )

            result[j] = {
                "B_source_energy": B_energy,
                "Q_source_energy": Q_energy,
                "B_source_norm": float(
                    np.sqrt(B_energy)
                ),
                "Q_source_norm": float(
                    np.sqrt(Q_energy)
                ),
            }

        return result

    # ----------------------------------------------------------------
    # Aliasing
    # ----------------------------------------------------------------

    def two_thirds_mask(self):
        cutoff = self.fl.N / 3.0

        return (
            (np.abs(self.fl.Kx) <= cutoff)
            & (np.abs(self.fl.Ky) <= cutoff)
            & (np.abs(self.fl.Kz) <= cutoff)
        )

    def dealias_field(
        self,
        u: np.ndarray,
    ) -> np.ndarray:
        u_hat = np.fft.fftn(
            u,
            axes=(1, 2, 3),
        )

        mask = self.two_thirds_mask()

        filtered = np.zeros_like(u_hat)
        filtered[:, mask] = (
            u_hat[:, mask]
        )

        return self.fl.leray_project(
            np.real(
                np.fft.ifftn(
                    filtered,
                    axes=(1, 2, 3),
                )
            )
        )

    def aliasing_audit(
        self,
        u: np.ndarray,
    ):
        c_raw = self.critical_quotient(u)

        u_dealiased = (
            self.dealias_field(u)
        )

        c_dealiased = (
            self.critical_quotient(
                u_dealiased
            )
        )

        absolute_difference = abs(
            c_raw - c_dealiased
        )

        relative_difference = (
            absolute_difference
            / max(
                abs(c_raw),
                1e-30,
            )
        )

        return {
            "C3_raw": float(c_raw),
            "C3_dealiased": float(
                c_dealiased
            ),
            "absolute_difference": float(
                absolute_difference
            ),
            "relative_difference": float(
                relative_difference
            ),
        }

    # ----------------------------------------------------------------
    # Amplitude invariance
    # ----------------------------------------------------------------

    def amplitude_invariance_test(
        self,
        u: np.ndarray,
        amplitudes: Iterable[float] = (
            1.0,
            2.0,
            4.0,
            8.0,
        ),
    ):
        base_norm = self.fl.norm_l2(u)

        if base_norm <= 0:
            raise ValueError(
                "Input field has zero L2 norm."
            )

        u0 = u / base_norm

        results = []

        for amplitude in amplitudes:
            ua = amplitude * u0

            l3 = self.fl.norm_l3(ua)
            d3 = self.critical_dissipation(ua)
            wp = self.pressure_work(ua)

            c3 = (
                wp
                / max(
                    l3 * d3,
                    1e-30,
                )
            )

            r3 = self.chain_rule_defect(
                ua
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

    # ----------------------------------------------------------------
    # Concentration family
    # ----------------------------------------------------------------

    def concentration_family(
        self,
        u: np.ndarray,
        sigma: float,
        delta: float = 1.0,
    ) -> np.ndarray:
        if sigma <= 0:
            raise ValueError(
                "sigma must be positive."
            )

        u_hat = np.fft.fftn(
            u,
            axes=(1, 2, 3),
        )

        effective_k = np.sqrt(
            (sigma * self.fl.Kx) ** 2
            + (sigma * self.fl.Ky) ** 2
            + (
                sigma ** delta
                * self.fl.Kz
            ) ** 2
        )

        envelope = np.exp(
            -0.5 * effective_k ** 2
        )

        concentrated_hat = (
            u_hat * envelope
        )

        concentrated = np.real(
            np.fft.ifftn(
                concentrated_hat,
                axes=(1, 2, 3),
            )
        )

        return self.fl.leray_project(
            concentrated
        )

    def concentration_sweep(
        self,
        u: np.ndarray,
        sigmas: Iterable[float] = (
            1.0,
            0.75,
            0.5,
            0.25,
            0.125,
        ),
        deltas: Iterable[float] = (
            1.0,
            0.5,
            2.0,
        ),
    ):
        results = []

        for delta in deltas:
            for sigma in sigmas:
                us = self.concentration_family(
                    u,
                    sigma=sigma,
                    delta=delta,
                )

                l3 = self.fl.norm_l3(us)
                d3 = self.critical_dissipation(us)
                wp = self.pressure_work(us)

                c3 = (
                    wp
                    / max(
                        l3 * d3,
                        1e-30,
                    )
                )

                results.append(
                    ConcentrationResult(
                        sigma=float(sigma),
                        delta=float(delta),
                        c3=float(c3),
                        l3=float(l3),
                        d3=float(d3),
                        pressure_work=float(wp),
                    )
                )

        return results

    # ----------------------------------------------------------------
    # Resolution measurement
    # ----------------------------------------------------------------

    def resolution_measurement(
        self,
        u: np.ndarray,
    ) -> ResolutionResult:
        l3 = self.fl.norm_l3(u)
        d3 = self.critical_dissipation(u)
        wp = self.pressure_work(u)

        c3 = (
            wp
            / max(
                l3 * d3,
                1e-30,
            )
        )

        return ResolutionResult(
            N=self.fl.N,
            c3=float(c3),
            l3=float(l3),
            d3=float(d3),
            pressure_work=float(wp),
        )

    # ----------------------------------------------------------------
    # Fourier phase randomization
    # ----------------------------------------------------------------

    def randomize_fourier_phases(
        self,
        u: np.ndarray,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Randomize phases while maintaining Hermitian symmetry by
        applying random phases in physical-space-compatible Fourier
        pairs.

        For this audit the resulting field is finally projected.
        """

        rng = np.random.default_rng(seed)

        u_hat = np.fft.fftn(
            u,
            axes=(1, 2, 3),
        )

        randomized = u_hat.copy()

        # Build the conjugate partner indices.
        n = self.fl.N

        for i in range(n):
            ii = (-i) % n

            for j in range(n):
                jj = (-j) % n

                for k in range(n):
                    kk = (-k) % n

                    partner = (
                        ii,
                        jj,
                        kk,
                    )

                    current = (
                        i,
                        j,
                        k,
                    )

                    # Process each pair once.
                    if current > partner:
                        continue

                    if current == partner:
                        randomized[
                            :, i, j, k
                        ] = np.real(
                            u_hat[
                                :, i, j, k
                            ]
                        )
                    else:
                        phase = rng.uniform(
                            0.0,
                            2.0 * np.pi,
                        )

                        randomized[
                            :, i, j, k
                        ] = (
                            np.abs(
                                u_hat[
                                    :, i, j, k
                                ]
                            )
                            * np.exp(
                                1j * phase
                            )
                        )

                        randomized[
                            :, ii, jj, kk
                        ] = np.conj(
                            randomized[
                                :, i, j, k
                            ]
                        )

        result = np.real(
            np.fft.ifftn(
                randomized,
                axes=(1, 2, 3),
            )
        )

        return self.fl.leray_project(
            result
        )

    # ----------------------------------------------------------------
    # Adversarial search
    # ----------------------------------------------------------------

    def adversarial_search(
        self,
        starts: int = 5,
        steps: int = 10,
        lr: float = 0.01,
        seed: int = 101,
        normalize: str = "L3",
    ):
        """
        Finite-dimensional heuristic search for large C3.

        This is a falsification tool, NOT a proof of a supremum.
        """

        if starts < 1:
            raise ValueError(
                "starts must be >= 1."
            )

        if steps < 1:
            raise ValueError(
                "steps must be >= 1."
            )

        rng = np.random.default_rng(seed)

        best_c3 = -np.inf
        best_field = None
        history = []

        for start in range(starts):
            field_seed = int(
                rng.integers(
                    0,
                    2 ** 31 - 1,
                )
            )

            u = (
                self.fl
                .random_divergence_free_field(
                    seed=field_seed,
                    amplitude=1.0,
                )
            )

            for step in range(steps):
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

                direction = self.fl.leray_project(
                    gp + gq
                )

                u = u + lr * direction
                u = self.fl.leray_project(u)

                if normalize.upper() == "L3":
                    norm = self.fl.norm_l3(u)
                else:
                    norm = self.fl.norm_l2(u)

                if norm > 1e-30:
                    u /= norm

                c3 = self.critical_quotient(u)

                history.append(
                    {
                        "start": start,
                        "step": step,
                        "C3": float(c3),
                    }
                )

                if c3 > best_c3:
                    best_c3 = c3
                    best_field = u.copy()

        return {
            "best_C3": float(best_c3),
            "best_field": best_field,
            "history": history,
        }

    # ----------------------------------------------------------------
    # Power-law fit
    # ----------------------------------------------------------------

    @staticmethod
    def power_law_fit(
        x: Sequence[float],
        y: Sequence[float],
    ):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        valid = (
            (x > 0)
            & (y > 0)
            & np.isfinite(x)
            & np.isfinite(y)
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
            1,
        )

        prediction = (
            slope * X
            + intercept
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
            float(r2),
        )

    # ----------------------------------------------------------------
    # Shell decay / zeta audit
    # ----------------------------------------------------------------

    def shell_decay_audit(
        self,
        shell_values: Dict[int, float],
    ):
        shells = np.array(
            sorted(shell_values.keys()),
            dtype=float,
        )

        values = np.array(
            [
                shell_values[int(j)]
                for j in shells
            ],
            dtype=float,
        )

        A, alpha, r2 = (
            self.power_law_fit(
                shells,
                values,
            )
        )

        return {
            "A": A,
            "alpha": alpha,
            "R2": r2,
            "partial_sums": np.cumsum(values),
            "shells": shells,
            "values": values,
        }

    # ----------------------------------------------------------------
    # Directional finite-difference check
    # ----------------------------------------------------------------

    def variational_gradient_check(
        self,
        u: np.ndarray,
        seed: int = 123,
        epsilon: float = 1e-5,
    ):
        rng = np.random.default_rng(seed)

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

        fp = self.critical_quotient(
            u + epsilon * direction
        )

        fm = self.critical_quotient(
            u - epsilon * direction
        )

        derivative = (
            fp - fm
        ) / (
            2.0 * epsilon
        )

        return {
            "finite_difference_directional_derivative":
                float(derivative),
            "epsilon": float(epsilon),
        }


# ====================================================================
# DISCRETE OPERATOR TESTS
# ====================================================================

def run_operator_tests(
    N: int = 16,
    L: float = 2.0 * np.pi,
    nu: float = 0.01,
) -> bool:
    print("=" * 78)
    print("STRUCTURE-PRESERVING LATTICE NAVIER-STOKES TEST SUITE")
    print("=" * 78)

    fluid = StructurePreservingLatticeFluid3D(
        N=N,
        L=L,
        nu=nu,
    )

    rng = np.random.default_rng(42)

    # Test 1: D0 skew-adjointness.
    f = rng.standard_normal(
        (N, N, N)
    )

    g = rng.standard_normal(
        (N, N, N)
    )

    lhs = fluid.inner_product(
        fluid.D0(f, 0),
        g,
    )

    rhs = -fluid.inner_product(
        f,
        fluid.D0(g, 0),
    )

    skew_defect = abs(lhs - rhs) / max(
        1.0,
        abs(lhs),
        abs(rhs),
    )

    print(
        f"Test 1: D0 skew-adjoint defect = "
        f"{skew_defect:.3e}"
    )

    # Test 2: divergence-free projection.
    u = fluid.random_divergence_free_field(
        seed=123,
        amplitude=1.0,
    )

    div_inf = fluid.divergence_linf(u)

    print(
        f"Test 2: ||div_h u||_inf = "
        f"{div_inf:.3e}"
    )

    # Test 3: projection idempotence.
    random_u = rng.standard_normal(
        (3, N, N, N)
    )

    Pu = fluid.leray_project(
        random_u
    )

    P2u = fluid.leray_project(
        Pu
    )

    projection_defect = (
        fluid.norm_l2(P2u - Pu)
        / max(
            1.0,
            fluid.norm_l2(Pu),
        )
    )

    print(
        f"Test 3: ||P^2u-Pu||/||Pu|| = "
        f"{projection_defect:.3e}"
    )

    # Test 4: skew convection energy cancellation.
    B = fluid.skew_convection(
        u,
        u,
    )

    nonlinear_energy = (
        fluid.inner_product(B, u)
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

    # Test 5: RHS energy identity.
    rhs_u = fluid.rhs(u)

    numerical_dE = (
        fluid.inner_product(
            rhs_u,
            u,
        )
    )

    theoretical_dE = (
        -nu * fluid.enstrophy(u)
    )

    energy_defect = abs(
        numerical_dE
        - theoretical_dE
    ) / max(
        1.0,
        abs(theoretical_dE),
    )

    print(
        f"Test 5: Energy identity relative defect = "
        f"{energy_defect:.3e}"
    )

    # Test 6: Taylor-Green divergence.
    tg = fluid.taylor_green()

    print(
        f"Test 6: Taylor-Green ||div_h u||_inf = "
        f"{fluid.divergence_linf(tg):.3e}"
    )

    # These tolerances are deliberately conservative. They detect
    # structural failures without pretending floating-point results
    # are exact symbolic identities.
    checks = [
        skew_defect < 1e-12,
        div_inf < 1e-10,
        projection_defect < 1e-12,
        nonlinear_relative < 1e-12,
        energy_defect < 1e-10,
    ]

    passed = all(checks)

    print(
        f"Operator tests: "
        f"{'PASS' if passed else 'FAIL'}"
    )

    return passed


# ====================================================================
# COMPLETE NUMERICAL AUDIT
# ====================================================================

def run_discrete_proof_audit(
    resolutions: Sequence[int] = (
        16,
        32,
        64,
    ),
    seed: int = 42,
    adversarial_starts: int = 5,
    adversarial_steps: int = 10,
):
    print("=" * 78)
    print("DISCRETE NAVIER-STOKES PROOF / AUDIT")
    print("=" * 78)
    print()
    print(
        "STATUS: EXPERIMENTAL — NOT A CONTINUUM PROOF"
    )
    print()

    resolution_results = []

    for N in resolutions:
        print("-" * 78)
        print(f"GRID RESOLUTION N = {N}")
        print("-" * 78)

        fluid = StructurePreservingLatticeFluid3D(
            N=N,
            L=2.0 * np.pi,
            nu=0.01,
        )

        audit = DiscreteProofAudit(
            fluid
        )

        u = fluid.random_divergence_free_field(
            seed=seed,
            amplitude=1.0,
        )

        # ------------------------------------------------------------
        # Basic quotient
        # ------------------------------------------------------------

        l3 = fluid.norm_l3(u)
        d3 = audit.critical_dissipation(u)
        wp = audit.pressure_work(u)
        c3 = audit.critical_quotient(u)
        r3 = audit.chain_rule_defect(u)

        print(
            f"C3                       = "
            f"{c3:.8e}"
        )

        print(
            f"R_h                      = "
            f"{r3:.8e}"
        )

        print(
            f"||u||_3                  = "
            f"{l3:.8e}"
        )

        print(
            f"D3                       = "
            f"{d3:.8e}"
        )

        print(
            f"|<grad p, grad q>|       = "
            f"{wp:.8e}"
        )

        resolution_results.append(
            audit.resolution_measurement(u)
        )

        # ------------------------------------------------------------
        # Amplitude invariance
        # ------------------------------------------------------------

        print()
        print("AMPLITUDE INVARIANCE")

        amp_results = (
            audit.amplitude_invariance_test(u)
        )

        for result in amp_results:
            print(
                f"A={result.amplitude:6.2f}  "
                f"C3={result.c3:.8e}  "
                f"R_h={result.r3:.8e}"
            )

        # ------------------------------------------------------------
        # Schur audit
        # ------------------------------------------------------------

        print()
        print("DYADIC SCHUR AUDIT")

        schur = audit.schur_audit(
            u,
            shell_type="dyadic",
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
            diagonal_max = float(
                np.max(
                    np.diag(Gamma)
                )
            )

            print(
                f"Diagonal max Gamma        : "
                f"{diagonal_max:.8e}"
            )

            if Gamma.shape[0] > 1:
                offdiag = (
                    Gamma
                    - np.diag(
                        np.diag(Gamma)
                    )
                )

                offdiag_max = float(
                    np.max(
                        np.abs(offdiag)
                    )
                )

                print(
                    f"Off-diagonal max Gamma   : "
                    f"{offdiag_max:.8e}"
                )

        # ------------------------------------------------------------
        # Triadic source audit
        # ------------------------------------------------------------

        print()
        print(
            "NONLINEAR SOURCE / TRIADIC AUDIT"
        )

        triadic = (
            audit.triadic_source_audit(
                u,
                shell_type="dyadic",
            )
        )

        for j, data in triadic.items():
            print(
                f"shell {j:2d}: "
                f"B={data['B_source_norm']:.6e}  "
                f"Q={data['Q_source_norm']:.6e}"
            )

        # ------------------------------------------------------------
        # Aliasing
        # ------------------------------------------------------------

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

        # ------------------------------------------------------------
        # Concentration
        # ------------------------------------------------------------

        print()
        print("CONCENTRATION SWEEP")

        concentration = (
            audit.concentration_sweep(
                u,
                sigmas=(
                    1.0,
                    0.5,
                    0.25,
                ),
                deltas=(
                    1.0,
                    0.5,
                    2.0,
                ),
            )
        )

        max_concentration = max(
            concentration,
            key=lambda x: x.c3,
        )

        print(
            f"Maximum concentration C3 : "
            f"{max_concentration.c3:.8e}"
        )

        # ------------------------------------------------------------
        # Adversarial search
        # ------------------------------------------------------------

        print()
        print("MULTI-START ADVERSARIAL SEARCH")

        adversary = (
            audit.adversarial_search(
                starts=adversarial_starts,
                steps=adversarial_steps,
                lr=0.01,
                seed=seed + N,
                normalize="L3",
            )
        )

        print(
            f"Best discovered C3       : "
            f"{adversary['best_C3']:.8e}"
        )

        # ------------------------------------------------------------
        # Variational finite difference
        # ------------------------------------------------------------

        print()
        print(
            "VARIATIONAL FINITE-DIFFERENCE CHECK"
        )

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
    # Final resolution summary
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
This computation tests the finite-dimensional quantity

    C3(u) =
        |<grad p, grad q>|
        -----------------
        ||u||_3 D3(u).

The experiment may provide evidence for or against candidate
discrete estimates.

It does NOT establish

    sup_u C3(u) < infinity

uniformly over continuum divergence-free fields.

Likewise,

    <Delta_j grad p, Delta_k grad q> = 0,  j != k

is an output-shell orthogonality identity resulting from disjoint
Fourier support. It is not a proof of nonlinear cross-scale decay.

The unresolved analytical target is a uniform estimate controlling
the nonlinear triadic source interactions in a continuum-compatible
way.

Numerical experiments are therefore used here primarily as
falsification and diagnostics.
"""
    )

    print("=" * 78)
    print("AUDIT COMPLETE")
    print("=" * 78)

    return resolution_results


# ====================================================================
# COMMAND-LINE INTERFACE
# ====================================================================

def parse_args(
    argv: Optional[Sequence[str]] = None,
):
    parser = argparse.ArgumentParser(
        description=(
            "Experimental discrete Navier-Stokes "
            "proof/audit framework."
        )
    )

    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=[16, 32, 64],
        help="Grid resolutions to audit.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--starts",
        type=int,
        default=5,
        help="Adversarial-search starting fields.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Adversarial-search steps per field.",
    )

    parser.add_argument(
        "--operators-only",
        action="store_true",
        help="Run only the structural operator tests.",
    )

    return parser.parse_args(argv)


# ====================================================================
# MAIN
# ====================================================================

def main(
    argv: Optional[Sequence[str]] = None,
) -> int:
    args = parse_args(argv)

    try:
        operator_ok = run_operator_tests(
            N=16,
            L=2.0 * np.pi,
            nu=0.01,
        )

        if not operator_ok:
            return 1

        if args.operators_only:
            return 0

        run_discrete_proof_audit(
            resolutions=args.resolutions,
            seed=args.seed,
            adversarial_starts=args.starts,
            adversarial_steps=args.steps,
        )

        return 0

    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
