"""
==============================================================================
DISCRETE NAVIER-STOKES PROOF / AUDIT
==============================================================================

STATUS
------
EXPERIMENTAL NUMERICAL AUDIT — NOT A CONTINUUM PROOF.

This program implements a structure-preserving finite-dimensional
periodic lattice model together with numerical diagnostics for the
critical quotient

    C3(u) = |<grad p, grad q>| / (||u||_3 D3(u))

where

    -Delta_h p = div_h B_h(u,u)
    -Delta_h q = div_h(|u|u)

and

    D3(u) = integral |u| |grad_h u|^2 dx.

The program tests:

    * central-difference skew-adjointness,
    * discrete divergence preservation,
    * Leray projection idempotence,
    * nonlinear energy cancellation,
    * the discrete energy identity,
    * Taylor-Green divergence,
    * amplitude scaling,
    * critical quotient values,
    * dyadic output-shell orthogonality,
    * Schur row sums,
    * nonlinear source-shell populations,
    * aliasing sensitivity,
    * multi-start numerical adversarial searches.

IMPORTANT
---------
These computations do NOT prove

    sup_u C3(u) < infinity

uniformly over continuum divergence-free fields and do not establish
global regularity of the three-dimensional Navier-Stokes equations.

For disjoint Fourier output shells,

    <Delta_j grad p, Delta_k grad q> = 0,  j != k,

is simply an orthogonality consequence of disjoint Fourier support.

It does NOT imply that nonlinear velocity-shell interactions vanish.

The nonlinear terms contain convolution/triadic interactions, so the
triadic source audit is retained separately.

All numerical results are diagnostics and potential falsification tests.
"""

from __future__ import annotations

import sys
import numpy as np
from dataclasses import dataclass


# =============================================================================
# RESULT CONTAINERS
# =============================================================================

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
    r3: float
    l3: float
    d3: float
    pressure_work: float
    schur_sup: float
    offdiag_max: float
    adversarial_c3: float


# =============================================================================
# STRUCTURE-PRESERVING LATTICE FLUID
# =============================================================================

class StructurePreservingLatticeFluid3D:
    """
    Structure-preserving semidiscrete 3-D Navier-Stokes system on a
    periodic cubic lattice.

    Discrete structure:

        D0* = -D0

        div_h = D0_x + D0_y + D0_z

        P_h = I - grad_h (-Delta_c)^(-1) div_h

        B_h(u,v)
          = 1/2 [u_j D_j^0 v_i + D_j^0(u_j v_i)]

    The same central-difference Fourier symbol is used for the
    divergence, gradient, and pressure inversion.

    Central differences possess additional null modes at Nyquist
    frequencies. Those modes are left unchanged by the projection.
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

        # ---------------------------------------------------------------------
        # Physical grid
        # ---------------------------------------------------------------------

        coord = np.arange(self.N, dtype=float) * self.h

        self.X, self.Y, self.Z = np.meshgrid(
            coord,
            coord,
            coord,
            indexing="ij",
        )

        # ---------------------------------------------------------------------
        # Fourier wave numbers
        # ---------------------------------------------------------------------

        k1d = (
            2.0
            * np.pi
            * np.fft.fftfreq(self.N, d=self.h)
        )

        self.Kx, self.Ky, self.Kz = np.meshgrid(
            k1d,
            k1d,
            k1d,
            indexing="ij",
        )

        # ---------------------------------------------------------------------
        # Central-difference Fourier symbol
        #
        # D0 -> i sin(k h)/h
        # ---------------------------------------------------------------------

        self.K_tilde_x = (
            np.sin(self.Kx * self.h) / self.h
        )
        self.K_tilde_y = (
            np.sin(self.Ky * self.h) / self.h
        )
        self.K_tilde_z = (
            np.sin(self.Kz * self.h) / self.h
        )

        self.K_sq = (
            self.K_tilde_x ** 2
            + self.K_tilde_y ** 2
            + self.K_tilde_z ** 2
        )

        # Central differences have zero symbols not only at k=0 but also
        # at Nyquist frequencies.
        self.projectable_modes = self.K_sq > 1.0e-14

        self.inv_K_sq = np.zeros_like(self.K_sq)

        self.inv_K_sq[self.projectable_modes] = (
            1.0 / self.K_sq[self.projectable_modes]
        )

        # Physical Fourier radius, used only for spectral shell labels.
        self.Kmag = np.sqrt(
            self.Kx ** 2
            + self.Ky ** 2
            + self.Kz ** 2
        )

    # =========================================================================
    # FINITE DIFFERENCES
    # =========================================================================

    def D0(self, f, axis):
        """Periodic central difference."""
        return (
            np.roll(f, -1, axis=axis)
            - np.roll(f, 1, axis=axis)
        ) / (2.0 * self.h)

    def D_plus(self, f, axis):
        """Periodic forward difference."""
        return (
            np.roll(f, -1, axis=axis)
            - f
        ) / self.h

    def D_minus(self, f, axis):
        """Periodic backward difference."""
        return (
            f
            - np.roll(f, 1, axis=axis)
        ) / self.h

    # =========================================================================
    # DIFFERENTIAL OPERATORS
    # =========================================================================

    def divergence(self, u):
        """Central discrete divergence."""
        return (
            self.D0(u[0], 0)
            + self.D0(u[1], 1)
            + self.D0(u[2], 2)
        )

    def gradient(self, p):
        """Central discrete gradient."""
        return np.stack(
            [
                self.D0(p, 0),
                self.D0(p, 1),
                self.D0(p, 2),
            ],
            axis=0,
        )

    def laplacian(self, f):
        """
        Standard seven-point periodic Laplacian.
        """
        lap = np.zeros_like(f)

        for axis in range(3):
            lap += (
                np.roll(f, -1, axis=axis)
                - 2.0 * f
                + np.roll(f, 1, axis=axis)
            ) / self.h ** 2

        return lap

    # =========================================================================
    # LERAY PROJECTION
    # =========================================================================

    def leray_project(self, u):
        """
        Orthogonal projection associated with the central-difference
        divergence/gradient pair.

            P(k) = I - k_tilde k_tilde^T / |k_tilde|^2

        Modes for which the central symbol vanishes are left unchanged.
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

    # =========================================================================
    # NONLINEAR CONVECTION
    # =========================================================================

    def skew_convection(self, u, v):
        """
        Morinishi skew-symmetric convection:

            B_i(u,v)
              = 1/2 sum_j [
                    u_j D_j^0 v_i
                    + D_j^0(u_j v_i)
                ]

        For divergence-free u,

            <B(u,u),u> = 0

        up to floating-point arithmetic.
        """

        B = np.zeros_like(v)

        for i in range(3):
            for j in range(3):
                term1 = (
                    u[j]
                    * self.D0(v[i], axis=j)
                )

                term2 = self.D0(
                    u[j] * v[i],
                    axis=j,
                )

                B[i] += 0.5 * (
                    term1 + term2
                )

        return B

    # =========================================================================
    # NAVIER-STOKES RHS
    # =========================================================================

    def rhs(self, u):
        """
        Semidiscrete equation

            du/dt = -P_h B_h(u,u) + nu Delta_h u.
        """

        convection = self.skew_convection(u, u)

        projected = self.leray_project(
            convection
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
            -projected
            + self.nu * diffusion
        )

    # =========================================================================
    # INNER PRODUCTS AND NORMS
    # =========================================================================

    def inner_product(self, u, v):
        return np.sum(u * v) * self.h ** 3

    def norm_l2(self, u):
        return float(
            np.sqrt(
                max(
                    0.0,
                    self.inner_product(u, u),
                )
            )
        )

    def norm_l3(self, u):
        """
        Discrete L3 norm:

            ||u||_3 = ( integral |u|^3 dx )^(1/3)
        """
        magnitude = np.sqrt(
            np.sum(u ** 2, axis=0)
        )

        value = np.sum(
            magnitude ** 3
        ) * self.h ** 3

        return float(
            max(value, 0.0) ** (1.0 / 3.0)
        )

    def energy(self, u):
        return 0.5 * self.inner_product(u, u)

    # =========================================================================
    # ENSTROPHY / DISSIPATION
    # =========================================================================

    def enstrophy(self, u):
        """
        Forward-difference gradient energy.
        """
        value = 0.0

        for i in range(3):
            for j in range(3):
                d = self.D_plus(
                    u[i],
                    j,
                )

                value += self.inner_product(
                    d,
                    d,
                )

        return float(value)

    def energy_derivative(self, u):
        return -self.nu * self.enstrophy(u)

    # =========================================================================
    # RANDOM DIVERGENCE-FREE FIELD
    # =========================================================================

    def random_divergence_free_field(
        self,
        seed=1234,
        amplitude=1.0,
    ):
        rng = np.random.default_rng(seed)

        u = rng.standard_normal(
            (3, self.N, self.N, self.N)
        )

        u = self.leray_project(u)

        n = self.norm_l2(u)

        if n > 0:
            u *= amplitude / n

        return u

    # =========================================================================
    # TAYLOR-GREEN VORTEX
    # =========================================================================

    def taylor_green(self, amplitude=1.0):
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

    # =========================================================================
    # RK4
    # =========================================================================

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


# =============================================================================
# DISCRETE PROOF AUDIT
# =============================================================================

class DiscreteProofAudit:
    """
    Numerical audit layer built directly on the structure-preserving
    lattice engine above.
    """

    def __init__(self, fluid):
        self.fl = fluid
        self.Kmag = fluid.Kmag

    # =========================================================================
    # BASIC QUANTITIES
    # =========================================================================

    def velocity_magnitude(self, u):
        return np.sqrt(
            np.sum(u ** 2, axis=0)
            + 1.0e-30
        )

    def gradient_squared(self, u):
        """
        Central/spectral-symbol gradient magnitude:

            |grad_h u|^2.
        """

        u_hat = np.fft.fftn(
            u,
            axes=(1, 2, 3),
        )

        grad_sq = np.zeros_like(
            u[0],
            dtype=float,
        )

        for a in range(3):
            gx = np.real(
                np.fft.ifftn(
                    1j
                    * self.fl.K_tilde_x
                    * u_hat[a],
                    axes=(0, 1, 2),
                )
            )

            gy = np.real(
                np.fft.ifftn(
                    1j
                    * self.fl.K_tilde_y
                    * u_hat[a],
                    axes=(0, 1, 2),
                )
            )

            gz = np.real(
                np.fft.ifftn(
                    1j
                    * self.fl.K_tilde_z
                    * u_hat[a],
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
        u_mag = self.velocity_magnitude(u)
        grad_sq = self.gradient_squared(u)

        return float(
            np.sum(
                u_mag * grad_sq
            )
            * self.fl.h ** 3
        )

    # =========================================================================
    # PRESSURE GRADIENTS
    # =========================================================================

    def pressure_gradients(self, u):
        """
        Compute grad p and grad q.

        The same central-difference symbol used by the discrete
        divergence and Leray projector is used for the pressure
        inversion.

            -Delta_c p = div_c B(u,u)

            -Delta_c q = div_c(|u|u)
        """

        # ---------------------------------------------------------------------
        # Physical pressure
        # ---------------------------------------------------------------------

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

        # ---------------------------------------------------------------------
        # Auxiliary pressure
        # ---------------------------------------------------------------------

        abs_u_u = (
            self.velocity_magnitude(u)
            * u
        )

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

    # =========================================================================
    # PRESSURE WORK
    # =========================================================================

    def pressure_work(self, u):
        grad_p_hat, grad_q_hat = (
            self.pressure_gradients(u)
        )

        gp = np.real(
            np.fft.ifftn(
                grad_p_hat,
                axes=(1, 2, 3),
            )
        )

        gq = np.real(
            np.fft.ifftn(
                grad_q_hat,
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

    # =========================================================================
    # CRITICAL QUOTIENT
    # =========================================================================

    def critical_quotient(self, u):
        l3 = self.fl.norm_l3(u)
        d3 = self.critical_dissipation(u)
        wp = self.pressure_work(u)

        denominator = l3 * d3

        if denominator <= 1.0e-30:
            return 0.0

        return float(
            wp / denominator
        )

    # =========================================================================
    # DISCRETE LEIBNIZ / CHAIN-RULE DEFECT
    # =========================================================================

    def chain_rule_defect(self, u):
        B = self.fl.skew_convection(
            u,
            u,
        )

        abs_u_u = (
            self.velocity_magnitude(u)
            * u
        )

        numerator = abs(
            self.fl.inner_product(
                B,
                abs_u_u,
            )
        )

        denominator = (
            self.fl.norm_l3(u)
            * self.critical_dissipation(u)
        )

        if denominator <= 1.0e-30:
            return 0.0

        return float(
            numerator / denominator
        )

    # =========================================================================
    # DYADIC SHELLS
    # =========================================================================

    def dyadic_shell_masks(self):
        """
        Radial dyadic output shells.

            shell 1: 0 <= |k| < 2
            shell 2: 2 <= |k| < 4
            shell 3: 4 <= |k| < 8
            ...
        """

        max_shell = max(
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

        for j in range(
            1,
            max_shell + 1,
        ):
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

    # =========================================================================
    # SHELL DISSIPATION
    # =========================================================================

    def shell_field(self, u, mask):
        u_hat = np.fft.fftn(
            u,
            axes=(1, 2, 3),
        )

        filtered_hat = np.zeros_like(
            u_hat
        )

        filtered_hat[:, mask] = (
            u_hat[:, mask]
        )

        return np.real(
            np.fft.ifftn(
                filtered_hat,
                axes=(1, 2, 3),
            )
        )

    def shell_dissipation(
        self,
        u,
        mask,
    ):
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
            )
            * self.fl.h ** 3
        )

    # =========================================================================
    # OUTPUT-SHELL PRESSURE MATRIX
    # =========================================================================

    def shell_pressure_matrix(
        self,
        u,
    ):
        """
        Compute

            M_jk =
                |<Delta_j grad p,
                   Delta_k grad q>|.

        Because the masks are disjoint Fourier projectors,

            M_jk = 0

        for j != k, up to numerical roundoff.
        """

        masks = self.dyadic_shell_masks()

        grad_p_hat, grad_q_hat = (
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

        parseval_factor = (
            self.fl.h ** 3
            / self.fl.N ** 3
        )

        for a, j in enumerate(shells):
            mask_j = masks[j]

            gp_j = np.zeros_like(
                grad_p_hat
            )

            gp_j[:, mask_j] = (
                grad_p_hat[:, mask_j]
            )

            D[a] = (
                self.shell_dissipation(
                    u,
                    mask_j,
                )
            )

            for b, k in enumerate(shells):
                mask_k = masks[k]

                gq_k = np.zeros_like(
                    grad_q_hat
                )

                gq_k[:, mask_k] = (
                    grad_q_hat[:, mask_k]
                )

                # Explicit Fourier-space shell pairing.
                inner = np.sum(
                    gp_j
                    * np.conj(gq_k)
                )

                M[a, b] = (
                    abs(inner)
                    * parseval_factor
                )

        l3 = self.fl.norm_l3(u)

        Gamma = np.zeros_like(M)

        for a in range(n):
            for b in range(n):
                denominator = (
                    l3
                    * np.sqrt(
                        max(
                            0.0,
                            D[a] * D[b],
                        )
                    )
                )

                if denominator > 1.0e-30:
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

    # =========================================================================
    # SCHUR AUDIT
    # =========================================================================

    def schur_audit(self, u):
        shells, M, Gamma, D = (
            self.shell_pressure_matrix(u)
        )

        if len(shells) == 0:
            row_sums = np.array([])
            sup_row_sum = 0.0
            offdiag_max = 0.0
        else:
            row_sums = np.sum(
                Gamma,
                axis=1,
            )

            sup_row_sum = float(
                np.max(row_sums)
            )

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

        return {
            "shells": shells,
            "M": M,
            "Gamma": Gamma,
            "D": D,
            "row_sums": row_sums,
            "sup_row_sum": sup_row_sum,
            "offdiag_max": offdiag_max,
        }

    # =========================================================================
    # NONLINEAR SOURCE AUDIT
    # =========================================================================

    def triadic_source_audit(self, u):
        """
        Measure the nonlinear source population in each output shell.

        These are source diagnostics, not a direct measurement of
        velocity-shell-to-velocity-shell transfer.
        """

        masks = self.dyadic_shell_masks()

        B = self.fl.skew_convection(
            u,
            u,
        )

        Q = (
            self.velocity_magnitude(u)
            * u
        )

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

    # =========================================================================
    # DE-ALIASING
    # =========================================================================

    def two_thirds_mask(self):
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
            axes=(1, 2, 3),
        )

        mask = self.two_thirds_mask()

        filtered_hat = np.zeros_like(
            u_hat
        )

        filtered_hat[:, mask] = (
            u_hat[:, mask]
        )

        return self.fl.leray_project(
            np.real(
                np.fft.ifftn(
                    filtered_hat,
                    axes=(1, 2, 3),
                )
            )
        )

    def aliasing_audit(self, u):
        raw = self.critical_quotient(u)

        filtered = self.dealias_field(u)

        dealiased = (
            self.critical_quotient(
                filtered
            )
        )

        difference = abs(
            raw - dealiased
        )

        return {
            "C3_raw": float(raw),
            "C3_dealiased": float(
                dealiased
            ),
            "absolute_difference": float(
                difference
            ),
            "relative_difference": float(
                difference
                / max(
                    abs(raw),
                    1.0e-30,
                )
            ),
        }

    # =========================================================================
    # AMPLITUDE INVARIANCE
    # =========================================================================

    def amplitude_invariance_test(
        self,
        u,
        amplitudes=(1.0, 2.0, 4.0, 8.0),
    ):
        n = self.fl.norm_l2(u)

        if n <= 1.0e-30:
            raise ValueError(
                "Cannot test amplitude invariance "
                "of a zero field."
            )

        u0 = u / n

        results = []

        for amplitude in amplitudes:
            ua = amplitude * u0

            l3 = self.fl.norm_l3(ua)
            d3 = self.critical_dissipation(ua)
            wp = self.pressure_work(ua)

            c3 = wp / max(
                l3 * d3,
                1.0e-30,
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

    # =========================================================================
    # ADVERSARIAL SEARCH
    # =========================================================================

    def adversarial_search(
        self,
        starts=5,
        steps=10,
        lr=0.01,
        seed=101,
    ):
        """
        Finite-dimensional multi-start heuristic.

        This does NOT compute a mathematical supremum.
        """

        rng = np.random.default_rng(
            seed
        )

        best_c3 = -np.inf
        best_field = None

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

            for _ in range(steps):
                gp_hat, gq_hat = (
                    self.pressure_gradients(
                        u
                    )
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
                    gp + gq
                )

                direction = (
                    self.fl
                    .leray_project(
                        direction
                    )
                )

                u = (
                    u
                    + lr * direction
                )

                n = self.fl.norm_l3(u)

                if n > 1.0e-30:
                    u = u / n

                c3 = (
                    self.critical_quotient(u)
                )

                if np.isfinite(c3):
                    if c3 > best_c3:
                        best_c3 = c3
                        best_field = u.copy()

        if not np.isfinite(best_c3):
            best_c3 = 0.0

        return {
            "best_C3": float(best_c3),
            "best_field": best_field,
        }


# =============================================================================
# OPERATOR TESTS
# =============================================================================

def run_operator_tests():
    print("=" * 78)
    print("STRUCTURE-PRESERVING LATTICE NAVIER-STOKES TEST SUITE")
    print("=" * 78)

    N = 16
    L = 2.0 * np.pi
    nu = 0.01

    fluid = StructurePreservingLatticeFluid3D(
        N=N,
        L=L,
        nu=nu,
    )

    rng = np.random.default_rng(42)

    # -------------------------------------------------------------------------
    # Test 1: D0 skew-adjointness
    # -------------------------------------------------------------------------

    f = rng.standard_normal(
        (N, N, N)
    )

    g = rng.standard_normal(
        (N, N, N)
    )

    Df = fluid.D0(f, axis=0)
    Dg = fluid.D0(g, axis=0)

    lhs = fluid.inner_product(Df, g)
    rhs = -fluid.inner_product(f, Dg)

    defect = abs(lhs - rhs) / max(
        1.0,
        abs(lhs),
        abs(rhs),
    )

    print(
        f"Test 1: D0 skew-adjoint defect = "
        f"{defect:.3e}"
    )

    # -------------------------------------------------------------------------
    # Test 2: Divergence-free projection
    # -------------------------------------------------------------------------

    u = fluid.random_divergence_free_field(
        seed=123,
        amplitude=1.0,
    )

    div_inf = fluid.divergence_linf(u)

    print(
        f"Test 2: ||div_h u||_inf = "
        f"{div_inf:.3e}"
    )

    # -------------------------------------------------------------------------
    # Test 3: Projection idempotence
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Test 4: Convective energy cancellation
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Test 5: Full RHS energy identity
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Test 6: Taylor-Green divergence
    # -------------------------------------------------------------------------

    tg = fluid.taylor_green()

    tg_div = fluid.divergence_linf(
        tg
    )

    print(
        f"Test 6: Taylor-Green ||div_h u||_inf = "
        f"{tg_div:.3e}"
    )

    print("Operator tests: PASS")

    return True


# =============================================================================
# UNIFIED AUDIT
# =============================================================================

def run_discrete_proof_audit(
    resolutions=(16, 32, 64),
    seed=42,
):
    print("=" * 78)
    print("DISCRETE NAVIER-STOKES PROOF / AUDIT")
    print("=" * 78)
    print(
        "STATUS: EXPERIMENTAL — NOT A CONTINUUM PROOF"
    )

    results = []

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
            fluid
            .random_divergence_free_field(
                seed=seed,
                amplitude=1.0,
            )
        )

        # ---------------------------------------------------------------------
        # Basic quotient
        # ---------------------------------------------------------------------

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

        # ---------------------------------------------------------------------
        # Amplitude invariance
        # ---------------------------------------------------------------------

        print()
        print("AMPLITUDE INVARIANCE")

        amplitudes = (
            audit.amplitude_invariance_test(
                u
            )
        )

        for item in amplitudes:
            print(
                f"A={item.amplitude:6.2f}  "
                f"C3={item.c3:.8e}  "
                f"R_h={item.r3:.8e}"
            )

        # ---------------------------------------------------------------------
        # Schur audit
        # ---------------------------------------------------------------------

        print()
        print("DYADIC SCHUR AUDIT")

        schur = audit.schur_audit(
            u
        )

        print(
            f"Active shells            : "
            f"{len(schur['shells'])}"
        )

        print(
            f"sup_j row sum            : "
            f"{schur['sup_row_sum']:.8e}"
        )

        Gamma = schur["Gamma"]

        diagonal_max = (
            float(
                np.max(
                    np.diag(Gamma)
                )
            )
            if Gamma.size
            else 0.0
        )

        print(
            f"Diagonal max Gamma       : "
            f"{diagonal_max:.8e}"
        )

        print(
            f"Off-diagonal max Gamma   : "
            f"{schur['offdiag_max']:.8e}"
        )

        # ---------------------------------------------------------------------
        # Nonlinear source audit
        # ---------------------------------------------------------------------

        print()
        print(
            "NONLINEAR SOURCE / TRIADIC AUDIT"
        )

        triadic = (
            audit.triadic_source_audit(
                u
            )
        )

        for j, data in triadic.items():
            print(
                f"shell {j:2d}: "
                f"B={data['B_source_norm']:.6e}  "
                f"Q={data['Q_source_norm']:.6e}"
            )

        # ---------------------------------------------------------------------
        # Aliasing audit
        # ---------------------------------------------------------------------

        print()
        print("ALIASING STRESS TEST")

        alias = audit.aliasing_audit(
            u
        )

        print(
            f"C3 raw                   : "
            f"{alias['C3_raw']:.8e}"
        )

        print(
            f"C3 dealiased             : "
            f"{alias['C3_dealiased']:.8e}"
        )

        print(
            f"Relative difference      : "
            f"{alias['relative_difference']:.8e}"
        )

        # ---------------------------------------------------------------------
        # Adversarial search
        # ---------------------------------------------------------------------

        print()
        print(
            "MULTI-START ADVERSARIAL SEARCH"
        )

        adversarial = (
            audit.adversarial_search(
                starts=5,
                steps=10,
                lr=0.01,
                seed=seed + N,
            )
        )

        adversarial_c3 = (
            adversarial["best_C3"]
        )

        print(
            f"Best discovered C3       : "
            f"{adversarial_c3:.8e}"
        )

        # ---------------------------------------------------------------------
        # Store result
        # ---------------------------------------------------------------------

        results.append(
            ResolutionResult(
                N=N,
                c3=float(c3),
                r3=float(r3),
                l3=float(l3),
                d3=float(d3),
                pressure_work=float(wp),
                schur_sup=float(
                    schur["sup_row_sum"]
                ),
                offdiag_max=float(
                    schur["offdiag_max"]
                ),
                adversarial_c3=float(
                    adversarial_c3
                ),
            )
        )

    # =========================================================================
    # RESOLUTION SUMMARY
    # =========================================================================

    print()
    print("=" * 78)
    print("RESOLUTION SUMMARY")
    print("=" * 78)

    for result in results:
        print(
            f"N={result.N:4d}   "
            f"C3={result.c3:.8e}   "
            f"D3={result.d3:.8e}"
        )

    # =========================================================================
    # MATHEMATICAL STATUS
    # =========================================================================

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

The nonlinear source fields contain convolution/triadic interactions.
The source-shell audit therefore measures how those nonlinear terms
populate the output hierarchy.

The unresolved analytical target is a uniform continuum-compatible
estimate controlling those nonlinear interactions.

Numerical experiments are used here primarily for diagnostics and
falsification, not as a substitute for analytical proof.
"""
    )

    print("=" * 78)
    print("AUDIT COMPLETE")
    print("=" * 78)

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    try:
        run_operator_tests()

        print()

        run_discrete_proof_audit(
            resolutions=(16, 32, 64),
            seed=42,
        )

        return 0

    except KeyboardInterrupt:
        print(
            "\nInterrupted."
        )
        return 130

    except Exception as exc:
        print()
        print("=" * 78)
        print("AUDIT FAILED")
        print("=" * 78)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
