"""
==============================================================================
DISCRETE NAVIER-STOKES PROOF / AUDIT — LEVEL 1 / LEVEL 2
==============================================================================

STATUS
------
EXPERIMENTAL NUMERICAL AUDIT — NOT A CONTINUUM PROOF.

This program tests finite-dimensional periodic lattice quantities associated
with a structure-preserving discretization of 3-D Navier-Stokes.

It does NOT establish global regularity of the continuum 3-D
Navier-Stokes equations.

MATHEMATICAL LEVELS
-------------------
LEVEL 1 — EXACT DISCRETE ALGEBRA
    The code audits:

        * central-difference skew-adjointness;
        * Leray projection properties;
        * discrete energy cancellation;
        * compatible Laplacian identity;
        * pressure/splitting identity;
        * weighted edge quantities.

    The algebraic identities are finite-dimensional identities.

LEVEL 2 — NUMERICAL EXTREMAL SEARCH
    The principal unresolved numerical target is the weighted edge quotient

        C_edge(u)
          =
        sum_{x,j} |u| |delta_j u|^3
        ----------------------------------------------
        h ||u||_3 sum_{x,j} |u| |delta_j u|^2 / h^2.

    Numerical searches for large values of

        C_edge, C_P, C_S, R_h

    provide numerical evidence only. They do NOT establish a uniform
    N-independent bound.

LEVEL 3 — CONTINUUM
    No continuum theorem is claimed.

DISCRETE OPERATORS
------------------
The primary derivative is the periodic central difference

    D0 f(x) = [f(x+h) - f(x-h)] / (2h).

Its Fourier symbol is

    i Ktilde_j,
    Ktilde_j = sin(k_j h) / h.

The compatible Laplacian is

    Delta_h -> -|Ktilde|^2.

The same symbol is used for

    * divergence,
    * gradient,
    * Leray projection,
    * pressure Poisson solve,
    * diffusion.

NYQUIST MODES
-------------
Central differences have additional null modes at Nyquist frequencies.
For Ktilde = 0 the discrete Leray projection leaves the vector mode
unchanged.

The program therefore explicitly measures null-mode content.

NONLINEAR TERM
--------------
The Morinishi skew form is

    B_i(u,v)
      =
    1/2 sum_j [
        u_j D_j v_i
        + D_j(u_j v_i)
    ].

For discretely divergence-free u,

    <B(u,u),u> = 0.

PRESSURE SPLIT
--------------
With

    q = |u|u,

the program audits

    <grad p, grad q>
      =
    <B,q> - <B,Pq>

subject to the sign convention used by the discrete Poisson solve.

The individual normalized pieces are reported as

    C_P = |<B,(I-P)q>| / (||u||_3 D3),

    C_S = |<B,Pq>| / (||u||_3 D3),

    R_h = |<B,q>| / (||u||_3 D3).

IMPORTANT
---------
These quantities should not be interpreted as independent bounds merely
because their sum reconstructs another quantity. Cancellation is possible.

The optimizer can therefore target C_P, C_S, R_h, and C_edge independently.

DEPENDENCIES
------------
Required:
    Python >= 3.10
    NumPy

Optional:
    SciPy

SciPy is required only for the true L-BFGS numerical optimizer.

USAGE
-----
Quick structural audit:

    python discrete_proof.py --quick

Standard audit:

    python discrete_proof.py

More adversarial randomized searching:

    python discrete_proof.py --search-starts 16 --search-steps 20

True L-BFGS optimization:

    python discrete_proof.py --optimize

More optimizer starts:

    python discrete_proof.py --optimize --opt-starts 8

Full resolution run:

    python discrete_proof.py --full --optimize

==============================================================================
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, Sequence, Optional, Callable

import numpy as np


# ============================================================================
# RESULT CONTAINERS
# ============================================================================

@dataclass
class ScalingResult:
    amplitude: float
    c3: float
    r3: float
    cp: float
    cs: float
    l3: float
    d3: float
    pressure_work: float


@dataclass
class ResolutionResult:
    N: int
    c3: float
    rh: float
    cp: float
    cs: float
    edge: float
    edge_central: float
    l3: float
    d3: float
    pressure_work: float
    null_fraction: float


@dataclass
class SearchResult:
    objective: str
    best_value: float
    best_field: Optional[np.ndarray]
    starts: int
    steps: int


@dataclass
class OptimizationResult:
    objective: str
    best_value: float
    best_field: Optional[np.ndarray]
    starts: int
    iterations: int
    scipy_available: bool


# ============================================================================
# STRUCTURE-PRESERVING LATTICE FLUID
# ============================================================================

class StructurePreservingLatticeFluid3D:
    """
    Structure-preserving semidiscrete 3-D Navier-Stokes system on a
    periodic cubic lattice.
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
        # Central difference Fourier symbols
        # ------------------------------------------------------------------

        self.K_tilde_x = np.sin(
            self.Kx * self.h
        ) / self.h

        self.K_tilde_y = np.sin(
            self.Ky * self.h
        ) / self.h

        self.K_tilde_z = np.sin(
            self.Kz * self.h
        ) / self.h

        self.K_sq = (
            self.K_tilde_x ** 2
            + self.K_tilde_y ** 2
            + self.K_tilde_z ** 2
        )

        self.projectable_modes = (
            self.K_sq > 1e-14
        )

        self.null_modes = ~self.projectable_modes

        self.inv_K_sq = np.zeros_like(
            self.K_sq
        )

        self.inv_K_sq[self.projectable_modes] = (
            1.0
            / self.K_sq[self.projectable_modes]
        )

    # ======================================================================
    # FINITE DIFFERENCES
    # ======================================================================

    def D0(self, f, axis: int):
        """
        Periodic central difference.
        """
        return (
            np.roll(f, -1, axis=axis)
            - np.roll(f, 1, axis=axis)
        ) / (2.0 * self.h)

    def D_plus(self, f, axis: int):
        """
        Periodic forward difference divided by h.
        """
        return (
            np.roll(f, -1, axis=axis)
            - f
        ) / self.h

    def D_minus(self, f, axis: int):
        """
        Periodic backward difference divided by h.
        """
        return (
            f
            - np.roll(f, 1, axis=axis)
        ) / self.h

    def delta_plus(self, f, axis: int):
        """
        Raw forward edge increment:

            delta_j f = f(x+h e_j) - f(x).
        """
        return (
            np.roll(f, -1, axis=axis)
            - f
        )

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

        return float(
            np.sqrt(
                np.sum(d * d)
                * self.h ** 3
            )
        )

    def divergence_linf(self, u):
        return float(
            np.max(
                np.abs(
                    self.divergence(u)
                )
            )
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
    # COMPATIBLE LAPLACIAN
    # ======================================================================

    def laplacian(self, f):
        f_hat = np.fft.fftn(f)

        lap_hat = (
            -self.K_sq
            * f_hat
        )

        return np.real(
            np.fft.ifftn(lap_hat)
        )

    # ======================================================================
    # LERAY PROJECTION
    # ======================================================================

    def leray_project(self, u):
        """
        Orthogonal projection onto the kernel of the central-difference
        divergence.

        For Ktilde != 0,

            P = I - Ktilde Ktilde^T / |Ktilde|^2.

        For Ktilde = 0 the vector mode is left unchanged.
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
            kdotu
            * self.inv_K_sq
        )

        projected_hat = np.empty_like(
            u_hat
        )

        projected_hat[0] = (
            u_hat[0]
            - self.K_tilde_x
            * correction
        )

        projected_hat[1] = (
            u_hat[1]
            - self.K_tilde_y
            * correction
        )

        projected_hat[2] = (
            u_hat[2]
            - self.K_tilde_z
            * correction
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
        Morinishi skew form:

            B_i(u,v)
              =
            1/2 sum_j [
                u_j D_j v_i
                + D_j(u_j v_i)
            ].
        """

        B = np.zeros_like(v)

        for i in range(3):
            for j in range(3):

                term1 = (
                    u[j]
                    * self.D0(v[i], j)
                )

                term2 = self.D0(
                    u[j] * v[i],
                    j,
                )

                B[i] += 0.5 * (
                    term1
                    + term2
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
            self.leray_project(
                convection
            )
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
            np.sum(u * v)
            * self.h ** 3
        )

    def norm_l2(self, u):
        value = self.inner_product(
            u,
            u,
        )

        return float(
            np.sqrt(
                max(value, 0.0)
            )
        )

    def norm_l3(self, u):
        magnitude = np.sqrt(
            np.sum(
                u ** 2,
                axis=0,
            )
        )

        return float(
            np.sum(
                magnitude ** 3
            )
            * self.h ** 3
        ) ** (1.0 / 3.0)

    def energy(self, u):
        return 0.5 * self.inner_product(
            u,
            u,
        )

    # ======================================================================
    # DISCRETE H1 / ENERGY DISSIPATION
    # ======================================================================

    def enstrophy(self, u):
        result = 0.0

        for i in range(3):
            for j in range(3):

                d = self.D0(
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
            -self.nu
            * self.enstrophy(u)
        )

    # ======================================================================
    # RANDOM DIVERGENCE-FREE FIELD
    # ======================================================================

    def random_divergence_free_field(
        self,
        seed: int = 1234,
        amplitude: float = 1.0,
    ):
        rng = np.random.default_rng(
            seed
        )

        u = rng.standard_normal(
            (
                3,
                self.N,
                self.N,
                self.N,
            )
        )

        u = self.leray_project(u)

        current = self.norm_l2(u)

        if current > 0:
            u *= amplitude / current

        return u

    # ======================================================================
    # TAYLOR-GREEN
    # ======================================================================

    def taylor_green(
        self,
        amplitude: float = 1.0,
    ):
        u = np.zeros(
            (
                3,
                self.N,
                self.N,
                self.N,
            ),
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


# ============================================================================
# DISCRETE AUDIT
# ============================================================================

class DiscreteProofAudit:

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

    # ======================================================================
    # BASIC QUANTITIES
    # ======================================================================

    def velocity_magnitude(self, u):
        return np.sqrt(
            np.sum(
                u ** 2,
                axis=0,
            )
        )

    def spectral_gradient_squared(self, u):
        """
        Pointwise sum of squared central-difference gradients.
        """

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
        magnitude = (
            self.velocity_magnitude(u)
        )

        grad_sq = (
            self.spectral_gradient_squared(u)
        )

        return float(
            np.sum(
                magnitude
                * grad_sq
            )
            * self.fl.h ** 3
        )

    # ======================================================================
    # PRESSURE GRADIENTS
    # ======================================================================

    def pressure_gradients(self, u):
        """
        Compute grad p and grad q using

            -Delta p = div B,
            -Delta q = div(|u|u).

        Since -Delta -> |Ktilde|^2,

            p_hat = div_B_hat / |Ktilde|^2,

            q_hat = div_Q_hat / |Ktilde|^2.

        The corresponding gradients are returned directly.
        """

        B = self.fl.skew_convection(
            u,
            u,
        )

        B_hat = np.fft.fftn(
            B,
            axes=(1, 2, 3),
        )

        div_B_hat = (
            1j
            * self.fl.K_tilde_x
            * B_hat[0]
            + 1j
            * self.fl.K_tilde_y
            * B_hat[1]
            + 1j
            * self.fl.K_tilde_z
            * B_hat[2]
        )

        p_hat = (
            div_B_hat
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

        magnitude = (
            self.velocity_magnitude(u)
        )

        Q = (
            magnitude
            * u
        )

        Q_hat = np.fft.fftn(
            Q,
            axes=(1, 2, 3),
        )

        div_Q_hat = (
            1j
            * self.fl.K_tilde_x
            * Q_hat[0]
            + 1j
            * self.fl.K_tilde_y
            * Q_hat[1]
            + 1j
            * self.fl.K_tilde_z
            * Q_hat[2]
        )

        q_hat = (
            div_Q_hat
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

        return (
            grad_p_hat,
            grad_q_hat,
        )

    def pressure_work_signed(self, u):
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

        return self.fl.inner_product(
            gp,
            gq,
        )

    def pressure_work(self, u):
        return abs(
            self.pressure_work_signed(u)
        )

    # ======================================================================
    # EXACT PRESSURE / SOLENOIDAL SPLIT
    # ======================================================================

    def pressure_split_audit(self, u):
        """
        Audit

            <grad p, grad q>
              =
            <B,q> - <B,Pq>.

        This identity is checked numerically rather than assumed.
        """

        magnitude = (
            self.velocity_magnitude(u)
        )

        B = self.fl.skew_convection(
            u,
            u,
        )

        q = magnitude * u

        Pq = self.fl.leray_project(q)

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

        pressure = self.fl.inner_product(
            gp,
            gq,
        )

        local_term = self.fl.inner_product(
            B,
            q,
        )

        solenoidal_term = self.fl.inner_product(
            B,
            Pq,
        )

        reconstructed = (
            local_term
            - solenoidal_term
        )

        residual = (
            pressure
            - reconstructed
        )

        scale = max(
            1.0,
            abs(pressure),
            abs(local_term),
            abs(solenoidal_term),
        )

        return {
            "pressure": float(pressure),
            "local_term": float(local_term),
            "solenoidal_term": float(solenoidal_term),
            "reconstructed": float(reconstructed),
            "residual": float(residual),
            "relative_residual":
                float(abs(residual) / scale),
        }

    # ======================================================================
    # CP / CS / RH
    # ======================================================================

    def pressure_solenoidal_split(
        self,
        u,
    ):
        magnitude = (
            self.velocity_magnitude(u)
        )

        B = self.fl.skew_convection(
            u,
            u,
        )

        q = magnitude * u

        Pq = self.fl.leray_project(q)

        a = self.fl.inner_product(
            B,
            q - Pq,
        )

        b = self.fl.inner_product(
            B,
            Pq,
        )

        scale = (
            self.fl.norm_l3(u)
            * self.critical_dissipation(u)
        )

        scale = max(
            scale,
            1e-30,
        )

        return {
            "a": float(a),
            "b": float(b),
            "a_abs_normalized":
                float(abs(a) / scale),
            "b_abs_normalized":
                float(abs(b) / scale),
            "sum_normalized":
                float(abs(a + b) / scale),
            "C_P":
                float(abs(a) / scale),
            "C_S":
                float(abs(b) / scale),
            "R_h":
                float(abs(a + b) / scale),
        }

    # ======================================================================
    # CRITICAL QUOTIENT
    # ======================================================================

    def critical_quotient(self, u):
        l3 = self.fl.norm_l3(u)
        d3 = self.critical_dissipation(u)
        wp = self.pressure_work(u)

        denominator = (
            l3 * d3
        )

        if denominator <= 1e-30:
            return 0.0

        return float(
            wp / denominator
        )

    # ======================================================================
    # CHAIN-RULE / NONLINEAR WORK
    # ======================================================================

    def chain_rule_defect(self, u):
        magnitude = (
            self.velocity_magnitude(u)
        )

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
    # LEVEL-2 WEIGHTED EDGE QUOTIENT
    # ======================================================================

    def weighted_edge_quotient(
        self,
        u,
    ):
        """
        Evaluate

            C_edge =
            sum |u| |delta u|^3
            ----------------------------------------
            h ||u||_3 sum |u| |delta u|^2 / h^2

        with

            delta_j u(x) = u(x+h e_j) - u(x).

        This directly targets the proposed Level-2 inequality.
        """

        magnitude = (
            self.velocity_magnitude(u)
        )

        cubic_sum = 0.0
        quadratic_sum = 0.0

        for j in range(3):

            delta = (
                self.fl.delta_plus(
                    u,
                    j + 1,
                )
            )

            delta_mag = np.sqrt(
                np.sum(
                    delta ** 2,
                    axis=0,
                )
            )

            cubic_sum += np.sum(
                magnitude
                * delta_mag ** 3
            )

            quadratic_sum += np.sum(
                magnitude
                * delta_mag ** 2
            )

        cubic_sum *= self.fl.h ** 3

        quadratic_sum *= (
            self.fl.h ** 3
            / self.fl.h ** 2
        )

        l3 = self.fl.norm_l3(u)

        denominator = (
            self.fl.h
            * l3
            * quadratic_sum
        )

        quotient = (
            cubic_sum / denominator
            if denominator > 1e-30
            else 0.0
        )

        return {
            "edge_cubic":
                float(cubic_sum),

            "weighted_quadratic":
                float(quadratic_sum),

            "l3":
                float(l3),

            "quotient":
                float(quotient),
        }

    def weighted_edge_quotient_central(
        self,
        u,
    ):
        """
        Central-difference comparison:

            sum |u| |D0 u|^3
            --------------------------------
            ||u||_3 sum |u| |D0 u|^2.
        """

        magnitude = (
            self.velocity_magnitude(u)
        )

        cubic_sum = 0.0
        quadratic_sum = 0.0

        for j in range(3):

            d = self.fl.D0(
                u,
                j + 1,
            )

            dmag = np.sqrt(
                np.sum(
                    d ** 2,
                    axis=0,
                )
            )

            cubic_sum += np.sum(
                magnitude
                * dmag ** 3
            )

            quadratic_sum += np.sum(
                magnitude
                * dmag ** 2
            )

        cubic_sum *= self.fl.h ** 3
        quadratic_sum *= self.fl.h ** 3

        l3 = self.fl.norm_l3(u)

        denominator = (
            l3
            * quadratic_sum
        )

        quotient = (
            cubic_sum / denominator
            if denominator > 1e-30
            else 0.0
        )

        return {
            "central_cubic":
                float(cubic_sum),

            "central_quadratic":
                float(quadratic_sum),

            "l3":
                float(l3),

            "quotient":
                float(quotient),
        }

    # ======================================================================
    # NULL-MODE AUDIT
    # ======================================================================

    def null_mode_fraction(
        self,
        u,
    ):
        """
        Fraction of Fourier L2 energy lying in Ktilde = 0 modes.
        """

        u_hat = np.fft.fftn(
            u,
            axes=(1, 2, 3),
        )

        total = np.sum(
            np.abs(u_hat) ** 2
        )

        null = np.sum(
            np.abs(
                u_hat[:, self.fl.null_modes]
            ) ** 2
        )

        if total <= 1e-30:
            return 0.0

        return float(
            null / total
        )

    def remove_null_modes(
        self,
        u,
    ):
        """
        Remove all Ktilde = 0 Fourier modes and reproject.
        """

        u_hat = np.fft.fftn(
            u,
            axes=(1, 2, 3),
        )

        filtered = u_hat.copy()

        filtered[
            :,
            self.fl.null_modes
        ] = 0.0

        filtered_field = np.real(
            np.fft.ifftn(
                filtered,
                axes=(1, 2, 3),
            )
        )

        return self.fl.leray_project(
            filtered_field
        )

    # ======================================================================
    # SCALE-CONCENTRATION AUDIT
    # ======================================================================

    def scale_concentration(
        self,
        u,
    ):
        """
        Report Fourier and dissipation concentration at high frequencies.
        """

        u_hat = np.fft.fftn(
            u,
            axes=(1, 2, 3),
        )

        energy_density = np.sum(
            np.abs(u_hat) ** 2,
            axis=0,
        )

        total_energy = np.sum(
            energy_density
        )

        high_energy = np.sum(
            energy_density[
                self.Kmag >= self.fl.N / 4.0
            ]
        )

        high2_energy = np.sum(
            energy_density[
                self.Kmag >= self.fl.N / 3.0
            ]
        )

        grad_sq = (
            self.spectral_gradient_squared(u)
        )

        weighted_grad = (
            self.velocity_magnitude(u)
            * grad_sq
        )

        # A physical-space proxy for concentration:
        # large gradients relative to total critical dissipation.
        threshold = np.percentile(
            weighted_grad,
            90.0,
        )

        top10 = np.sum(
            weighted_grad[
                weighted_grad >= threshold
            ]
        )

        total_d3 = np.sum(
            weighted_grad
        )

        return {
            "energy_high_N_over_4":
                float(
                    high_energy
                    / max(
                        total_energy,
                        1e-30,
                    )
                ),

            "energy_high_N_over_3":
                float(
                    high2_energy
                    / max(
                        total_energy,
                        1e-30,
                    )
                ),

            "top10_percent_D3_fraction":
                float(
                    top10
                    / max(
                        total_d3,
                        1e-30,
                    )
                ),
        }

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

        for j in range(
            1,
            maximum + 1,
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

    def shell_field(
        self,
        u,
        mask,
    ):
        u_hat = np.fft.fftn(
            u,
            axes=(1, 2, 3),
        )

        filtered = np.zeros_like(
            u_hat
        )

        filtered[
            :,
            mask
        ] = u_hat[
            :,
            mask
        ]

        return np.real(
            np.fft.ifftn(
                filtered,
                axes=(1, 2, 3),
            )
        )

    def shell_dissipation(
        self,
        u,
        mask,
    ):
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
                magnitude
                * grad_sq
            )
            * self.fl.h ** 3
        )

    def shell_pressure_matrix(
        self,
        u,
    ):
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

        for a, j in enumerate(shells):

            D[a] = (
                self.shell_dissipation(
                    u,
                    masks[j],
                )
            )

        for a, j in enumerate(shells):

            for b, k in enumerate(shells):

                gp_j = np.zeros_like(
                    gp_hat
                )

                gq_k = np.zeros_like(
                    gq_hat
                )

                gp_j[
                    :,
                    masks[j]
                ] = gp_hat[
                    :,
                    masks[j]
                ]

                gq_k[
                    :,
                    masks[k]
                ] = gq_hat[
                    :,
                    masks[k]
                ]

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

    def shell_orthogonality_audit(
        self,
        u,
    ):
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
            "sup_row_sum":
                float(
                    np.max(row_sums)
                )
                if row_sums.size
                else 0.0,
        }

    def schur_audit(self, u):
        return self.shell_orthogonality_audit(u)

    # ======================================================================
    # TRIADIC SOURCE AUDIT
    # ======================================================================

    def triadic_source_audit(
        self,
        u,
    ):
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
                    float(
                        np.sqrt(
                            B_energy
                        )
                    ),

                "Q_source_norm":
                    float(
                        np.sqrt(
                            Q_energy
                        )
                    ),
            }

        return result

    # ======================================================================
    # ALIASING / HIGH-FREQUENCY SENSITIVITY
    # ======================================================================

    def two_thirds_mask(self):
        cutoff = self.fl.N / 3.0

        return (
            (np.abs(self.fl.Kx) <= cutoff)
            & (np.abs(self.fl.Ky) <= cutoff)
            & (np.abs(self.fl.Kz) <= cutoff)
        )

    def spectral_filter(
        self,
        u,
        mask,
    ):
        u_hat = np.fft.fftn(
            u,
            axes=(1, 2, 3),
        )

        filtered = np.zeros_like(
            u_hat
        )

        filtered[
            :,
            mask
        ] = u_hat[
            :,
            mask
        ]

        return np.real(
            np.fft.ifftn(
                filtered,
                axes=(1, 2, 3),
            )
        )

    def dealias_field(
        self,
        u,
    ):
        filtered = self.spectral_filter(
            u,
            self.two_thirds_mask(),
        )

        return self.fl.leray_project(
            filtered
        )

    def aliasing_audit(
        self,
        u,
    ):
        raw = self.critical_quotient(
            u
        )

        dealiased_field = (
            self.dealias_field(u)
        )

        dealiased = (
            self.critical_quotient(
                dealiased_field
            )
        )

        difference = abs(
            raw
            - dealiased
        )

        return {
            "C3_raw":
                float(raw),

            "C3_dealiased":
                float(dealiased),

            "absolute_difference":
                float(difference),

            "relative_difference":
                float(
                    difference
                    / max(
                        abs(raw),
                        1e-30,
                    )
                ),
        }

    # ======================================================================
    # AMPLITUDE INVARIANCE
    # ======================================================================

    def amplitude_invariance_test(
        self,
        u,
        amplitudes=(
            1.0,
            2.0,
            4.0,
            8.0,
        ),
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
            d3 = self.critical_dissipation(
                ua
            )
            wp = self.pressure_work(
                ua
            )

            denominator = (
                l3 * d3
            )

            c3 = (
                wp / denominator
                if denominator > 1e-30
                else 0.0
            )

            r3 = (
                self.chain_rule_defect(
                    ua
                )
            )

            split = (
                self.pressure_solenoidal_split(
                    ua
                )
            )

            results.append(
                ScalingResult(
                    amplitude=float(
                        amplitude
                    ),
                    c3=float(c3),
                    r3=float(
                        split["R_h"]
                    ),
                    cp=float(
                        split["C_P"]
                    ),
                    cs=float(
                        split["C_S"]
                    ),
                    l3=float(l3),
                    d3=float(d3),
                    pressure_work=float(wp),
                )
            )

        return results

    # ======================================================================
    # RESOLUTION MEASUREMENT
    # ======================================================================

    def resolution_measurement(
        self,
        u,
    ):
        l3 = self.fl.norm_l3(u)
        d3 = self.critical_dissipation(u)
        wp = self.pressure_work(u)

        denominator = (
            l3 * d3
        )

        c3 = (
            wp / denominator
            if denominator > 1e-30
            else 0.0
        )

        split = (
            self.pressure_solenoidal_split(
                u
            )
        )

        edge = (
            self.weighted_edge_quotient(
                u
            )["quotient"]
        )

        edge_central = (
            self.weighted_edge_quotient_central(
                u
            )["quotient"]
        )

        return ResolutionResult(
            N=self.fl.N,
            c3=float(c3),
            rh=float(split["R_h"]),
            cp=float(split["C_P"]),
            cs=float(split["C_S"]),
            edge=float(edge),
            edge_central=float(
                edge_central
            ),
            l3=float(l3),
            d3=float(d3),
            pressure_work=float(wp),
            null_fraction=float(
                self.null_mode_fraction(u)
            ),
        )

    # ======================================================================
    # RANDOMIZED SEARCH
    # ======================================================================

    def objective_value(
        self,
        u,
        objective: str,
    ):
        if objective == "C3":
            return self.critical_quotient(u)

        split = (
            self.pressure_solenoidal_split(
                u
            )
        )

        if objective == "CP":
            return split["C_P"]

        if objective == "CS":
            return split["C_S"]

        if objective == "Rh":
            return split["R_h"]

        if objective == "Cedge":
            return (
                self.weighted_edge_quotient(
                    u
                )["quotient"]
            )

        if objective == "CedgeCentral":
            return (
                self.weighted_edge_quotient_central(
                    u
                )["quotient"]
            )

        raise ValueError(
            f"Unknown objective: {objective}"
        )

    def random_search(
        self,
        objective: str = "C3",
        starts: int = 8,
        steps: int = 10,
        perturbation: float = 0.05,
        seed: int = 101,
    ):
        """
        Randomized projected search.

        This is NOT an optimizer and does NOT compute a supremum.
        """

        if starts < 1:
            raise ValueError(
                "starts must be >= 1."
            )

        if steps < 0:
            raise ValueError(
                "steps must be >= 0."
            )

        if perturbation <= 0:
            raise ValueError(
                "perturbation must be positive."
            )

        rng = np.random.default_rng(
            seed
        )

        best_value = -np.inf
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

            value = self.objective_value(
                u,
                objective,
            )

            if value > best_value:
                best_value = value
                best_field = u.copy()

            for _ in range(steps):

                direction = (
                    rng.standard_normal(
                        u.shape
                    )
                )

                direction = (
                    self.fl.leray_project(
                        direction
                    )
                )

                direction_norm = (
                    self.fl.norm_l2(
                        direction
                    )
                )

                if direction_norm <= 1e-30:
                    continue

                direction /= (
                    direction_norm
                )

                candidate = (
                    u
                    + perturbation
                    * direction
                )

                candidate = (
                    self.fl.leray_project(
                        candidate
                    )
                )

                candidate_norm = (
                    self.fl.norm_l2(
                        candidate
                    )
                )

                if candidate_norm <= 1e-30:
                    continue

                candidate /= (
                    candidate_norm
                )

                candidate_value = (
                    self.objective_value(
                        candidate,
                        objective,
                    )
                )

                if candidate_value > value:
                    u = candidate
                    value = candidate_value

                if value > best_value:
                    best_value = value
                    best_field = u.copy()

        return SearchResult(
            objective=objective,
            best_value=float(
                best_value
            ),
            best_field=best_field,
            starts=starts,
            steps=steps,
        )

    # Backward-compatible wrapper.
    def adversarial_search(
        self,
        starts=3,
        steps=5,
        lr=0.01,
        seed=101,
    ):
        result = self.random_search(
            objective="C3",
            starts=starts,
            steps=steps,
            perturbation=lr,
            seed=seed,
        )

        return {
            "best_C3":
                result.best_value,
            "best_field":
                result.best_field,
        }


# ============================================================================
# ADVERSARIAL FIELD FAMILIES
# ============================================================================

def single_spike_field(
    fluid: StructurePreservingLatticeFluid3D,
):
    """
    Localized Gaussian-like vector spike, then projected.
    """

    X = fluid.X
    Y = fluid.Y
    Z = fluid.Z

    # Periodic distance to origin.
    dx = np.angle(
        np.exp(1j * X)
    )
    dy = np.angle(
        np.exp(1j * Y)
    )
    dz = np.angle(
        np.exp(1j * Z)
    )

    sigma = 0.35

    bump = np.exp(
        -(
            dx ** 2
            + dy ** 2
            + dz ** 2
        )
        / (2.0 * sigma ** 2)
    )

    u = np.zeros(
        (
            3,
            fluid.N,
            fluid.N,
            fluid.N,
        )
    )

    u[0] = bump

    u = fluid.leray_project(u)

    n = fluid.norm_l2(u)

    if n > 0:
        u /= n

    return u


def dipole_field(
    fluid: StructurePreservingLatticeFluid3D,
):
    X = fluid.X
    Y = fluid.Y
    Z = fluid.Z

    sigma = 0.4

    dx1 = np.angle(
        np.exp(1j * X)
    )

    dx2 = np.angle(
        np.exp(1j * (X - np.pi))
    )

    dy = np.angle(
        np.exp(1j * Y)
    )

    dz = np.angle(
        np.exp(1j * Z)
    )

    bump1 = np.exp(
        -(
            dx1 ** 2
            + dy ** 2
            + dz ** 2
        )
        / (2.0 * sigma ** 2)
    )

    bump2 = np.exp(
        -(
            dx2 ** 2
            + dy ** 2
            + dz ** 2
        )
        / (2.0 * sigma ** 2)
    )

    scalar = bump1 - bump2

    u = np.zeros(
        (
            3,
            fluid.N,
            fluid.N,
            fluid.N,
        )
    )

    u[1] = scalar

    u = fluid.leray_project(u)

    n = fluid.norm_l2(u)

    if n > 0:
        u /= n

    return u


def checkerboard_field(
    fluid: StructurePreservingLatticeFluid3D,
):
    idx = np.arange(
        fluid.N
    )

    checker = (
        (-1.0)
        ** (
            idx[:, None, None]
            + idx[None, :, None]
            + idx[None, None, :]
        )
    )

    u = np.zeros(
        (
            3,
            fluid.N,
            fluid.N,
            fluid.N,
        )
    )

    u[0] = checker

    u = fluid.leray_project(u)

    n = fluid.norm_l2(u)

    if n > 0:
        u /= n

    return u


def dyadic_shell_field(
    fluid: StructurePreservingLatticeFluid3D,
):
    """
    Construct a single-band Fourier field.
    """

    rng = np.random.default_rng(777)

    u = rng.standard_normal(
        (
            3,
            fluid.N,
            fluid.N,
            fluid.N,
        )
    )

    u_hat = np.fft.fftn(
        u,
        axes=(1, 2, 3),
    )

    radius = np.sqrt(
        fluid.Kx ** 2
        + fluid.Ky ** 2
        + fluid.Kz ** 2
    )

    mask = (
        (radius >= 2.0)
        & (radius < 4.0)
    )

    u_hat[
        :,
        ~mask
    ] = 0.0

    u = np.real(
        np.fft.ifftn(
            u_hat,
            axes=(1, 2, 3),
        )
    )

    u = fluid.leray_project(u)

    n = fluid.norm_l2(u)

    if n > 0:
        u /= n

    return u


def nested_multiscale_field(
    fluid: StructurePreservingLatticeFluid3D,
):
    """
    Superposition of three localized scales.
    """

    X = fluid.X
    Y = fluid.Y
    Z = fluid.Z

    dx = np.angle(
        np.exp(1j * X)
    )
    dy = np.angle(
        np.exp(1j * Y)
    )
    dz = np.angle(
        np.exp(1j * Z)
    )

    r2 = (
        dx ** 2
        + dy ** 2
        + dz ** 2
    )

    u = np.zeros(
        (
            3,
            fluid.N,
            fluid.N,
            fluid.N,
        )
    )

    for sigma, amplitude in [
        (0.20, 1.0),
        (0.40, 0.6),
        (0.80, 0.3),
    ]:
        bump = np.exp(
            -r2
            / (2.0 * sigma ** 2)
        )

        u[0] += (
            amplitude
            * bump
        )

        u[1] += (
            0.5
            * amplitude
            * np.roll(
                bump,
                fluid.N // 8,
                axis=1,
            )
        )

    u = fluid.leray_project(u)

    n = fluid.norm_l2(u)

    if n > 0:
        u /= n

    return u


def abc_field(
    fluid: StructurePreservingLatticeFluid3D,
):
    """
    ABC / Beltrami-type periodic field.
    """

    A = 1.0
    B = 1.0
    C = 1.0

    u = np.zeros(
        (
            3,
            fluid.N,
            fluid.N,
            fluid.N,
        )
    )

    u[0] = (
        A * np.sin(fluid.Z)
        + C * np.cos(fluid.Y)
    )

    u[1] = (
        B * np.sin(fluid.X)
        + A * np.cos(fluid.Z)
    )

    u[2] = (
        C * np.sin(fluid.Y)
        + B * np.cos(fluid.X)
    )

    u = fluid.leray_project(u)

    n = fluid.norm_l2(u)

    if n > 0:
        u /= n

    return u


def shear_layer_field(
    fluid: StructurePreservingLatticeFluid3D,
):
    """
    Smooth shear depending primarily on one coordinate.
    """

    u = np.zeros(
        (
            3,
            fluid.N,
            fluid.N,
            fluid.N,
        )
    )

    u[0] = np.tanh(
        8.0
        * np.sin(fluid.Y)
    )

    u = fluid.leray_project(u)

    n = fluid.norm_l2(u)

    if n > 0:
        u /= n

    return u


def adversarial_families(
    fluid: StructurePreservingLatticeFluid3D,
):
    return {
        "single_spike":
            single_spike_field(fluid),

        "opposite_sign_dipole":
            dipole_field(fluid),

        "near_nyquist_checkerboard":
            checkerboard_field(fluid),

        "mid_dyadic_shell":
            dyadic_shell_field(fluid),

        "nested_multiscale":
            nested_multiscale_field(fluid),

        "ABC_Beltrami":
            abc_field(fluid),

        "shear_layer":
            shear_layer_field(fluid),
    }


# ============================================================================
# TRUE L-BFGS OPTIMIZATION
# ============================================================================

class LBFGSAdversarialOptimizer:
    """
    Optional true L-BFGS optimization.

    SciPy is required.

    The optimization variable is an unconstrained vector v. The physical
    field is

        u(v) = P(v) / ||P(v)||_2.

    Thus the candidate field remains discretely divergence-free.

    IMPORTANT
    ---------
    This is a numerical optimizer, not a proof of a supremum.
    """

    def __init__(
        self,
        audit: DiscreteProofAudit,
    ):
        self.audit = audit
        self.fl = audit.fl

        try:
            import scipy.optimize as scipy_opt
        except ImportError:
            scipy_opt = None

        self.scipy_opt = scipy_opt

    def available(self):
        return self.scipy_opt is not None

    def normalize_field(
        self,
        raw,
    ):
        raw = raw.reshape(
            (
                3,
                self.fl.N,
                self.fl.N,
                self.fl.N,
            )
        )

        projected = (
            self.fl.leray_project(raw)
        )

        norm = self.fl.norm_l2(
            projected
        )

        if norm <= 1e-30:
            return None

        return projected / norm

    def objective_from_vector(
        self,
        x,
        objective,
    ):
        u = self.normalize_field(x)

        if u is None:
            return 1e6

        value = (
            self.audit.objective_value(
                u,
                objective,
            )
        )

        # scipy minimizes.
        return -float(value)

    def optimize_one(
        self,
        initial_field,
        objective="CP",
        maxiter=100,
        ftol=1e-9,
    ):
        if not self.available():
            raise RuntimeError(
                "SciPy is not installed. "
                "Install scipy to use L-BFGS."
            )

        from scipy.optimize import minimize

        initial = self.normalize_field(
            initial_field
        )

        if initial is None:
            raise ValueError(
                "Initial field has zero projected norm."
            )

        x0 = initial.ravel().copy()

        result = minimize(
            lambda x:
                self.objective_from_vector(
                    x,
                    objective,
                ),
            x0,
            method="L-BFGS-B",
            options={
                "maxiter": int(maxiter),
                "ftol": float(ftol),
                "maxls": 20,
            },
        )

        final_field = (
            self.normalize_field(
                result.x
            )
        )

        if final_field is None:
            return OptimizationResult(
                objective=objective,
                best_value=-np.inf,
                best_field=None,
                starts=1,
                iterations=int(
                    result.nit
                ),
                scipy_available=True,
            )

        value = (
            self.audit.objective_value(
                final_field,
                objective,
            )
        )

        return OptimizationResult(
            objective=objective,
            best_value=float(value),
            best_field=final_field,
            starts=1,
            iterations=int(
                result.nit
            ),
            scipy_available=True,
        )

    def optimize(
        self,
        objective="CP",
        starts=4,
        maxiter=100,
        seed=12345,
        initial_fields=None,
    ):
        if not self.available():
            return OptimizationResult(
                objective=objective,
                best_value=np.nan,
                best_field=None,
                starts=0,
                iterations=0,
                scipy_available=False,
            )

        rng = np.random.default_rng(
            seed
        )

        candidates = []

        if initial_fields is not None:
            candidates.extend(
                initial_fields
            )

        while len(candidates) < starts:

            u = (
                self.fl
                .random_divergence_free_field(
                    seed=int(
                        rng.integers(
                            0,
                            2**31 - 1,
                        )
                    ),
                    amplitude=1.0,
                )
            )

            candidates.append(u)

        best_value = -np.inf
        best_field = None
        total_iterations = 0

        for index, initial in enumerate(
            candidates[:starts],
            start=1,
        ):

            print(
                f"    L-BFGS start "
                f"{index}/{starts} ..."
            )

            result = self.optimize_one(
                initial,
                objective=objective,
                maxiter=maxiter,
            )

            total_iterations += (
                result.iterations
            )

            print(
                f"        value = "
                f"{result.best_value:.8e}"
            )

            if (
                result.best_field is not None
                and result.best_value > best_value
            ):
                best_value = (
                    result.best_value
                )

                best_field = (
                    result.best_field.copy()
                )

        return OptimizationResult(
            objective=objective,
            best_value=float(best_value),
            best_field=best_field,
            starts=starts,
            iterations=total_iterations,
            scipy_available=True,
        )


# ============================================================================
# FIELD AUDIT REPORT
# ============================================================================

def print_field_report(
    name: str,
    audit: DiscreteProofAudit,
    u,
):
    fl = audit.fl

    result = audit.resolution_measurement(
        u
    )

    split = (
        audit.pressure_solenoidal_split(
            u
        )
    )

    edge = (
        audit.weighted_edge_quotient(
            u
        )
    )

    edge_central = (
        audit.weighted_edge_quotient_central(
            u
        )
    )

    pressure_split = (
        audit.pressure_split_audit(u)
    )

    concentration = (
        audit.scale_concentration(u)
    )

    print()
    print(
        f"  {name}"
    )
    print(
        "  " + "-" * 70
    )

    print(
        f"    divergence_inf       = "
        f"{fl.divergence_linf(u):.6e}"
    )

    print(
        f"    ||u||_3             = "
        f"{result.l3:.6e}"
    )

    print(
        f"    D3                  = "
        f"{result.d3:.6e}"
    )

    print(
        f"    C3                  = "
        f"{result.c3:.8e}"
    )

    print(
        f"    CP                  = "
        f"{result.cp:.8e}"
    )

    print(
        f"    CS                  = "
        f"{result.cs:.8e}"
    )

    print(
        f"    Rh                  = "
        f"{result.rh:.8e}"
    )

    print(
        f"    C_edge              = "
        f"{edge['quotient']:.8e}"
    )

    print(
        f"    C_edge_central      = "
        f"{edge_central['quotient']:.8e}"
    )

    print(
        f"    null-mode fraction  = "
        f"{result.null_fraction:.8e}"
    )

    print(
        f"    pressure split err  = "
        f"{pressure_split['relative_residual']:.3e}"
    )

    print(
        f"    high-k energy >=N/4 = "
        f"{concentration['energy_high_N_over_4']:.6e}"
    )

    print(
        f"    high-k energy >=N/3 = "
        f"{concentration['energy_high_N_over_3']:.6e}"
    )

    print(
        f"    top 10% D3 fraction  = "
        f"{concentration['top10_percent_D3_fraction']:.6e}"
    )

    print(
        f"    split check:"
    )

    print(
        f"        pressure         = "
        f"{pressure_split['pressure']:.8e}"
    )

    print(
        f"        <B,q>            = "
        f"{pressure_split['local_term']:.8e}"
    )

    print(
        f"        <B,Pq>           = "
        f"{pressure_split['solenoidal_term']:.8e}"
    )

    print(
        f"        reconstructed     = "
        f"{pressure_split['reconstructed']:.8e}"
    )


# ============================================================================
# STRUCTURAL OPERATOR TESTS
# ============================================================================

def run_operator_tests():
    print("=" * 78)
    print(
        "STRUCTURE-PRESERVING LATTICE NAVIER-STOKES TEST SUITE"
    )
    print("=" * 78)

    N = 16

    fluid = (
        StructurePreservingLatticeFluid3D(
            N=N,
            L=2.0 * np.pi,
            nu=0.01,
        )
    )

    rng = np.random.default_rng(
        42
    )

    # ------------------------------------------------------------------
    # Test 1: skew adjointness
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Test 2: divergence-free projection
    # ------------------------------------------------------------------

    u = (
        fluid.random_divergence_free_field(
            seed=123,
            amplitude=1.0,
        )
    )

    div_inf = (
        fluid.divergence_linf(u)
    )

    print(
        f"Test 2: ||div_h u||_inf = "
        f"{div_inf:.3e}"
    )

    # ------------------------------------------------------------------
    # Test 3: projection idempotence
    # ------------------------------------------------------------------

    u_random = rng.standard_normal(
        (
            3,
            N,
            N,
            N,
        )
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

    # ------------------------------------------------------------------
    # Test 4: convective energy cancellation
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Test 5: RHS energy identity
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Test 6: Taylor-Green divergence
    # ------------------------------------------------------------------

    tg = fluid.taylor_green()

    tg_div = (
        fluid.divergence_linf(tg)
    )

    print(
        f"Test 6: Taylor-Green ||div_h u||_inf = "
        f"{tg_div:.3e}"
    )

    # ------------------------------------------------------------------
    # Test 7: Laplacian consistency
    # ------------------------------------------------------------------

    test_field = rng.standard_normal(
        (N, N, N)
    )

    d0_laplacian = np.zeros_like(
        test_field
    )

    for axis in range(3):

        d0_laplacian += (
            fluid.D0(
                fluid.D0(
                    test_field,
                    axis,
                ),
                axis,
            )
        )

    spectral_laplacian = (
        fluid.laplacian(
            test_field
        )
    )

    laplacian_defect = (
        np.linalg.norm(
            d0_laplacian
            - spectral_laplacian
        )
        / max(
            1.0,
            np.linalg.norm(
                spectral_laplacian
            ),
        )
    )

    print(
        f"Test 7: D0^2/Laplacian relative defect = "
        f"{laplacian_defect:.3e}"
    )

    # ------------------------------------------------------------------
    # Test 8: pressure split identity
    # ------------------------------------------------------------------

    audit = DiscreteProofAudit(
        fluid
    )

    split = (
        audit.pressure_split_audit(
            u
        )
    )

    print(
        f"Test 8: pressure split relative defect = "
        f"{split['relative_residual']:.3e}"
    )

    # ------------------------------------------------------------------
    # Test 9: P orthogonality
    # ------------------------------------------------------------------

    q = (
        audit.velocity_magnitude(u)
        * u
    )

    Pq = fluid.leray_project(q)

    gradient_part = (
        q - Pq
    )

    orthogonality = (
        abs(
            fluid.inner_product(
                Pq,
                gradient_part,
            )
        )
        / max(
            1.0,
            fluid.norm_l2(Pq)
            * fluid.norm_l2(
                gradient_part
            ),
        )
    )

    print(
        f"Test 9: Pq/(I-P)q orthogonality = "
        f"{orthogonality:.3e}"
    )

    # ------------------------------------------------------------------
    # PASS / FAIL
    # ------------------------------------------------------------------

    tolerances = {
        "skew": 1e-12,
        "div": 1e-12,
        "projection": 1e-12,
        "nonlinear": 1e-12,
        "energy": 1e-12,
        "tg": 1e-12,
        "laplacian": 1e-12,
        "split": 1e-12,
        "orthogonality": 1e-12,
    }

    passed = (
        skew_defect
        < tolerances["skew"]

        and div_inf
        < tolerances["div"]

        and projection_defect
        < tolerances["projection"]

        and nonlinear_relative
        < tolerances["nonlinear"]

        and energy_defect
        < tolerances["energy"]

        and tg_div
        < tolerances["tg"]

        and laplacian_defect
        < tolerances["laplacian"]

        and split["relative_residual"]
        < tolerances["split"]

        and orthogonality
        < tolerances["orthogonality"]
    )

    print(
        "Operator tests: "
        + (
            "PASS"
            if passed
            else "FAIL"
        )
    )

    if not passed:
        raise RuntimeError(
            "Structure-preserving operator tests failed."
        )

    return fluid


# ============================================================================
# ADVERSARIAL FAMILY AUDIT
# ============================================================================

def run_adversarial_families(
    resolutions=(16, 32),
):
    print()
    print("=" * 78)
    print(
        "SEVEN ADVERSARIAL GEOMETRIC FAMILIES"
    )
    print("=" * 78)

    summary = []

    for N in resolutions:

        print()
        print(
            f"N = {N}"
        )
        print(
            "-" * 78
        )

        fluid = (
            StructurePreservingLatticeFluid3D(
                N=N,
                L=2.0 * np.pi,
                nu=0.01,
            )
        )

        audit = (
            DiscreteProofAudit(
                fluid
            )
        )

        fields = adversarial_families(
            fluid
        )

        for name, u in fields.items():

            print_field_report(
                name,
                audit,
                u,
            )

            measurement = (
                audit.resolution_measurement(
                    u
                )
            )

            summary.append(
                (
                    name,
                    measurement,
                )
            )

    print()
    print(
        "ADVERSARIAL FAMILY SUMMARY"
    )
    print(
        "-" * 78
    )

    print(
        f"{'Family':30s} "
        f"{'N':>4s} "
        f"{'CP':>10s} "
        f"{'CS':>10s} "
        f"{'Rh':>10s} "
        f"{'Edge':>10s}"
    )

    for name, result in summary:

        print(
            f"{name:30s} "
            f"{result.N:4d} "
            f"{result.cp:10.4e} "
            f"{result.cs:10.4e} "
            f"{result.rh:10.4e} "
            f"{result.edge:10.4e}"
        )

    return summary


# ============================================================================
# SEARCH AUDIT
# ============================================================================

def run_randomized_searches(
    N=32,
    starts=8,
    steps=10,
):
    print()
    print("=" * 78)
    print(
        f"RANDOMIZED LARGE-VALUE SEARCH — N={N}"
    )
    print("=" * 78)

    fluid = (
        StructurePreservingLatticeFluid3D(
            N=N,
            L=2.0 * np.pi,
            nu=0.01,
        )
    )

    audit = (
        DiscreteProofAudit(
            fluid
        )
    )

    objectives = [
        "C3",
        "CP",
        "CS",
        "Rh",
        "Cedge",
        "CedgeCentral",
    ]

    results = {}

    for objective in objectives:

        print()
        print(
            f"Objective: {objective}"
        )

        result = audit.random_search(
            objective=objective,
            starts=starts,
            steps=steps,
            perturbation=0.05,
            seed=1000 + len(
                results
            ),
        )

        results[objective] = result

        print(
            f"  best discovered value = "
            f"{result.best_value:.10e}"
        )

        if result.best_field is not None:

            print_field_report(
                f"best randomized {objective}",
                audit,
                result.best_field,
            )

    return results


# ============================================================================
# L-BFGS AUDIT
# ============================================================================

def run_lbfgs_optimization(
    N=32,
    starts=4,
    maxiter=100,
):
    print()
    print("=" * 78)
    print(
        f"TRUE L-BFGS ADVERSARIAL OPTIMIZATION — N={N}"
    )
    print("=" * 78)

    fluid = (
        StructurePreservingLatticeFluid3D(
            N=N,
            L=2.0 * np.pi,
            nu=0.01,
        )
    )

    audit = (
        DiscreteProofAudit(
            fluid
        )
    )

    optimizer = (
        LBFGSAdversarialOptimizer(
            audit
        )
    )

    if not optimizer.available():

        print()
        print(
            "SciPy is not installed."
        )
        print(
            "Install with:"
        )
        print(
            "    python -m pip install scipy"
        )

        return {}

    objectives = [
        "CP",
        "CS",
        "Rh",
        "Cedge",
    ]

    results = {}

    for objective in objectives:

        print()
        print(
            "=" * 50
        )

        result = optimizer.optimize(
            objective=objective,
            starts=starts,
            maxiter=maxiter,
            seed=9000 + len(
                results
            ),
        )

        results[objective] = result

        print()
        print(
            f"BEST L-BFGS {objective}"
        )

        print(
            f"  value      = "
            f"{result.best_value:.10e}"
        )

        print(
            f"  starts     = "
            f"{result.starts}"
        )

        print(
            f"  iterations = "
            f"{result.iterations}"
        )

        if result.best_field is not None:

            print_field_report(
                f"L-BFGS maximizer of {objective}",
                audit,
                result.best_field,
            )

    return results


# ============================================================================
# RESOLUTION AUDIT
# ============================================================================

def run_resolution_audit(
    resolutions=(16, 32, 64),
    seed=42,
):
    print()
    print("=" * 78)
    print(
        "RESOLUTION AUDIT"
    )
    print("=" * 78)

    results = []

    for N in resolutions:

        print()
        print(
            f"N = {N}"
        )

        fluid = (
            StructurePreservingLatticeFluid3D(
                N=N,
                L=2.0 * np.pi,
                nu=0.01,
            )
        )

        audit = (
            DiscreteProofAudit(
                fluid
            )
        )

        u = (
            fluid.random_divergence_free_field(
                seed=seed,
                amplitude=1.0,
            )
        )

        result = (
            audit.resolution_measurement(
                u
            )
        )

        results.append(result)

        print(
            f"  C3              = "
            f"{result.c3:.8e}"
        )

        print(
            f"  CP              = "
            f"{result.cp:.8e}"
        )

        print(
            f"  CS              = "
            f"{result.cs:.8e}"
        )

        print(
            f"  Rh              = "
            f"{result.rh:.8e}"
        )

        print(
            f"  C_edge          = "
            f"{result.edge:.8e}"
        )

        print(
            f"  C_edge central  = "
            f"{result.edge_central:.8e}"
        )

        print(
            f"  null fraction   = "
            f"{result.null_fraction:.8e}"
        )

    print()
    print(
        "SUMMARY"
    )
    print(
        "-" * 78
    )

    print(
        f"{'N':>5s} "
        f"{'C3':>12s} "
        f"{'CP':>12s} "
        f"{'CS':>12s} "
        f"{'Rh':>12s} "
        f"{'Cedge':>12s} "
        f"{'null':>12s}"
    )

    for r in results:

        print(
            f"{r.N:5d} "
            f"{r.c3:12.5e} "
            f"{r.cp:12.5e} "
            f"{r.cs:12.5e} "
            f"{r.rh:12.5e} "
            f"{r.edge:12.5e} "
            f"{r.null_fraction:12.5e}"
        )

    return results


# ============================================================================
# COMPLETE AUDIT
# ============================================================================

def run_discrete_proof_audit(
    resolutions=(16, 32, 64),
    seed=42,
    adversarial=True,
    optimize=False,
    search_starts=8,
    search_steps=10,
    optimizer_starts=4,
    optimizer_iterations=100,
):
    print("=" * 78)
    print(
        "DISCRETE NAVIER-STOKES PROOF / AUDIT"
    )
    print("=" * 78)

    print()
    print(
        "STATUS:"
    )

    print(
        "    LEVEL 1: exact finite-dimensional identities."
    )

    print(
        "    LEVEL 2: numerical search for large quotients."
    )

    print(
        "    LEVEL 3: continuum theorem NOT established."
    )

    # ----------------------------------------------------------------------
    # Basic random-field resolution audit
    # ----------------------------------------------------------------------

    resolution_results = (
        run_resolution_audit(
            resolutions=resolutions,
            seed=seed,
        )
    )

    # ----------------------------------------------------------------------
    # Seven adversarial families
    # ----------------------------------------------------------------------

    if adversarial:

        family_resolutions = tuple(
            N
            for N in resolutions
            if N <= 64
        )

        if not family_resolutions:
            family_resolutions = (
                resolutions[0],
            )

        run_adversarial_families(
            resolutions=family_resolutions
        )

    # ----------------------------------------------------------------------
    # Randomized searches
    # ----------------------------------------------------------------------

    if adversarial:

        search_N = max(
            resolutions
        )

        run_randomized_searches(
            N=search_N,
            starts=search_starts,
            steps=search_steps,
        )

    # ----------------------------------------------------------------------
    # True optimizer
    # ----------------------------------------------------------------------

    if optimize:

        optimize_N = min(
            max(resolutions),
            32,
        )

        run_lbfgs_optimization(
            N=optimize_N,
            starts=optimizer_starts,
            maxiter=optimizer_iterations,
        )

    # ----------------------------------------------------------------------
    # Mathematical status
    # ----------------------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "MATHEMATICAL STATUS"
    )
    print("=" * 78)

    print(
        """
LEVEL 1 — EXACT DISCRETE ALGEBRA
--------------------------------
The program verifies finite-dimensional identities including:

    * D0 skew-adjointness;
    * divergence-free projection;
    * projection idempotence;
    * skew-form energy cancellation;
    * compatible Laplacian identity;
    * pressure/splitting identity;
    * P/(I-P) orthogonality.

LEVEL 2 — NUMERICAL EVIDENCE
----------------------------
The primary unresolved target is the weighted edge quotient

    C_edge(u)
      =
    sum |u| |delta_j u|^3
    ----------------------------------------------
    h ||u||_3 sum |u| |delta_j u|^2 / h^2.

Large-value searches also target

    C_P,
    C_S,
    R_h.

Randomized searches discover large values only.

L-BFGS searches are genuine numerical optimizations, but their output
is still only a discovered local/global numerical candidate at a finite
resolution. It is NOT a proof of the supremum.

NYQUIST / NULL MODES
--------------------
Central differences possess additional Ktilde = 0 modes. The program
reports their Fourier energy fraction and can remove them for a
sensitivity audit.

SCALE CONCENTRATION
-------------------
For large-value candidates the program reports high-frequency Fourier
energy and concentration of the critical dissipation.

SHELLS
------
Explicitly filtered disjoint Fourier shells are orthogonal in the
ordinary L2 pairing. This does NOT establish nonlinear triadic
decoupling.

CONTINUUM STATUS
----------------
None of the numerical observations establish an N-independent analytic
bound or a continuum Navier-Stokes regularity theorem.

The central mathematical question remains whether an N-independent
analytic estimate can be proved for the weighted edge quotient or an
equivalent structure-preserving formulation.
"""
    )

    print("=" * 78)
    print(
        "AUDIT COMPLETE"
    )
    print("=" * 78)

    return resolution_results


# ============================================================================
# COMMAND LINE
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Structure-preserving discrete "
            "Navier-Stokes proof/audit."
        )
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Run structural tests and a single N=16 "
            "resolution audit."
        ),
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Run N=16,32,64 with adversarial "
            "families and searches."
        ),
    )

    parser.add_argument(
        "--optimize",
        action="store_true",
        help=(
            "Run true L-BFGS optimization. "
            "Requires SciPy."
        ),
    )

    parser.add_argument(
        "--search-starts",
        type=int,
        default=8,
        help=(
            "Number of starts for randomized search."
        ),
    )

    parser.add_argument(
        "--search-steps",
        type=int,
        default=10,
        help=(
            "Number of local perturbation steps."
        ),
    )

    parser.add_argument(
        "--opt-starts",
        type=int,
        default=4,
        help=(
            "Number of L-BFGS starts per objective."
        ),
    )

    parser.add_argument(
        "--opt-iter",
        type=int,
        default=100,
        help=(
            "Maximum L-BFGS iterations per start."
        ),
    )

    args = parser.parse_args()

    print(
        "Python:",
        sys.version.split()[0],
    )

    print(
        "NumPy:",
        np.__version__,
    )

    print()

    # Always run Level-1 operator tests first.
    run_operator_tests()

    print()

    if args.quick:

        run_discrete_proof_audit(
            resolutions=(16,),
            seed=42,
            adversarial=False,
            optimize=False,
            search_starts=args.search_starts,
            search_steps=args.search_steps,
            optimizer_starts=args.opt_starts,
            optimizer_iterations=args.opt_iter,
        )

        return

    if args.full:

        resolutions = (
            16,
            32,
            64,
        )

        run_discrete_proof_audit(
            resolutions=resolutions,
            seed=42,
            adversarial=True,
            optimize=args.optimize,
            search_starts=args.search_starts,
            search_steps=args.search_steps,
            optimizer_starts=args.opt_starts,
            optimizer_iterations=args.opt_iter,
        )

        return

    # Default run.
    run_discrete_proof_audit(
        resolutions=(
            16,
            32,
            64,
        ),
        seed=42,
        adversarial=True,
        optimize=args.optimize,
        search_starts=args.search_starts,
        search_steps=args.search_steps,
        optimizer_starts=args.opt_starts,
        optimizer_iterations=args.opt_iter,
    )


if __name__ == "__main__":
    main()
