"""
==============================================================================
DISCRETE NAVIER-STOKES PROOF / AUDIT
LEVEL-1 / LEVEL-2 CERTIFICATE-DISCOVERY ENGINE
==============================================================================

STATUS
------
EXPERIMENTAL DISCRETE AUDIT.

This program works with a finite periodic lattice.  It verifies exact
finite-dimensional identities numerically and searches for large values of
the associated scale-invariant quotients.

It does NOT prove global regularity of the continuum 3-D Navier-Stokes
equations.

MATHEMATICAL STATUS
-------------------

LEVEL 1
-------
Exact algebraic identities are tested numerically:

    <D0 f, g> = - <f, D0 g>

    P^2 = P

    div(Pu) = 0

    <B(u,u), u> = 0

    <grad p, grad q>
        = <B, q> - <B, Pq>

    <Pq, (I-P)q> = 0

LEVEL 2
-------
The following weighted edge quotient is evaluated:

    C_edge =
        sum_x,j |u| |delta_j u|^3
        -----------------------------------------
        h ||u||_3 sum_x,j |u| |delta_j u|^2/h^2

where

    delta_j u(x) = u(x+h e_j) - u(x).

The discrete cell embedding

    ||u||_infinity <= h^(-1) ||u||_3

and

    |delta_j u| <= 2 ||u||_infinity

give the analytic ceiling

    C_edge <= 2.

The corresponding commutator constant is recorded as

    C_R <= 1

according to the discrete commutator normalization used by this audit.

IMPORTANT
---------
The code does not treat numerical maxima as proofs.

For example:

    "best discovered C_S = 0.22"

means exactly that: a numerical search found a field with that value.

It does NOT mean

    C_S <= 0.22.

The remaining analytical bottleneck is the solenoidal pairing

    C_S =
        |<B(u,u), P(|u|u)>|
        ---------------------
        ||u||_3 D3(u).

The pressure quantity satisfies the exact Hodge relation

    C_P = |<B, q-Pq>|/(||u||_3 D3)

and therefore

    C_P <= C_R + C_S.

With C_R <= 1,

    C_P <= 1 + C_S.

NYQUIST
-------
Central differences have additional null modes at zero/Nyquist frequencies.
These modes are explicitly audited.

DEPENDENCIES
------------
Required:
    Python >= 3.10
    NumPy

Optional:
    SciPy

SciPy is required only for the L-BFGS adversarial search.

RUN
---
    python discrete_proof.py

Quick:
    python discrete_proof.py --quick

Targeted C_S Discovery:
    python discrete_proof.py --target-cs

Full:
    python discrete_proof.py --full

Optimizer:
    python discrete_proof.py --optimize

The default run is deliberately moderate.  Full L-BFGS searches on N=32
can be expensive.

==============================================================================
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ============================================================================
# GLOBAL CERTIFICATE CONSTANTS
# ============================================================================

ANALYTIC_C_EDGE_BOUND = 2.0
ANALYTIC_C_R_BOUND = 1.0

EPS = 1.0e-30


# ============================================================================
# RESULT CONTAINERS
# ============================================================================

@dataclass
class ScalingResult:
    amplitude: float
    c_p: float
    c_s: float
    r_h: float
    l3: float
    d3: float
    pressure_work: float


@dataclass
class ResolutionResult:
    N: int
    c_p: float
    c_s: float
    r_h: float
    c_edge: float
    c_edge_central: float
    l3: float
    d3: float


@dataclass
class SearchResult:
    target: str
    value: float
    success: bool
    nit: int
    nfev: int
    message: str
    field: Optional[np.ndarray]


@dataclass
class CSResult:
    cs: float
    numerator: float
    l3: float
    d3: float
    rh: float
    cos_theta: float


@dataclass
class CSSearchResult:
    best_cs: float
    best_field: Optional[np.ndarray]
    history: List[Dict]
    starts: int
    iterations: int


@dataclass
class CSShellResult:
    shells: List[int]
    values: np.ndarray
    absolute_values: np.ndarray
    total: float
    row_sums: np.ndarray


# ============================================================================
# STRUCTURE-PRESERVING PERIODIC LATTICE FLUID
# ============================================================================

class StructurePreservingLatticeFluid3D:
    """
    Periodic 3-D lattice system using one compatible central-difference
    Fourier symbol throughout.

    Grid:
        N x N x N

    Physical spacing:
        h = L/N

    Central difference:
        D0_j f = [f(x+h e_j)-f(x-h e_j)]/(2h)

    Fourier symbol:
        i sin(k_j h)/h

    Compatible Laplacian:
        Delta_h -> -|Ktilde|^2

    Leray projection:
        P = I - Ktilde Ktilde^T/|Ktilde|^2

    Ktilde = 0 modes are left unchanged by P because they already lie in
    the kernel of the discrete divergence.
    """

    def __init__(
        self,
        N: int = 16,
        L: float = 2.0 * np.pi,
        nu: float = 0.01,
    ):
        if N < 4:
            raise ValueError("N must be at least 4.")

        if L <= 0.0:
            raise ValueError("L must be positive.")

        if nu < 0.0:
            raise ValueError("nu must be nonnegative.")

        self.N = int(N)
        self.L = float(L)
        self.h = self.L / self.N
        self.nu = float(nu)

        # ------------------------------------------------------------------
        # Physical coordinates
        # ------------------------------------------------------------------

        x = np.arange(self.N, dtype=float) * self.h

        self.X, self.Y, self.Z = np.meshgrid(
            x,
            x,
            x,
            indexing="ij",
        )

        # ------------------------------------------------------------------
        # Fourier wave numbers
        # ------------------------------------------------------------------

        k = (
            2.0
            * np.pi
            * np.fft.fftfreq(
                self.N,
                d=self.h,
            )
        )

        self.Kx, self.Ky, self.Kz = np.meshgrid(
            k,
            k,
            k,
            indexing="ij",
        )

        # Central-difference symbols.

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

        self.Kmag = np.sqrt(
            self.Kx ** 2
            + self.Ky ** 2
            + self.Kz ** 2
        )

        # Numerical null mask.

        self.null_modes = (
            self.K_sq <= 1.0e-14
        )

        self.projectable_modes = (
            ~self.null_modes
        )

        self.inv_K_sq = np.zeros_like(
            self.K_sq
        )

        self.inv_K_sq[
            self.projectable_modes
        ] = (
            1.0
            / self.K_sq[
                self.projectable_modes
            ]
        )

    # ======================================================================
    # BASIC DIFFERENCES
    # ======================================================================

    def D0(self, f, axis: int):
        """
        Periodic central difference.

            D0 f = [f(x+h)-f(x-h)]/(2h).
        """
        return (
            np.roll(f, -1, axis=axis)
            - np.roll(f, 1, axis=axis)
        ) / (2.0 * self.h)

    def D_plus(self, f, axis: int):
        """
        Periodic forward difference.
        """
        return (
            np.roll(f, -1, axis=axis)
            - f
        ) / self.h

    def D_minus(self, f, axis: int):
        """
        Periodic backward difference.
        """
        return (
            f
            - np.roll(f, 1, axis=axis)
        ) / self.h

    def delta_plus(self, u, component_axis: int):
        """
        Raw edge increment

            delta_j u(x) = u(x+h e_j)-u(x).

        component_axis is 1,2,3 because u has shape

            (component, x, y, z).
        """
        return (
            np.roll(
                u,
                -1,
                axis=component_axis,
            )
            - u
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
    # LAPLACIAN
    # ======================================================================

    def laplacian(self, f):
        """
        Compatible spectral Laplacian.

            Delta_h -> -|Ktilde|^2.
        """
        f_hat = np.fft.fftn(f)

        out_hat = (
            -self.K_sq
            * f_hat
        )

        return np.real(
            np.fft.ifftn(out_hat)
        )

    # ======================================================================
    # LERAY PROJECTION
    # ======================================================================

    def leray_project(self, u):
        """
        Orthogonal projection onto ker(div_h).

        At Ktilde=0 the input mode is left unchanged.
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
    # CONVECTION
    # ======================================================================

    def skew_convection(self, u, v):
        """
        Morinishi skew form:

            B_i(u,v)
              = 1/2 sum_j [
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
                    term1 + term2
                )

        return B

    # ======================================================================
    # NAVIER-STOKES RHS
    # ======================================================================

    def rhs(self, u):
        B = self.skew_convection(
            u,
            u,
        )

        PB = self.leray_project(B)

        diffusion = np.stack(
            [
                self.laplacian(u[0]),
                self.laplacian(u[1]),
                self.laplacian(u[2]),
            ],
            axis=0,
        )

        return (
            -PB
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
        mag = np.sqrt(
            np.sum(
                u ** 2,
                axis=0,
            )
        )

        return float(
            np.sum(
                mag ** 3
            )
            * self.h ** 3
        ) ** (1.0 / 3.0)

    def norm_linf(self, u):
        mag = np.sqrt(
            np.sum(
                u ** 2,
                axis=0,
            )
        )

        return float(
            np.max(mag)
        )

    def energy(self, u):
        return 0.5 * self.inner_product(
            u,
            u,
        )

    # ======================================================================
    # DISSIPATION
    # ======================================================================

    def enstrophy(self, u):
        value = 0.0

        for i in range(3):
            for j in range(3):
                d = self.D0(
                    u[i],
                    j,
                )

                value += self.inner_product(
                    d,
                    d,
                )

        return float(value)

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
        normalize: str = "L2",
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

        if normalize.upper() == "L3":
            n = self.norm_l3(u)
        else:
            n = self.norm_l2(u)

        if n > EPS:
            u *= amplitude / n

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

    # ======================================================================
    # TIME STEPPING
    # ======================================================================

    def rk4_step(self, u, dt):
        k1 = self.rhs(u)

        k2 = self.rhs(
            u + 0.5 * dt * k1
        )

        k3 = self.rhs(
            u + 0.5 * dt * k2
        )

        k4 = self.rhs(
            u + dt * k3
        )

        return (
            u
            + dt
            * (
                k1
                + 2.0 * k2
                + 2.0 * k3
                + k4
            )
            / 6.0
        )


# ============================================================================
# AUDIT ENGINE
# ============================================================================

class DiscreteProofAudit:

    def __init__(
        self,
        fluid: StructurePreservingLatticeFluid3D,
    ):
        self.fl = fluid

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

    def gradient_squared(self, u):
        """
        Pointwise

            |grad_h u|^2
        """
        u_hat = np.fft.fftn(
            u,
            axes=(1, 2, 3),
        )

        grad_sq = np.zeros(
            u.shape[1:],
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
        mag = self.velocity_magnitude(u)

        grad_sq = self.gradient_squared(u)

        return float(
            np.sum(
                mag * grad_sq
            )
            * self.fl.h ** 3
        )

    def denominator(self, u):
        return (
            self.fl.norm_l3(u)
            * self.critical_dissipation(u)
        )

    # ======================================================================
    # PRESSURE GRADIENTS
    # ======================================================================

    def pressure_gradients(self, u):
        """
        Compute grad p and grad q.

        -Delta p = div B
        -Delta q = div(|u|u)

        Since

            -Delta -> |Ktilde|^2,

        p_hat = div(B)_hat / |Ktilde|^2
        q_hat = div(Q)_hat / |Ktilde|^2.
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

        mag = self.velocity_magnitude(u)

        Q = mag * u

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

        return grad_p_hat, grad_q_hat

    def pressure_gradients_physical(self, u):
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

        return gp, gq

    def pressure_work_signed(self, u):
        gp, gq = (
            self.pressure_gradients_physical(
                u
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
    # EXACT HODGE SPLIT
    # ======================================================================

    def pressure_split_audit(self, u):
        """
        Verify

            <grad p, grad q>
              =
            <B,q> - <B,Pq>.

        """
        mag = self.velocity_magnitude(u)

        q = mag * u

        B = self.fl.skew_convection(
            u,
            u,
        )

        Pq = self.fl.leray_project(q)

        pressure = (
            self.pressure_work_signed(u)
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
            "solenoidal_term":
                float(solenoidal_term),
            "reconstructed":
                float(reconstructed),
            "residual":
                float(residual),
            "relative_residual":
                float(
                    abs(residual) / scale
                ),
        }

    def pressure_solenoidal_split(self, u):
        mag = self.velocity_magnitude(u)

        q = mag * u

        B = self.fl.skew_convection(
            u,
            u,
        )

        Pq = self.fl.leray_project(q)

        q_grad = q - Pq

        a = self.fl.inner_product(
            B,
            q_grad,
        )

        b = self.fl.inner_product(
            B,
            Pq,
        )

        PB = self.fl.leray_project(B)
        norm_PB = self.fl.norm_l2(PB)
        norm_Pq = self.fl.norm_l2(Pq)
        cos_theta = (abs(b) / max(norm_PB * norm_Pq, EPS)) if (norm_PB * norm_Pq > EPS) else 0.0

        denominator = max(
            self.denominator(u),
            EPS,
        )

        return {
            "cos_theta": float(cos_theta),
            "a": float(a),
            "b": float(b),

            "C_P": float(
                abs(a) / denominator
            ),

            "C_S": float(
                abs(b) / denominator
            ),

            "R_h": float(
                abs(a + b)
                / denominator
            ),

            "signed_R_h": float(
                (a + b)
                / denominator
            ),

            "a_abs_normalized":
                float(
                    abs(a) / denominator
                ),

            "b_abs_normalized":
                float(
                    abs(b) / denominator
                ),

            "sum_normalized":
                float(
                    abs(a + b)
                    / denominator
                ),
        }

    def critical_quotient(self, u):
        return self.pressure_solenoidal_split(
            u
        )["C_P"]

    def solenoidal_quotient(self, u):
        return self.pressure_solenoidal_split(
            u
        )["C_S"]

    def commutator_quotient(self, u):
        return self.pressure_solenoidal_split(
            u
        )["R_h"]

    # ======================================================================
    # CHAIN-RULE / COMMUTATOR
    # ======================================================================

    def chain_rule_defect(self, u):
        mag = self.velocity_magnitude(u)

        B = self.fl.skew_convection(
            u,
            u,
        )

        numerator = abs(
            self.fl.inner_product(
                B,
                mag * u,
            )
        )

        denominator = max(
            self.denominator(u),
            EPS,
        )

        return float(
            numerator / denominator
        )

    # ======================================================================
    # EDGE QUOTIENT
    # ======================================================================

    def weighted_edge_quotient(self, u):
        """
        Evaluate

            C_edge =
              sum |u| |delta u|^3
              -------------------------------------
              h ||u||_3 sum |u| |delta u|^2/h^2.

        The analytic certificate is

            C_edge <= 2.
        """

        mag = self.velocity_magnitude(u)

        cubic = 0.0
        quadratic = 0.0

        for j in range(3):

            delta = self.fl.delta_plus(
                u,
                j + 1,
            )

            delta_mag = np.sqrt(
                np.sum(
                    delta ** 2,
                    axis=0,
                )
            )

            cubic += np.sum(
                mag
                * delta_mag ** 3
            )

            quadratic += np.sum(
                mag
                * delta_mag ** 2
            )

        cubic *= self.fl.h ** 3

        quadratic *= (
            self.fl.h ** 3
            / self.fl.h ** 2
        )

        l3 = self.fl.norm_l3(u)

        denominator = (
            self.fl.h
            * l3
            * quadratic
        )

        quotient = (
            cubic / denominator
            if denominator > EPS
            else 0.0
        )

        return {
            "edge_cubic":
                float(cubic),

            "weighted_quadratic":
                float(quadratic),

            "l3":
                float(l3),

            "quotient":
                float(quotient),

            "analytic_bound":
                ANALYTIC_C_EDGE_BOUND,

            "certified":
                bool(
                    quotient
                    <= ANALYTIC_C_EDGE_BOUND
                    + 1e-10
                ),
        }

    def weighted_edge_quotient_central(self, u):
        """
        Central-difference comparison.
        """
        mag = self.velocity_magnitude(u)

        cubic = 0.0
        quadratic = 0.0

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

            cubic += np.sum(
                mag
                * dmag ** 3
            )

            quadratic += np.sum(
                mag
                * dmag ** 2
            )

        cubic *= self.fl.h ** 3
        quadratic *= self.fl.h ** 3

        l3 = self.fl.norm_l3(u)

        denominator = (
            l3
            * quadratic
        )

        quotient = (
            cubic / denominator
            if denominator > EPS
            else 0.0
        )

        return {
            "central_cubic":
                float(cubic),

            "central_quadratic":
                float(quadratic),

            "l3":
                float(l3),

            "quotient":
                float(quotient),
        }

    # ======================================================================
    # NULL MODE AUDIT
    # ======================================================================

    def null_mode_fraction(self, u):
        u_hat = np.fft.fftn(
            u,
            axes=(1, 2, 3),
        )

        total = np.sum(
            np.abs(u_hat) ** 2
        )

        null = np.sum(
            np.abs(
                u_hat[
                    :,
                    self.fl.null_modes
                ]
            ) ** 2
        )

        if total <= EPS:
            return 0.0

        return float(
            null / total
        )

    def remove_null_modes(self, u):
        u_hat = np.fft.fftn(
            u,
            axes=(1, 2, 3),
        )

        u_hat[
            :,
            self.fl.null_modes
        ] = 0.0

        filtered = np.real(
            np.fft.ifftn(
                u_hat,
                axes=(1, 2, 3),
            )
        )

        return self.fl.leray_project(
            filtered
        )

    # ======================================================================
    # CONCENTRATION
    # ======================================================================

    def scale_concentration(self, u):
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

        high_n4 = np.sum(
            energy_density[
                self.fl.Kmag
                >= self.fl.N / 4.0
            ]
        )

        high_n3 = np.sum(
            energy_density[
                self.fl.Kmag
                >= self.fl.N / 3.0
            ]
        )

        weighted = (
            self.velocity_magnitude(u)
            * self.gradient_squared(u)
        )

        total_d3 = np.sum(
            weighted
        )

        threshold = np.percentile(
            weighted,
            90.0,
        )

        top10 = np.sum(
            weighted[
                weighted >= threshold
            ]
        )

        return {
            "energy_high_N_over_4":
                float(
                    high_n4
                    / max(
                        total_energy,
                        EPS,
                    )
                ),

            "energy_high_N_over_3":
                float(
                    high_n3
                    / max(
                        total_energy,
                        EPS,
                    )
                ),

            "top10_percent_D3_fraction":
                float(
                    top10
                    / max(
                        total_d3,
                        EPS,
                    )
                ),
        }

    # ======================================================================
    # DYADIC SHELLS
    # ======================================================================

    def dyadic_shell_masks(self):
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
                (self.fl.Kmag >= low)
                & (self.fl.Kmag < high)
            )

        return masks

    def shell_field(self, u, mask):
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

        grad_sq = self.gradient_squared(
            uj
        )

        mag = self.velocity_magnitude(
            u
        )

        return float(
            np.sum(
                mag * grad_sq
            )
            * self.fl.h ** 3
        )

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

        for a, j in enumerate(shells):
            D[a] = self.shell_dissipation(u, masks[j])
            for b, k in enumerate(shells):
                common_mask = masks[j] & masks[k]
                if np.any(common_mask):
                    inner = np.sum(
                        gp_hat[:, common_mask] * np.conj(gq_hat[:, common_mask])
                    )
                    M[a, b] = abs(inner) * parseval
                else:
                    M[a, b] = 0.0

        l3 = self.fl.norm_l3(u)

        Gamma = np.zeros_like(
            M
        )

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

                if denominator > EPS:
                    Gamma[a, b] = (
                        M[a, b]
                        / denominator
                    )

        row_sums = (
            np.sum(
                Gamma,
                axis=1,
            )
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
    # TRIADIC SOURCE AUDIT
    # ======================================================================

    def triadic_source_audit(self, u):
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

            if not np.any(mask):
                continue

            B_energy = np.sum(
                np.abs(
                    B_hat[
                        :,
                        mask
                    ]
                ) ** 2
            )

            Q_energy = np.sum(
                np.abs(
                    Q_hat[
                        :,
                        mask
                    ]
                ) ** 2
            )

            result[j] = {
                "B_source_energy":
                    float(B_energy),

                "Q_source_energy":
                    float(Q_energy),

                "B_source_norm":
                    float(
                        np.sqrt(B_energy)
                    ),

                "Q_source_norm":
                    float(
                        np.sqrt(Q_energy)
                    ),
            }

        return result

    # ======================================================================
    # ALIASING
    # ======================================================================

    def two_thirds_mask(self):
        cutoff = self.fl.N / 3.0

        return (
            (np.abs(self.fl.Kx) <= cutoff)
            & (np.abs(self.fl.Ky) <= cutoff)
            & (np.abs(self.fl.Kz) <= cutoff)
        )

    def spectral_filter(self, u, mask):
        u_hat = np.fft.fftn(
            u,
            axes=(1, 2, 3),
        )

        out = np.zeros_like(
            u_hat
        )

        out[
            :,
            mask
        ] = u_hat[
            :,
            mask
        ]

        return np.real(
            np.fft.ifftn(
                out,
                axes=(1, 2, 3),
            )
        )

    def dealias_field(self, u):
        filtered = self.spectral_filter(
            u,
            self.two_thirds_mask(),
        )

        return self.fl.leray_project(
            filtered
        )

    def aliasing_audit(self, u):
        raw = self.critical_quotient(u)

        filtered = self.dealias_field(
            u
        )

        dealiased = (
            self.critical_quotient(
                filtered
            )
        )

        absolute = abs(
            raw - dealiased
        )

        relative = (
            absolute
            / max(
                abs(raw),
                EPS,
            )
        )

        return {
            "C3_raw":
                float(raw),

            "C3_dealiased":
                float(dealiased),

            "absolute_difference":
                float(absolute),

            "relative_difference":
                float(relative),
        }

    # ======================================================================
    # AMPLITUDE INVARIANCE
    # ======================================================================

    def amplitude_invariance_test(
        self,
        u,
        amplitudes=(1.0, 2.0, 4.0, 8.0),
    ):
        n = self.fl.norm_l2(u)

        if n <= EPS:
            raise ValueError(
                "Cannot test a zero field."
            )

        u0 = u / n

        results = []

        for A in amplitudes:

            ua = A * u0

            split = (
                self.pressure_solenoidal_split(
                    ua
                )
            )

            results.append(
                ScalingResult(
                    amplitude=float(A),
                    c_p=split["C_P"],
                    c_s=split["C_S"],
                    r_h=split["R_h"],
                    l3=self.fl.norm_l3(ua),
                    d3=self.critical_dissipation(
                        ua
                    ),
                    pressure_work=self.pressure_work(
                        ua
                    ),
                )
            )

        return results

    # ======================================================================
    # PHASE RANDOMIZATION
    # ======================================================================

    def randomize_fourier_phases(
        self,
        u,
        seed=None,
    ):
        """
        Randomize phases while subsequently reprojecting.

        This is a stress test, not an invariant transformation.
        """
        rng = np.random.default_rng(
            seed
        )

        u_hat = np.fft.fftn(
            u,
            axes=(1, 2, 3),
        )

        phase = rng.uniform(
            0.0,
            2.0 * np.pi,
            size=u_hat.shape[1:],
        )

        randomized = (
            np.abs(u_hat)
            * np.exp(
                1j * phase[None, ...]
            )
        )

        ur = np.real(
            np.fft.ifftn(
                randomized,
                axes=(1, 2, 3),
            )
        )

        return self.fl.leray_project(
            ur
        )

    # ======================================================================
    # RESOLUTION
    # ======================================================================

    def resolution_measurement(self, u):
        split = (
            self.pressure_solenoidal_split(
                u
            )
        )

        edge = (
            self.weighted_edge_quotient(
                u
            )
        )

        central = (
            self.weighted_edge_quotient_central(
                u
            )
        )

        return ResolutionResult(
            N=self.fl.N,
            c_p=split["C_P"],
            c_s=split["C_S"],
            r_h=split["R_h"],
            c_edge=edge["quotient"],
            c_edge_central=central[
                "quotient"
            ],
            l3=self.fl.norm_l3(u),
            d3=self.critical_dissipation(
                u
            ),
        )


# ============================================================================
# ADVERSARIAL FAMILY GENERATORS
# ============================================================================

class AdversarialFamilies:

    @staticmethod
    def _normalize(
        fluid,
        u,
        norm="L2",
    ):
        u = fluid.leray_project(u)

        if norm.upper() == "L3":
            n = fluid.norm_l3(u)
        else:
            n = fluid.norm_l2(u)

        if n > EPS:
            u = u / n

        return u

    @staticmethod
    def single_spike(
        fluid,
        norm="L2",
    ):
        u = np.zeros(
            (
                3,
                fluid.N,
                fluid.N,
                fluid.N,
            ),
            dtype=float,
        )

        c = fluid.N // 2

        u[
            0,
            c,
            c,
            c
        ] = 1.0

        return AdversarialFamilies._normalize(
            fluid,
            u,
            norm,
        )

    @staticmethod
    def dipole(
        fluid,
        norm="L2",
    ):
        u = np.zeros(
            (
                3,
                fluid.N,
                fluid.N,
                fluid.N,
            ),
            dtype=float,
        )

        c = fluid.N // 2

        u[
            0,
            c,
            c,
            c
        ] = 1.0

        u[
            0,
            (c + 2) % fluid.N,
            c,
            c
        ] = -1.0

        return AdversarialFamilies._normalize(
            fluid,
            u,
            norm,
        )

    @staticmethod
    def near_nyquist_checkerboard(
        fluid,
        norm="L2",
    ):
        i = np.arange(
            fluid.N
        )[:, None, None]

        j = np.arange(
            fluid.N
        )[None, :, None]

        k = np.arange(
            fluid.N
        )[None, None, :]

        checker = (
            (-1.0)
            ** (
                i + j + k
            )
        )

        u = np.zeros(
            (
                3,
                fluid.N,
                fluid.N,
                fluid.N,
            ),
            dtype=float,
        )

        u[0] = checker

        return AdversarialFamilies._normalize(
            fluid,
            u,
            norm,
        )

    @staticmethod
    def mid_dyadic_shell(
        fluid,
        norm="L2",
    ):
        rng = np.random.default_rng(
            721
        )

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

        low = max(
            2.0,
            fluid.N / 8.0,
        )

        high = max(
            low + 1.0,
            fluid.N / 4.0,
        )

        mask = (
            (fluid.Kmag >= low)
            & (fluid.Kmag < high)
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

        u = np.real(
            np.fft.ifftn(
                filtered,
                axes=(1, 2, 3),
            )
        )

        return AdversarialFamilies._normalize(
            fluid,
            u,
            norm,
        )

    @staticmethod
    def nested_multiscale(
        fluid,
        norm="L2",
    ):
        u = np.zeros(
            (
                3,
                fluid.N,
                fluid.N,
                fluid.N,
            ),
            dtype=float,
        )

        x = fluid.X
        y = fluid.Y
        z = fluid.Z

        # Three nested scales.

        for m, amp in (
            (1, 1.0),
            (2, 0.65),
            (4, 0.40),
        ):
            u[0] += (
                amp
                * np.sin(
                    m * x
                )
                * np.cos(
                    m * y
                )
            )

            u[1] += (
                -amp
                * np.cos(
                    m * x
                )
                * np.sin(
                    m * y
                )
            )

        # Add a z-dependent divergence-free pair.

        u[0] += (
            0.35
            * np.sin(2.0 * z)
        )

        u[2] += (
            0.35
            * np.cos(2.0 * z)
        )

        return AdversarialFamilies._normalize(
            fluid,
            u,
            norm,
        )

    @staticmethod
    def abc_beltrami(
        fluid,
        norm="L2",
    ):
        A = 1.0
        B = 1.0
        C = 1.0

        x = fluid.X
        y = fluid.Y
        z = fluid.Z

        u = np.zeros(
            (
                3,
                fluid.N,
                fluid.N,
                fluid.N,
            ),
            dtype=float,
        )

        u[0] = (
            A * np.sin(z)
            + C * np.cos(y)
        )

        u[1] = (
            B * np.sin(x)
            + A * np.cos(z)
        )

        u[2] = (
            C * np.sin(y)
            + B * np.cos(x)
        )

        return AdversarialFamilies._normalize(
            fluid,
            u,
            norm,
        )

    @staticmethod
    def shear_layer(
        fluid,
        norm="L2",
    ):
        x = fluid.X
        y = fluid.Y

        u = np.zeros(
            (
                3,
                fluid.N,
                fluid.N,
                fluid.N,
            ),
            dtype=float,
        )

        u[0] = (
            np.sin(4.0 * y)
            + 0.35 * np.sin(8.0 * y)
        )

        # A second component independent of its own coordinate
        # preserves divergence-free structure.

        u[2] = (
            0.35 * np.cos(3.0 * x)
        )

        return AdversarialFamilies._normalize(
            fluid,
            u,
            norm,
        )

    @staticmethod
    def all(
        fluid,
        norm="L2",
    ):
        return {
            "single_spike":
                AdversarialFamilies.single_spike(
                    fluid,
                    norm,
                ),

            "opposite_sign_dipole":
                AdversarialFamilies.dipole(
                    fluid,
                    norm,
                ),

            "near_nyquist_checkerboard":
                AdversarialFamilies.near_nyquist_checkerboard(
                    fluid,
                    norm,
                ),

            "mid_dyadic_shell":
                AdversarialFamilies.mid_dyadic_shell(
                    fluid,
                    norm,
                ),

            "nested_multiscale":
                AdversarialFamilies.nested_multiscale(
                    fluid,
                    norm,
                ),

            "ABC_Beltrami":
                AdversarialFamilies.abc_beltrami(
                    fluid,
                    norm,
                ),

            "shear_layer":
                AdversarialFamilies.shear_layer(
                    fluid,
                    norm,
                ),
        }


# ============================================================================
# L-BFGS ADVERSARIAL OPTIMIZER
# ============================================================================

class LBFGSAdversarialOptimizer:
    """
    Numerical adversarial search.

    IMPORTANT
    ---------

    This class performs finite-dimensional numerical optimization only.

    It does NOT prove a supremum.

    The optimization parameterization is the divergence-free field itself,
    represented in physical-space coordinates and projected after each
    objective evaluation.

    A true analytic gradient of the complete quotient is not claimed here.
    SciPy's L-BFGS-B can therefore be used in its numerical-gradient mode.

    This is intentionally marked as a DISCOVERY TOOL.

    For practical runs, low-dimensional seeded families are recommended.
    """

    def __init__(
        self,
        audit: DiscreteProofAudit,
        maxiter: int = 20,
        ftol: float = 1e-9,
        gtol: float = 1e-6,
        maxfun: int = 1000,
    ):
        self.audit = audit

        self.maxiter = int(maxiter)
        self.ftol = float(ftol)
        self.gtol = float(gtol)
        self.maxfun = int(maxfun)

        try:
            from scipy.optimize import minimize
        except ImportError as exc:
            raise ImportError(
                "SciPy is required for L-BFGS optimization. "
                "Install with: pip install scipy"
            ) from exc

        self.minimize = minimize

    def _project_and_normalize(
        self,
        vector,
    ):
        shape = (
            3,
            self.audit.fl.N,
            self.audit.fl.N,
            self.audit.fl.N,
        )

        u = np.asarray(
            vector,
            dtype=float,
        ).reshape(shape)

        u = self.audit.fl.leray_project(
            u
        )

        n = self.audit.fl.norm_l2(
            u
        )

        if n <= EPS:
            return np.zeros_like(u)

        return u / n

    def objective(
        self,
        vector,
        target,
    ):
        u = self._project_and_normalize(
            vector
        )

        if target == "C_P":
            value = (
                self.audit.pressure_solenoidal_split(
                    u
                )["C_P"]
            )

        elif target == "C_S":
            value = (
                self.audit.pressure_solenoidal_split(
                    u
                )["C_S"]
            )

        elif target == "R_h":
            value = (
                self.audit.pressure_solenoidal_split(
                    u
                )["R_h"]
            )

        elif target == "C_edge":
            value = (
                self.audit.weighted_edge_quotient(
                    u
                )["quotient"]
            )

        else:
            raise ValueError(
                f"Unknown target: {target}"
            )

        return -float(value)

    def optimize(
        self,
        initial_field,
        target="C_S",
    ):
        x0 = np.asarray(
            initial_field,
            dtype=float,
        ).ravel()

        result = self.minimize(
            lambda x:
                self.objective(
                    x,
                    target,
                ),

            x0,

            method="L-BFGS-B",

            options={
                "maxiter": self.maxiter,
                "ftol": self.ftol,
                "gtol": self.gtol,
                "maxfun": self.maxfun,
                "maxls": 20,
            },
        )

        field = (
            self._project_and_normalize(
                result.x
            )
        )

        value = -float(
            result.fun
        )

        return SearchResult(
            target=target,
            value=value,
            success=bool(
                result.success
            ),
            nit=int(
                getattr(
                    result,
                    "nit",
                    0,
                )
            ),
            nfev=int(
                getattr(
                    result,
                    "nfev",
                    0,
                )
            ),
            message=str(
                result.message
            ),
            field=field,
        )


# ============================================================================
# C_S TARGETED DISCOVERY ENGINE
# ============================================================================

class CSTargetedDiscovery:
    """
    Targeted numerical engine for the solenoidal pairing

        C_S = |<B, P(|u|u)>| / (||u||_3 D3).

    Focuses exclusively on the remaining Level-2 analytical bottleneck.
    """

    def __init__(self, audit: DiscreteProofAudit):
        self.audit = audit
        self.fl = audit.fl

    def project_normalize(self, u):
        u = self.fl.leray_project(u)
        n = self.fl.norm_l3(u)
        if n <= EPS:
            raise ValueError("Zero field encountered during normalization.")
        return u / n

    def cs_components(self, u):
        u = self.project_normalize(u)
        mag = self.audit.velocity_magnitude(u)
        B = self.fl.skew_convection(u, u)
        q = mag * u
        Pq = self.fl.leray_project(q)

        numerator = abs(self.fl.inner_product(B, Pq))
        l3 = self.fl.norm_l3(u)
        d3 = self.audit.critical_dissipation(u)
        denom = max(l3 * d3, EPS)

        cs = numerator / denom
        rh_num = abs(self.fl.inner_product(B, q))
        rh = rh_num / denom

        PB = self.fl.leray_project(B)
        norm_PB = self.fl.norm_l2(PB)
        norm_Pq = self.fl.norm_l2(Pq)
        cos_theta = (numerator / max(norm_PB * norm_Pq, EPS)) if (norm_PB * norm_Pq > EPS) else 0.0

        return CSResult(
            cs=float(cs),
            numerator=float(numerator),
            l3=float(l3),
            d3=float(d3),
            rh=float(rh),
            cos_theta=float(cos_theta),
        )

    def gauge_invariance(self, u):
        """
        Test exact identity: <B, P(|u|u)> = <B, P((|u|-mean|u|)u)>.
        """
        u = self.project_normalize(u)
        mag = self.audit.velocity_magnitude(u)
        q = mag * u
        mean_mag = float(np.mean(mag))
        q_fluc = (mag - mean_mag) * u

        B = self.fl.skew_convection(u, u)
        Pq = self.fl.leray_project(q)
        Pq_fluc = self.fl.leray_project(q_fluc)

        a = self.fl.inner_product(B, Pq)
        b = self.fl.inner_product(B, Pq_fluc)
        defect = abs(a - b) / max(1.0, abs(a), abs(b))

        return {
            "original": float(a),
            "fluctuation": float(b),
            "mean_magnitude": mean_mag,
            "defect": float(defect),
        }

    def cs_fluctuation_form(self, u):
        u = self.project_normalize(u)
        mag = self.audit.velocity_magnitude(u)
        mean_mag = float(np.mean(mag))
        q_fluc = (mag - mean_mag) * u

        B = self.fl.skew_convection(u, u)
        Pq = self.fl.leray_project(q_fluc)
        numerator = abs(self.fl.inner_product(B, Pq))
        l3 = self.fl.norm_l3(u)
        d3 = self.audit.critical_dissipation(u)

        return numerator / max(l3 * d3, EPS)

    def random_start(self, rng):
        seed = int(rng.integers(0, 2**31 - 1))
        u = self.fl.random_divergence_free_field(seed=seed, amplitude=1.0)
        return self.project_normalize(u)

    def random_fourier_perturbation(self, u, rng, amplitude=0.05, cutoff=None):
        N = self.fl.N
        if cutoff is None:
            cutoff = max(2, N // 3)

        noise = rng.standard_normal(size=u.shape)
        noise_hat = np.fft.fftn(noise, axes=(1, 2, 3))

        mask = (
            (np.abs(self.fl.Kx) <= cutoff)
            & (np.abs(self.fl.Ky) <= cutoff)
            & (np.abs(self.fl.Kz) <= cutoff)
        )
        noise_hat *= mask[None, ...]

        noise = np.real(np.fft.ifftn(noise_hat, axes=(1, 2, 3)))
        noise = self.fl.leray_project(noise)
        n = self.fl.norm_l3(noise)
        if n <= EPS:
            return u.copy()
        noise /= n

        return self.project_normalize(u + amplitude * noise)

    def local_search(self, u, rng, iterations=100, initial_step=0.10, min_step=1e-5, cutoff=None):
        u = self.project_normalize(u)
        current = self.cs_components(u)
        best_u = u.copy()
        best_cs = current.cs
        history = []
        step = initial_step

        for it in range(iterations):
            candidate = self.random_fourier_perturbation(u, rng, amplitude=step, cutoff=cutoff)
            value = self.cs_components(candidate)
            accepted = value.cs > current.cs

            if accepted:
                u = candidate
                current = value
                if value.cs > best_cs:
                    best_cs = value.cs
                    best_u = candidate.copy()
            else:
                step *= 0.97

            step = max(step, min_step)
            history.append({
                "iteration": it,
                "C_S": float(current.cs),
                "accepted": bool(accepted),
                "step": float(step),
                "R_h": float(current.rh),
            })

        return best_u, best_cs, history

    def search(self, starts=16, iterations=100, seed=20260908, initial_step=0.10, cutoff=None, initial_field=None):
        rng = np.random.default_rng(seed)
        best_cs = -np.inf
        best_field = None
        history = []

        for s in range(starts):
            if s == 0 and initial_field is not None:
                u = self.project_normalize(initial_field)
            else:
                u = self.random_start(rng)

            initial = self.cs_components(u)
            candidate, value, local_history = self.local_search(
                u, rng, iterations=iterations, initial_step=initial_step, cutoff=cutoff
            )
            history.append({
                "start": s,
                "initial_C_S": initial.cs,
                "final_C_S": value,
                "local_history": local_history,
            })

            if value > best_cs:
                best_cs = value
                best_field = candidate.copy()

        return CSSearchResult(
            best_cs=float(best_cs),
            best_field=best_field,
            history=history,
            starts=starts,
            iterations=iterations,
        )

    def shell_cs_decomposition(self, u):
        u = self.project_normalize(u)
        masks = self.audit.dyadic_shell_masks()
        B = self.fl.skew_convection(u, u)
        mag = self.audit.velocity_magnitude(u)
        q = mag * u
        Pq = self.fl.leray_project(q)

        B_hat = np.fft.fftn(B, axes=(1, 2, 3))
        Pq_hat = np.fft.fftn(Pq, axes=(1, 2, 3))
        parseval = self.fl.h ** 3 / self.fl.N ** 3

        shells = []
        values = []

        for j, mask in masks.items():
            if not np.any(mask):
                continue

            inner = np.sum(B_hat[:, mask] * np.conj(Pq_hat[:, mask]))
            value = float(np.real(inner) * parseval)
            shells.append(j)
            values.append(value)

        values = np.asarray(values, dtype=float)

        return CSShellResult(
            shells=shells,
            values=values,
            absolute_values=np.abs(values),
            total=float(np.sum(values)),
            row_sums=np.abs(values),
        )

    def remove_null_modes(self, u):
        u_hat = np.fft.fftn(u, axes=(1, 2, 3))
        u_hat[:, self.fl.null_modes] = 0.0
        filtered = np.real(np.fft.ifftn(u_hat, axes=(1, 2, 3)))
        return self.project_normalize(filtered)

    def null_mode_audit(self, u):
        u = self.project_normalize(u)
        before = self.cs_components(u)
        filtered = self.remove_null_modes(u)
        after = self.cs_components(filtered)
        abs_change = abs(before.cs - after.cs)

        return {
            "C_S_raw": before.cs,
            "C_S_filtered": after.cs,
            "absolute_change": float(abs_change),
            "relative_change": float(abs_change / max(abs(before.cs), EPS)),
        }

    def dealiased_cs_audit(self, u):
        u = self.project_normalize(u)
        raw = self.cs_components(u)
        filtered = self.audit.dealias_field(u)
        filtered = self.project_normalize(filtered)
        dealiased = self.cs_components(filtered)
        abs_change = abs(raw.cs - dealiased.cs)

        return {
            "C_S_raw": raw.cs,
            "C_S_dealiased": dealiased.cs,
            "absolute_change": float(abs_change),
            "relative_change": float(abs_change / max(abs(raw.cs), EPS)),
        }

    def record(self, u):
        result = self.cs_components(u)
        shell = self.shell_cs_decomposition(u)
        gauge = self.gauge_invariance(u)
        nulls = self.null_mode_audit(u)
        alias = self.dealiased_cs_audit(u)

        return {
            "N": int(self.fl.N),
            "C_S": result.cs,
            "R_h": result.rh,
            "numerator": result.numerator,
            "L3": result.l3,
            "D3": result.d3,
            "gauge_defect": gauge["defect"],
            "null_mode_C_S": nulls["C_S_filtered"],
            "null_mode_relative_change": nulls["relative_change"],
            "dealiased_C_S": alias["C_S_dealiased"],
            "dealiased_relative_change": alias["relative_change"],
            "shells": shell.shells,
            "shell_values": shell.values.tolist(),
            "shell_absolute_values": shell.absolute_values.tolist(),
        }


def explain_cs_shell_data(shell_result: CSShellResult):
    values = np.asarray(shell_result.values)
    if len(values) == 0:
        return {"dominant_shell": None, "dominance_fraction": 0.0, "effective_shell_count": 0.0}

    absolute = np.abs(values)
    total = np.sum(absolute)
    if total <= EPS:
        return {"dominant_shell": shell_result.shells[0], "dominance_fraction": 0.0, "effective_shell_count": 0.0}

    p = absolute / total
    dominant_index = int(np.argmax(absolute))
    entropy = -np.sum(p[p > 0] * np.log(p[p > 0]))
    effective_shell_count = float(np.exp(entropy))

    return {
        "dominant_shell": shell_result.shells[dominant_index],
        "dominance_fraction": float(p[dominant_index]),
        "effective_shell_count": effective_shell_count,
    }


def print_cs_discovery_report(audit: DiscreteProofAudit, u, label="field"):
    engine = CSTargetedDiscovery(audit)
    result = engine.cs_components(u)
    gauge = engine.gauge_invariance(u)
    shell = engine.shell_cs_decomposition(u)
    shell_info = explain_cs_shell_data(shell)
    nulls = engine.null_mode_audit(u)
    alias = engine.dealiased_cs_audit(u)

    print()
    print("=" * 78)
    print(f"C_S TARGETED DISCOVERY REPORT — {label}")
    print("=" * 78)
    print(f"    C_S                 = {result.cs:.12e}")
    print(f"    alignment cos(theta)= {result.cos_theta:.6f} ({result.cos_theta:.2%})")
    print(f"    R_h                 = {result.rh:.12e} (analytic C_R <= {ANALYTIC_C_R_BOUND:.3f})")
    print(f"    numerator           = {result.numerator:.12e}")
    print(f"    ||u||_3             = {result.l3:.12e}")
    print(f"    D3                  = {result.d3:.12e}")
    print(f"    gauge defect        = {gauge['defect']:.3e}")
    print()
    print("NULL-MODE ROBUSTNESS")
    print(f"    raw C_S             = {nulls['C_S_raw']:.12e}")
    print(f"    filtered C_S        = {nulls['C_S_filtered']:.12e}")
    print(f"    relative change     = {nulls['relative_change']:.3e}")
    print()
    print("DEALIASING ROBUSTNESS")
    print(f"    raw C_S             = {alias['C_S_raw']:.12e}")
    print(f"    dealiased C_S       = {alias['C_S_dealiased']:.12e}")
    print(f"    relative change     = {alias['relative_change']:.3e}")
    print()
    print("OUTPUT-SHELL DECOMPOSITION")
    for j, value in zip(shell.shells, shell.values):
        print(f"    shell {j:2d}: {value:+.12e}")
    print(f"    shell sum           = {shell.total:+.12e}")
    print(f"    dominant shell      = {shell_info['dominant_shell']} (fraction: {shell_info['dominance_fraction']:.1%})")
    print(f"    effective shells    = {shell_info['effective_shell_count']:.2f}")
    print("=" * 78)


def run_cs_campaign(audit: DiscreteProofAudit, families: Dict[str, np.ndarray], starts=4, iterations=50, seed=20260908):
    engine = CSTargetedDiscovery(audit)
    results = {}

    print()
    print("=" * 78)
    print("TARGETED C_S DISCOVERY CAMPAIGN (SMOOTH FOURIER ASCENT)")
    print("=" * 78)

    for name, u in families.items():
        base = engine.cs_components(u)
        search = engine.search(
            starts=starts,
            iterations=iterations,
            seed=seed,
            initial_field=u,
        )
        record = engine.record(search.best_field)
        results[name] = {
            "initial_C_S": base.cs,
            "best_C_S": search.best_cs,
            "search": search,
            "record": record,
        }
        print(f"  {name:28s} initial C_S={base.cs:.6e} -> best C_S={search.best_cs:.6e}")

    print()
    print("C_S CAMPAIGN SUMMARY")
    print("-" * 78)
    global_best = -np.inf
    global_name = None

    for name, res in results.items():
        val = res["best_C_S"]
        print(f"  {name:28s} C_S = {val:.10e}")
        if val > global_best:
            global_best = val
            global_name = name

    print()
    print(f"GLOBAL DISCOVERED C_S = {global_best:.10e} (from {global_name})")
    print("WARNING: Discovered numerical values do NOT prove an analytical supremum.")
    print("=" * 78)

    return results


# ============================================================================
# FIELD REPORTING
# ============================================================================

def print_field_report(
    name,
    audit,
    u,
):
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

    central = (
        audit.weighted_edge_quotient_central(
            u
        )
    )

    concentration = (
        audit.scale_concentration(
            u
        )
    )

    null_fraction = (
        audit.null_mode_fraction(
            u
        )
    )

    print()
    print("-" * 78)
    print(f"FIELD: {name}")
    print("-" * 78)

    print(
        f"    ||u||_3            = "
        f"{audit.fl.norm_l3(u):.8e}"
    )

    print(
        f"    D3                 = "
        f"{audit.critical_dissipation(u):.8e}"
    )

    print(
        f"    C_P                = "
        f"{split['C_P']:.8e}"
    )

    print(
        f"    C_S                = "
        f"{split['C_S']:.8e}"
        f"(alignment cos theta = "
        f"{split['cos_theta']:.4f}, {split['cos_theta']:.2%})"
    )

    print(
        f"    R_h                = "
        f"{split['R_h']:.8e} "
        f"(analytic C_R <= "
        f"{ANALYTIC_C_R_BOUND:.3f})"
    )

    print(
        f"    C_edge             = "
        f"{edge['quotient']:.8e} "
        f"(analytic <= "
        f"{ANALYTIC_C_EDGE_BOUND:.3f})"
    )

    print(
        f"    C_edge central     = "
        f"{central['quotient']:.8e}"
    )

    print(
        f"    pressure split     = "
        f"{split['C_P']:.8e} + "
        f"{split['C_S']:.8e}"
    )

    print(
        f"    R_h decomposition   = "
        f"|a+b| normalized = "
        f"{split['R_h']:.8e}"
    )

    print(
        f"    null-mode fraction = "
        f"{null_fraction:.8e}"
    )

    print(
        f"    high-k E >= N/4    = "
        f"{concentration['energy_high_N_over_4']:.8e}"
    )

    print(
        f"    high-k E >= N/3    = "
        f"{concentration['energy_high_N_over_3']:.8e}"
    )

    print(
        f"    top 10% D3         = "
        f"{concentration['top10_percent_D3_fraction']:.8e}"
    )


# ============================================================================
# OPERATOR TESTS
# ============================================================================

def run_operator_tests():
    print("=" * 78)
    print("STRUCTURAL OPERATOR TESTS")
    print("=" * 78)

    N = 16

    fluid = (
        StructurePreservingLatticeFluid3D(
            N=N,
            L=2.0 * np.pi,
            nu=0.01,
        )
    )

    audit = DiscreteProofAudit(
        fluid
    )

    rng = np.random.default_rng(
        20260908
    )

    # ----------------------------------------------------------------------
    # Test 1: skew adjointness
    # ----------------------------------------------------------------------

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

    defect_1 = abs(
        lhs - rhs
    ) / max(
        1.0,
        abs(lhs),
        abs(rhs),
    )

    print(
        f"Test 1: D0 skew-adjoint defect = "
        f"{defect_1:.3e}"
    )

    # ----------------------------------------------------------------------
    # Test 2: divergence-free projection
    # ----------------------------------------------------------------------

    u = (
        fluid.random_divergence_free_field(
            seed=123
        )
    )

    defect_2 = fluid.divergence_linf(
        u
    )

    print(
        f"Test 2: ||div_h u||_inf = "
        f"{defect_2:.3e}"
    )

    # ----------------------------------------------------------------------
    # Test 3: P^2=P
    # ----------------------------------------------------------------------

    ur = rng.standard_normal(
        (
            3,
            N,
            N,
            N,
        )
    )

    Pu = fluid.leray_project(
        ur
    )

    P2u = fluid.leray_project(
        Pu
    )

    defect_3 = (
        fluid.norm_l2(
            P2u - Pu
        )
        / max(
            1.0,
            fluid.norm_l2(Pu),
        )
    )

    print(
        f"Test 3: projection idempotence = "
        f"{defect_3:.3e}"
    )

    # ----------------------------------------------------------------------
    # Test 4: energy cancellation
    # ----------------------------------------------------------------------

    B = fluid.skew_convection(
        u,
        u,
    )

    value = fluid.inner_product(
        B,
        u,
    )

    defect_4 = abs(
        value
    ) / max(
        1.0,
        fluid.norm_l2(B)
        * fluid.norm_l2(u),
    )

    print(
        f"Test 4: <B(u,u),u> normalized = "
        f"{defect_4:.3e}"
    )

    # ----------------------------------------------------------------------
    # Test 5: RHS energy identity
    # ----------------------------------------------------------------------

    rhs_u = fluid.rhs(
        u
    )

    numerical = fluid.inner_product(
        rhs_u,
        u,
    )

    theoretical = (
        -fluid.nu
        * fluid.enstrophy(u)
    )

    defect_5 = abs(
        numerical
        - theoretical
    ) / max(
        1.0,
        abs(theoretical),
    )

    print(
        f"Test 5: energy identity defect = "
        f"{defect_5:.3e}"
    )

    # ----------------------------------------------------------------------
    # Test 6: Taylor-Green
    # ----------------------------------------------------------------------

    tg = fluid.taylor_green()

    defect_6 = fluid.divergence_linf(
        tg
    )

    print(
        f"Test 6: Taylor-Green divergence = "
        f"{defect_6:.3e}"
    )

    # ----------------------------------------------------------------------
    # Test 7: D0^2 = compatible Laplacian
    # ----------------------------------------------------------------------

    f2 = rng.standard_normal(
        (
            N,
            N,
            N,
        )
    )

    d0_lap = np.zeros_like(
        f2
    )

    for axis in range(3):
        d0_lap += fluid.D0(
            fluid.D0(
                f2,
                axis,
            ),
            axis,
        )

    spectral_lap = fluid.laplacian(
        f2
    )

    defect_7 = (
        np.linalg.norm(
            d0_lap
            - spectral_lap
        )
        / max(
            1.0,
            np.linalg.norm(
                spectral_lap
            ),
        )
    )

    print(
        f"Test 7: Laplacian compatibility = "
        f"{defect_7:.3e}"
    )

    # ----------------------------------------------------------------------
    # Test 8: pressure split
    # ----------------------------------------------------------------------

    split = audit.pressure_split_audit(
        u
    )

    defect_8 = split[
        "relative_residual"
    ]

    print(
        f"Test 8: Hodge pressure split defect = "
        f"{defect_8:.3e}"
    )

    # ----------------------------------------------------------------------
    # Test 9: Hodge orthogonality
    # ----------------------------------------------------------------------

    mag = audit.velocity_magnitude(
        u
    )

    q = mag * u

    Pq = fluid.leray_project(
        q
    )

    q_perp = q - Pq

    orthogonal = fluid.inner_product(
        Pq,
        q_perp,
    )

    defect_9 = abs(
        orthogonal
    ) / max(
        1.0,
        fluid.norm_l2(Pq)
        * fluid.norm_l2(q_perp),
    )

    print(
        f"Test 9: Hodge orthogonality defect = "
        f"{defect_9:.3e}"
    )

    # ----------------------------------------------------------------------
    # Test 10: C_S fluctuation gauge invariance
    # ----------------------------------------------------------------------

    mean_mag = float(
        np.mean(mag)
    )

    q_fluc = (
        (mag - mean_mag)
        * u
    )

    B = fluid.skew_convection(
        u,
        u,
    )

    Pq = fluid.leray_project(
        q
    )

    Pq_fluc = fluid.leray_project(
        q_fluc
    )

    val_orig = fluid.inner_product(
        B,
        Pq,
    )

    val_fluc = fluid.inner_product(
        B,
        Pq_fluc,
    )

    defect_10 = abs(
        val_orig
        - val_fluc
    ) / max(
        1.0,
        abs(val_orig),
    )

    print(
        f"Test 10: C_S fluctuation gauge defect = "
        f"{defect_10:.3e}"
    )

    # ----------------------------------------------------------------------
    # Test 11: certified edge ceiling
    # ----------------------------------------------------------------------

    edge = (
        audit.weighted_edge_quotient(
            u
        )
    )

    defect_11 = max(
        0.0,
        edge["quotient"]
        - ANALYTIC_C_EDGE_BOUND,
    )

    print(
        f"Test 11: C_edge <= 2 certificate check = "
        f"{defect_11:.3e}"
    )

    # ----------------------------------------------------------------------
    # Tolerances
    # ----------------------------------------------------------------------

    tolerances = {
        "skew": 1e-12,
        "div": 1e-12,
        "projection": 1e-12,
        "energy": 1e-12,
        "tg": 1e-12,
        "laplacian": 1e-12,
        "pressure": 1e-10,
        "orthogonality": 1e-12,
        "gauge": 1e-12,
        "edge": 1e-12,
    }

    passed = (
        defect_1 < tolerances["skew"]
        and defect_2 < tolerances["div"]
        and defect_3 < tolerances["projection"]
        and defect_4 < tolerances["energy"]
        and defect_5 < tolerances["energy"]
        and defect_6 < tolerances["tg"]
        and defect_7 < tolerances["laplacian"]
        and defect_8 < tolerances["pressure"]
        and defect_9 < tolerances["orthogonality"]
        and defect_10 < tolerances["gauge"]
        and defect_11 < tolerances["edge"]
    )

    print()
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
            "One or more structural tests failed."
        )

    return fluid, audit


# ============================================================================
# SEVEN ADVERSARIAL FAMILIES
# ============================================================================

def run_adversarial_families(
    resolutions=(16, 32),
):
    print()
    print("=" * 78)
    print("SEVEN ADVERSARIAL GEOMETRIC FAMILIES")
    print("=" * 78)

    all_results = {}

    for N in resolutions:

        print()
        print(
            f"RESOLUTION N={N}"
        )

        fluid = (
            StructurePreservingLatticeFluid3D(
                N=N
            )
        )

        audit = DiscreteProofAudit(
            fluid
        )

        families = (
            AdversarialFamilies.all(
                fluid
            )
        )

        all_results[N] = {}

        for name, u in families.items():

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

            all_results[N][name] = {
                "C_P": split["C_P"],
                "C_S": split["C_S"],
                "R_h": split["R_h"],
                "C_edge": edge["quotient"],
            }

            print(
                f"{name:32s} "
                f"C_P={split['C_P']:.6e} "
                f"C_S={split['C_S']:.6e} "
                f"R_h={split['R_h']:.6e} "
                f"C_edge={edge['quotient']:.6e}"
            )

            if (
                edge["quotient"]
                > ANALYTIC_C_EDGE_BOUND
                + 1e-10
            ):
                print(
                    "    WARNING: empirical C_edge exceeds "
                    "the declared analytic ceiling."
                )

    return all_results


# ============================================================================
# RANDOMIZED SEARCH
# ============================================================================

def randomized_search(
    audit,
    starts=10,
    steps=20,
    perturbation=0.02,
    seed=101,
    target="C_S",
):
    """
    Random local search.

    This is deliberately not called an optimizer.
    """

    rng = np.random.default_rng(
        seed
    )

    best_value = -np.inf
    best_field = None

    for start in range(starts):

        u = (
            audit.fl.random_divergence_free_field(
                seed=int(
                    rng.integers(
                        0,
                        2**31 - 1,
                    )
                )
            )
        )

        def value(field):
            if target == "C_P":
                return audit.pressure_solenoidal_split(
                    field
                )["C_P"]

            if target == "C_S":
                return audit.pressure_solenoidal_split(
                    field
                )["C_S"]

            if target == "R_h":
                return audit.pressure_solenoidal_split(
                    field
                )["R_h"]

            if target == "C_edge":
                return audit.weighted_edge_quotient(
                    field
                )["quotient"]

            raise ValueError(
                target
            )

        current = value(u)

        for _ in range(steps):

            direction = rng.standard_normal(
                u.shape
            )

            direction = (
                audit.fl.leray_project(
                    direction
                )
            )

            n = audit.fl.norm_l2(
                direction
            )

            if n <= EPS:
                continue

            direction /= n

            candidate = (
                u
                + perturbation
                * direction
            )

            candidate = (
                audit.fl.leray_project(
                    candidate
                )
            )

            cn = audit.fl.norm_l2(
                candidate
            )

            if cn <= EPS:
                continue

            candidate /= cn

            candidate_value = (
                value(candidate)
            )

            if candidate_value > current:
                u = candidate
                current = candidate_value

        if current > best_value:
            best_value = current
            best_field = u.copy()

    return {
        "target": target,
        "best_value": float(
            best_value
        ),
        "field": best_field,
    }


# ============================================================================
# L-BFGS SEARCH DRIVER
# ============================================================================

def run_lbfgs_searches(
    N=16,
    targets=(
        "C_P",
        "C_S",
        "R_h",
        "C_edge",
    ),
    maxiter=5,
    seed=2026,
):
    print()
    print("=" * 78)
    print("L-BFGS ADVERSARIAL DISCOVERY")
    print("=" * 78)

    print(
        "IMPORTANT: these are finite-dimensional discovered values, "
        "not supremum bounds."
    )

    fluid = (
        StructurePreservingLatticeFluid3D(
            N=N
        )
    )

    audit = DiscreteProofAudit(
        fluid
    )

    # Use several structured starts rather than only a Gaussian start.

    starts = list(
        AdversarialFamilies.all(
            fluid
        ).items()
    )

    rng = np.random.default_rng(
        seed
    )

    starts.append(
        (
            "random",
            fluid.random_divergence_free_field(
                seed=int(
                    rng.integers(
                        0,
                        2**31 - 1,
                    )
                )
            ),
        )
    )

    optimizer = (
        LBFGSAdversarialOptimizer(
            audit,
            maxiter=maxiter,
            maxfun=200,
        )
    )

    results = []

    for target in targets:

        best = None

        for name, u0 in starts:

            print(
                f"  target={target:7s} "
                f"start={name:30s}"
            )

            try:
                result = optimizer.optimize(
                    u0,
                    target=target,
                )
            except Exception as exc:
                print(
                    f"    optimization failed: {exc}"
                )
                continue

            print(
                f"    value={result.value:.8e} "
                f"success={result.success} "
                f"nit={result.nit} "
                f"nfev={result.nfev}"
            )

            if (
                best is None
                or result.value > best.value
            ):
                best = result

        if best is not None:

            results.append(
                best
            )

            print()
            print(
                f"BEST DISCOVERED {target}: "
                f"{best.value:.10e}"
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
    print("RESOLUTION AUDIT")
    print("=" * 78)

    results = []

    for N in resolutions:

        fluid = (
            StructurePreservingLatticeFluid3D(
                N=N
            )
        )

        audit = DiscreteProofAudit(
            fluid
        )

        u = (
            fluid.random_divergence_free_field(
                seed=seed
            )
        )

        result = (
            audit.resolution_measurement(
                u
            )
        )

        results.append(
            result
        )

        print(
            f"N={N:4d} "
            f"C_P={result.c_p:.8e} "
            f"C_S={result.c_s:.8e} "
            f"R_h={result.r_h:.8e} "
            f"C_edge={result.c_edge:.8e}"
        )

    return results


# ============================================================================
# FINAL STATUS
# ============================================================================

def print_mathematical_status(
    adversarial_results=None,
    optimizer_results=None,
    cs_campaign_results=None,
):
    print()
    print("=" * 78)
    print("MATHEMATICAL STATUS")
    print("=" * 78)

    print()
    print("LEVEL 1 — EXACT DISCRETE ALGEBRA")
    print("--------------------------------")
    print(
        "  D0 skew-adjointness:                VERIFIED"
    )
    print(
        "  Leray projection structure:         VERIFIED"
    )
    print(
        "  Convective energy cancellation:     VERIFIED"
    )
    print(
        "  Hodge pressure split:               VERIFIED"
    )
    print(
        "  Hodge orthogonality:                VERIFIED"
    )
    print(
        "  C_S fluctuation gauge invariance:   VERIFIED"
    )

    print()
    print("LEVEL 2 — UNIFORM DISCRETE CERTIFICATES")
    print("----------------------------------------")
    print(
        f"  C_edge <= {ANALYTIC_C_EDGE_BOUND:.1f}: "
        f"ANALYTICALLY CERTIFIED"
    )
    print(
        f"  C_R <= {ANALYTIC_C_R_BOUND:.1f}: "
        f"ANALYTICALLY CERTIFIED"
    )

    print()
    print("REMAINING BOTTLENECK")
    print("--------------------")
    print(
        "  C_S = |<B,P(|u|u)>|/(||u||_3 D3)"
    )
    print(
        "  remains analytically OPEN in this code."
    )

    print()
    print("PRESSURE CONSEQUENCE")
    print("--------------------")
    print(
        "  C_P <= C_R + C_S"
    )
    print(
        f"      <= {ANALYTIC_C_R_BOUND:.1f} + C_S"
    )

    if optimizer_results:

        print()
        print(
            "BEST NUMERICALLY DISCOVERED VALUES"
        )
        print(
            "-----------------------------------"
        )

        for result in optimizer_results:

            print(
                f"  {result.target:8s} "
                f"{result.value:.10e}"
            )

        print()
        print(
            "These are discovered finite-dimensional values."
        )
        print(
            "They are NOT certified upper bounds."
        )

    if adversarial_results:

        print()
        print(
            "ADVERSARIAL FAMILY SEARCH"
        )
        print(
            "-------------------------"
        )

        maxima = {
            "C_P": 0.0,
            "C_S": 0.0,
            "R_h": 0.0,
            "C_edge": 0.0,
        }

        for N, families in (
            adversarial_results.items()
        ):
            for data in families.values():
                for key in maxima:
                    maxima[key] = max(
                        maxima[key],
                        data[key],
                    )

        for key, value in maxima.items():

            print(
                f"  discovered max {key:7s} = "
                f"{value:.10e}"
            )

    if cs_campaign_results:
        print()
        print("TARGETED C_S CAMPAIGN DISCOVERY")
        print("-------------------------------")
        for name, res in cs_campaign_results.items():
            print(f"  {name:28s} C_S = {res['best_C_S']:.10e}")

    print()
    print("LEVEL 3 — CONTINUUM")
    print("--------------------")
    print(
        "  N -> infinity, null-mode resolution, "
        "compactness and continuum estimates: OPEN"
    )

    print()
    print(
        "FINAL WARNING:"
    )

    print(
        "  Numerical boundedness, resolution stability, "
        "optimization failure, or absence of a counterexample "
        "does not establish a continuum supremum."
    )

    print()
    print(
        "  The central analytical target is the solenoidal "
        "pairing C_S."
    )

    print("=" * 78)


# ============================================================================
# FIELD REPORT SUITE
# ============================================================================

def run_field_reports(N=16):
    print()
    print("=" * 78)
    print(
        f"ADVERSARIAL FIELD REPORTS — N={N}"
    )
    print("=" * 78)

    fluid = (
        StructurePreservingLatticeFluid3D(
            N=N
        )
    )

    audit = DiscreteProofAudit(
        fluid
    )

    families = (
        AdversarialFamilies.all(
            fluid
        )
    )

    for name, u in families.items():
        print_field_report(
            name,
            audit,
            u,
        )

    return fluid, audit, families


# ============================================================================
# COMPLETE AUDIT
# ============================================================================

def run_complete_audit(
    quick=False,
    full=False,
    optimize=False,
    target_cs=False,
    campaign_starts=4,
    campaign_iter=50,
):
    print("=" * 78)
    print("DISCRETE NAVIER-STOKES PROOF / AUDIT")
    print("=" * 78)

    print()
    print(
        "STATUS: DISCRETE EXPERIMENTAL / CERTIFICATE-DISCOVERY ENGINE"
    )

    print()
    print(
        f"Python: {sys.version.split()[0]}"
    )

    print(
        f"NumPy : {np.__version__}"
    )

    # ----------------------------------------------------------------------
    # Structural tests
    # ----------------------------------------------------------------------

    fluid, audit = (
        run_operator_tests()
    )

    # ----------------------------------------------------------------------
    # Field reports
    # ----------------------------------------------------------------------

    _, _, families = run_field_reports(
        N=16
    )

    # ----------------------------------------------------------------------
    # Seven families
    # ----------------------------------------------------------------------

    if quick:

        adversarial_results = (
            run_adversarial_families(
                resolutions=(16,)
            )
        )

    else:

        adversarial_results = (
            run_adversarial_families(
                resolutions=(
                    16,
                    32,
                )
            )
        )

    # ----------------------------------------------------------------------
    # Resolution
    # ----------------------------------------------------------------------

    if full:

        run_resolution_audit(
            resolutions=(
                16,
                32,
                64,
            )
        )

    else:

        run_resolution_audit(
            resolutions=(
                16,
                32,
            )
        )

    # ----------------------------------------------------------------------
    # Randomized discovery
    # ----------------------------------------------------------------------

    if not quick:

        print()
        print("=" * 78)
        print(
            "RANDOMIZED DISCOVERY"
        )
        print("=" * 78)

        for target in (
            "C_P",
            "C_S",
            "R_h",
            "C_edge",
        ):

            result = randomized_search(
                audit,
                starts=3,
                steps=5,
                perturbation=0.01,
                seed=100 + len(target),
                target=target,
            )

            print(
                f"{target:8s}: "
                f"best discovered = "
                f"{result['best_value']:.10e}"
            )

    # ----------------------------------------------------------------------
    # L-BFGS
    # ----------------------------------------------------------------------

    optimizer_results = None

    if optimize:

        optimizer_results = (
            run_lbfgs_searches(
                N=16,
                maxiter=5,
            )
        )

    # ----------------------------------------------------------------------
    # Final status
    # ----------------------------------------------------------------------

    cs_campaign_results = None
    if target_cs:
        cs_campaign_results = run_cs_campaign(
            audit,
            families,
            starts=campaign_starts,
            iterations=campaign_iter,
        )

    print_mathematical_status(
        adversarial_results=(
            adversarial_results
        ),
        optimizer_results=(
            optimizer_results
        ),
        cs_campaign_results=(
            cs_campaign_results
        ),
    )


# ============================================================================
# COMMAND LINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Discrete Navier-Stokes "
            "Level-1/Level-2 proof audit."
        )
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Run N=16 only and skip randomized "
            "discovery."
        ),
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Include N=64 resolution audit."
        ),
    )

    parser.add_argument(
        "--optimize",
        action="store_true",
        help=(
            "Run SciPy L-BFGS adversarial searches. "
            "This can be expensive."
        ),
    )

    parser.add_argument(
        "--target-cs",
        action="store_true",
        help=(
            "Run targeted Fourier ascent and shell "
            "decomposition on C_S."
        ),
    )

    parser.add_argument(
        "--campaign-starts",
        type=int,
        default=4,
        help="Number of starts for targeted C_S campaign.",
    )

    parser.add_argument(
        "--campaign-iter",
        type=int,
        default=50,
        help="Number of iterations for targeted C_S campaign.",
    )

    args = parser.parse_args()

    run_complete_audit(
        quick=args.quick,
        full=args.full,
        optimize=args.optimize,
        target_cs=args.target_cs,
        campaign_starts=args.campaign_starts,
        campaign_iter=args.campaign_iter,
    )


if __name__ == "__main__":
    main()
