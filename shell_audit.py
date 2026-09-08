"""
shell_audit.py
====================================================================

EXPERIMENTAL SPECTRAL SHELL AUDIT
---------------------------------

This module provides numerical diagnostics for the critical quotient

    C3(u) = |<grad p, grad q>| / (||u||_3 D3(u))

where

    -Delta p = div((u · grad)u)
    -Delta q = div(|u|u)

and

    D3(u) = integral |u| |grad u|^2 dx.

IMPORTANT SCIENTIFIC STATUS
---------------------------

This file is an EXPERIMENTAL numerical diagnostic.

It does NOT prove:

    sup_u C3(u) < infinity
    C3(j) <= C j^(-1-epsilon)
    a uniform Schur estimate
    a continuum triadic decay theorem
    global Navier-Stokes regularity.

In particular:

    <(grad p)_j, (grad q)_k> = 0, j != k

is expected when (grad p)_j and (grad q)_k are formed using
disjoint Fourier output masks. This is simply Fourier orthogonality
of the OUTPUT shells.

It does NOT mean that nonlinear cross-scale interactions vanish.

The nonlinear interaction occurs before the pressure projection:

    u_m x u_n  --->  pressure source shell j.

For that reason, this module also exposes source-level diagnostics
which can be extended to a full triad audit.

The numerical results should therefore be interpreted as evidence
about finite-dimensional discretizations and selected test fields,
not as continuum proofs.

====================================================================
"""

import numpy as np


class ShellAuditFramework:
    """
    Experimental spectral-shell analysis for
    StructurePreservingLatticeFluid3D.

    Expected fluid interface:

        fluid.N
        fluid.h

        fluid.Kx
        fluid.Ky
        fluid.Kz
        fluid.K_tilde_x
        fluid.K_tilde_y
        fluid.K_tilde_z
        fluid.inv_K_sq

        fluid.skew_convection(u, u)
        fluid.inner_product(a, b)
        fluid.norm_l2(u)
        fluid.norm_l3(u)
        fluid.random_divergence_free_field(...)
        fluid.leray_project(u)

    Arrays are assumed to use the convention

        u.shape == (3, N, N, N)

    with Fourier transforms taken over axes (1, 2, 3).
    """

    def __init__(self, fluid):
        self.fl = fluid

        self.Kmag = np.sqrt(
            self.fl.Kx**2
            + self.fl.Ky**2
            + self.fl.Kz**2
        )

    # ================================================================
    # SHELL CONSTRUCTION
    # ================================================================

    def linear_shell_masks(self, jmax=None):
        """
        Linear radial shells:

            shell j = { k : j-1 <= |k| < j }.
        """

        if jmax is None:
            jmax = self.fl.N // 3

        masks = {}

        for j in range(1, jmax + 1):
            masks[j] = (
                (self.Kmag >= float(j - 1))
                & (self.Kmag < float(j))
            )

        return masks

    def dyadic_shell_masks(self, oct_max=None):
        """
        Dyadic Littlewood-Paley-style shells:

            shell 1: 0 <= |k| < 2
            shell j: 2^(j-1) <= |k| < 2^j.
        """

        if oct_max is None:
            oct_max = int(np.log2(max(2, self.fl.N // 2)))

        masks = {}

        for j in range(1, oct_max + 1):
            k_low = 0.0 if j == 1 else 2.0 ** (j - 1)
            k_high = 2.0 ** j

            masks[j] = (
                (self.Kmag >= k_low)
                & (self.Kmag < k_high)
            )

        return masks

    def shell_mode_counts(self, masks):
        """Return the number of Fourier modes in each shell."""

        return {
            j: int(np.count_nonzero(mask))
            for j, mask in masks.items()
        }

    # ================================================================
    # PRESSURE CALCULATION
    # ================================================================

    def compute_pressure_gradients(self, u):
        """
        Compute Fourier representations of grad p and grad q.

        Physical pressure:

            -Delta p = div(B(u,u))

        Auxiliary pressure:

            -Delta q = div(|u|u)

        Returns
        -------
        grad_p_hat, grad_q_hat
            Arrays of shape (3,N,N,N).
        """

        u_mag = np.sqrt(
            np.sum(u**2, axis=0)
        )

        # ------------------------------------------------------------
        # Physical pressure
        # ------------------------------------------------------------

        B = self.fl.skew_convection(u, u)

        B_hat = np.fft.fftn(
            B,
            axes=(1, 2, 3)
        )

        divB_hat = (
            1j * self.fl.K_tilde_x * B_hat[0]
            + 1j * self.fl.K_tilde_y * B_hat[1]
            + 1j * self.fl.K_tilde_z * B_hat[2]
        )

        p_hat = -divB_hat * self.fl.inv_K_sq

        grad_p_hat = np.stack(
            [
                1j * self.fl.K_tilde_x * p_hat,
                1j * self.fl.K_tilde_y * p_hat,
                1j * self.fl.K_tilde_z * p_hat,
            ],
            axis=0,
        )

        # ------------------------------------------------------------
        # Auxiliary pressure
        # ------------------------------------------------------------

        abs_u_u = u_mag * u

        abs_u_u_hat = np.fft.fftn(
            abs_u_u,
            axes=(1, 2, 3)
        )

        div_abs_hat = (
            1j * self.fl.K_tilde_x * abs_u_u_hat[0]
            + 1j * self.fl.K_tilde_y * abs_u_u_hat[1]
            + 1j * self.fl.K_tilde_z * abs_u_u_hat[2]
        )

        q_hat = -div_abs_hat * self.fl.inv_K_sq

        grad_q_hat = np.stack(
            [
                1j * self.fl.K_tilde_x * q_hat,
                1j * self.fl.K_tilde_y * q_hat,
                1j * self.fl.K_tilde_z * q_hat,
            ],
            axis=0,
        )

        return grad_p_hat, grad_q_hat

    # ================================================================
    # FOURIER SHELL PROJECTION
    # ================================================================

    def project_fourier_shell(self, field_hat, mask):
        """Return field_hat restricted to one Fourier shell."""

        result = np.zeros_like(field_hat)
        result[:, mask] = field_hat[:, mask]
        return result

    def inverse_fourier_field(self, field_hat):
        """Inverse FFT of a vector-valued Fourier field."""

        return np.real(
            np.fft.ifftn(
                field_hat,
                axes=(1, 2, 3)
            )
        )

    # ================================================================
    # CRITICAL DISSIPATION
    # ================================================================

    def critical_dissipation_shell(
        self,
        mask,
        u,
        u_mag=None,
    ):
        """
        Compute

            D_{3,j}
              = integral |u| |grad u_j|^2 dx.

        NOTE:

        The weight |u| is the FULL velocity magnitude, not |u_j|.
        Consequently these shell contributions should not be
        automatically interpreted as an orthogonal decomposition
        of the nonlinear functional D3.
        """

        if u_mag is None:
            u_mag = np.sqrt(
                np.sum(u**2, axis=0)
            )

        u_hat = np.fft.fftn(
            u,
            axes=(1, 2, 3)
        )

        uj_hat = np.zeros_like(u_hat)
        uj_hat[:, mask] = u_hat[:, mask]

        grad_sq = np.zeros_like(u_mag)

        for component in range(3):

            gx = np.real(
                np.fft.ifftn(
                    1j * self.fl.K_tilde_x * uj_hat[component],
                    axes=(0, 1, 2),
                )
            )

            gy = np.real(
                np.fft.ifftn(
                    1j * self.fl.K_tilde_y * uj_hat[component],
                    axes=(0, 1, 2),
                )
            )

            gz = np.real(
                np.fft.ifftn(
                    1j * self.fl.K_tilde_z * uj_hat[component],
                    axes=(0, 1, 2),
                )
            )

            grad_sq += gx**2 + gy**2 + gz**2

        return (
            np.sum(u_mag * grad_sq)
            * self.fl.h**3
        )

    # ================================================================
    # FULL SHELL MATRIX
    # ================================================================

    def full_shell_matrix_audit(
        self,
        u,
        shell_type="dyadic",
    ):
        """
        Compute the shell interaction matrix

            M_jk =
              | <(grad p)_j, (grad q)_k> |.

        Also compute

            Gamma_jk =
                M_jk /
                ( ||u||_3 sqrt(D3_j D3_k) ).

        Returns
        -------

        active_shells
        M_matrix
        Gamma_matrix
        row_sums
        """

        if shell_type == "dyadic":
            masks = self.dyadic_shell_masks()
        elif shell_type == "linear":
            masks = self.linear_shell_masks()
        else:
            raise ValueError(
                "shell_type must be 'dyadic' or 'linear'"
            )

        grad_p_hat, grad_q_hat = (
            self.compute_pressure_gradients(u)
        )

        u_mag = np.sqrt(
            np.sum(u**2, axis=0)
        )

        u_3_norm = self.fl.norm_l3(u)

        active_shells = [
            j
            for j, mask in masks.items()
            if np.count_nonzero(mask) > 0
        ]

        n_shells = len(active_shells)

        D3 = {}

        for j in active_shells:
            D3[j] = self.critical_dissipation_shell(
                masks[j],
                u,
                u_mag,
            )

        M = np.zeros(
            (n_shells, n_shells),
            dtype=float,
        )

        Gamma = np.zeros_like(M)

        # ------------------------------------------------------------
        # IMPORTANT:
        #
        # Fourier-shell projections have disjoint supports.
        # Therefore j != k gives zero analytically for the projected
        # output fields.
        #
        # We nevertheless calculate it numerically to monitor leakage.
        # ------------------------------------------------------------

        for row, j in enumerate(active_shells):

            gp_hat_j = self.project_fourier_shell(
                grad_p_hat,
                masks[j],
            )

            gp_j = self.inverse_fourier_field(
                gp_hat_j
            )

            for col, k in enumerate(active_shells):

                gq_hat_k = self.project_fourier_shell(
                    grad_q_hat,
                    masks[k],
                )

                gq_k = self.inverse_fourier_field(
                    gq_hat_k
                )

                inner = abs(
                    self.fl.inner_product(
                        gp_j,
                        gq_k,
                    )
                )

                M[row, col] = inner

                denom = (
                    u_3_norm
                    * np.sqrt(
                        max(D3[j], 0.0)
                        * max(D3[k], 0.0)
                    )
                )

                Gamma[row, col] = (
                    inner / denom
                    if denom > 1e-30
                    else 0.0
                )

        row_sums = np.sum(
            Gamma,
            axis=1,
        )

        return (
            active_shells,
            M,
            Gamma,
            row_sums,
        )

    # ================================================================
    # SHELL SPECTRUM
    # ================================================================

    def shell_spectrum(
        self,
        u,
        shell_type="linear",
    ):
        """
        Compute shell-level pressure work, mode count,
        per-mode work, dissipation and critical quotient.

        Returns a dictionary keyed by shell index.
        """

        if shell_type == "linear":
            masks = self.linear_shell_masks()
        elif shell_type == "dyadic":
            masks = self.dyadic_shell_masks()
        else:
            raise ValueError(
                "shell_type must be 'linear' or 'dyadic'"
            )

        grad_p_hat, grad_q_hat = (
            self.compute_pressure_gradients(u)
        )

        u_mag = np.sqrt(
            np.sum(u**2, axis=0)
        )

        u3 = self.fl.norm_l3(u)

        results = {}

        for j, mask in masks.items():

            multiplicity = int(
                np.count_nonzero(mask)
            )

            if multiplicity == 0:
                continue

            gp_hat = self.project_fourier_shell(
                grad_p_hat,
                mask,
            )

            gq_hat = self.project_fourier_shell(
                grad_q_hat,
                mask,
            )

            gp = self.inverse_fourier_field(
                gp_hat
            )

            gq = self.inverse_fourier_field(
                gq_hat
            )

            a_j = abs(
                self.fl.inner_product(
                    gp,
                    gq,
                )
            )

            D_j = self.critical_dissipation_shell(
                mask,
                u,
                u_mag,
            )

            C_j = (
                a_j / (u3 * D_j)
                if D_j > 1e-30
                else 0.0
            )

            results[j] = {
                "multiplicity": multiplicity,
                "work": a_j,
                "work_per_mode": (
                    a_j / multiplicity
                ),
                "D3": D_j,
                "C3": C_j,
            }

        return results

    # ================================================================
    # POWER-LAW FIT
    # ================================================================

    @staticmethod
    def power_law_fit(
        x,
        y,
        xmin=None,
        xmax=None,
    ):
        """
        Fit

            y ~ C x^alpha

        using linear regression in log-log coordinates.

        Returns

            C, alpha, r_squared

        This is an empirical finite-range fit only.
        """

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        valid = (
            (x > 0)
            & (y > 0)
        )

        if xmin is not None:
            valid &= x >= xmin

        if xmax is not None:
            valid &= x <= xmax

        x = x[valid]
        y = y[valid]

        if len(x) < 2:
            return np.nan, np.nan, np.nan

        lx = np.log(x)
        ly = np.log(y)

        slope, intercept = np.polyfit(
            lx,
            ly,
            1,
        )

        prediction = (
            intercept
            + slope * lx
        )

        ss_res = np.sum(
            (ly - prediction) ** 2
        )

        ss_tot = np.sum(
            (ly - np.mean(ly)) ** 2
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

    # ================================================================
    # SCHUR DIAGNOSTIC
    # ================================================================

    def schur_summary(
        self,
        u,
        shell_type="dyadic",
    ):
        """
        Compute the normalized interaction matrix and
        its Schur row-sum supremum.
        """

        shells, M, Gamma, row_sums = (
            self.full_shell_matrix_audit(
                u,
                shell_type=shell_type,
            )
        )

        return {
            "shells": shells,
            "M": M,
            "Gamma": Gamma,
            "row_sums": row_sums,
            "sup_row_sum": (
                float(np.max(row_sums))
                if len(row_sums)
                else 0.0
            ),
            "max_off_diagonal": (
                float(
                    np.max(
                        Gamma
                        - np.diag(
                            np.diag(Gamma)
                        )
                    )
                )
                if len(shells)
                else 0.0
            ),
        }

    # ================================================================
    # ADVERSARIAL PROJECTED GRADIENT SEARCH
    # ================================================================

    def adversarial_projected_gradient_search(
        self,
        steps=30,
        lr=0.05,
        seed=101,
    ):
        """
        Experimental projected-gradient ascent for

            C3(u)
              = |<grad p,grad q>|
                / (||u||_3 D3(u)).

        IMPORTANT:

        The update direction below is deliberately simple and is
        NOT the exact Fréchet derivative of C3.

        It is therefore an adversarial heuristic, not a variational
        proof of optimality.
        """

        u = self.fl.random_divergence_free_field(
            seed=seed,
            amplitude=1.0,
        )

        best_c3 = 0.0
        history = []

        for step in range(steps):

            u_mag = np.sqrt(
                np.sum(u**2, axis=0)
            ) + 1e-14

            u3 = self.fl.norm_l3(u)

            grad_p_hat, grad_q_hat = (
                self.compute_pressure_gradients(u)
            )

            grad_p = self.inverse_fourier_field(
                grad_p_hat
            )

            grad_q = self.inverse_fourier_field(
                grad_q_hat
            )

            Wp = abs(
                self.fl.inner_product(
                    grad_p,
                    grad_q,
                )
            )

            # --------------------------------------------------------
            # D3
            # --------------------------------------------------------

            u_hat = np.fft.fftn(
                u,
                axes=(1, 2, 3),
            )

            grad_sq = np.zeros_like(u_mag)

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
                    gx**2
                    + gy**2
                    + gz**2
                )

            D3 = (
                np.sum(
                    u_mag * grad_sq
                )
                * self.fl.h**3
            )

            C3 = (
                Wp / (u3 * D3)
                if D3 > 1e-30
                else 0.0
            )

            best_c3 = max(
                best_c3,
                C3,
            )

            history.append(
                {
                    "step": step,
                    "C3": C3,
                    "best_C3": best_c3,
                    "Wp": Wp,
                    "D3": D3,
                    "L3": u3,
                }
            )

            # --------------------------------------------------------
            # Heuristic ascent direction
            # --------------------------------------------------------

            direction = (
                grad_p
                + grad_q
            )

            u = (
                u
                + lr * direction
            )

            # Enforce incompressibility.
            u = self.fl.leray_project(u)

            # Fix overall L2 amplitude so the optimizer cannot
            # increase C3 merely through numerical amplitude drift.
            l2 = self.fl.norm_l2(u)

            if l2 > 1e-30:
                u *= 1.0 / l2

        return best_c3, history


# ====================================================================
# REPORTING UTILITIES
# ====================================================================

def print_shell_spectrum(results):
    """Pretty-print a shell spectrum."""

    print(
        "\n"
        + "=" * 105
    )
    print(
        "SHELL SPECTRUM"
    )
    print(
        "=" * 105
    )

    print(
        f"{'Shell':>7s} | "
        f"{'Modes':>10s} | "
        f"{'Work':>16s} | "
        f"{'Work/mode':>16s} | "
        f"{'D3':>16s} | "
        f"{'C3':>16s}"
    )

    print("-" * 105)

    for j in sorted(results):

        r = results[j]

        print(
            f"{j:7d} | "
            f"{r['multiplicity']:10d} | "
            f"{r['work']:16.6e} | "
            f"{r['work_per_mode']:16.6e} | "
            f"{r['D3']:16.6e} | "
            f"{r['C3']:16.6e}"
        )


def print_schur_report(summary):
    """Print Schur diagnostic."""

    print(
        "\n"
        + "=" * 80
    )
    print(
        "SCHUR INTERACTION AUDIT"
    )
    print(
        "=" * 80
    )

    print(
        "Active shells:"
        f" {summary['shells']}"
    )

    print(
        "Maximum normalized off-diagonal entry:"
        f" {summary['max_off_diagonal']:.6e}"
    )

    print(
        "Schur row-sum supremum:"
        f" {summary['sup_row_sum']:.6e}"
    )

    print(
        "\nInterpretation:"
    )

    print(
        "The small off-diagonal values measure Fourier-shell "
        "projection leakage/roundoff."
    )

    print(
        "They do NOT establish absence of nonlinear cross-scale "
        "interactions in the pressure source."
    )


# ====================================================================
# AUTOMATED TEST HARNESS
# ====================================================================

def run_shell_audit_tests():
    """
    Reproducible numerical audit.

    Requires StructurePreservingLatticeFluid3D to be defined/importable
    from the repository's fluid implementation.
    """

    print("=" * 100)
    print("INFINITE-SHELL INTERACTION & SCHUR AUDIT")
    print("=" * 100)

    try:
        StructurePreservingLatticeFluid3D
    except NameError:
        raise RuntimeError(
            "StructurePreservingLatticeFluid3D is not defined. "
            "Import the repository's fluid class before running "
            "run_shell_audit_tests()."
        )

    for N_res in [16, 32, 64]:

        print(
            f"\n{'=' * 100}"
        )

        print(
            f"GRID RESOLUTION N = {N_res}"
        )

        print(
            f"{'=' * 100}"
        )

        fluid = StructurePreservingLatticeFluid3D(
            N=N_res,
            L=2.0 * np.pi,
            nu=0.01,
        )

        audit = ShellAuditFramework(
            fluid
        )

        u_test = (
            fluid.random_divergence_free_field(
                seed=42,
                amplitude=1.0,
            )
        )

        # ------------------------------------------------------------
        # Linear shell spectrum
        # ------------------------------------------------------------

        spectrum = audit.shell_spectrum(
            u_test,
            shell_type="linear",
        )

        print_shell_spectrum(
            spectrum
        )

        # ------------------------------------------------------------
        # Empirical power-law fits
        # ------------------------------------------------------------

        shell_ids = np.array(
            sorted(spectrum),
            dtype=float,
        )

        multiplicity = np.array(
            [
                spectrum[j]["multiplicity"]
                for j in sorted(spectrum)
            ],
            dtype=float,
        )

        work_per_mode = np.array(
            [
                spectrum[j]["work_per_mode"]
                for j in sorted(spectrum)
            ],
            dtype=float,
        )

        dissipation = np.array(
            [
                spectrum[j]["D3"]
                for j in sorted(spectrum)
            ],
            dtype=float,
        )

        quotient = np.array(
            [
                spectrum[j]["C3"]
                for j in sorted(spectrum)
            ],
            dtype=float,
        )

        _, alpha_modes, r2_modes = (
            audit.power_law_fit(
                shell_ids,
                work_per_mode,
                xmin=4,
            )
        )

        _, alpha_D3, r2_D3 = (
            audit.power_law_fit(
                shell_ids,
                dissipation,
                xmin=4,
            )
        )

        _, alpha_C3, r2_C3 = (
            audit.power_law_fit(
                shell_ids,
                quotient,
                xmin=4,
            )
        )

        print(
            "\nEMPIRICAL POWER-LAW FITS"
        )

        print(
            f"Per-mode work exponent : "
            f"{alpha_modes:+.6f}"
            f"    R^2 = {r2_modes:.6f}"
        )

        print(
            f"D3 shell exponent      : "
            f"{alpha_D3:+.6f}"
            f"    R^2 = {r2_D3:.6f}"
        )

        print(
            f"C3 shell exponent      : "
            f"{alpha_C3:+.6f}"
            f"    R^2 = {r2_C3:.6f}"
        )

        # ------------------------------------------------------------
        # Schur audit
        # ------------------------------------------------------------

        summary = audit.schur_summary(
            u_test,
            shell_type="dyadic",
        )

        print_schur_report(
            summary
        )

        # ------------------------------------------------------------
        # Adversarial search
        # ------------------------------------------------------------

        print(
            "\n"
            + "=" * 80
        )

        print(
            "ADVERSARIAL PROJECTED-GRADIENT SEARCH"
        )

        print(
            "=" * 80
        )

        best_c3, history = (
            audit.adversarial_projected_gradient_search(
                steps=25,
                lr=0.05,
                seed=101,
            )
        )

        print(
            f"Best C3 discovered: "
            f"{best_c3:.6e}"
        )

        if history:
            print(
                f"Initial C3: "
                f"{history[0]['C3']:.6e}"
            )

            print(
                f"Final C3:   "
                f"{history[-1]['C3']:.6e}"
            )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "AUDIT COMPLETE"
    )

    print(
        "=" * 100
    )


# ====================================================================
# OPTIONAL MODULE ENTRY POINT
# ====================================================================

if __name__ == "__main__":
    run_shell_audit_tests()
