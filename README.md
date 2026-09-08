# Discrete Navier–Stokes Proof / Audit

> **Status: EXPERIMENTAL — NOT A PROOF OF CONTINUUM GLOBAL REGULARITY**

This repository contains a structure-preserving numerical framework for investigating
critical estimates associated with the three-dimensional incompressible
Navier–Stokes equations.

The central goal is **falsifiable computational analysis**: construct discrete
fields, evaluate the relevant nonlinear and pressure terms, perform spectral
shell audits, and search adversarially for configurations that could violate
candidate bounds.

Numerical evidence in this repository must not be interpreted as a proof of
global regularity for the continuum Navier–Stokes equations.

---

## Mathematical setting

We consider the incompressible Navier–Stokes equations

\[
\partial_t u +(u\cdot\nabla)u
 =-\nabla p+\nu\Delta u,
\qquad
\nabla\cdot u=0.
\]

The principal diagnostic is the critical quotient

\[
\mathfrak C_3(u)
 =
 \frac{
 \left|\langle\nabla p,\nabla q\rangle\right|
 }{
 \|u\|_{L^3}\,\mathcal D_3(u)
 },
\]

where

\[
-\Delta p
 =
 \operatorname{div}\big((u\cdot\nabla)u\big),
\]

\[
-\Delta q
 =
 \operatorname{div}(|u|u),
\]

and

\[
\mathcal D_3(u)
 =
 \int |u|\,|\nabla u|^2\,dx.
\]

The quotient is invariant under the critical Navier–Stokes amplitude scaling

\[
u_\sigma(x)
 =
 \sigma^{-1}U(x/\sigma).
\]

Consequently, simple isotropic concentration does not by itself produce a
divergence of the quotient.

---

## What the repository investigates

The unified audit framework examines several possible mechanisms that could
challenge a critical estimate.

### 1. Critical scaling

The implementation checks numerical invariance under

\[
u\mapsto Au.
\]

Since

\[
\|Au\|_3=A\|u\|_3,
\qquad
\mathcal D_3(Au)=A^3\mathcal D_3(u),
\]

and the pressure-work numerator is quartic,

\[
W_p(Au)=A^4W_p(u),
\]

the quotient satisfies

\[
\mathfrak C_3(Au)=\mathfrak C_3(u).
\]

---

### 2. Spectral shell analysis

The audit decomposes the pressure gradients into radial or dyadic Fourier
shells.

For output-filtered fields with disjoint Fourier support,

\[
\langle(\nabla p)_j,(\nabla q)_k\rangle=0,
\qquad j\ne k,
\]

up to numerical roundoff.

This is an exact consequence of Fourier orthogonality for the corresponding
discrete spectral projection. It should **not** be confused with a proof that
the nonlinear source terms themselves decouple across scales.

The nonlinear sources

\[
(u\cdot\nabla)u
\quad\text{and}\quad
|u|u
\]

contain triadic and broadband interactions before the pressure projection is
applied. The repository therefore separately audits source-shell populations.

---

### 3. Schur-type diagnostics

For shell interactions we form a normalized matrix

\[
\Gamma_{jk}
 =
 \frac{
 |\langle(\nabla p)_j,(\nabla q)_k\rangle|
 }{
 \|u\|_3
 \sqrt{\mathcal D_{3,j}\mathcal D_{3,k}}
 }.
\]

The associated row sum is

\[
S_j=\sum_k\Gamma_{jk}.
\]

The computational quantity

\[
\sup_j S_j
\]

is useful as a diagnostic for candidate uniform estimates.

A bounded numerical value is **evidence only**. It does not establish a
continuum Schur estimate.

---

### 4. Nonlinear source / triadic audit

The code separately measures how

\[
B(u,u)=(u\cdot\nabla)u
\]

and

\[
Q(u)=|u|u
\]

populate spectral shells.

This distinction is important:

> Output-shell orthogonality does not eliminate cross-scale nonlinear
> interactions. Those interactions occur inside the source convolutions and
> can transfer energy between frequency bands before the pressure projection.

---

### 5. Aliasing stress tests

The discrete nonlinear terms can be contaminated by spectral aliasing.

The audit therefore compares raw and dealiased calculations where supported.

Large differences between the two should be treated as a warning that a
numerical observation is resolution- or discretization-dependent.

---

### 6. Adversarial optimization

The framework includes a projected-gradient search attempting to increase

\[
\mathfrak C_3(u)
\]

while maintaining the divergence-free constraint.

The purpose is **falsification**:

* Can an apparently reasonable configuration produce a large quotient?
* Does increasing resolution expose growth?
* Do anisotropic or multiscale configurations behave differently?
* Does a candidate estimate survive adversarial perturbations?

Failure to find a counterexample is not a proof that none exists.

---

## Repository philosophy

This project deliberately separates three categories of statements.

### Exact discrete identities

These can be established directly from the implemented discrete operators,
for example:

* Fourier-shell orthogonality;
* algebraic scaling identities;
* properties of explicitly implemented projections;
* reproducibility of deterministic numerical calculations.

### Numerical observations

Examples include:

* measured values of \(\mathfrak C_3\);
* apparent shell-decay exponents;
* resolution trends;
* optimizer results;
* observed differences between aliased and dealiased calculations.

These depend on the grid, implementation, initial data, tolerances, and numerical
precision.

### Continuum conjectures

Examples include statements such as

\[
\sup_u\mathfrak C_3(u)<\infty
\]

or a universal shell estimate of the form

\[
\Gamma_{jk}\le C2^{-\varepsilon|j-k|}.
\]

Such statements require independent mathematical proofs. They are not established
merely because numerical experiments support them.

---

## Current experimental question

A major unresolved question investigated by the repository is whether a
uniform critical estimate of the schematic form

\[
|\langle\nabla p,\nabla q\rangle|
\le
C\,
\|u\|_3\,
\mathcal D_3(u)
\]

can hold for all sufficiently regular divergence-free velocity fields with a
finite universal constant \(C\).

Even if such a constant exists, that alone does **not** automatically prove
global regularity of arbitrary three-dimensional Navier–Stokes data.

For example, an energy inequality of the schematic form

\[
\frac{d}{dt}\|u\|_3^3
+
c\nu\mathcal D_3(u)
\le
C\|u\|_3\mathcal D_3(u)
\]

still requires an appropriate mechanism to control the coefficient of
\(\mathcal D_3\) for large data.

The repository therefore treats the critical quotient as an analytical target,
not as a completed Millennium Prize proof.

---

## Reproducibility

The primary experiment should be run at multiple resolutions.

Typical resolutions include

```text
N = 16
N = 32
N = 64
```

A useful audit should examine:

1. critical amplitude scaling;
2. resolution convergence;
3. shell-by-shell behavior;
4. source-shell growth;
5. aliasing sensitivity;
6. Schur row sums;
7. adversarial initial conditions;
8. numerical stability and reproducibility.

A result that disappears under resolution refinement should not be treated as
evidence for a continuum estimate.

---

## Interpreting shell decay

A fitted law such as

\[
\mathcal C_3(j)\sim j^{-\alpha}
\]

can be useful experimentally.

If a sufficiently robust continuum argument eventually established

\[
\alpha>1,
\]

then the linear shell series

\[
\sum_{j=1}^{\infty}j^{-\alpha}
\]

would converge.

Likewise, dyadic grouping would convert algebraic decay into geometric decay.

However:

> A fitted exponent obtained from a finite numerical inertial range is not
> evidence by itself that the same exponent holds uniformly for arbitrarily
> high frequencies.

This distinction is central to the scientific purpose of the repository.

---

## Numerical limitations

The following issues must be considered before interpreting results:

* finite spatial resolution;
* finite frequency range;
* discrete-vs-continuum operator differences;
* Fourier truncation;
* aliasing;
* boundary/periodicity assumptions;
* floating-point roundoff;
* optimizer initialization;
* optimizer convergence;
* incomplete exploration of configuration space;
* shell definitions and normalization conventions.

In particular, observing values near machine precision in off-diagonal shell
entries is expected when exact disjoint Fourier masks are used. It does not
constitute evidence for an analytic decay law such as

\[
2^{-\varepsilon|j-k|}.
\]

---

## Suggested workflow for new experiments

When introducing a new candidate configuration:

1. Generate a smooth divergence-free field.
2. Verify the discrete divergence.
3. Measure \(\|u\|_3\).
4. Measure \(\mathcal D_3(u)\).
5. Compute \(p\) and \(q\).
6. Compute \(\mathfrak C_3(u)\).
7. Repeat after amplitude rescaling.
8. Repeat at increasing resolution.
9. Run the shell audit.
10. Run the aliasing/dealiasing comparison.
11. Run multiple adversarial initializations.
12. Record all parameters and random seeds.

The objective is to make potential counterexamples easier to reproduce and
candidate estimates easier to falsify.

---

## Scientific disclaimer

**This repository does not prove the global regularity or finite-time blow-up
problem for the three-dimensional incompressible Navier–Stokes equations.**

No finite numerical experiment can establish a statement quantified over all
continuum solutions and arbitrarily small scales.

The code is intended as a computational laboratory for:

* testing structural identities;
* discovering potentially useful estimates;
* finding counterexamples to proposed estimates;
* studying spectral interactions;
* measuring numerical scaling;
* guiding future analytical work.

Any claim of a continuum theorem must be independently justified analytically.

---

## License

Add the project's chosen license here.

If no license has yet been selected, do not imply that the code is licensed for
unrestricted reuse.
