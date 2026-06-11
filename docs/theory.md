# Mathematical background

This page summarizes the mathematics implemented in SIES. For complete
statements and proofs, see Ammari & Kang, *Polarization and Moment Tensors*
(Springer, 2007) and the papers listed on the [home page](index.md).

## The conductivity problem

Let $D_1, \dots, D_L$ be small, disjoint, $C^2$-smooth inclusions in
$\mathbb{R}^2$ with conductivities $k_l \neq 1$ in a background of
conductivity $1$. A point source at $x_s$ generates the potential $u$
solving

$$
\nabla \cdot \big( 1 + \textstyle\sum_l (k_l - 1) \chi_{D_l} \big) \nabla u = \delta_{x_s},
\qquad u(x) - G(x - x_s) = O(|x|^{-1}),
$$

where $G(x) = \frac{1}{2\pi}\log|x|$ is the fundamental solution of the
Laplacian. The measured data are the perturbations $u - G$ collected at
receivers for every source: the **multistatic response (MSR) matrix**.

## Layer potentials

The perturbation admits the representation

$$
u - G = \sum_{l} S_{D_l}[\phi_l],
$$

where $S_D$ is the **single layer potential** and the densities $\phi_l$
solve a system of boundary integral equations driven by the **adjoint
Neumann–Poincaré operator** $K_D^*$:

$$
(\lambda_l I - K_{D_l}^*)[\phi_l]
- \sum_{m \neq l} \frac{\partial S_{D_m}[\phi_m]}{\partial \nu_l}
 = \frac{\partial G(\cdot - x_s)}{\partial \nu_l},
\qquad
\lambda_l = \frac{k_l + 1}{2(k_l - 1)}.
$$

SIES discretizes these operators with P0 boundary elements
(`sies.operators`), with analytic treatment of the singular diagonals.

## Generalized polarization tensors

The far-field expansion of $u - G$ is governed by the **contracted
generalized polarization tensors** (CGPT). With harmonic polynomials
$\mathrm{Re}(z^m)$, $\mathrm{Im}(z^m)$:

$$
M^{cc}_{mn} = \int_{\partial D} \mathrm{Re}(z^n)\,
(\lambda I - K_D^*)^{-1}\!\left[ \frac{\partial\, \mathrm{Re}(z^m)}{\partial \nu} \right] ds,
$$

and similarly for the $cs$, $sc$, $ss$ blocks (`sies.asymptotics`). The MSR
matrix is then, to leading order, the *linear* image of the CGPT:

$$
\mathrm{MSR} \approx A_s \, M \, A_r^T,
$$

where $A_s, A_r$ depend only on the acquisition geometry. Inverting this
linear system reconstructs the CGPT from measurements (`sies.pde`); for
equispaced circular acquisitions the least-squares inverse is known in closed
form.

## Invariant shape descriptors

Under a translation $z_0$, rotation $\theta$ and scaling $s$ of the shape,
the *complex* CGPT pair $(N_1, N_2)$ transforms explicitly through
lower-triangular binomial matrices and diagonal phase matrices. Normalizing
out these factors yields descriptors $(\mathcal{I}_1, \mathcal{I}_2)$
**invariant** under all three transformations (`sies.dictionary`).
Identification then amounts to a nearest-neighbor search in descriptor space,
which is robust to measurement noise.

## Tracking

When the target's CGPT is known, the position and orientation
$(x_t, \theta_t)$ become the unknowns of the observation model

$$
h(x_t, \theta_t) = A_s \, M(x_t, \theta_t)\, A_r^T,
$$

which is differentiable in closed form. An **Extended Kalman Filter** with a
constant-velocity motion model estimates the trajectory online
(`sies.tracking`).
