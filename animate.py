"""
Bayes' theorem for Media Mix Modeling (MMM)
===========================================

Bayes' theorem:   P(theta | data) = P(data | theta) * P(theta) / P(data)

In MMM language:
    theta      = a channel's true ROI (or its regression coefficient)
    prior      = what we believe about ROI before this dataset
                 (past studies, lift tests, benchmarks)
    likelihood = what the observed weekly sales data says
    posterior  = the updated belief: a precision-weighted blend of both

Part 1 uses a closed-form Normal-Normal conjugate update (same math as
the interactive animation). Part 2 sketches a real Bayesian MMM in PyMC.

Run:  pip install numpy matplotlib   (Part 1)
      pip install pymc               (Part 2, optional)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import stats

# ----------------------------------------------------------------------
# Part 1: Conjugate Bayesian update, animated week by week
# ----------------------------------------------------------------------

# --- Prior belief about TV ROI ---------------------------------------
prior_mean = 2.0        # we believe TV returns ~2x per dollar
prior_n    = 3          # "worth 3 weeks of data" — prior confidence
sigma      = 0.9        # week-to-week noise in observed ROI

# --- Simulated "true" world ------------------------------------------
rng      = np.random.default_rng(42)
true_roi = 3.2          # reality is better than we believed
n_weeks  = 26
weekly_obs = rng.normal(true_roi, sigma, size=n_weeks)


def posterior_params(obs: np.ndarray):
    """Normal-Normal conjugate update.

    Precision (1/variance) is additive: the posterior mean is the
    precision-weighted average of prior mean and data mean. This IS
    Bayes' theorem, solved analytically for the Gaussian case.
    """
    n = len(obs)
    prior_prec = prior_n / sigma**2          # confidence in the prior
    data_prec  = n / sigma**2                # confidence in the data
    total_prec = prior_prec + data_prec

    post_mean = (prior_prec * prior_mean + data_prec * obs.mean()) / total_prec \
        if n > 0 else prior_mean
    post_sd = np.sqrt(1.0 / total_prec)
    weight_on_data = data_prec / total_prec
    return post_mean, post_sd, weight_on_data


# --- Animation: watch the posterior sharpen as weeks arrive ----------
x = np.linspace(0, 5, 400)
fig, ax = plt.subplots(figsize=(9, 5))

prior_pdf = stats.norm.pdf(x, prior_mean, sigma / np.sqrt(prior_n))
ax.plot(x, prior_pdf, "--", color="gray", label="Prior")
post_line, = ax.plot([], [], color="#7F77DD", lw=2, label="Posterior")
post_fill = None
title = ax.set_title("")
ax.axvline(true_roi, color="#1D9E75", ls=":", label=f"True ROI = {true_roi}x")
ax.set_xlabel("Return on ad spend (ROI)")
ax.set_ylabel("Density")
ax.set_xlim(0, 5)
ax.set_ylim(0, 3.2)
ax.legend(loc="upper left")


def update(frame):
    global post_fill
    obs = weekly_obs[:frame]
    mu, sd, w = posterior_params(obs)
    pdf = stats.norm.pdf(x, mu, sd)
    post_line.set_data(x, pdf)
    if post_fill is not None:
        post_fill.remove()
    post_fill = ax.fill_between(x, pdf, alpha=0.15, color="#7F77DD")
    title.set_text(
        f"Week {frame}: posterior ROI = {mu:.2f}x  "
        f"(±{1.96*sd:.2f}),  weight on data = {w:.0%}"
    )
    return post_line,


anim = FuncAnimation(fig, update, frames=range(0, n_weeks + 1),
                     interval=250, repeat=False)
# plt.tight_layout()
# plt.show()
#To save instead:  
anim.save("bayes_mmm.gif", writer="pillow", fps=4)


# ----------------------------------------------------------------------
# Part 2: What this looks like in a real Bayesian MMM (PyMC sketch)
# ----------------------------------------------------------------------
# In practice MMM estimates many channel coefficients at once, with
# adstock (carryover) and saturation (diminishing returns). Priors do
# the same job as above: they keep noisy, collinear media data from
# producing implausible ROIs, and MCMC gives full posteriors instead
# of point estimates.

PYMC_SKETCH = """
import pymc as pm
import numpy as np

def geometric_adstock(x, alpha, l_max=8):
    w = alpha ** np.arange(l_max)
    w = w / w.sum()
    return np.convolve(x, w)[: len(x)]

with pm.Model() as mmm:
    # --- Priors (the Bayesian part) ---
    intercept = pm.Normal("intercept", mu=sales.mean(), sigma=sales.std())

    # HalfNormal prior: media effects can't be negative.
    # This single choice eliminates the nonsense negative-ROI
    # estimates that plain OLS produces on collinear media data.
    beta_tv     = pm.HalfNormal("beta_tv", sigma=1.0)
    beta_search = pm.HalfNormal("beta_search", sigma=1.0)

    # Adstock decay: Beta prior keeps it in (0, 1)
    alpha_tv = pm.Beta("alpha_tv", alpha=2, beta=2)

    # Saturation (Hill/logistic) half-saturation point
    lam_tv = pm.Gamma("lam_tv", alpha=3, beta=1)

    noise = pm.HalfNormal("noise", sigma=sales.std())

    # --- Likelihood: how spend maps to sales ---
    tv_effect = beta_tv * pm.math.tanh(
        geometric_adstock(tv_spend, 0.5) / lam_tv
    )
    mu = intercept + tv_effect + beta_search * search_spend

    pm.Normal("sales", mu=mu, sigma=noise, observed=sales)

    # --- Posterior: Bayes' theorem, computed by MCMC ---
    trace = pm.sample(2000, tune=1000, target_accept=0.9)

# trace now holds the full posterior for every parameter:
#   ROI point estimate  -> posterior mean of beta_tv
#   uncertainty         -> credible interval, e.g.
#                          az.hdi(trace, var_names=["beta_tv"])
# Budget decisions can use the whole distribution, e.g.
# P(ROI_tv > ROI_search) computed directly from posterior samples.
"""

if __name__ == "__main__":
    print(PYMC_SKETCH)