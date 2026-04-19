import os
os.environ["MPLCONFIGDIR"] = os.path.join(os.getcwd(), ".mplconfig")

from pathlib import Path
import json
import math

import matplotlib

# Use a non-interactive backend so figures can still be saved
# when the script is run without a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import the epidemic-network simulator used in the project.
from simulator import simulate



# Main settings

# These control the overall experiment and output locations.
BASE_DIR = Path(".")
OUTPUT_DIR = BASE_DIR / "outputs_all_methods"
INFECTED_CSV = BASE_DIR / "infected_timeseries.csv"
REWIRING_CSV = BASE_DIR / "rewiring_timeseries.csv"
DEGREE_CSV = BASE_DIR / "final_degree_histograms.csv"

# Settings for the shared rejection ABC baseline
N_DRAWS = 5000
ACCEPT_FRACTION = 0.05
MASTER_SEED = 12345
N_POSTERIOR_PRED_REPS = 40
N_PPC_SAMPLES = 50

# The strongest summary set will be used for the advanced methods.
ADVANCED_SUMMARY_GROUP = "full"

# ABC-MCMC settings
ABC_MCMC_N_ITER = 3500
ABC_MCMC_BURN = 800
ABC_MCMC_THIN = 3
ABC_MCMC_PROPOSAL_SD = np.array([0.018, 0.008, 0.025], dtype=float)
ABC_MCMC_EPS_MULTIPLIER = 1.00  # epsilon is set relative to the baseline rejection ABC epsilon

# SMC-ABC / ABC-PMC settings
SMC_N_PARTICLES = 220
SMC_N_ROUNDS = 6
SMC_EPS_QUANTILE = 0.55
SMC_JITTER_SCALE = 1.10
SMC_FINAL_PPC_SAMPLES = 60

# Synthetic likelihood MCMC settings
SL_MCMC_N_ITER = 2200
SL_MCMC_BURN = 600
SL_MCMC_THIN = 4
SL_ESTIMATION_REPS = 24
SL_PROPOSAL_SD = np.array([0.018, 0.010, 0.020], dtype=float)
SL_COV_RIDGE = 1e-6
SL_CACHE_ROUND = 6

# Simulator settings taken from the project setup
SIM_N = 200
SIM_P_EDGE = 0.05
SIM_N_INFECTED0 = 5
SIM_T = 200

# Prior ranges used in the report / project statement
BETA_MIN, BETA_MAX = 0.05, 0.50
GAMMA_MIN, GAMMA_MAX = 0.02, 0.20
RHO_MIN, RHO_MAX = 0.00, 0.80
PRIOR_BOUNDS = np.array([
    [BETA_MIN, BETA_MAX],
    [GAMMA_MIN, GAMMA_MAX],
    [RHO_MIN, RHO_MAX],
], dtype=float)



# Summary statistics

SUMMARY_NAMES = [
    "peak_infected",        # highest infected fraction reached
    "peak_time",            # time when the infection reaches its peak
    "final_infected",       # infected fraction at the final time step
    "mean_infected",        # average infected fraction over time
    "auc_infected",         # total infected mass across the whole path
    "early_growth_5",       # infected(t=5) - infected(t=0)
    "mid_growth_20",        # infected(t=20) - infected(t=0)
    "time_above_10pct",     # number of time points with infected fraction at least 10%
    "total_rewires",        # total number of rewiring events
    "peak_rewires",         # largest rewiring count at a single time step
    "mean_rewires",         # average rewiring count
    "early_rewires_20",     # total rewiring during the first 20 time steps
    "late_rewires_100_200", # total rewiring from time 100 to 200
    "mean_degree",          # mean final network degree
    "var_degree",           # variance of the final degree distribution
    "share_degree_0_4",     # share of nodes with degree 0 to 4
    "share_degree_5_9",     # share of nodes with degree 5 to 9
    "share_degree_10_14",   # share of nodes with degree 10 to 14
    "share_degree_15_plus", # share of nodes with degree 15 or above
]


# We compare three different summary sets:
# 1. infected_only: infection information only
# 2. infected_rewiring: infection + rewiring information
# 3. full: infection + rewiring + degree information
SUMMARY_GROUPS = {
    "infected_only": [
        "peak_infected",
        "peak_time",
        "final_infected",
        "mean_infected",
        "auc_infected",
        "early_growth_5",
        "mid_growth_20",
        "time_above_10pct",
    ],
    "infected_rewiring": [
        "peak_infected",
        "peak_time",
        "final_infected",
        "mean_infected",
        "auc_infected",
        "early_growth_5",
        "mid_growth_20",
        "time_above_10pct",
        "total_rewires",
        "peak_rewires",
        "mean_rewires",
        "early_rewires_20",
        "late_rewires_100_200",
    ],
    "full": SUMMARY_NAMES.copy(),
}

# Order used when processing the three summary sets
SUMMARY_SET_ORDER = ["infected_only", "infected_rewiring", "full"]

# Parameter names used throughout the script
PARAM_NAMES = ["beta", "gamma", "rho"]

# Column names for the regression-adjusted ABC samples
ADJ_PARAM_NAMES = ["beta_adj", "gamma_adj", "rho_adj"]



# Basic helper functions

def load_observed_data():
    return pd.read_csv(INFECTED_CSV), pd.read_csv(REWIRING_CSV), pd.read_csv(DEGREE_CSV)


def safe_index_value(arr, idx):
    # Keep the index inside the valid range.
    idx = min(max(idx, 0), len(arr) - 1)
    return arr[idx]


def summarize_arrays(infected_series, rewiring_series, degree_hist):
    infected_series = np.asarray(infected_series, dtype=float)
    rewiring_series = np.asarray(rewiring_series, dtype=float)
    degree_hist = np.asarray(degree_hist, dtype=float)

    t = np.arange(len(infected_series))
    total_nodes = degree_hist.sum()
    degree_values = np.arange(len(degree_hist))
    if total_nodes <= 0:
        raise ValueError("Degree histogram has non-positive total count.")

    peak_idx = int(np.argmax(infected_series))
    peak_infected = float(infected_series[peak_idx])
    peak_time = float(t[peak_idx])
    final_infected = float(infected_series[-1])
    mean_infected = float(np.mean(infected_series))
    auc_infected = float(np.sum(infected_series))
    early_growth_5 = float(safe_index_value(infected_series, 5) - infected_series[0])
    mid_growth_20 = float(safe_index_value(infected_series, 20) - infected_series[0])
    time_above_10pct = float(np.sum(infected_series >= 0.10))

    total_rewires = float(np.sum(rewiring_series))
    peak_rewires = float(np.max(rewiring_series))
    mean_rewires = float(np.mean(rewiring_series))
    early_rewires_20 = float(np.sum(rewiring_series[:20]))
    late_rewires_100_200 = float(np.sum(rewiring_series[100:201]))

    mean_degree = float(np.sum(degree_values * degree_hist) / total_nodes)
    var_degree = float(np.sum(((degree_values - mean_degree) ** 2) * degree_hist) / total_nodes)
    share_degree_0_4 = float(np.sum(degree_hist[0:5]) / total_nodes)
    share_degree_5_9 = float(np.sum(degree_hist[5:10]) / total_nodes)
    share_degree_10_14 = float(np.sum(degree_hist[10:15]) / total_nodes)
    share_degree_15_plus = float(np.sum(degree_hist[15:]) / total_nodes)

    return np.array([
        peak_infected,
        peak_time,
        final_infected,
        mean_infected,
        auc_infected,
        early_growth_5,
        mid_growth_20,
        time_above_10pct,
        total_rewires,
        peak_rewires,
        mean_rewires,
        early_rewires_20,
        late_rewires_100_200,
        mean_degree,
        var_degree,
        share_degree_0_4,
        share_degree_5_9,
        share_degree_10_14,
        share_degree_15_plus,
    ], dtype=float)


def summarize_observed_dataset(infected_df, rewiring_df, degree_df):
    replicate_ids = sorted(infected_df["replicate_id"].unique())
    summaries = []

    for rep_id in replicate_ids:
        inf_rep = infected_df[infected_df["replicate_id"] == rep_id].sort_values("time")
        rew_rep = rewiring_df[rewiring_df["replicate_id"] == rep_id].sort_values("time")
        deg_rep = degree_df[degree_df["replicate_id"] == rep_id].sort_values("degree")
        summaries.append(
            summarize_arrays(
                inf_rep["infected_fraction"].to_numpy(),
                rew_rep["rewire_count"].to_numpy(),
                deg_rep["count"].to_numpy(),
            )
        )

    rep_summaries = np.vstack(summaries)
    return rep_summaries.mean(axis=0), rep_summaries


def get_summary_indices(summary_group_name):
    return [SUMMARY_NAMES.index(name) for name in SUMMARY_GROUPS[summary_group_name]]


def clip_theta(theta):
    # Force the parameter vector to stay inside the prior box.
    theta = np.asarray(theta, dtype=float).copy()
    theta[0] = np.clip(theta[0], BETA_MIN, BETA_MAX)
    theta[1] = np.clip(theta[1], GAMMA_MIN, GAMMA_MAX)
    theta[2] = np.clip(theta[2], RHO_MIN, RHO_MAX)
    return theta


def in_prior_bounds(theta):
    theta = np.asarray(theta, dtype=float)
    return bool(np.all(theta >= PRIOR_BOUNDS[:, 0]) and np.all(theta <= PRIOR_BOUNDS[:, 1]))


def sample_prior(rng):
    # Draw one parameter vector from the uniform prior.
    return np.array([
        rng.uniform(BETA_MIN, BETA_MAX),
        rng.uniform(GAMMA_MIN, GAMMA_MAX),
        rng.uniform(RHO_MIN, RHO_MAX),
    ], dtype=float)


def standardized_distance(sim_summary, obs_summary, scale):
    # Standardized Euclidean distance used in the ABC steps.
    z = (sim_summary - obs_summary) / scale
    return float(np.sqrt(np.sum(z ** 2)))


def equal_tailed_interval(x, alpha=0.05):
    return float(np.quantile(x, alpha / 2)), float(np.quantile(x, 1 - alpha / 2))


def posterior_summary_table(samples_df, column_map):
    rows = []
    for label, col in column_map.items():
        x = samples_df[col].to_numpy()
        q_low, q_high = equal_tailed_interval(x)
        rows.append({
            "parameter": label,
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "sd": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
            "q2.5": q_low,
            "q97.5": q_high,
        })
    return pd.DataFrame(rows)


def method_summary_from_array(samples, method_name, summary_group):
    df = pd.DataFrame(samples, columns=PARAM_NAMES)
    tab = posterior_summary_table(df, {k: k for k in PARAM_NAMES})
    tab.insert(0, "summary_group", summary_group)
    tab.insert(0, "method", method_name)
    return tab



# Observed data plots

def save_observed_data_plots(infected_df, rewiring_df, degree_df, outdir):
    outdir.mkdir(parents=True, exist_ok=True)

    infected_mean = infected_df.groupby("time")["infected_fraction"].mean()
    infected_q10 = infected_df.groupby("time")["infected_fraction"].quantile(0.10)
    infected_q90 = infected_df.groupby("time")["infected_fraction"].quantile(0.90)
    plt.figure(figsize=(8, 4.5))
    plt.plot(infected_mean.index, infected_mean.values)
    plt.fill_between(infected_mean.index, infected_q10.values, infected_q90.values, alpha=0.25)
    plt.xlabel("Time")
    plt.ylabel("Infected fraction")
    plt.title("Observed infected time series")
    plt.tight_layout()
    plt.savefig(outdir / "observed_infected_timeseries.png", dpi=220)
    plt.close()

    rew_mean = rewiring_df.groupby("time")["rewire_count"].mean()
    rew_q10 = rewiring_df.groupby("time")["rewire_count"].quantile(0.10)
    rew_q90 = rewiring_df.groupby("time")["rewire_count"].quantile(0.90)
    plt.figure(figsize=(8, 4.5))
    plt.plot(rew_mean.index, rew_mean.values)
    plt.fill_between(rew_mean.index, rew_q10.values, rew_q90.values, alpha=0.25)
    plt.xlabel("Time")
    plt.ylabel("Rewire count")
    plt.title("Observed rewiring time series")
    plt.tight_layout()
    plt.savefig(outdir / "observed_rewiring_timeseries.png", dpi=220)
    plt.close()

    mean_degree_hist = degree_df.groupby("degree")["count"].mean()
    plt.figure(figsize=(8, 4.5))
    plt.bar(mean_degree_hist.index, mean_degree_hist.values)
    plt.xlabel("Degree")
    plt.ylabel("Mean count across replicates")
    plt.title("Observed final degree histogram")
    plt.tight_layout()
    plt.savefig(outdir / "observed_degree_histogram.png", dpi=220)
    plt.close()



# Simulation helpers

def simulate_replicate_summaries(beta, gamma, rho, n_replicates, seed):
    rng = np.random.default_rng(seed)
    rep_summaries = []
    infected_matrix = []
    rewiring_matrix = []
    degree_matrix = []

    # Run several simulator replicates under the same parameter vector.
    for _ in range(n_replicates):
        rep_seed = int(rng.integers(0, 2**32 - 1))
        infected_series, rewiring_series, degree_hist = simulate(
            beta=beta,
            gamma=gamma,
            rho=rho,
            N=SIM_N,
            p_edge=SIM_P_EDGE,
            n_infected0=SIM_N_INFECTED0,
            T=SIM_T,
            rng=np.random.default_rng(rep_seed),
        )
        rep_summaries.append(summarize_arrays(infected_series, rewiring_series, degree_hist))
        infected_matrix.append(np.asarray(infected_series, dtype=float))
        rewiring_matrix.append(np.asarray(rewiring_series, dtype=float))
        degree_matrix.append(np.asarray(degree_hist, dtype=float))

    return (
        np.vstack(rep_summaries),
        np.vstack(infected_matrix),
        np.vstack(rewiring_matrix),
        np.vstack(degree_matrix),
    )


def simulate_mean_summary(theta, n_replicates, seed):
    # Average the replicate summaries so the simulation matches the structure
    # of the observed dataset.
    rep_summaries, _, _, _ = simulate_replicate_summaries(
        theta[0], theta[1], theta[2], n_replicates=n_replicates, seed=seed
    )
    return rep_summaries.mean(axis=0)



# Shared-baseline rejection ABC

def generate_shared_abc_draws(obs_rep_summaries, n_draws, seed):
    rng = np.random.default_rng(seed)
    n_replicates = obs_rep_summaries.shape[0]
    params = []
    summary_rows = []

    # Generate one shared bank of simulations so all summary sets
    # are compared fairly using the same underlying draws.
    for draw in range(n_draws):
        theta = sample_prior(rng)
        sim_seed = int(rng.integers(0, 2**32 - 1))
        sim_mean = simulate_mean_summary(theta, n_replicates=n_replicates, seed=sim_seed)
        params.append({"draw_id": draw, "beta": theta[0], "gamma": theta[1], "rho": theta[2]})
        summary_rows.append(sim_mean)
        if (draw + 1) % 50 == 0:
            print(f"[shared baseline] completed {draw + 1}/{n_draws} draws")

    return {
        "params_df": pd.DataFrame(params),
        "summary_matrix": np.vstack(summary_rows),
    }


def build_result_for_summary_group(summary_group_name, shared_draws, obs_mean_summary, obs_rep_summaries):
    group_idx = get_summary_indices(summary_group_name)
    obs_group_mean = obs_mean_summary[group_idx]
    obs_group_reps = obs_rep_summaries[:, group_idx]
    scale = obs_group_reps.std(axis=0, ddof=1)
    scale = np.where(scale <= 1e-12, 1.0, scale)

    sim_group_matrix = shared_draws["summary_matrix"][:, group_idx]
    distances = np.array([
        standardized_distance(sim_group_matrix[i], obs_group_mean, scale)
        for i in range(sim_group_matrix.shape[0])
    ])

    all_draws = shared_draws["params_df"].copy()
    all_draws["distance"] = distances
    all_draws = all_draws.sort_values("distance").reset_index(drop=True)

    n_accept = max(1, int(np.floor(ACCEPT_FRACTION * len(all_draws))))
    accepted = all_draws.iloc[:n_accept].copy().reset_index(drop=True)
    accepted_summary_matrix = sim_group_matrix[accepted["draw_id"].to_numpy()]
    accepted["weight_uniform"] = 1.0 / len(accepted)

    return {
        "summary_group": summary_group_name,
        "group_idx": group_idx,
        "group_names": [SUMMARY_NAMES[i] for i in group_idx],
        "obs_group_mean": obs_group_mean,
        "obs_group_reps": obs_group_reps,
        "scale": scale,
        "all_draws": all_draws,
        "accepted": accepted,
        "accepted_summary_matrix": accepted_summary_matrix,
        "epsilon": float(accepted["distance"].max()),
    }



# Regression adjustment

def epanechnikov_weights(distances, eps):
    if eps <= 0:
        return np.ones_like(distances)
    u = np.clip(distances / eps, 0.0, 1.0)
    return 1.0 - u ** 2


def weighted_linear_regression(X, y, w):
    sqrt_w = np.sqrt(w)
    coef, _, _, _ = np.linalg.lstsq(X * sqrt_w[:, None], y * sqrt_w, rcond=None)
    return coef


def apply_regression_adjustment(accepted_df, accepted_summary_matrix, obs_group_mean):
    theta = accepted_df[PARAM_NAMES].to_numpy()
    s = accepted_summary_matrix
    distances = accepted_df["distance"].to_numpy()
    eps = float(np.max(distances))

    if eps <= 0:
        adjusted = theta.copy()
    else:
        # Beaumont-style local linear adjustment:
        # shift accepted draws toward the observed summaries.
        X_core = s - obs_group_mean[None, :]
        X = np.column_stack([np.ones(len(s)), X_core])
        w = epanechnikov_weights(distances, eps)
        adjusted = np.zeros_like(theta)
        for j in range(theta.shape[1]):
            coef = weighted_linear_regression(X, theta[:, j], w)
            adjusted[:, j] = theta[:, j] - X_core @ coef[1:]

    adjusted = np.column_stack([
        np.clip(adjusted[:, 0], BETA_MIN, BETA_MAX),
        np.clip(adjusted[:, 1], GAMMA_MIN, GAMMA_MAX),
        np.clip(adjusted[:, 2], RHO_MIN, RHO_MAX),
    ])
    return pd.DataFrame(adjusted, columns=ADJ_PARAM_NAMES)



# ABC-MCMC

def propose_theta_rw(current, proposal_sd, rng):
    # Random-walk Gaussian proposal.
    prop = np.asarray(current, dtype=float) + rng.normal(0.0, proposal_sd, size=3)
    return prop


def abc_mcmc(obs_group_mean, scale, n_replicates, epsilon, n_iter, burn, thin, proposal_sd, seed):
    rng = np.random.default_rng(seed)

    # First find a valid starting point that already satisfies the ABC threshold.
    init_theta = None
    init_dist = None
    for _ in range(5000):
        theta_try = sample_prior(rng)
        sim_mean = simulate_mean_summary(theta_try, n_replicates, seed=int(rng.integers(0, 2**32 - 1)))
        dist = standardized_distance(sim_mean, obs_group_mean, scale)
        if dist <= epsilon:
            init_theta = theta_try
            init_dist = dist
            break
    if init_theta is None:
        raise RuntimeError("ABC-MCMC could not find an initial accepted state. Increase epsilon.")

    current = init_theta.copy()
    current_dist = float(init_dist)
    chain = []
    accepted_moves = 0

    for it in range(n_iter):
        proposal = propose_theta_rw(current, proposal_sd, rng)
        if not in_prior_bounds(proposal):
            chain.append(np.concatenate([current, [current_dist, 0.0]]))
            continue

        sim_mean = simulate_mean_summary(proposal, n_replicates, seed=int(rng.integers(0, 2**32 - 1)))
        dist = standardized_distance(sim_mean, obs_group_mean, scale)
        if dist <= epsilon:
            current = proposal
            current_dist = dist
            accepted_moves += 1
            accepted_indicator = 1.0
        else:
            accepted_indicator = 0.0
        chain.append(np.concatenate([current, [current_dist, accepted_indicator]]))

        if (it + 1) % 250 == 0:
            print(f"[ABC-MCMC] iteration {it + 1}/{n_iter}")

    chain = np.vstack(chain)
    chain_df = pd.DataFrame(chain, columns=PARAM_NAMES + ["distance", "accepted_move"])
    kept = chain_df.iloc[burn::thin].reset_index(drop=True)
    meta = {
        "epsilon": float(epsilon),
        "n_iter": int(n_iter),
        "burn": int(burn),
        "thin": int(thin),
        "proposal_sd": proposal_sd.tolist(),
        "acceptance_rate": float(accepted_moves / max(1, n_iter)),
    }
    return chain_df, kept, meta



# SMC-ABC / ABC-PMC

def weighted_covariance(x, w):
    w = np.asarray(w, dtype=float)
    w = w / np.sum(w)
    mu = np.sum(x * w[:, None], axis=0)
    xc = x - mu
    return (xc * w[:, None]).T @ xc


def multivariate_gaussian_density(x, mean, cov):
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    d = len(x)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return 0.0
    diff = x - mean
    quad = diff @ np.linalg.solve(cov, diff)
    logpdf = -0.5 * (d * np.log(2 * np.pi) + logdet + quad)
    return float(np.exp(logpdf))


def smc_abc(obs_group_mean, scale, n_replicates, n_particles, n_rounds, eps_quantile, jitter_scale, seed):
    rng = np.random.default_rng(seed)
    particles = []
    distances = []

    # Round 1: sample everything from the prior.
    for i in range(n_particles):
        theta = sample_prior(rng)
        sim_mean = simulate_mean_summary(theta, n_replicates, seed=int(rng.integers(0, 2**32 - 1)))
        dist = standardized_distance(sim_mean, obs_group_mean, scale)
        particles.append(theta)
        distances.append(dist)
        if (i + 1) % 50 == 0:
            print(f"[SMC round 1] particle {i + 1}/{n_particles}")

    particles = np.vstack(particles)
    distances = np.asarray(distances, dtype=float)
    weights = np.ones(n_particles, dtype=float) / n_particles
    history = []

    for round_id in range(1, n_rounds + 1):
        epsilon = float(np.quantile(distances, eps_quantile))
        keep = distances <= epsilon
        prev_particles = particles[keep]
        prev_weights = weights[keep]
        prev_weights = prev_weights / np.sum(prev_weights)
        prev_distances = distances[keep]

        cov = weighted_covariance(prev_particles, prev_weights)
        cov = jitter_scale * 2.0 * cov + 1e-8 * np.eye(3)

        history.append({
            "round": round_id,
            "epsilon": epsilon,
            "n_kept_from_previous": int(np.sum(keep)),
            "mean_distance_prev_kept": float(np.mean(prev_distances)),
        })

        if round_id == n_rounds:
            particles = prev_particles
            weights = prev_weights
            distances = prev_distances
            break

        new_particles = []
        new_distances = []
        new_weights = []

        # Later rounds: resample previous particles and jitter them.
        while len(new_particles) < n_particles:
            ancestor_idx = int(rng.choice(len(prev_particles), p=prev_weights))
            ancestor = prev_particles[ancestor_idx]
            proposal = rng.multivariate_normal(mean=ancestor, cov=cov)
            if not in_prior_bounds(proposal):
                continue

            sim_mean = simulate_mean_summary(proposal, n_replicates, seed=int(rng.integers(0, 2**32 - 1)))
            dist = standardized_distance(sim_mean, obs_group_mean, scale)
            if dist > epsilon:
                continue

            denom = 0.0
            for j in range(len(prev_particles)):
                denom += prev_weights[j] * multivariate_gaussian_density(proposal, prev_particles[j], cov)
            numer = 1.0 / np.prod(PRIOR_BOUNDS[:, 1] - PRIOR_BOUNDS[:, 0])
            weight = numer / max(denom, 1e-300)

            new_particles.append(proposal)
            new_distances.append(dist)
            new_weights.append(weight)

            if len(new_particles) % 50 == 0:
                print(f"[SMC round {round_id + 1}] accepted particle {len(new_particles)}/{n_particles}")

        particles = np.vstack(new_particles)
        distances = np.asarray(new_distances, dtype=float)
        weights = np.asarray(new_weights, dtype=float)
        weights = weights / np.sum(weights)

    final_df = pd.DataFrame(particles, columns=PARAM_NAMES)
    final_df["weight"] = weights
    final_df["distance"] = distances
    history_df = pd.DataFrame(history)
    return final_df, history_df



# Synthetic likelihood MCMC

def log_mvn_density(x, mean, cov):
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return -np.inf
    diff = x - mean
    quad = diff @ np.linalg.solve(cov, diff)
    return float(-0.5 * (len(x) * np.log(2 * np.pi) + logdet + quad))


def make_sl_cache_key(theta):
    # Round parameters before caching so nearly identical values
    # do not create too many separate cache entries.
    return tuple(np.round(np.asarray(theta, dtype=float), SL_CACHE_ROUND))


def synthetic_likelihood_value(theta, obs_group_mean, n_replicates, n_estimation_reps, rng, cache):
    key = make_sl_cache_key(theta)
    if key in cache:
        return cache[key]

    sim_means = []
    for _ in range(n_estimation_reps):
        sim_means.append(
            simulate_mean_summary(theta, n_replicates=n_replicates, seed=int(rng.integers(0, 2**32 - 1)))
        )
    sim_means = np.vstack(sim_means)

    mu_hat = sim_means.mean(axis=0)
    cov_hat = np.cov(sim_means, rowvar=False)
    if cov_hat.ndim == 0:
        cov_hat = np.array([[float(cov_hat)]])
    cov_hat = cov_hat + SL_COV_RIDGE * np.eye(cov_hat.shape[0])
    loglik = log_mvn_density(obs_group_mean, mu_hat, cov_hat)

    cache[key] = {
        "loglik": float(loglik),
        "mu_hat": mu_hat,
        "cov_hat": cov_hat,
    }
    return cache[key]


def synthetic_likelihood_mcmc(obs_group_mean, baseline_center, n_replicates, n_iter, burn, thin, proposal_sd, n_estimation_reps, seed):
    rng = np.random.default_rng(seed)
    cache = {}

    # Start near the adjusted ABC center rather than from a random prior draw.
    current = clip_theta(baseline_center)
    current_eval = synthetic_likelihood_value(
        current, obs_group_mean, n_replicates, n_estimation_reps, rng, cache
    )
    current_ll = current_eval["loglik"]

    chain = []
    accepted_moves = 0

    for it in range(n_iter):
        proposal = propose_theta_rw(current, proposal_sd, rng)
        if in_prior_bounds(proposal):
            prop_eval = synthetic_likelihood_value(
                proposal, obs_group_mean, n_replicates, n_estimation_reps, rng, cache
            )
            prop_ll = prop_eval["loglik"]
            log_alpha = prop_ll - current_ll  # uniform prior + symmetric random-walk proposal
            if np.log(rng.random()) < log_alpha:
                current = proposal
                current_ll = prop_ll
                accepted_moves += 1
                accepted_indicator = 1.0
            else:
                accepted_indicator = 0.0
        else:
            accepted_indicator = 0.0

        chain.append(np.concatenate([current, [current_ll, accepted_indicator]]))
        if (it + 1) % 200 == 0:
            print(f"[Synthetic Likelihood MCMC] iteration {it + 1}/{n_iter}")

    chain = np.vstack(chain)
    chain_df = pd.DataFrame(chain, columns=PARAM_NAMES + ["log_synth_like", "accepted_move"])
    kept = chain_df.iloc[burn::thin].reset_index(drop=True)
    meta = {
        "n_iter": int(n_iter),
        "burn": int(burn),
        "thin": int(thin),
        "proposal_sd": proposal_sd.tolist(),
        "n_estimation_reps": int(n_estimation_reps),
        "acceptance_rate": float(accepted_moves / max(1, n_iter)),
        "cache_size": int(len(cache)),
    }
    return chain_df, kept, meta



# Plotting helpers

def save_histogram(series, title, xlabel, filename, weights=None):
    plt.figure(figsize=(6, 4))
    plt.hist(series, bins=20, weights=weights)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()


def save_scatter(x, y, xlabel, ylabel, title, filename):
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, s=18)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()


def save_weighted_scatter(x, y, w, xlabel, ylabel, title, filename):
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, s=200 * (w / np.max(w)), alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()


def save_posterior_plots_baseline(result, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    accepted = result["accepted"]
    adjusted = result["adjusted"]

    for param in PARAM_NAMES:
        save_histogram(accepted[param], f"Rejection ABC posterior: {param}", param, outdir / f"rejection_{param}_hist.png")

    for xcol, ycol in [("beta", "gamma"), ("beta", "rho"), ("gamma", "rho")]:
        save_scatter(accepted[xcol], accepted[ycol], xcol, ycol, f"Rejection ABC: {xcol} vs {ycol}", outdir / f"rejection_{xcol}_{ycol}_scatter.png")

    for param, col in zip(PARAM_NAMES, ADJ_PARAM_NAMES):
        save_histogram(adjusted[col], f"Regression-adjusted posterior: {param}", param, outdir / f"adjusted_{param}_hist.png")

    label_map = {"beta_adj": "beta", "gamma_adj": "gamma", "rho_adj": "rho"}
    for xcol, ycol in [("beta_adj", "gamma_adj"), ("beta_adj", "rho_adj"), ("gamma_adj", "rho_adj")]:
        save_scatter(adjusted[xcol], adjusted[ycol], label_map[xcol], label_map[ycol], f"Adjusted ABC: {label_map[xcol]} vs {label_map[ycol]}", outdir / f"adjusted_{xcol}_{ycol}_scatter.png")

    save_histogram(result["all_draws"]["distance"], f"Distance distribution: {result['summary_group']}", "Distance", outdir / "distance_histogram.png")


def save_posterior_plots_samples(samples_df, outdir, prefix, weighted=False, weight_col=None):
    outdir.mkdir(parents=True, exist_ok=True)
    weights = None
    if weighted and weight_col is not None:
        weights = samples_df[weight_col].to_numpy(dtype=float)
    for p in PARAM_NAMES:
        save_histogram(samples_df[p], f"{prefix}: {p}", p, outdir / f"{prefix}_{p}_hist.png", weights=weights)
    if weighted and weight_col is not None:
        save_weighted_scatter(samples_df["beta"], samples_df["rho"], samples_df[weight_col], "beta", "rho", f"{prefix}: beta vs rho", outdir / f"{prefix}_beta_rho_scatter.png")
    else:
        save_scatter(samples_df["beta"], samples_df["rho"], "beta", "rho", f"{prefix}: beta vs rho", outdir / f"{prefix}_beta_rho_scatter.png")



# Posterior predictive check helpers

def weighted_resample_indices(weights, size, rng):
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)
    return rng.choice(len(weights), size=size, replace=True, p=weights)


def ppc_from_parameter_samples(param_array, infected_obs, rewiring_obs, degree_obs, outdir, suffix):
    outdir.mkdir(parents=True, exist_ok=True)
    obs_inf_mean = infected_obs.groupby("time")["infected_fraction"].mean().to_numpy()
    obs_rew_mean = rewiring_obs.groupby("time")["rewire_count"].mean().to_numpy()
    obs_deg_mean = degree_obs.groupby("degree")["count"].mean().to_numpy()

    rng = np.random.default_rng(MASTER_SEED + 202)
    infected_means = []
    rewiring_means = []
    degree_means = []

    # For each sampled parameter vector, simulate posterior predictive replicates.
    for theta in param_array:
        _, infected_matrix, rewiring_matrix, degree_matrix = simulate_replicate_summaries(
            theta[0], theta[1], theta[2], n_replicates=N_POSTERIOR_PRED_REPS,
            seed=int(rng.integers(0, 2**32 - 1))
        )
        infected_means.append(infected_matrix.mean(axis=0))
        rewiring_means.append(rewiring_matrix.mean(axis=0))
        degree_means.append(degree_matrix.mean(axis=0))

    infected_means = np.vstack(infected_means)
    rewiring_means = np.vstack(rewiring_means)
    degree_means = np.vstack(degree_means)

    def save_band(obs, sims, xlabel, ylabel, title, filename):
        x = np.arange(len(obs))
        q10 = np.quantile(sims, 0.10, axis=0)
        q90 = np.quantile(sims, 0.90, axis=0)
        m = sims.mean(axis=0)
        plt.figure(figsize=(8, 4.5))
        plt.plot(x, obs, label="Observed mean")
        plt.plot(x, m, label="Posterior predictive mean")
        plt.fill_between(x, q10, q90, alpha=0.25, label="Predictive 10%-90% band")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=220)
        plt.close()

    save_band(obs_inf_mean, infected_means, "Time", "Infected fraction", f"PPC infected ({suffix})", outdir / f"ppc_infected_{suffix}.png")
    save_band(obs_rew_mean, rewiring_means, "Time", "Rewire count", f"PPC rewiring ({suffix})", outdir / f"ppc_rewiring_{suffix}.png")
    save_band(obs_deg_mean, degree_means, "Degree", "Count", f"PPC degree histogram ({suffix})", outdir / f"ppc_degree_{suffix}.png")

    return {
        "infected_rmse": float(np.sqrt(np.mean((infected_means.mean(axis=0) - obs_inf_mean) ** 2))),
        "rewiring_rmse": float(np.sqrt(np.mean((rewiring_means.mean(axis=0) - obs_rew_mean) ** 2))),
        "degree_rmse": float(np.sqrt(np.mean((degree_means.mean(axis=0) - obs_deg_mean) ** 2))),
    }



# Text report

def write_text_report(all_baseline_results, advanced_summary, outdir):
    lines = []
    lines.append("ST3246 Final Project - Unified all-methods automatic summary\n")
    lines.append(f"Shared baseline draws = {N_DRAWS}")
    lines.append(f"Acceptance fraction = {ACCEPT_FRACTION}")
    lines.append(f"Advanced summary group = {ADVANCED_SUMMARY_GROUP}")
    lines.append("")

    lines.append("BASELINE SUMMARY-SET COMPARISON")
    lines.append("--------------------------------")
    for name, result in all_baseline_results.items():
        lines.append(f"Summary set: {name}")
        lines.append("Included summaries: " + ", ".join(result["group_names"]))
        lines.append("Rejection ABC posterior summary")
        lines.append(result["posterior_rejection"].to_string(index=False))
        lines.append("")
        lines.append("Regression-adjusted ABC posterior summary")
        lines.append(result["posterior_adjusted"].to_string(index=False))
        lines.append("")

    lines.append("ADVANCED METHODS ON STRONGEST SUMMARY SET")
    lines.append("----------------------------------------")
    for method_name, info in advanced_summary.items():
        lines.append(f"Method: {method_name}")
        if "posterior" in info:
            lines.append(info["posterior"].to_string(index=False))
        if "meta" in info:
            lines.append("Meta:")
            lines.append(json.dumps(info["meta"], indent=2))
        if "ppc" in info:
            lines.append("PPC RMSE:")
            lines.append(json.dumps(info["ppc"], indent=2))
        lines.append("")

    (outdir / "analysis_summary_all_methods.txt").write_text("\n".join(lines), encoding="utf-8")



# Main pipeline

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    infected_obs, rewiring_obs, degree_obs = load_observed_data()
    save_observed_data_plots(infected_obs, rewiring_obs, degree_obs, OUTPUT_DIR)

    obs_mean_summary, obs_rep_summaries = summarize_observed_dataset(infected_obs, rewiring_obs, degree_obs)
    n_observed_replicates = obs_rep_summaries.shape[0]

    # 1) Build one shared simulation bank for the baseline comparison
    print("=" * 90)
    print("Generating shared baseline simulations for all three summary sets...")
    shared_draws = generate_shared_abc_draws(obs_rep_summaries, n_draws=N_DRAWS, seed=MASTER_SEED)
    shared_draws["params_df"].to_csv(OUTPUT_DIR / "shared_draw_parameters.csv", index=False)
    pd.DataFrame(shared_draws["summary_matrix"], columns=SUMMARY_NAMES).to_csv(
        OUTPUT_DIR / "shared_draw_full_mean_summaries.csv", index=False
    )

    all_baseline_results = {}
    baseline_comparison_rows = []

    for summary_group_name in SUMMARY_SET_ORDER:
        print("=" * 90)
        print(f"Building baseline result for summary set: {summary_group_name}")
        result = build_result_for_summary_group(summary_group_name, shared_draws, obs_mean_summary, obs_rep_summaries)
        result["adjusted"] = apply_regression_adjustment(
            result["accepted"], result["accepted_summary_matrix"], result["obs_group_mean"]
        )
        result["posterior_rejection"] = posterior_summary_table(result["accepted"], {k: k for k in PARAM_NAMES})
        result["posterior_adjusted"] = posterior_summary_table(result["adjusted"], dict(zip(PARAM_NAMES, ADJ_PARAM_NAMES)))

        group_outdir = OUTPUT_DIR / "baseline" / summary_group_name
        group_outdir.mkdir(parents=True, exist_ok=True)
        result["all_draws"].to_csv(group_outdir / "all_draws.csv", index=False)
        result["accepted"].to_csv(group_outdir / "accepted_rejection_abc.csv", index=False)
        result["adjusted"].to_csv(group_outdir / "accepted_adjusted_abc.csv", index=False)
        pd.DataFrame(result["accepted_summary_matrix"], columns=result["group_names"]).to_csv(
            group_outdir / "accepted_summary_matrix.csv", index=False
        )
        result["posterior_rejection"].to_csv(group_outdir / "posterior_summary_rejection.csv", index=False)
        result["posterior_adjusted"].to_csv(group_outdir / "posterior_summary_adjusted.csv", index=False)
        save_posterior_plots_baseline(result, group_outdir)

        # Use adjusted ABC for the baseline PPC since it is the strongest baseline version.
        adjusted_param_array = result["adjusted"][ADJ_PARAM_NAMES].to_numpy()
        ppc_idx = np.random.default_rng(MASTER_SEED + 111).choice(
            len(adjusted_param_array), size=min(N_PPC_SAMPLES, len(adjusted_param_array)), replace=False
        )
        result["ppc_adjusted"] = ppc_from_parameter_samples(
            adjusted_param_array[ppc_idx], infected_obs, rewiring_obs, degree_obs, group_outdir, suffix="adjusted"
        )

        meta = {
            "summary_group": summary_group_name,
            "summary_names": result["group_names"],
            "n_draws": N_DRAWS,
            "accept_fraction": ACCEPT_FRACTION,
            "n_accepted": int(len(result["accepted"])),
            "epsilon": float(result["epsilon"]),
            "shared_simulations_used": True,
        }
        (group_outdir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        baseline_row = {"summary_group": summary_group_name, "epsilon": result["epsilon"]}
        for _, r in result["posterior_rejection"].iterrows():
            baseline_row[f"rejection_mean_{r['parameter']}"] = r["mean"]
            baseline_row[f"rejection_sd_{r['parameter']}"] = r["sd"]
        for _, r in result["posterior_adjusted"].iterrows():
            baseline_row[f"adjusted_mean_{r['parameter']}"] = r["mean"]
            baseline_row[f"adjusted_sd_{r['parameter']}"] = r["sd"]
        baseline_row.update({f"ppc_{k}": v for k, v in result["ppc_adjusted"].items()})
        baseline_comparison_rows.append(baseline_row)
        all_baseline_results[summary_group_name] = result

    pd.DataFrame(baseline_comparison_rows).to_csv(OUTPUT_DIR / "baseline_summary_set_comparison.csv", index=False)

    # 2) Run the heavier methods on the strongest summary set only
    print("=" * 90)
    print(f"Running advanced methods on summary set: {ADVANCED_SUMMARY_GROUP}")
    adv_base = all_baseline_results[ADVANCED_SUMMARY_GROUP]
    adv_outdir = OUTPUT_DIR / "advanced_methods" / ADVANCED_SUMMARY_GROUP
    adv_outdir.mkdir(parents=True, exist_ok=True)

    advanced_summary = {}

    # Advanced benchmark: adjusted ABC on the full summary set
    adjusted_benchmark = adv_base["adjusted"][ADJ_PARAM_NAMES].copy()
    adjusted_benchmark.columns = PARAM_NAMES
    adjusted_benchmark_posterior = posterior_summary_table(adjusted_benchmark, {k: k for k in PARAM_NAMES})
    benchmark_idx = np.random.default_rng(MASTER_SEED + 222).choice(
        len(adjusted_benchmark), size=min(N_PPC_SAMPLES, len(adjusted_benchmark)), replace=False
    )
    benchmark_ppc = ppc_from_parameter_samples(
        adjusted_benchmark.to_numpy()[benchmark_idx], infected_obs, rewiring_obs, degree_obs,
        adv_outdir, suffix="adjusted_benchmark"
    )
    adjusted_benchmark.to_csv(adv_outdir / "adjusted_abc_benchmark_samples.csv", index=False)
    adjusted_benchmark_posterior.to_csv(adv_outdir / "adjusted_abc_benchmark_posterior.csv", index=False)
    advanced_summary["adjusted_abc_benchmark"] = {
        "posterior": adjusted_benchmark_posterior,
        "ppc": benchmark_ppc,
        "meta": {"source": "shared baseline adjusted ABC", "summary_group": ADVANCED_SUMMARY_GROUP},
    }

    # Advanced method 1: ABC-MCMC
    abc_mcmc_chain, abc_mcmc_kept, abc_mcmc_meta = abc_mcmc(
        obs_group_mean=adv_base["obs_group_mean"],
        scale=adv_base["scale"],
        n_replicates=n_observed_replicates,
        epsilon=ABC_MCMC_EPS_MULTIPLIER * adv_base["epsilon"],
        n_iter=ABC_MCMC_N_ITER,
        burn=ABC_MCMC_BURN,
        thin=ABC_MCMC_THIN,
        proposal_sd=ABC_MCMC_PROPOSAL_SD,
        seed=MASTER_SEED + 300,
    )
    abc_mcmc_chain.to_csv(adv_outdir / "abc_mcmc_chain.csv", index=False)
    abc_mcmc_kept.to_csv(adv_outdir / "abc_mcmc_kept_samples.csv", index=False)
    save_posterior_plots_samples(abc_mcmc_kept, adv_outdir, prefix="abc_mcmc")
    abc_mcmc_ppc_idx = np.random.default_rng(MASTER_SEED + 333).choice(
        len(abc_mcmc_kept), size=min(N_PPC_SAMPLES, len(abc_mcmc_kept)), replace=False
    )
    abc_mcmc_ppc = ppc_from_parameter_samples(
        abc_mcmc_kept[PARAM_NAMES].to_numpy()[abc_mcmc_ppc_idx], infected_obs, rewiring_obs, degree_obs,
        adv_outdir, suffix="abc_mcmc"
    )
    abc_mcmc_posterior = posterior_summary_table(abc_mcmc_kept, {k: k for k in PARAM_NAMES})
    abc_mcmc_posterior.to_csv(adv_outdir / "abc_mcmc_posterior.csv", index=False)
    advanced_summary["abc_mcmc"] = {"posterior": abc_mcmc_posterior, "ppc": abc_mcmc_ppc, "meta": abc_mcmc_meta}

    # Advanced method 2: SMC-ABC
    smc_df, smc_history = smc_abc(
        obs_group_mean=adv_base["obs_group_mean"],
        scale=adv_base["scale"],
        n_replicates=n_observed_replicates,
        n_particles=SMC_N_PARTICLES,
        n_rounds=SMC_N_ROUNDS,
        eps_quantile=SMC_EPS_QUANTILE,
        jitter_scale=SMC_JITTER_SCALE,
        seed=MASTER_SEED + 400,
    )
    smc_df.to_csv(adv_outdir / "smc_abc_final_particles.csv", index=False)
    smc_history.to_csv(adv_outdir / "smc_abc_history.csv", index=False)
    save_posterior_plots_samples(smc_df, adv_outdir, prefix="smc_abc", weighted=True, weight_col="weight")
    smc_rng = np.random.default_rng(MASTER_SEED + 444)
    smc_ppc_idx = weighted_resample_indices(smc_df["weight"].to_numpy(), size=min(SMC_FINAL_PPC_SAMPLES, len(smc_df)), rng=smc_rng)
    smc_ppc = ppc_from_parameter_samples(
        smc_df[PARAM_NAMES].to_numpy()[smc_ppc_idx], infected_obs, rewiring_obs, degree_obs,
        adv_outdir, suffix="smc_abc"
    )
    smc_posterior = posterior_summary_table(smc_df, {k: k for k in PARAM_NAMES})
    smc_posterior.to_csv(adv_outdir / "smc_abc_posterior.csv", index=False)
    advanced_summary["smc_abc"] = {
        "posterior": smc_posterior,
        "ppc": smc_ppc,
        "meta": {"n_particles": SMC_N_PARTICLES, "n_rounds": SMC_N_ROUNDS, "final_round_epsilon": float(smc_history.iloc[-1]["epsilon"] if len(smc_history) else np.nan)},
    }

    # Advanced method 3: synthetic likelihood MCMC
    sl_chain, sl_kept, sl_meta = synthetic_likelihood_mcmc(
        obs_group_mean=adv_base["obs_group_mean"],
        baseline_center=adjusted_benchmark[["beta", "gamma", "rho"]].mean().to_numpy(),
        n_replicates=n_observed_replicates,
        n_iter=SL_MCMC_N_ITER,
        burn=SL_MCMC_BURN,
        thin=SL_MCMC_THIN,
        proposal_sd=SL_PROPOSAL_SD,
        n_estimation_reps=SL_ESTIMATION_REPS,
        seed=MASTER_SEED + 500,
    )
    sl_chain.to_csv(adv_outdir / "synthetic_likelihood_chain.csv", index=False)
    sl_kept.to_csv(adv_outdir / "synthetic_likelihood_kept_samples.csv", index=False)
    save_posterior_plots_samples(sl_kept, adv_outdir, prefix="synthetic_likelihood")
    sl_ppc_idx = np.random.default_rng(MASTER_SEED + 555).choice(
        len(sl_kept), size=min(N_PPC_SAMPLES, len(sl_kept)), replace=False
    )
    sl_ppc = ppc_from_parameter_samples(
        sl_kept[PARAM_NAMES].to_numpy()[sl_ppc_idx], infected_obs, rewiring_obs, degree_obs,
        adv_outdir, suffix="synthetic_likelihood"
    )
    sl_posterior = posterior_summary_table(sl_kept, {k: k for k in PARAM_NAMES})
    sl_posterior.to_csv(adv_outdir / "synthetic_likelihood_posterior.csv", index=False)
    advanced_summary["synthetic_likelihood"] = {"posterior": sl_posterior, "ppc": sl_ppc, "meta": sl_meta}

    # 3) Save one comparison table across the advanced methods
    method_tables = []
    method_tables.append(method_summary_from_array(adjusted_benchmark.to_numpy(), "adjusted_abc_benchmark", ADVANCED_SUMMARY_GROUP))
    method_tables.append(method_summary_from_array(abc_mcmc_kept[PARAM_NAMES].to_numpy(), "abc_mcmc", ADVANCED_SUMMARY_GROUP))
    method_tables.append(method_summary_from_array(smc_df[PARAM_NAMES].to_numpy(), "smc_abc", ADVANCED_SUMMARY_GROUP))
    method_tables.append(method_summary_from_array(sl_kept[PARAM_NAMES].to_numpy(), "synthetic_likelihood", ADVANCED_SUMMARY_GROUP))
    pd.concat(method_tables, ignore_index=True).to_csv(OUTPUT_DIR / "advanced_method_comparison.csv", index=False)

    ppc_comp = pd.DataFrame([
        {"method": name, **info["ppc"]} for name, info in advanced_summary.items() if "ppc" in info
    ])
    ppc_comp.to_csv(OUTPUT_DIR / "advanced_method_ppc_comparison.csv", index=False)

    run_meta = {
        "shared_baseline": True,
        "baseline_draws": N_DRAWS,
        "accept_fraction": ACCEPT_FRACTION,
        "advanced_summary_group": ADVANCED_SUMMARY_GROUP,
        "simulator_settings": {
            "N": SIM_N,
            "p_edge": SIM_P_EDGE,
            "n_infected0": SIM_N_INFECTED0,
            "T": SIM_T,
        },
        "methods_included": [
            "rejection_abc",
            "regression_adjusted_abc",
            "abc_mcmc",
            "smc_abc",
            "synthetic_likelihood",
        ],
    }
    (OUTPUT_DIR / "run_metadata_all_methods.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    write_text_report(all_baseline_results, advanced_summary, OUTPUT_DIR)

    print("\nDone")

if __name__ == "__main__":
    main()