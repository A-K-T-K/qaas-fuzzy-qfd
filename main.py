import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tabulate import tabulate

# ==============================================================================
# GLOBAL CONFIGURATION & JOURNAL-GRADE PLOT SETTINGS
# ==============================================================================
OUTPUT_DIR = "outputs/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SHOW_TITLES = False  # Toggle True to render titles on plots, False for print-ready camera figures
FIGSIZE = (16 / 2.5, 9 / 2.5)  # Aspect ratio (Width x Height in inches)
DPI_RESOLUTION = 600  # High-resolution vector export

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "bold",
        "axes.edgecolor": "#222222",
        "axes.linewidth": 0.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": 9,
        "figure.autolayout": True,
    }
)

# ==============================================================================
# 1. SETUP DEFINITIONS & TFN LINGUISTIC MAPS
# ==============================================================================
TFN_MAP = {
    "S": np.array([0.6, 0.9, 1.0]),
    "M": np.array([0.2, 0.5, 0.8]),
    "W": np.array([0.0, 0.1, 0.4]),
    "--": np.array([0.0, 0.0, 0.2]),
}

CRISP_MAP = {"S": 9, "M": 3, "W": 1, "--": 0}

whats_data = [
    ("W1", "Computational acceleration for complex optimization", "Finance & Government", 5, "M"),
    ("W2", "Data sovereignty, cryptographic integrity, & export controls", "Finance & Government", 6, "M"),
    ("W3", "Platform quality, reproducibility, & accessible workflows", "Research", 5, "M"),
    ("W4", "Secure handling of sensitive patient/logistical data", "Healthcare & Logistics", 5, "M"),
    ("W5", "Decision support, routing optimization, & operational workflow", "Healthcare & Logistics", 4, "W"),
    ("W6", "Interoperability, runtime orchestration, & platform abstraction", "Service-Provider", 5, "M"),
]

capabilities_data = [
    ("C1", "Quantum API gateways & OpenAPI exposure"),
    ("C2", "Hybrid orchestration runtime layers"),
    ("C3", "Classical-quantum execution delay minimization"),
    ("C4", "Feed-forward control & multi-programming"),
    ("C5", "Multi-tenancy isolation controls"),
    ("C6", "Decentralized governance & compliance"),
    ("C7", "Secure service abstractions"),
    ("C8", "Quality-assessed software components & standards"),
]

R_matrix_linguistic = np.array([
    ["M", "S", "S", "M", "--", "--", "W", "W"],  # W1
    ["M", "W", "--", "--", "S", "S", "S", "M"],  # W2
    ["S", "M", "W", "W", "M", "W", "M", "S"],  # W3
    ["W", "W", "--", "--", "S", "M", "S", "W"],  # W4
    ["S", "S", "M", "M", "W", "--", "W", "M"],  # W5
    ["S", "S", "M", "M", "M", "M", "M", "S"],  # W6
])

# ==============================================================================
# 2. COMPUTATIONAL ENGINE & AGGREGATION
# ==============================================================================
w_fuzzy = np.array([TFN_MAP[row[4]] for row in whats_data])

R_fuzzy = np.zeros((6, 8, 3))
for i in range(6):
    for j in range(8):
        R_fuzzy[i, j] = TFN_MAP[R_matrix_linguistic[i, j]]

S_fuzzy = np.zeros((8, 3))
for j in range(8):
    S_fuzzy[j] = np.sum(w_fuzzy * R_fuzzy[:, j, :], axis=0)

S_crisp = np.mean(S_fuzzy, axis=1)
S_yager = (S_fuzzy[:, 0] + 2 * S_fuzzy[:, 1] + S_fuzzy[:, 2]) / 4
N_fuzzy = (S_crisp / np.sum(S_crisp)) * 100

# ==============================================================================
# 3. PRINT TABLE 1: Stakeholder Requirements
# ==============================================================================
print("=" * 90)
print("TABLE 1: Stakeholder requirements (WHATs), domain origins, counts, and weights")
print("=" * 90)
df_whats = pd.DataFrame(whats_data, columns=["ID", "Stakeholder requirement (WHAT)", "Domain origin", "Count", "Code"])
df_whats["TFN weight"] = [f"({TFN_MAP[c][0]:.1f}, {TFN_MAP[c][1]:.1f}, {TFN_MAP[c][2]:.1f})" for c in df_whats["Code"]]
print(tabulate(df_whats, headers="keys", tablefmt="grid", showindex=False))

# ==============================================================================
# 4. PRINT TABLE 2 & GENERATE FIGURE 1: QFD Relationship Heatmap
# ==============================================================================
print("\n" + "=" * 90)
print("TABLE 2: QFD relationship matrix (WHATs x Capabilities)")
print("=" * 90)
df_hoq = pd.DataFrame(
    R_matrix_linguistic,
    index=[w[0] for w in whats_data],
    columns=[c[0] for c in capabilities_data],
)
print(tabulate(df_hoq, headers="keys", tablefmt="grid", showindex=True))

# Figure 1: Density Heatmap
fig, ax = plt.subplots(figsize=(16 / 3, 9 / 3), dpi=DPI_RESOLUTION)
code_to_val = {"S": 3, "M": 2, "W": 1, "--": 0}
numeric_R = np.vectorize(code_to_val.get)(R_matrix_linguistic)

cax = ax.matshow(numeric_R, cmap="YlGnBu", vmin=0, vmax=3)

for i in range(6):
    for j in range(8):
        text_color = "white" if numeric_R[i, j] >= 2 else "black"
        ax.text(
            j, i, R_matrix_linguistic[i, j],
            ha="center", va="center", color=text_color, fontweight="bold", fontsize=10
        )

ax.set_xticks(range(8))
ax.set_yticks(range(6))
ax.set_xticklabels([c[0] for c in capabilities_data], fontsize=10, fontweight="bold")
ax.set_yticklabels([w[0] for w in whats_data], fontsize=10, fontweight="bold")
ax.xaxis.set_ticks_position("bottom")

if SHOW_TITLES:
    plt.title("QFD relationship matrix density heatmap", fontsize=12, pad=15, fontweight="bold")

plt.savefig(os.path.join(OUTPUT_DIR, "fig1_hoq_heatmap.pdf"), format="pdf", bbox_inches="tight")
plt.close()

# ==============================================================================
# 5. PRINT TABLE 3 & GENERATE FIGURE 2: Capability Ranking Bar Chart
# ==============================================================================
capability_results = []
for j in range(8):
    capability_results.append({
        "ID": capabilities_data[j][0],
        "Architectural capability": capabilities_data[j][1],
        "Fuzzy score": f"({S_fuzzy[j, 0]:.2f}, {S_fuzzy[j, 1]:.2f}, {S_fuzzy[j, 2]:.2f})",
        "Crisp": S_crisp[j],
        "Yager": S_yager[j],
        "Weight (%)": N_fuzzy[j],
    })

df_capabilities = pd.DataFrame(capability_results)
df_capabilities = df_capabilities.sort_values(by="Crisp", ascending=False).reset_index(drop=True)
df_capabilities["Rank"] = range(1, 9)

print("\n" + "=" * 90)
print("TABLE 3: Architectural capability (C_j) ranking and scores")
print("=" * 90)
print(tabulate(df_capabilities, headers="keys", tablefmt="grid", showindex=False))

top_2_ids = df_capabilities["ID"].iloc[:2].tolist()

# Figure 2: Prioritized Capability Ranking Bar Chart
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI_RESOLUTION)
df_sorted_plot = df_capabilities.sort_values(by="Weight (%)", ascending=True)

colors = ["#1f77b4" if id_ in top_2_ids else "#aec7e8" for id_ in df_sorted_plot["ID"]]
bars = ax.barh(
    df_sorted_plot["ID"] + " - " + df_sorted_plot["Architectural capability"],
    df_sorted_plot["Weight (%)"],
    color=colors,
    edgecolor="#333333",
    height=0.65,
)

for bar in bars:
    width = bar.get_width()
    ax.text(
        width + 0.2, bar.get_y() + bar.get_height() / 2,
        f"{width:.2f}%", ha="left", va="center", fontsize=9, fontweight="bold"
    )

ax.set_xlabel("Normalized weight ($N_j$ %)", fontsize=10, fontweight="bold")
ax.set_xlim(0, 18)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

if SHOW_TITLES:
    plt.title("Prioritized ranking and normalized weight distribution of QaaS capabilities", fontsize=12, fontweight="bold", pad=12)

plt.savefig(os.path.join(OUTPUT_DIR, "fig2_capability_ranking_bar.pdf"), format="pdf", bbox_inches="tight")
plt.close()

# ==============================================================================
# 6. PRINT TABLE 4: Crisp Benchmark Comparison & Spearman Correlation
# ==============================================================================
w_crisp = np.array([CRISP_MAP[row[4]] for row in whats_data])
R_crisp = np.zeros((6, 8))
for i in range(6):
    for j in range(8):
        R_crisp[i, j] = CRISP_MAP[R_matrix_linguistic[i, j]]

S_crisp_benchmark = np.dot(w_crisp, R_crisp)
N_crisp_benchmark = (S_crisp_benchmark / np.sum(S_crisp_benchmark)) * 100

df_comp = pd.DataFrame({
    "ID": [c[0] for c in capabilities_data],
    "Architectural capability description": [c[1] for c in capabilities_data],
    "Crisp weight (%)": N_crisp_benchmark,
    "Fuzzy weight (%)": N_fuzzy,
})

rho_benchmark, p_val_benchmark = spearmanr(N_crisp_benchmark, N_fuzzy)

print("\n" + "=" * 90)
print("TABLE 4: Crisp benchmark vs. fuzzy model comparison")
print("=" * 90)
print(tabulate(df_comp, headers="keys", tablefmt="grid", showindex=False))
print(f"\n[STATISTICAL METRIC] Spearman's Rank Correlation Coefficient (rho): {rho_benchmark:.4f}")
print(f"[STATISTICAL METRIC] Two-tailed p-value: {p_val_benchmark:.4e}")

# ==============================================================================
# 7. PRINT TABLE 5: Optimism-Weighted Index Trajectory across Alpha-Cuts
# ==============================================================================
alpha_vals = [0.00, 0.25, 0.50, 0.75, 1.00]
attitudes = [
    "Fully pessimistic (lower bound)",
    "Moderately pessimistic",
    "Hurwicz Midpoint Index (alpha=0.50)",
    "Moderately optimistic",
    "Fully optimistic (upper bound)",
]

alpha_rows = []
for a, att in zip(alpha_vals, attitudes):
    I_C1 = a * S_fuzzy[0, 2] + (1 - a) * S_fuzzy[0, 0]
    I_C7 = a * S_fuzzy[6, 2] + (1 - a) * S_fuzzy[6, 0]

    if np.isclose(I_C1, I_C7):
        r1, r2 = "Tie", "--"
    elif I_C1 > I_C7:
        r1, r2 = "C1", "C7"
    else:
        r1, r2 = "C7", "C1"

    alpha_rows.append({
        "Index (alpha)": f"{a:.2f}",
        "Attitude": att,
        "I_alpha(C1)": f"{I_C1:.2f}",
        "I_alpha(C7)": f"{I_C7:.2f}",
        "Rank 1": r1,
        "Rank 2": r2,
    })

df_alpha = pd.DataFrame(alpha_rows)
print("\n" + "=" * 90)
print("TABLE 5: Optimism-weighted index trajectory across alpha-cuts")
print("=" * 90)
print(tabulate(df_alpha, headers="keys", tablefmt="grid", showindex=False))

# ==============================================================================
# 8. GENERATE FIGURE 3: Sensitivity Boxplot for Concurrent Perturbations
# ==============================================================================
np.random.seed(42)
num_scenarios = 20
concurrent_weights = np.zeros((num_scenarios, 8))

for s in range(num_scenarios):
    delta = np.random.uniform(-0.10, 0.10, size=(6, 1))
    w_perturbed = w_fuzzy * (1 + delta)
    w_perturbed = np.clip(w_perturbed, 0.0, 1.0)

    S_pert = np.zeros((8, 3))
    for j in range(8):
        S_pert[j] = np.sum(w_perturbed * R_fuzzy[:, j, :], axis=0)

    crisp_p = np.mean(S_pert, axis=1)
    concurrent_weights[s] = (crisp_p / np.sum(crisp_p)) * 100

fig, ax = plt.subplots(figsize=(16 / 3, 9 / 3), dpi=DPI_RESOLUTION)

bp = ax.boxplot(
    [concurrent_weights[:, j] for j in range(8)],
    patch_artist=True,
    tick_labels=[c[0] for c in capabilities_data],
)

for box in bp["boxes"]:
    box.set(facecolor="#d9e5f5", color="#1f77b4", linewidth=1.2)
for median in bp["medians"]:
    median.set(color="#b22222", linewidth=1.5)

ax.set_ylabel("Normalized weight ($N_j$ %)", fontsize=10, fontweight="bold")
ax.set_xlabel("Architectural capability ($C_j$)", fontsize=10, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.spines["top"].set_visible(True)
ax.spines["right"].set_visible(True)

if SHOW_TITLES:
    plt.title("Distribution of capability weights across 20 concurrent perturbation scenarios", fontsize=12, fontweight="bold", pad=12)

# SAVED AS FIG3 TO MATCH MANUSCRIPT FIGURE 3 REFERENCE
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_sensitivity_concurrent_boxplot.pdf"), format="pdf", bbox_inches="tight")
plt.close()

# ==============================================================================
# 9. COMPLETE MANUSCRIPT SECTION 2.6 DIAGNOSTIC SENSITIVITY SUITE
# ==============================================================================
print("\n" + "=" * 90)
print("SECTION 2.6: MULTI-TIERED SENSITIVITY AND DIAGNOSTIC PROTOCOLS")
print("=" * 90)

# Diagnostic 1: Monte Carlo Matrix Noise
print("\n[DIAGNOSTIC 1] Monte Carlo Matrix Noise Perturbations (R_ij)")
print("Running 2,000 trials for noise levels k in {1, 3, 5} with non-zero shift guarantees...")

level_order = ["--", "W", "M", "S"]
idx_map = {l: i for i, l in enumerate(level_order)}
baseline_ranks = np.argsort(-S_crisp)

mc_results = []
for k_noise in [1, 3, 5]:
    top1_stable_count = 0
    top2_tier_stable_count = 0
    rhos = []

    for _ in range(2000):
        R_perturbed_ling = R_matrix_linguistic.copy()
        flat_indices = np.random.choice(48, size=k_noise, replace=False)
        
        for idx in flat_indices:
            r, c = divmod(idx, 8)
            curr_idx = idx_map[R_perturbed_ling[r, c]]
            
            possible_shifts = []
            if curr_idx > 0:
                possible_shifts.append(-1)
            if curr_idx < 3:
                possible_shifts.append(1)
            
            shift = np.random.choice(possible_shifts)
            new_idx = curr_idx + shift
            R_perturbed_ling[r, c] = level_order[new_idx]

        R_pert_fuz = np.array([
            [TFN_MAP[R_perturbed_ling[i, j]] for j in range(8)]
            for i in range(6)
        ])
        S_pert_fuz = np.sum(w_fuzzy[:, np.newaxis, :] * R_pert_fuz, axis=0)
        crisp_pert = np.mean(S_pert_fuz, axis=1)

        p_ranks = np.argsort(-crisp_pert)
        if p_ranks[0] == baseline_ranks[0]:
            top1_stable_count += 1
        if set(p_ranks[:2]) == set(baseline_ranks[:2]):
            top2_tier_stable_count += 1

        rho_trial, _ = spearmanr(S_crisp, crisp_pert)
        rhos.append(rho_trial)

    mc_results.append({
        "Noise level (k)": k_noise,
        "Trials": 2000,
        "Top-1 Stability (%)": f"{(top1_stable_count / 2000) * 100:.2f}%",
        "Top-2 Tier Stability (%)": f"{(top2_tier_stable_count / 2000) * 100:.2f}%",
        "Mean Spearman Rho": f"{np.mean(rhos):.4f}",
    })

df_mc = pd.DataFrame(mc_results)
print(tabulate(df_mc, headers="keys", tablefmt="grid", showindex=False))

# Diagnostic 2: Discrete Requirement Reclassifications
print("\n[DIAGNOSTIC 2] Discrete Requirement Reclassifications (W_i +/- 1 level)")

code_levels = ["W", "M", "S"]
code_idx_map = {"W": 0, "M": 1, "S": 2}
discrete_results = []

for i, row in enumerate(whats_data):
    req_id = row[0]
    curr_code = row[4]
    curr_idx = code_idx_map[curr_code]

    for direction, shift in [("-1 Level", -1), ("+1 Level", 1)]:
        new_idx = curr_idx + shift
        if 0 <= new_idx < len(code_levels):
            new_code = code_levels[new_idx]
            w_pert = w_fuzzy.copy()
            w_pert[i] = TFN_MAP[new_code]

            S_pert = np.sum(w_pert[:, np.newaxis, :] * R_fuzzy, axis=0)
            crisp_pert = np.mean(S_pert, axis=1)
            p_ranks = np.argsort(-crisp_pert)

            rho_val, _ = spearmanr(S_crisp, crisp_pert)
            discrete_results.append({
                "Requirement": req_id,
                "Shift": direction,
                "Original": curr_code,
                "Perturbed": new_code,
                "Rank 1": capabilities_data[p_ranks[0]][0],
                "Rank 2": capabilities_data[p_ranks[1]][0],
                "Spearman Rho": f"{rho_val:.4f}",
            })

df_discrete = pd.DataFrame(discrete_results)
print(tabulate(df_discrete, headers="keys", tablefmt="grid", showindex=False))

# Diagnostic 3: Single-Variable Relative Weight Scaling
print("\n[DIAGNOSTIC 3a] Single-Variable Weight Scaling (+/- 10%, 12 scenarios)")

single_var_results = []
for i in range(6):
    req_id = f"W{i+1}"
    for shift_factor in [-0.10, +0.10]:
        w_pert = w_fuzzy.copy()
        w_pert[i] = np.clip(w_pert[i] * (1 + shift_factor), 0.0, 1.0)

        S_pert = np.sum(w_pert[:, np.newaxis, :] * R_fuzzy, axis=0)
        crisp_pert = np.mean(S_pert, axis=1)
        p_ranks = np.argsort(-crisp_pert)

        rho_val, _ = spearmanr(S_crisp, crisp_pert)
        single_var_results.append({
            "Requirement": req_id,
            "Scaling": f"{shift_factor*100:+.0f}%",
            "Rank 1": capabilities_data[p_ranks[0]][0],
            "Rank 2": capabilities_data[p_ranks[1]][0],
            "Spearman Rho": f"{rho_val:.4f}",
        })

df_single_var = pd.DataFrame(single_var_results)
print(tabulate(df_single_var.head(6), headers="keys", tablefmt="grid", showindex=False))
print(f"... ({len(df_single_var)} total single-variable scenarios evaluated successfully)")

# Diagnostic 4: TFN Scale Constant Anchor Perturbations
print("\n[DIAGNOSTIC 4] TFN Scale Constant Anchor Perturbations (+/- 10%)")

anchor_results = []
for shift_factor in [-0.10, +0.10]:
    tfn_pert_map = {
        "S": np.clip(np.array([0.6, 0.9 * (1 + shift_factor), 1.0]), 0.0, 1.0),
        "M": np.clip(np.array([0.2, 0.5 * (1 + shift_factor), 0.8]), 0.0, 1.0),
        "W": np.clip(np.array([0.0, 0.1 * (1 + shift_factor), 0.4]), 0.0, 1.0),
        "--": np.array([0.0, 0.0, 0.2]),
    }

    w_p = np.array([tfn_pert_map[row[4]] for row in whats_data])
    R_p = np.zeros((6, 8, 3))
    for i in range(6):
        for j in range(8):
            R_p[i, j] = tfn_pert_map[R_matrix_linguistic[i, j]]

    S_p = np.sum(w_p[:, np.newaxis, :] * R_p, axis=0)
    crisp_p = np.mean(S_p, axis=1)
    p_ranks = np.argsort(-crisp_p)

    rho_val, _ = spearmanr(S_crisp, crisp_p)
    anchor_results.append({
        "Anchor Modal Shift": f"{shift_factor*100:+.0f}%",
        "Rank 1": capabilities_data[p_ranks[0]][0],
        "Rank 2": capabilities_data[p_ranks[1]][0],
        "Spearman Rho": f"{rho_val:.4f}",
    })

df_anchor = pd.DataFrame(anchor_results)
print(tabulate(df_anchor, headers="keys", tablefmt="grid", showindex=False))

print("\n[SUCCESS] Script execution complete.")