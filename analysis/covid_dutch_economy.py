"""
Estimate the effect of COVID-19 on the Dutch economy.

Approach: Counterfactual analysis using pre-COVID GDP trends.
We fit a linear trend model on quarterly Dutch GDP (2015-2019),
project what GDP *would* have been in 2020-2021 absent COVID,
and compare to actual outcomes to estimate the cumulative loss.

Data source: CBS (Statistics Netherlands) — quarterly GDP at market prices,
volume index (2015=100), seasonally adjusted.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

# --------------------------------------------------------------------------
# 1. Dutch quarterly GDP data (volume index, 2015=100, seasonally adjusted)
#    Source: CBS StatLine — BBP, productie en bestedingen; kwartalen
#    https://opendata.cbs.nl/statline/
# --------------------------------------------------------------------------

gdp_data = {
    "quarter": [
        # 2015
        "2015Q1", "2015Q2", "2015Q3", "2015Q4",
        # 2016
        "2016Q1", "2016Q2", "2016Q3", "2016Q4",
        # 2017
        "2017Q1", "2017Q2", "2017Q3", "2017Q4",
        # 2018
        "2018Q1", "2018Q2", "2018Q3", "2018Q4",
        # 2019
        "2019Q1", "2019Q2", "2019Q3", "2019Q4",
        # 2020
        "2020Q1", "2020Q2", "2020Q3", "2020Q4",
        # 2021
        "2021Q1", "2021Q2", "2021Q3", "2021Q4",
        # 2022
        "2022Q1", "2022Q2", "2022Q3", "2022Q4",
    ],
    # Real GDP volume index (2015 = 100), seasonally adjusted
    # Values approximate CBS published figures
    "gdp_index": [
        # 2015
        99.1, 99.7, 100.3, 100.9,
        # 2016
        101.1, 101.8, 102.6, 103.5,
        # 2017
        104.4, 105.3, 106.1, 107.0,
        # 2018
        107.8, 108.5, 109.0, 109.5,
        # 2019
        110.0, 110.5, 110.9, 111.4,
        # 2020 — COVID shock
        109.5, 100.8, 107.6, 108.2,
        # 2021 — recovery
        107.5, 111.2, 113.0, 114.5,
        # 2022 — post-COVID
        115.0, 116.2, 116.5, 116.4,
    ],
}

# Dutch GDP in current euros (billions) for loss calculation
# Approximate annual nominal GDP (CBS)
NOMINAL_GDP_2019_BN = 813  # billion EUR

df = pd.DataFrame(gdp_data)
df["date"] = pd.PeriodIndex(df["quarter"], freq="Q").to_timestamp()
df["t"] = np.arange(len(df))  # time index

# --------------------------------------------------------------------------
# 2. Split into pre-COVID (training) and COVID/post-COVID (evaluation)
# --------------------------------------------------------------------------

pre_covid = df[df["date"] < "2020-01-01"].copy()
covid_period = df[df["date"] >= "2020-01-01"].copy()

# --------------------------------------------------------------------------
# 3. Fit counterfactual model on pre-COVID data
#    Model: GDP_index = β0 + β1*t + β2*t² + seasonal dummies
# --------------------------------------------------------------------------

pre_covid["t2"] = pre_covid["t"] ** 2
pre_covid["q"] = pre_covid["date"].dt.quarter

# Seasonal dummies (Q1 as reference)
seasonal = pd.get_dummies(pre_covid["q"], prefix="Q", drop_first=True, dtype=float)
X_train = pd.concat(
    [pre_covid[["t", "t2"]].reset_index(drop=True), seasonal.reset_index(drop=True)],
    axis=1,
)
X_train = sm.add_constant(X_train)
y_train = pre_covid["gdp_index"].values

model = sm.OLS(y_train, X_train).fit()

print("=" * 65)
print("COUNTERFACTUAL MODEL — Pre-COVID Dutch GDP (2015Q1–2019Q4)")
print("=" * 65)
print(model.summary())

# --------------------------------------------------------------------------
# 4. Predict counterfactual GDP for 2020-2022
# --------------------------------------------------------------------------

covid_period = covid_period.copy()
covid_period["t2"] = covid_period["t"] ** 2
covid_period["q"] = covid_period["date"].dt.quarter

seasonal_pred = pd.get_dummies(covid_period["q"], prefix="Q", drop_first=True, dtype=float)
X_pred = pd.concat(
    [covid_period[["t", "t2"]].reset_index(drop=True), seasonal_pred.reset_index(drop=True)],
    axis=1,
)
X_pred = sm.add_constant(X_pred)

# Point predictions + confidence intervals
predictions = model.get_prediction(X_pred)
pred_summary = predictions.summary_frame(alpha=0.05)

covid_period = covid_period.reset_index(drop=True)
covid_period["counterfactual"] = pred_summary["mean"].values
covid_period["cf_lower"] = pred_summary["obs_ci_lower"].values
covid_period["cf_upper"] = pred_summary["obs_ci_upper"].values

# GDP gap = actual - counterfactual
covid_period["gap"] = covid_period["gdp_index"] - covid_period["counterfactual"]
covid_period["gap_pct"] = (covid_period["gap"] / covid_period["counterfactual"]) * 100

# --------------------------------------------------------------------------
# 5. Estimate cumulative GDP loss
# --------------------------------------------------------------------------

# Focus on 2020Q1-2021Q4 as the core COVID impact window
covid_window = covid_period[covid_period["date"] < "2022-01-01"]

avg_gap_pct = covid_window["gap_pct"].mean()
# Quarterly nominal GDP ≈ annual / 4
quarterly_nominal = NOMINAL_GDP_2019_BN / 4
cumulative_loss_bn = -(covid_window["gap_pct"] / 100 * quarterly_nominal).sum()

print("\n")
print("=" * 65)
print("ESTIMATED COVID IMPACT ON DUTCH GDP (2020Q1–2021Q4)")
print("=" * 65)

for _, row in covid_window.iterrows():
    print(
        f"  {row['quarter']:8s}  actual={row['gdp_index']:6.1f}  "
        f"counterfactual={row['counterfactual']:6.1f}  "
        f"gap={row['gap_pct']:+5.1f}%"
    )

print(f"\n  Average quarterly GDP gap:   {avg_gap_pct:+.1f}%")
print(f"  Cumulative GDP loss (2020-2021): ~€{cumulative_loss_bn:.0f} billion")
print(f"  (relative to 2019 nominal GDP of €{NOMINAL_GDP_2019_BN}bn)")

# Worst quarter
worst = covid_window.loc[covid_window["gap_pct"].idxmin()]
print(f"\n  Worst quarter: {worst['quarter']} — GDP {worst['gap_pct']:+.1f}% below trend")

# --------------------------------------------------------------------------
# 6. Statistical test: structural break at 2020Q1 (Chow test)
# --------------------------------------------------------------------------

# Fit model on full sample
df_full = df.copy()
df_full["t2"] = df_full["t"] ** 2
df_full["q"] = df_full["date"].dt.quarter
seasonal_full = pd.get_dummies(df_full["q"], prefix="Q", drop_first=True, dtype=float)
X_full = pd.concat(
    [df_full[["t", "t2"]].reset_index(drop=True), seasonal_full.reset_index(drop=True)],
    axis=1,
)
X_full = sm.add_constant(X_full)
y_full = df_full["gdp_index"].values
model_full = sm.OLS(y_full, X_full).fit()

# Chow test: compare RSS of pooled vs. separate regressions
RSS_pooled = model_full.ssr
RSS_pre = model.ssr

# Post-COVID regression
X_post = X_pred.copy()
y_post = covid_period["gdp_index"].values
model_post = sm.OLS(y_post, X_post).fit()
RSS_post = model_post.ssr

k = X_train.shape[1]  # number of parameters
n = len(y_full)

F_stat = ((RSS_pooled - (RSS_pre + RSS_post)) / k) / ((RSS_pre + RSS_post) / (n - 2 * k))
p_value = 1 - stats.f.cdf(F_stat, k, n - 2 * k)

print(f"\n  Chow test for structural break at 2020Q1:")
print(f"    F-statistic = {F_stat:.2f}")
print(f"    p-value     = {p_value:.4f}")
if p_value < 0.05:
    print("    → Reject H0: significant structural break (COVID shock confirmed)")
else:
    print("    → Fail to reject H0 at 5% level")

# --------------------------------------------------------------------------
# 7. Plot results
# --------------------------------------------------------------------------

fig, axes = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [3, 1]})

# --- Top panel: GDP actual vs counterfactual ---
ax1 = axes[0]

# Pre-COVID actual
ax1.plot(
    pre_covid["date"], pre_covid["gdp_index"],
    "o-", color="#2c3e50", markersize=4, label="Actual GDP (pre-COVID)", linewidth=1.5,
)

# COVID-period actual
ax1.plot(
    covid_period["date"], covid_period["gdp_index"],
    "o-", color="#e74c3c", markersize=5, label="Actual GDP (COVID & post)", linewidth=2,
)

# Counterfactual
ax1.plot(
    covid_period["date"], covid_period["counterfactual"],
    "--", color="#3498db", linewidth=2, label="Counterfactual (no-COVID trend)",
)

# Confidence band
ax1.fill_between(
    covid_period["date"],
    covid_period["cf_lower"],
    covid_period["cf_upper"],
    alpha=0.15, color="#3498db", label="95% prediction interval",
)

# Shade the gap for 2020
covid_2020 = covid_period[covid_period["date"] < "2021-01-01"]
ax1.fill_between(
    covid_2020["date"],
    covid_2020["gdp_index"],
    covid_2020["counterfactual"],
    alpha=0.3, color="#e74c3c", label="GDP loss (2020)",
)

# COVID onset line
ax1.axvline(pd.Timestamp("2020-03-01"), color="grey", linestyle=":", alpha=0.7)
ax1.text(
    pd.Timestamp("2020-03-15"), ax1.get_ylim()[0] + 1,
    "First lockdown\n(Mar 2020)", fontsize=8, color="grey", va="bottom",
)

ax1.set_ylabel("GDP Volume Index (2015 = 100)", fontsize=11)
ax1.set_title(
    "Effect of COVID-19 on the Dutch Economy\n"
    "Counterfactual analysis based on pre-COVID trend (2015–2019)",
    fontsize=13, fontweight="bold",
)
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, alpha=0.3)

# --- Bottom panel: GDP gap (%) ---
ax2 = axes[1]

colors = ["#e74c3c" if g < 0 else "#27ae60" for g in covid_period["gap_pct"]]
ax2.bar(covid_period["date"], covid_period["gap_pct"], width=60, color=colors, alpha=0.8)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_ylabel("GDP Gap (%)", fontsize=11)
ax2.set_xlabel("Quarter", fontsize=11)
ax2.set_title("Deviation from counterfactual trend", fontsize=11)
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("analysis/covid_dutch_economy.png", dpi=150, bbox_inches="tight")
plt.savefig("analysis/covid_dutch_economy.pdf", bbox_inches="tight")
print("\n\nFigures saved to analysis/covid_dutch_economy.png and .pdf")
print("=" * 65)
