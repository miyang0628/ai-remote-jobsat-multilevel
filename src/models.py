# ============================================================================
# src/models.py
#
# Verification Channels and Remote Work — multilevel model utilities
#
# Pure-Python mixed models (statsmodels), imported by 03_analysis.ipynb.
# Returns tidy result rows/frames; no file I/O here.
#
# Three statsmodels MixedLM pitfalls handled throughout:
#   (1) standardize the continuous control -> random-slope convergence
#   (2) use ONE optimizer for all models -> valid likelihood-ratio tests
#   (3) compute df from parameter counts (df_modelwc is unreliable)
# ============================================================================
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2
import warnings
warnings.filterwarnings("ignore")
from recode import build_sample

# All mixed models use the SAME optimizer so their log-likelihoods are comparable.
OPT = "powell"

REMOTE_LEVELS = ["In-person", "Hybrid", "Remote"]


def prep(d):
    """Set reference categories and standardize the continuous control."""
    d = d.copy()
    d["remote3"] = pd.Categorical(d["remote3"], REMOTE_LEVELS)
    d["devgroup"] = d["devgroup"].astype("category")
    # (1) standardization is what makes the random slope converge
    d["workexp_z"] = (d["workexp"] - d["workexp"].mean()) / d["workexp"].std()
    keep = ["jobsat", "ai_use", "remote3", "country", "workexp_z", "devgroup"]
    return d.dropna(subset=keep)


def _fit(formula, d, re_formula=None):
    return smf.mixedlm(formula, d, groups=d["country"],
                       re_formula=re_formula).fit(reml=False, method=OPT)


def fit_all(d_raw):
    """Fit H1/H2/H3 models and the interaction-only reduced model. Returns dict."""
    d = prep(d_raw)
    F = "jobsat ~ {} + workexp_z + C(devgroup)"
    m1 = _fit(F.format("ai_use"), d)                                  # H1 main
    m2 = _fit(F.format("ai_use*C(remote3)"), d)                       # H2 interaction (rand. intercept)
    m_red = _fit("jobsat ~ ai_use + C(remote3) + workexp_z + C(devgroup)", d)  # reduced (no interaction)
    m3 = _fit(F.format("ai_use*C(remote3)"), d, re_formula="~ai_use")  # H3 random slope
    return {"d": d, "m1": m1, "m2": m2, "m_red": m_red, "m3": m3}


def lrt(m_small, m_big):
    """Likelihood-ratio test; df from parameter-count difference."""
    df = len(m_big.params) - len(m_small.params)
    stat = 2 * (m_big.llf - m_small.llf)
    return stat, df, chi2.sf(stat, df)


def summarize_year(d_raw, label):
    """Return (summary_rows_df, fitted_models) for one year."""
    fit = fit_all(d_raw)
    m1, m2, m_red, m3, d = fit["m1"], fit["m2"], fit["m_red"], fit["m3"], fit["d"]
    sd_y = d["jobsat"].std()
    rows = []

    # H1 main effect
    rows.append(dict(year=label, test="H1_main_ai_use", term="ai_use",
                     coef=m1.params["ai_use"], p=m1.pvalues["ai_use"]))

    # H2 interaction terms + interaction-only LRT
    ci = m2.conf_int()
    for term in [t for t in m2.params.index if "ai_use:" in t]:
        lo, hi = ci.loc[term, 0], ci.loc[term, 1]
        rows.append(dict(year=label, test="H2_interaction", term=term,
                         coef=m2.params[term], p=m2.pvalues[term],
                         ci_lo=lo, ci_hi=hi, max_abs_d=max(abs(lo), abs(hi)) / sd_y))
    stat, df, pval = lrt(m_red, m2)
    rows.append(dict(year=label, test="H2_interaction_LRT",
                     term=f"chi2(df={df})", coef=stat, p=pval))

    # H3 random slope of ai_use across countries
    slope_var = m3.cov_re.iloc[1, 1]
    stat3 = 2 * (m3.llf - m2.llf)
    p3 = 0.5 * chi2.sf(stat3, 1) + 0.5 * chi2.sf(stat3, 2)  # boundary mixture
    rows.append(dict(year=label, test="H3_random_slope", term="var(ai_use|country)",
                     coef=slope_var, p=p3))

    return pd.DataFrame(rows), fit


def intensity_2025(d_raw):
    """
    Exploratory (2025 only): does AI-use FREQUENCY moderate the link?
    Requires an 'ai_freq' column (non-users -> 0). Returns (summary_df, simple_slopes_df).
    """
    d = d_raw.copy()
    d["ai_freq"] = d["ai_freq"].fillna(0)  # No* -> 0 (non-user)
    d["remote3"] = pd.Categorical(d["remote3"], REMOTE_LEVELS)
    d["devgroup"] = d["devgroup"].astype("category")
    d = d.dropna(subset=["jobsat", "remote3", "country", "workexp", "devgroup"])
    d["workexp_z"] = (d["workexp"] - d["workexp"].mean()) / d["workexp"].std()

    rows = []

    # Full-sample interaction (continuous ai_freq) vs reduced
    red = _fit("jobsat ~ ai_freq + C(remote3) + workexp_z + C(devgroup)", d)
    full = _fit("jobsat ~ ai_freq*C(remote3) + workexp_z + C(devgroup)", d)
    stat, df, pval = lrt(red, full)
    rows.append(dict(spec="continuous_all", stat=stat, df=df, p=pval, n=len(d)))

    # Users-only (drop non-users) -> intensity net of use/non-use jump
    users = d[d["ai_freq"] > 0].copy()
    users["workexp_z"] = (users["workexp"] - users["workexp"].mean()) / users["workexp"].std()
    red_u = _fit("jobsat ~ ai_freq + C(remote3) + workexp_z + C(devgroup)", users)
    full_u = _fit("jobsat ~ ai_freq*C(remote3) + workexp_z + C(devgroup)", users)
    stat, df, pval = lrt(red_u, full_u)
    rows.append(dict(spec="continuous_users_only", stat=stat, df=df, p=pval, n=len(users)))

    # Categorical ai_freq (drop equal-interval assumption)
    d["ai_freq_c"] = d["ai_freq"].astype(int).astype("category")
    red_c = _fit("jobsat ~ C(ai_freq_c) + C(remote3) + workexp_z + C(devgroup)", d)
    full_c = _fit("jobsat ~ C(ai_freq_c)*C(remote3) + workexp_z + C(devgroup)", d)
    stat, df, pval = lrt(red_c, full_c)
    rows.append(dict(spec="categorical", stat=stat, df=df, p=pval, n=len(d)))

    # Simple slopes: ai_freq effect within each remote3 level
    slopes = []
    for lvl in REMOTE_LEVELS:
        order = [lvl] + [x for x in REMOTE_LEVELS if x != lvl]
        d["r_rel"] = pd.Categorical(d["remote3"], order)
        m = _fit("jobsat ~ ai_freq*C(r_rel) + workexp_z + C(devgroup)", d)
        slopes.append(dict(remote3=lvl, slope=m.params["ai_freq"], p=m.pvalues["ai_freq"]))

    return pd.DataFrame(rows), pd.DataFrame(slopes)
    
def intensity_predictions_2025(d_raw):
    """
    Model-based predicted jobsat for the intensity figure (2025, exploratory).

    Fits jobsat ~ ai_freq * C(remote3) + workexp_z + C(devgroup) with a random
    intercept, then predicts at each (remote3, ai_freq) cell holding controls at
    reference/mean: workexp_z = 0 and devgroup = the most frequent group. This
    keeps intercept differences between work arrangements (unlike drawing common
    simple slopes), so the lines reflect actual fitted levels.

    Returns a tidy DataFrame: remote3, ai_freq, pred.
    """
    d = d_raw.copy()
    d["ai_freq"] = d["ai_freq"].fillna(0)  # No* -> 0 (non-user)
    d["remote3"] = pd.Categorical(d["remote3"], REMOTE_LEVELS)
    d["devgroup"] = d["devgroup"].astype("category")
    d = d.dropna(subset=["jobsat", "remote3", "country", "workexp", "devgroup"])
    d["workexp_z"] = (d["workexp"] - d["workexp"].mean()) / d["workexp"].std()

    m = _fit("jobsat ~ ai_freq*C(remote3) + workexp_z + C(devgroup)", d)

    # Hold controls fixed: workexp_z = 0 (its mean), devgroup = modal category.
    modal_dev = d["devgroup"].value_counts().idxmax()
    grid = pd.DataFrame(
        [(lvl, f) for lvl in REMOTE_LEVELS for f in [0, 1, 2, 3]],
        columns=["remote3", "ai_freq"],
    )
    grid["workexp_z"] = 0.0
    grid["devgroup"] = modal_dev
    grid["remote3"] = pd.Categorical(grid["remote3"], REMOTE_LEVELS)
    grid["devgroup"] = pd.Categorical(grid["devgroup"], categories=d["devgroup"].cat.categories)

    # Fixed-effects prediction (population-level; random intercept = 0)
    grid["pred"] = m.predict(exog=grid)
    return grid[["remote3", "ai_freq", "pred"]]
        
def descriptive_table(d_raw, label):
    """
    Table 1 building block: per-variable summary for one analytic sample.
    Continuous vars -> mean/SD/min/max/missing; categorical -> category shares.
    Uses the same complete-case CORE sample as the models (via prep).
    """
    d = prep(d_raw)  # same sample the models run on
    rows = []
    n = len(d)
    rows.append(dict(variable="N (analytic sample)", stat=f"{n}"))
    rows.append(dict(variable="Countries (level-2)", stat=f"{d['country'].nunique()}"))

    # Continuous: jobsat, workexp (report on raw workexp, not the z-scored one)
    for v, name in [("jobsat", "Job satisfaction (0-10)"),
                    ("workexp", "Work experience (years)")]:
        s = d[v]
        rows.append(dict(variable=name,
                         stat=f"M={s.mean():.2f}, SD={s.std():.2f}, "
                              f"range=[{s.min():.0f}, {s.max():.0f}]"))

    # Binary: ai_use
    p_use = d["ai_use"].mean() * 100
    rows.append(dict(variable="AI use = Yes (%)", stat=f"{p_use:.1f}%"))

    # Categorical shares: remote3, devgroup
    for v, name in [("remote3", "Work arrangement"), ("devgroup", "Role group")]:
        vc = (d[v].value_counts(normalize=True) * 100).round(1)
        share = ", ".join(f"{k} {v_:.1f}%" for k, v_ in vc.items())
        rows.append(dict(variable=name, stat=share))

    out = pd.DataFrame(rows)
    out.insert(0, "year", label)
    return out


def missingness_table(d_recoded, label):
    """
    Missingness on the raw analytic variables BEFORE complete-case filtering,
    so readers see how the analytic sample was formed. Input: recoded frame
    (r24 / r25), not the filtered sample.
    """
    vars_ = ["jobsat", "ai_use", "remote3", "country", "workexp", "devgroup"]
    rows = []
    for v in vars_:
        miss = d_recoded[v].isna().mean() * 100
        rows.append(dict(year=label, variable=v, missing_pct=round(miss, 1)))
    return pd.DataFrame(rows)


def correlations_and_vif(d_raw, label):
    """
    Numeric correlation matrix among model predictors and VIF for collinearity.
    Categorical predictors are dummy-coded (remote3, devgroup) to compute VIF on
    the actual design matrix the model uses.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import numpy as np

    d = prep(d_raw)
    # Correlation among the continuous/binary terms
    num = d[["jobsat", "ai_use", "workexp_z"]].copy()
    corr = num.corr().round(3)
    corr.insert(0, "year", label)

    # VIF on the model design matrix (predictors only, no outcome)
    X = pd.get_dummies(
        d[["ai_use", "remote3", "workexp_z", "devgroup"]],
        columns=["remote3", "devgroup"], drop_first=True
    ).astype(float)
    X["_const"] = 1.0
    vif_rows = []
    for i, col in enumerate(X.columns):
        if col == "_const":
            continue
        vif = variance_inflation_factor(X.values, i)
        vif_rows.append(dict(year=label, term=col, VIF=round(vif, 2)))
    return corr, pd.DataFrame(vif_rows)


def icc_country(d_raw, label):
    """
    Intraclass correlation: share of jobsat variance at the country level.
    Fitted from a null (intercept-only) multilevel model.
    ICC = var(country) / (var(country) + residual var).
    """
    d = prep(d_raw)
    m0 = smf.mixedlm("jobsat ~ 1", d, groups=d["country"]).fit(reml=True, method=OPT)
    var_country = float(m0.cov_re.iloc[0, 0])
    var_resid = float(m0.scale)
    icc = var_country / (var_country + var_resid)
    return pd.DataFrame([dict(year=label,
                              var_country=round(var_country, 4),
                              var_resid=round(var_resid, 4),
                              ICC=round(icc, 4))])


def sensitivity_country_filter(d_raw, label, thresholds=(50, 100, 200)):
    """
    Sensitivity of the H2 interaction-only LRT to the minimum-country-N filter.
    Refits at each threshold and reports N, #countries, and the clean LRT p.
    """
    rows = []
    for thr in thresholds:
        s = build_sample(d_raw, min_country_n=thr)
        d = prep(s)
        red = _fit("jobsat ~ ai_use + C(remote3) + workexp_z + C(devgroup)", d)
        full = _fit("jobsat ~ ai_use*C(remote3) + workexp_z + C(devgroup)", d)
        stat, df, pval = lrt(red, full)
        rows.append(dict(year=label, min_country_n=thr, N=len(d),
                         countries=d["country"].nunique(),
                         H2_LRT_chi2=round(stat, 3), H2_LRT_p=round(pval, 4)))
    return pd.DataFrame(rows)