# ============================================================================
# src/recode.py
#
# Verification Channels and Remote Work — recoding utilities
#
# Turns raw Stack Overflow Developer Survey columns into the common analytic
# schema used across notebooks. Imported by 02_recode.ipynb and the analysis
# notebook. No file I/O here — pure DataFrame transforms.
#
# Recoding decisions (fixed during design):
#   RemoteWork -> remote3 (In-person / Hybrid / Remote); 2025 "Your choice" -> NaN + flag
#   AISelect   -> ai_use (binary Yes*=1 / No*=0); 2025 frequency kept in ai_freq
#   JobSat     -> jobsat (0-10, unchanged)
#   WorkExp    -> workexp (continuous control)
#   DevType    -> devgroup (collapsed coarse roles)
#   Country    -> country (level-2 grouping)
# ============================================================================
import numpy as np
import pandas as pd

# Core variables that must be non-missing for a row to enter the analytic sample.
CORE = ["jobsat", "ai_use", "remote3", "country", "workexp", "devgroup"]

# Minimum respondents per country to keep it as a level-2 group.
MIN_COUNTRY_N = 100


def _devgroup(x):
    """Collapse the many DevType strings into a small set of role groups."""
    if not isinstance(x, str):
        return np.nan
    xl = x.lower()
    if "student" in xl:
        return "Student"
    if "manager" in xl:
        return "Manager"
    if "data" in xl or "machine learning" in xl or "ai/ml" in xl:
        return "Data/ML"
    if "devops" in xl or "sre" in xl or "system admin" in xl:
        return "DevOps/Infra"
    if "developer" in xl:
        return "Developer"
    if "academic" in xl or "research" in xl:
        return "Research"
    return "Other"


def recode(df, year):
    """Map raw survey columns onto the common analytic schema for one year."""
    d = df.copy()

    # --- RemoteWork -> remote3 (3 categories) + flag for 2025 "Your choice"
    if year == 2024:
        rmap = {
            "In-person": "In-person",
            "Hybrid (some remote, some in-person)": "Hybrid",
            "Remote": "Remote",
        }
        d["remote3"] = d["RemoteWork"].map(rmap)
        d["yourchoice"] = 0
    elif year == 2025:
        rmap = {
            "In-person": "In-person",
            "Hybrid (some remote, leans heavy to in-person)": "Hybrid",
            "Hybrid (some in-person, leans heavy to flexibility)": "Hybrid",
            "Remote": "Remote",
        }
        d["remote3"] = d["RemoteWork"].map(rmap)          # "Your choice" -> NaN
        d["yourchoice"] = d["RemoteWork"].str.startswith("Your choice", na=False).astype(int)
    else:
        raise ValueError(f"Unsupported year: {year}")

    # --- AISelect -> binary use (Yes* = 1, No* = 0)
    d["ai_use"] = d["AISelect"].apply(
        lambda x: 1 if isinstance(x, str) and x.startswith("Yes")
        else (0 if isinstance(x, str) and x.startswith("No") else np.nan)
    )

    # --- 2025 only: preserve usage frequency for the exploratory intensity analysis
    if year == 2025:
        freq = {
            "Yes, I use AI tools daily": 3,
            "Yes, I use AI tools weekly": 2,
            "Yes, I use AI tools monthly or infrequently": 1,
        }
        d["ai_freq"] = d["AISelect"].map(freq)            # No* -> NaN (non-user)

    # --- Outcome and continuous control
    d["jobsat"] = pd.to_numeric(d["JobSat"], errors="coerce")
    d["workexp"] = pd.to_numeric(d["WorkExp"], errors="coerce")

    # --- Coarse role and country
    d["devgroup"] = d["DevType"].apply(_devgroup)
    d["country"] = d["Country"]

    d["year"] = year
    return d


def build_sample(d, min_country_n=MIN_COUNTRY_N):
    """Complete-case sample on CORE variables, dropping tiny countries."""
    s = d.dropna(subset=CORE).copy()
    vc = s["country"].value_counts()
    keep = vc[vc >= min_country_n].index
    return s[s["country"].isin(keep)].copy()


def build_robust_2025(r25, min_country_n=MIN_COUNTRY_N):
    """Robustness variant: absorb 2025 'Your choice' into Remote, then sample."""
    d = r25.copy()
    d.loc[d["yourchoice"] == 1, "remote3"] = "Remote"
    return build_sample(d, min_country_n=min_country_n)