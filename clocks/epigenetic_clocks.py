"""
epigenetic_clocks.py
=====================

Generic infrastructure for elastic-net-style DNA methylation ("epigenetic")
age clocks.

Design
------
Most published epigenetic clocks (Horvath 2013, Hannum 2013, PhenoAge 2018,
GrimAge V2 2022, etc.) share the same basic shape:

    1. A (possibly non-linear) transform applied to chronological age
       during model training, and its inverse applied to the model's
       linear predictor to recover an age estimate.
    2. A linear combination of CpG beta-values with published
       coefficients + an intercept ("elastic-net" style prediction).
    3. A common set of evaluation metrics (correlation with chronological
       age, median absolute error, median age acceleration).
    4. Validation of a re-implementation against numbers published in the
       original paper / supplementary tables.

"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


# =====================================================================
# GENERIC — age transformation utilities
# =====================================================================
def log_linear_trafo(
    age: "float | np.ndarray",
    adult_age: float = 20.0,
) -> "float | np.ndarray":
    """
    Log-linear age transformation used by Horvath-style clocks.

    Compresses the age scale for individuals at or below ``adult_age``
    (log scale, since biological change is fast in childhood) and uses a
    linear scale above it. Works on scalars or array-likes.

    Parameters
    ----------
    age : float or array-like
        Chronological age(s) in years.
    adult_age : float
        Inflection point of the transformation (Horvath 2013 uses 20).

    Returns
    -------
    float or np.ndarray
        Transformed age(s), matching the shape of the input.
    """
    scalar_input = np.isscalar(age)
    age_arr = np.asarray(age, dtype=float)

    out = np.where(
        age_arr <= adult_age,
        np.log((age_arr + 1) / (adult_age + 1)),
        (age_arr - adult_age) / (adult_age + 1),
    )
    return float(out) if scalar_input else out


def log_linear_anti_trafo(
    x: "float | np.ndarray",
    adult_age: float = 20.0,
) -> "float | np.ndarray":
    """
    Inverse of :func:`log_linear_trafo`. Converts a model's linear
    predictor back into an age estimate in years.

    Works on scalars or array-likes.
    """
    scalar_input = np.isscalar(x)
    x_arr = np.asarray(x, dtype=float)

    out = np.where(
        x_arr < 0,
        (adult_age + 1) * np.exp(x_arr) - 1,
        (adult_age + 1) * x_arr + adult_age,
    )
    return float(out) if scalar_input else out


# =====================================================================
# GENERIC — linear-combination ("elastic net") clock predictor
# =====================================================================
def predict_methylation_age(
    methylation: pd.DataFrame,
    coefficients: pd.DataFrame,
    cpg_col: str = "CpGmarker",
    coef_col: str = "Coefficient",
    intercept_label: str = "(Intercept)",
    impute_missing: bool = True,
    impute_beta: float = 0.5,
    anti_trafo_fn: Optional[Callable] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Predict methylation ("epigenetic") age from a linear combination of
    CpG beta-values and published clock coefficients.

    This is the shared computational core of essentially every published
    linear epigenetic clock: ``age_hat = f(intercept + sum(coef_i * beta_i))``
    where ``f`` is an optional inverse-transform (identity if the clock
    was trained directly in years, e.g. Hannum; a log-linear anti-trafo
    for Horvath-style clocks).

    Parameters
    ----------
    methylation : pd.DataFrame
        Beta-value matrix, shape (n_CpGs, n_samples). Index = CpG probe
        IDs, columns = sample IDs. Values expected in [0, 1].
    coefficients : pd.DataFrame
        Clock coefficient table with columns ``cpg_col`` and ``coef_col``.
        Must contain one row with ``cpg_col == intercept_label`` holding
        the model intercept.
    cpg_col : str
        Column name for CpG identifiers.
    coef_col : str
        Column name for coefficient values.
    intercept_label : str
        Value in ``cpg_col`` that marks the intercept row.
    impute_missing : bool
        If True, CpGs absent from ``methylation`` are imputed with
        ``impute_beta`` (population-mean-like beta). If False, missing
        sites are simply skipped (equivalent to a zero contribution).
    impute_beta : float
        Beta value used to impute missing CpGs when ``impute_missing``.
    anti_trafo_fn : callable, optional
        Function mapping the raw linear predictor to an age estimate.
        Defaults to the identity function (i.e. the model output is
        already in years).
    verbose : bool
        Print a short report of missing CpGs.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``SampleID`` with a single ``DNAmAge`` column.
    """
    if anti_trafo_fn is None:
        anti_trafo_fn = lambda x: x  # noqa: E731 — identity by default

    # ── 1. Extract intercept ────────────────────────────────────────
    intercept_mask = coefficients[cpg_col] == intercept_label
    if not intercept_mask.any():
        raise ValueError(f"No '{intercept_label}' row found in coefficients table.")

    intercept = coefficients.loc[intercept_mask, coef_col].values[0]
    cpg_df = coefficients[~intercept_mask].copy()

    # ── 2. Check for missing CpGs ───────────────────────────────────
    all_cpgs = set(cpg_df[cpg_col])
    present_cpgs = all_cpgs & set(methylation.index)
    missing_cpgs = all_cpgs - present_cpgs

    if missing_cpgs and verbose:
        pct = 100 * len(missing_cpgs) / len(all_cpgs)
        print(
            f"\u26a0  {len(missing_cpgs)} / {len(all_cpgs)} clock CpGs "
            f"({pct:.1f}%) not found in methylation data."
        )
        if not impute_missing:
            print("   Proceeding with available sites only.")

    # ── 3. Compute linear predictor ─────────────────────────────────
    n_samples = methylation.shape[1]
    linear_pred = np.full(n_samples, intercept, dtype=float)

    present_coef_df = cpg_df[cpg_df[cpg_col].isin(present_cpgs)]
    if len(present_coef_df):
        sub_matrix = methylation.loc[present_coef_df[cpg_col].values].values
        linear_pred += present_coef_df[coef_col].values @ sub_matrix

    if impute_missing and missing_cpgs:
        missing_coef_df = cpg_df[cpg_df[cpg_col].isin(missing_cpgs)]
        linear_pred += missing_coef_df[coef_col].sum() * impute_beta

    # ── 4. Apply anti-transformation ────────────────────────────────
    ages = anti_trafo_fn(linear_pred)

    return pd.DataFrame(
        {"SampleID": methylation.columns, "DNAmAge": ages}
    ).set_index("SampleID")


# =====================================================================
# GENERIC — per-CpG imputation of missing values
# =====================================================================

def build_cpg_reference(
    datasets: Union[Dict[str, pd.DataFrame], List[pd.DataFrame]],
    cpgs: Optional[set] = None,
) -> pd.Series:
    """
    Per-CpG reference beta values for imputing CpGs that are missing in every
    sample of a dataset.

    Each CpG's reference is the average of its per-dataset medians, so every
    dataset counts equally regardless of its size. A CpG absent from all
    datasets gets NaN (the clock's own fallback then applies).

    Parameters
    ----------
    datasets : dict or list of DataFrames shaped (n_CpGs, n_samples),
        CpG IDs as index. NaNs allowed.
    cpgs : set, optional
        Only compute the reference for these CpGs (e.g. ``clock_cpgs(clocks)``).
        Strongly recommended: clocks use ~1,000 CpGs, arrays have ~450–865k.

    Returns
    -------
    pd.Series indexed by CpG ID.
    """
    frames = datasets.values() if isinstance(datasets, dict) else datasets
    medians = []
    for df in frames:
        sub = df if cpgs is None else df.loc[df.index.intersection(list(cpgs))]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN rows -> NaN
            med = np.nanmedian(sub.to_numpy(dtype=float), axis=1)
        medians.append(pd.Series(med, index=sub.index))
    if not medians:
        return pd.Series(dtype=float)
    return pd.concat(medians, axis=1).mean(axis=1)


def impute_per_cpg(
    methylation: pd.DataFrame,
    reference: Optional[pd.Series] = None,
    within_cohort: bool = True,
) -> pd.DataFrame:
    """
    Fill missing beta values CpG by CpG (never with a per-sample value).

    1. within_cohort : a CpG missing in some samples gets that CpG's median
                       across the dataset's other samples.
    2. reference     : a CpG still missing (i.e. missing in every sample of the
                       dataset) gets its value from ``reference``
                       (see :func:`build_cpg_reference`).

    CpGs that remain all-NaN are left as NaN; :func:`run_all_clocks` drops
    them, so the clock treats them as missing and applies its fallback.

    Parameters
    ----------
    methylation : DataFrame (n_CpGs, n_samples), CpG IDs as index.
    reference   : Series indexed by CpG ID, optional.
    within_cohort : bool

    Returns
    -------
    DataFrame of the same shape.
    """
    arr = methylation.to_numpy(dtype=float, copy=True)        # never modifies the input
    if within_cohort:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN rows -> NaN
            med = np.nanmedian(arr, axis=1)
        rows, cols = np.where(np.isnan(arr))
        arr[rows, cols] = med[rows]
    if reference is not None:
        ref = reference.reindex(methylation.index).to_numpy(dtype=float)
        rows, cols = np.where(np.isnan(arr))
        arr[rows, cols] = ref[rows]
    return pd.DataFrame(arr, index=methylation.index, columns=methylation.columns)


def clock_cpgs(clocks: Union[List[Any], Dict[str, Any]]) -> set:
    """Union of CpG IDs used by the given clocks (empty if any clock hides them)."""
    items = clocks.values() if isinstance(clocks, dict) else clocks
    sets = [_clock_cpgs(c) for c in items]
    if not sets or any(not s for s in sets):
        return set()
    return set().union(*sets)


def _clock_cpgs(clock: Any) -> set:
    """CpG IDs a clock uses, if it exposes ``coefficients`` and ``CPG_COL``."""
    coefs   = getattr(clock, "coefficients", None)
    cpg_col = getattr(clock, "CPG_COL", None)
    if coefs is None or cpg_col is None or cpg_col not in getattr(coefs, "columns", []):
        return set()
    return set(coefs[cpg_col])


# =====================================================================
# GENERIC — evaluation metrics
# =====================================================================
DEFAULT_METRIC_NAMES: Dict[str, str] = {
    "correlation": "Cor(Age, DNAmAge)",
    "median_error": "Median error (Age, DNAmAge)",
    "median_age_accel": "Median AgeAcceleration",
}


def compute_epigenetic_age_metrics(
    chron_ages: np.ndarray,
    dnam_ages: np.ndarray,
    age_accel_method: str = "residual",
    metric_names: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Compute the three standard epigenetic-clock evaluation metrics:
    correlation with chronological age, median absolute error, and
    median age acceleration.

    Applicable to any clock (Horvath, Hannum, PhenoAge, ...) — just pass
    that clock's predicted ages.

    Parameters
    ----------
    chron_ages : array-like
        Chronological ages of samples (years).
    dnam_ages : array-like
        DNAm ages predicted by the clock.
    age_accel_method : {'residual', 'difference'}
        'residual'   : Residuals from regressing DNAmAge ~ ChronAge
                       (recommended; the standard definition of "age
                       acceleration" in the epigenetic-clock literature).
        'difference' : Simple DNAmAge - ChronAge.
    metric_names : dict, optional
        Override the output column names. Keys: 'correlation',
        'median_error', 'median_age_accel'. Defaults to
        :data:`DEFAULT_METRIC_NAMES`. Useful for matching the exact
        column names used in a paper's supplementary tables so the
        output can be fed straight into :func:`validate_clock_results`.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with the three metrics as columns.
    """
    names = {**DEFAULT_METRIC_NAMES, **(metric_names or {})}

    chron_ages = np.asarray(chron_ages, dtype=float)
    dnam_ages = np.asarray(dnam_ages, dtype=float)

    # ── Validation ───────────────────────────────────────────────────
    if len(chron_ages) != len(dnam_ages):
        raise ValueError("chron_ages and dnam_ages must have the same length.")
    if len(chron_ages) < 3:
        raise ValueError("At least 3 samples are required.")

    nan_mask = np.isnan(chron_ages) | np.isnan(dnam_ages)
    if nan_mask.any():
        print(f"\u26a0  Dropping {nan_mask.sum()} sample(s) with NaN values.")
        chron_ages = chron_ages[~nan_mask]
        dnam_ages = dnam_ages[~nan_mask]

    # ── 1. Pearson correlation ──────────────────────────────────────
    corr, _ = pearsonr(chron_ages, dnam_ages)

    # ── 2. Median absolute error ────────────────────────────────────
    median_error = np.median(np.abs(dnam_ages - chron_ages))

    # ── 3. Median age acceleration ──────────────────────────────────
    if age_accel_method == "residual":
        reg = LinearRegression().fit(chron_ages.reshape(-1, 1), dnam_ages)
        age_accel = dnam_ages - reg.predict(chron_ages.reshape(-1, 1))
    elif age_accel_method == "difference":
        age_accel = dnam_ages - chron_ages
    else:
        raise ValueError(
            f"Unknown age_accel_method='{age_accel_method}'. "
            "Choose 'residual' or 'difference'."
        )

    median_age_accel = np.median(age_accel)

    return pd.DataFrame(
        [
            {
                names["correlation"]: round(float(corr), 4),
                names["median_error"]: round(float(median_error), 4),
                names["median_age_accel"]: round(float(median_age_accel), 4),
            }
        ]
    )


# =====================================================================
# GENERIC — validation against published benchmark results
# =====================================================================
def validate_clock_results(
    your_df: pd.DataFrame,
    paper_df: pd.DataFrame,
    column_map: Dict[str, str],
    tolerances: Dict[str, float],
    labels: Optional[Dict[str, str]] = None,
    dataset_name: str = "Dataset",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compare a re-implementation's computed clock metrics against
    published paper values, for any clock.

    Parameters
    ----------
    your_df : pd.DataFrame
        Your computed results (single row), e.g. the output of
        :func:`compute_epigenetic_age_metrics`.
    paper_df : pd.DataFrame
        Paper / published results (single row, or first row is used).
    column_map : dict
        Maps paper column name -> your column name, e.g.
        ``{"Cor(Age,m.age(Train.CpGs))": "Cor(Age, DNAmAge)"}``.
    tolerances : dict
        Maps paper column name -> allowed absolute difference.
    labels : dict, optional
        Maps paper column name -> human-readable label for display.
        Defaults to the paper column name itself.
    dataset_name : str
        Label for display.
    verbose : bool
        Print a summary table if True.

    Returns
    -------
    pd.DataFrame
        Comparison table (indexed by metric label) with 'Yours', 'Paper',
        'Abs. Diff', 'Tolerance', and 'Status' columns.
    """
    labels = labels or {col: col for col in column_map}

    paper_row = paper_df.iloc[0]
    your_row = your_df.iloc[0]

    rows = []
    for paper_col, your_col in column_map.items():
        paper_val = float(paper_row[paper_col])
        your_val = float(your_row[your_col])
        abs_diff = abs(your_val - paper_val)
        tol = tolerances[paper_col]
        passed = abs_diff <= tol

        rows.append(
            {
                "Metric": labels.get(paper_col, paper_col),
                "Yours": round(your_val, 4),
                "Paper": round(paper_val, 4),
                "Abs. Diff": round(abs_diff, 4),
                "Tolerance": tol,
                "Status": "\u2713 Pass" if passed else "\u2717 Fail",
            }
        )

    result_df = pd.DataFrame(rows).set_index("Metric")

    if verbose:
        all_pass = all(r["Status"].startswith("\u2713") for r in rows)
        width = 60
        print(f"\n{'=' * width}")
        print(f"  Clock Validation — {dataset_name}")
        print(f"{'=' * width}")
        print(result_df.to_string())
        print(f"{'=' * width}")
        if all_pass:
            print("  \u2713 All metrics within tolerance.")
            print("  Implementation appears correct.")
        else:
            failed = [r["Metric"] for r in rows if r["Status"].startswith("\u2717")]
            print(f"  \u2717 Outside tolerance: {', '.join(failed)}")
            print("  Check: CpG preprocessing, age transformation,")
            print("         sample filtering, or training CpG selection.")
        print(f"{'=' * width}\n")

    return result_df


# =====================================================================
# HORVATH — clock-specific implementation
# =====================================================================
class Horvath:
    """
    Horvath (2013) pan-tissue DNA methylation age ("epigenetic") clock.

    Reference
    ---------
    Horvath S. (2013) "DNA methylation age of human tissues and cell
    types." Genome Biology, 14:R115.
    https://doi.org/10.1186/gb-2013-14-10-r115

    The 353-CpG coefficient table is available as "AdditionalFile3.csv"
    in the paper's supplementary material.

    Usage
    -----
    >>> clock = Horvath(coefficients_path="horvath.csv")
    >>> dnam_ages = clock.predict(methylation)
    >>> metrics = clock.compute_metrics(chron_ages, dnam_ages["DNAmAge"])
    >>> clock.validate(metrics, paper_results)
    """

    # ── Clock-specific constants ────────────────────────────────────
    ADULT_AGE: float = 20.0
    CPG_COL: str = "CpGmarker"
    COEF_COL: str = "CoefficientTraining"
    INTERCEPT_LABEL: str = "(Intercept)"

    METRIC_NAMES: Dict[str, str] = {
        "correlation": "Cor(Age, m.age(Train.CpGs))",
        "median_error": "Median error (Age, m.age(Train.CpGs))",
        "median_age_accel": "Median AgeAcceleration(Train.CpGs)",
    }

    PAPER_COLUMN_MAP: Dict[str, str] = {
        "Cor(Age,m.age(Train.CpGs))": "Cor(Age, m.age(Train.CpGs))",
        "Median error (Age, m.age(Train.CpGs))": "Median error (Age, m.age(Train.CpGs))",
        "Median AgeAcceleration(Train.CpGs)": "Median AgeAcceleration(Train.CpGs)",
    }

    METRIC_LABELS: Dict[str, str] = {
        "Cor(Age,m.age(Train.CpGs))": "Cor(Age, DNAmAge)",
        "Median error (Age, m.age(Train.CpGs))": "Median Error",
        "Median AgeAcceleration(Train.CpGs)": "Median Age Acceleration",
    }

    DEFAULT_TOLERANCES: Dict[str, float] = {
        "Cor(Age,m.age(Train.CpGs))": 0.1,
        "Median error (Age, m.age(Train.CpGs))": 0.5,
        "Median AgeAcceleration(Train.CpGs)": 0.5,
    }

    def __init__(
        self,
        coefficients_path: Optional[str] = None,
        coefficients: Optional[pd.DataFrame] = None,
        adult_age: float = ADULT_AGE,
    ):
        """
        Parameters
        ----------
        coefficients_path : str, optional
            Path to the Horvath 'AdditionalFile3.csv' coefficient table.
            Loaded automatically via :meth:`load_coefficients`.
        coefficients : pd.DataFrame, optional
            Pre-loaded coefficient table (used instead of
            ``coefficients_path``, e.g. if already in memory).
        adult_age : float
            Inflection point of the age transformation (default 20,
            as used in the original paper).
        """
        self.adult_age = adult_age

        if coefficients is not None:
            self.coefficients = coefficients
        elif coefficients_path is not None:
            self.coefficients = self.load_coefficients(coefficients_path)
        else:
            self.coefficients = None

    # ── Age transformation ──────────────────────────────────────────
    def trafo(self, age: "float | np.ndarray") -> "float | np.ndarray":
        """Log-linear transform of chronological age (see Horvath 2013)."""
        return log_linear_trafo(age, adult_age=self.adult_age)

    def anti_trafo(self, x: "float | np.ndarray") -> "float | np.ndarray":
        """Inverse of :meth:`trafo`: linear predictor -> age in years."""
        return log_linear_anti_trafo(x, adult_age=self.adult_age)

    # ── Data loading ────────────────────────────────────────────────
    @classmethod
    def load_coefficients(cls, filepath: str) -> pd.DataFrame:
        """
        Load the Horvath 353-CpG coefficient file.

        The official file is 'AdditionalFile3.csv' from Horvath (2013),
        Genome Biology, 14:R115. Expected columns after the 2-row header
        preamble: ['CpGmarker', 'CoefficientTraining', ...].
        """
        df = pd.read_csv(filepath, skiprows=2)
        print(f"Loaded {len(df) - 1} CpG coefficients + intercept.")
        return df

    # ── Prediction ──────────────────────────────────────────────────
    def predict(
        self,
        methylation: pd.DataFrame,
        coefficients: Optional[pd.DataFrame] = None,
        impute_missing: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Predict DNAm age using the Horvath clock.

        Parameters
        ----------
        methylation : pd.DataFrame
            Beta-value matrix, shape (n_CpGs, n_samples). Index = CpG
            probe IDs, columns = sample IDs. Values in [0, 1].
        coefficients : pd.DataFrame, optional
            Overrides ``self.coefficients`` for this call.
        impute_missing : bool
            If True, missing CpGs are imputed assuming a population mean
            beta of 0.5. If False, missing sites are skipped.
        verbose : bool
            Print a report of missing CpGs.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by ``SampleID`` with a ``DNAmAge`` column.
        """
        coefficients = coefficients if coefficients is not None else self.coefficients
        if coefficients is None:
            raise ValueError(
                "No coefficients available. Pass `coefficients_path` / "
                "`coefficients` to __init__, or `coefficients` to predict()."
            )

        return predict_methylation_age(
            methylation=methylation,
            coefficients=coefficients,
            cpg_col=self.CPG_COL,
            coef_col=self.COEF_COL,
            intercept_label=self.INTERCEPT_LABEL,
            impute_missing=impute_missing,
            impute_beta=0.5,
            anti_trafo_fn=self.anti_trafo,
            verbose=verbose,
        )

    # ── Metrics ─────────────────────────────────────────────────────
    def compute_metrics(
        self,
        chron_ages: np.ndarray,
        dnam_ages: np.ndarray,
        age_accel_method: str = "residual",
    ) -> pd.DataFrame:
        """
        Compute the three standard Horvath clock evaluation metrics.
        See :func:`compute_epigenetic_age_metrics` for parameter details.
        """
        return compute_epigenetic_age_metrics(
            chron_ages=chron_ages,
            dnam_ages=dnam_ages,
            age_accel_method=age_accel_method,
            metric_names=self.METRIC_NAMES,
        )

    # ── Validation ──────────────────────────────────────────────────
    def validate(
        self,
        your_metrics: pd.DataFrame,
        paper_df: pd.DataFrame,
        tolerances: Optional[Dict[str, float]] = None,
        dataset_name: str = "Dataset",
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Compare computed Horvath clock metrics against published paper /
        supplementary values (e.g. Horvath 2013's per-dataset table).
        See :func:`validate_clock_results` for parameter details.
        """
        return validate_clock_results(
            your_df=your_metrics,
            paper_df=paper_df,
            column_map=self.PAPER_COLUMN_MAP,
            tolerances=tolerances or self.DEFAULT_TOLERANCES,
            labels=self.METRIC_LABELS,
            dataset_name=dataset_name,
            verbose=verbose,
        )

    # ── Convenience: end-to-end pipeline ────────────────────────────
    def run(
        self,
        methylation: pd.DataFrame,
        chron_ages: Optional[pd.Series] = None,
        age_accel_method: str = "residual",
        impute_missing: bool = True,
        verbose: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Convenience wrapper: predict DNAm age and, if ``chron_ages`` is
        given, join it and compute evaluation metrics in one call.

        Returns
        -------
        dict
            ``{"predictions": DataFrame}`` and, if ``chron_ages`` was
            provided, additionally ``{"comparison": DataFrame,
            "metrics": DataFrame}``.
        """
        predictions = self.predict(methylation, impute_missing=impute_missing, verbose=verbose)
        out = {"predictions": predictions}

        if chron_ages is not None:
            comparison = predictions.join(chron_ages.rename("ChronAge"))
            comparison["AgeAcceleration"] = comparison["DNAmAge"] - comparison["ChronAge"]
            metrics = self.compute_metrics(
                comparison["ChronAge"], comparison["DNAmAge"], age_accel_method=age_accel_method
            )
            out["comparison"] = comparison
            out["metrics"] = metrics

        return out


# =====================================================================
# HANNUM — clock-specific implementation
# =====================================================================
class Hannum:
    """
    Hannum et al. (2013) whole-blood DNA methylation age clock.

    Reference
    ---------
    Hannum G, et al. (2013) "Genome-wide Methylation Profiles Reveal
    Quantitative Views of Human Aging Rates." Molecular Cell, 49(2):
    359–367.
    https://doi.org/10.1016/j.molcel.2012.10.016

    The 71-CpG coefficient table is available as supplementary data
    from the above publication. Expected CSV columns (with numeric
    index in column 0):
    ['Marker', 'Chrom', 'Pos', 'Genes', 'CpG Island', 'Coefficient']

    Notes
    -----
    - Unlike Horvath, Hannum uses a **direct linear model** — no age
      transformation is applied before or after prediction.
    - Trained exclusively on **whole blood**; not suitable for other
      tissue types.
    - Sex and BMI were covariates during model training but are **not**
      required as inputs at prediction time.
    - The intercept is **not** stored in the coefficient CSV; supply it
      via the ``intercept`` parameter in ``__init__``.

    Usage
    -----
    >>> clock = Hannum(coefficients_path="hannum.csv", intercept=<value>)
    >>> dnam_ages = clock.predict(methylation)
    >>> metrics  = clock.compute_metrics(chron_ages, dnam_ages["DNAmAge"])
    >>> clock.validate(metrics, paper_results)
    """

    # ── Clock-specific constants ────────────────────────────────────
    CPG_COL: str = "Marker"
    COEF_COL: str = "Coefficient"

    # Synthetic label used to inject the intercept into the coefficient
    # table so that predict_methylation_age can extract it uniformly.
    # Not present in the CSV; created at predict time by _inject_intercept.
    _INTERCEPT_LABEL: str = "__hannum_intercept__"

    METRIC_NAMES: Dict[str, str] = {
        "correlation": "Cor(Age, DNAmAge)",
        "median_error": "Median Absolute Error (years)",
        "median_age_accel": "Median Age Acceleration",
    }

    PAPER_COLUMN_MAP: Dict[str, str] = {
        "Cor(Age, DNAmAge)": "Cor(Age, DNAmAge)",
        "Median Absolute Error (years)": "Median Absolute Error (years)",
        "Median Age Acceleration": "Median Age Acceleration",
    }

    METRIC_LABELS: Dict[str, str] = {
        "Cor(Age, DNAmAge)": "Cor(Age, DNAmAge)",
        "Median Absolute Error (years)": "Median Absolute Error",
        "Median Age Acceleration": "Median Age Acceleration",
    }

    # Blood benchmarks from Hannum et al. (2013): r ≈ 0.96, MAE ≈ 3.9 yrs
    DEFAULT_TOLERANCES: Dict[str, float] = {
        "Cor(Age, DNAmAge)": 0.05,
        "Median Absolute Error (years)": 0.5,
        "Median Age Acceleration": 0.5,
    }

    def __init__(
        self,
        coefficients_path: Optional[str] = None,
        coefficients: Optional[pd.DataFrame] = None,
        intercept: float = 0.0,
    ):
        """
        Parameters
        ----------
        coefficients_path : str, optional
            Path to the Hannum 71-CpG coefficient CSV file.
            Loaded automatically via :meth:`load_coefficients`.
        coefficients : pd.DataFrame, optional
            Pre-loaded coefficient table (used instead of
            ``coefficients_path``, e.g. if already in memory).
        intercept : float
            Linear model intercept. The Hannum coefficient CSV does not
            include an intercept row; supply the correct value from the
            paper's supplementary materials.

            .. warning::
                Defaults to ``0.0`` — this will bias all predictions.
                Always set this explicitly.
        """
        self.intercept = intercept

        if intercept == 0.0:
            warnings.warn(
                "Hannum intercept is set to 0.0 (the default). "
                "This is almost certainly incorrect and will bias every "
                "prediction. Supply the correct value from the paper's "
                "supplementary materials.",
                UserWarning,
                stacklevel=2,
            )

        if coefficients is not None:
            self.coefficients = coefficients
        elif coefficients_path is not None:
            self.coefficients = self.load_coefficients(coefficients_path)
        else:
            self.coefficients = None

    # ── No age transformation (direct linear model) ─────────────────
    def trafo(self, age: "float | np.ndarray") -> "float | np.ndarray":
        """
        Identity transform — Hannum uses a direct linear model and
        requires no age transformation prior to training, unlike Horvath.
        Retained here for API consistency.
        """
        return age

    def anti_trafo(self, x: "float | np.ndarray") -> "float | np.ndarray":
        """
        Identity inverse — the linear predictor is already in years;
        no back-transformation is needed. Retained for API consistency.
        """
        return x

    # ── Data loading ────────────────────────────────────────────────
    @classmethod
    def load_coefficients(cls, filepath: str) -> pd.DataFrame:
        """
        Load the Hannum 71-CpG coefficient file.

        The expected format matches the supplementary CSV from Hannum
        et al. (2013): a numeric index in column 0, followed by columns
        ['Marker', 'Chrom', 'Pos', 'Genes', 'CpG Island', 'Coefficient'].
        The numeric index column is consumed by ``index_col=0``.

        Parameters
        ----------
        filepath : str
            Path to the CSV file.

        Returns
        -------
        pd.DataFrame
            71-row coefficient table ready for use in :meth:`predict`.
        """
        df = pd.read_csv(filepath, index_col=None)
        print(f"Loaded {len(df)} CpG coefficients.")
        return df

    # ── Internal: inject intercept into coefficient table ────────────
    def _inject_intercept(self, coefficients: pd.DataFrame) -> pd.DataFrame:
        """
        Prepend a synthetic intercept row to the coefficient table so
        that :func:`predict_methylation_age` can extract it uniformly
        via its ``intercept_label`` mechanism.

        Only ``CPG_COL`` and ``COEF_COL`` are retained from the input
        (which also contains 'Chrom', 'Pos', 'Genes', 'CpG Island'),
        reducing it to the two-column format expected by the generic
        predictor. The methylation matrix is not modified because
        ``predict_methylation_age`` handles the intercept independently
        of the CpG dot-product loop.

        Parameters
        ----------
        coefficients : pd.DataFrame
            Raw 71-row Hannum coefficient table.

        Returns
        -------
        pd.DataFrame
            72-row table (intercept row prepended) with only
            ``CPG_COL`` and ``COEF_COL`` columns.
        """
        intercept_row = pd.DataFrame(
            {
                self.CPG_COL: [self._INTERCEPT_LABEL],
                self.COEF_COL: [self.intercept],
            }
        )
        return pd.concat(
            [intercept_row, coefficients[[self.CPG_COL, self.COEF_COL]]],
            ignore_index=True,
        )

    # ── Prediction ──────────────────────────────────────────────────
    def predict(
        self,
        methylation: pd.DataFrame,
        coefficients: Optional[pd.DataFrame] = None,
        impute_missing: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Predict DNAm age using the Hannum clock.

        Parameters
        ----------
        methylation : pd.DataFrame
            Beta-value matrix, shape (n_CpGs, n_samples). Index = CpG
            probe IDs, columns = sample IDs. Values in [0, 1].
        coefficients : pd.DataFrame, optional
            Overrides ``self.coefficients`` for this call.
        impute_missing : bool
            If True, missing CpGs are imputed with a population mean
            beta of 0.5. If False, missing sites are skipped.
        verbose : bool
            Print a report of missing CpGs.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by sample ID with a single ``DNAmAge``
            column (epigenetic age in years).
        """
        coefficients = coefficients if coefficients is not None else self.coefficients
        if coefficients is None:
            raise ValueError(
                "No coefficients available. Pass `coefficients_path` / "
                "`coefficients` to __init__, or `coefficients` to predict()."
            )

        return predict_methylation_age(
            methylation=methylation,
            coefficients=self._inject_intercept(coefficients),
            cpg_col=self.CPG_COL,
            coef_col=self.COEF_COL,
            intercept_label=self._INTERCEPT_LABEL,
            impute_missing=impute_missing,
            impute_beta=0.5,
            anti_trafo_fn=self.anti_trafo,  # identity — no transformation
            verbose=verbose,
        )

    # ── Metrics ─────────────────────────────────────────────────────
    def compute_metrics(
        self,
        chron_ages: np.ndarray,
        dnam_ages: np.ndarray,
        age_accel_method: str = "residual",
    ) -> pd.DataFrame:
        """
        Compute the three standard Hannum clock evaluation metrics:
        Cor(Age, DNAmAge), Median Absolute Error, Median Age Acceleration.

        On whole blood (GSE40279, n = 656) expect approximately:
        - Pearson r ≈ 0.96
        - Median absolute error ≈ 3.9 years

        See :func:`compute_epigenetic_age_metrics` for parameter details.
        """
        return compute_epigenetic_age_metrics(
            chron_ages=chron_ages,
            dnam_ages=dnam_ages,
            age_accel_method=age_accel_method,
            metric_names=self.METRIC_NAMES,
        )

    # ── Validation ──────────────────────────────────────────────────
    def validate(
        self,
        your_metrics: pd.DataFrame,
        paper_df: pd.DataFrame,
        tolerances: Optional[Dict[str, float]] = None,
        dataset_name: str = "Dataset",
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Compare computed metrics against published Hannum (2013) values.

        Reference benchmarks for whole blood (GSE40279):
        - Pearson r ≈ 0.96
        - Median absolute error ≈ 3.9 years

        See :func:`validate_clock_results` for parameter details.
        """
        return validate_clock_results(
            your_df=your_metrics,
            paper_df=paper_df,
            column_map=self.PAPER_COLUMN_MAP,
            tolerances=tolerances or self.DEFAULT_TOLERANCES,
            labels=self.METRIC_LABELS,
            dataset_name=dataset_name,
            verbose=verbose,
        )

    # ── Convenience: end-to-end pipeline ────────────────────────────
    def run(
        self,
        methylation: pd.DataFrame,
        chron_ages: Optional[pd.Series] = None,
        age_accel_method: str = "residual",
        impute_missing: bool = True,
        verbose: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Convenience wrapper: predict DNAm age and, if ``chron_ages`` is
        provided, join it and compute evaluation metrics in one call.

        Returns
        -------
        dict
            ``{"predictions": DataFrame}`` and, if ``chron_ages`` was
            provided, additionally ``{"comparison": DataFrame,
            "metrics": DataFrame}``.
        """
        predictions = self.predict(
            methylation, impute_missing=impute_missing, verbose=verbose
        )
        out = {"predictions": predictions}

        if chron_ages is not None:
            comparison = predictions.join(chron_ages.rename("ChronAge"))
            comparison["AgeAcceleration"] = (
                comparison["DNAmAge"] - comparison["ChronAge"]
            )
            metrics = self.compute_metrics(
                comparison["ChronAge"],
                comparison["DNAmAge"],
                age_accel_method=age_accel_method,
            )
            out["comparison"] = comparison
            out["metrics"] = metrics

        return out


# =====================================================================
# PHENOAGE — clock-specific implementation
# =====================================================================
class PhenoAge:
    """
    Levine et al. (2018) PhenoAge DNA methylation clock.

    Reference
    ---------
    Levine ME, et al. (2018) "An epigenetic biomarker of aging for
    lifespan and healthspan." Aging, 10(4): 573–591.
    https://doi.org/10.18632/aging.101414

    The 513-CpG coefficient table is available as supplementary data
    from the above publication. Expected CSV columns::

        CpG | Chromosome | Map Info | Gene Symbol | Entrez ID |
        Weight | Univariate Age Correlation | Horvath Overlap |
        Hannum Overlap

    The intercept is stored in the ``CpG`` column as ``"Intercept"``
    with its value in the ``Weight`` column.

    Notes
    -----
    - PhenoAge is a **direct linear model** in years — no age
      transformation is applied before or after prediction.
    - The clock was trained to predict "phenotypic age" derived from
      nine clinical chemistry biomarkers, not simply chronological age.
    - Validated primarily in whole blood; may perform differently in
      other tissue types.

    Usage
    -----
    >>> clock = PhenoAge(coefficients_path="phenoage.csv")
    >>> dnam_ages = clock.predict(methylation)
    >>> metrics = clock.compute_metrics(chron_ages, dnam_ages["DNAmAge"])
    >>> clock.validate(metrics, paper_results)
    """

    # ── Clock-specific constants ────────────────────────────────────
    CPG_COL: str = "CpG"
    COEF_COL: str = "Weight"
    INTERCEPT_LABEL: str = "Intercept"

    METRIC_NAMES: Dict[str, str] = {
        "correlation": "Cor(Age, DNAmPhenoAge)",
        "median_error": "Median Absolute Error (years)",
        "median_age_accel": "Median Age Acceleration",
    }

    PAPER_COLUMN_MAP: Dict[str, str] = {
        "Cor(Age, DNAmPhenoAge)": "Cor(Age, DNAmPhenoAge)",
        "Median Absolute Error (years)": "Median Absolute Error (years)",
        "Median Age Acceleration": "Median Age Acceleration",
    }

    METRIC_LABELS: Dict[str, str] = {
        "Cor(Age, DNAmPhenoAge)": "Cor(Age, DNAmPhenoAge)",
        "Median Absolute Error (years)": "Median Absolute Error",
        "Median Age Acceleration": "Median Age Acceleration",
    }

    # Blood benchmarks from Levine et al. (2018)
    DEFAULT_TOLERANCES: Dict[str, float] = {
        "Cor(Age, DNAmPhenoAge)": 0.05,
        "Median Absolute Error (years)": 0.5,
        "Median Age Acceleration": 0.5,
    }

    def __init__(
        self,
        coefficients_path: Optional[str] = None,
        coefficients: Optional[pd.DataFrame] = None,
    ):
        """
        Parameters
        ----------
        coefficients_path : str, optional
            Path to the PhenoAge coefficient CSV. Expected columns:
            ``CpG`` (probe IDs, intercept row labelled ``"Intercept"``)
            and ``Weight`` (model coefficients).
        coefficients : pd.DataFrame, optional
            Pre-loaded coefficient table (used instead of
            ``coefficients_path``).
        """
        if coefficients is not None:
            self.coefficients = coefficients
        elif coefficients_path is not None:
            self.coefficients = self.load_coefficients(coefficients_path)
        else:
            self.coefficients = None

    # ── No age transformation (direct linear model) ─────────────────
    def trafo(self, age: "float | np.ndarray") -> "float | np.ndarray":
        """
        Identity transform — PhenoAge uses a direct linear model and
        requires no age transformation. Retained for API consistency.
        """
        return age

    def anti_trafo(self, x: "float | np.ndarray") -> "float | np.ndarray":
        """
        Identity inverse — the linear predictor is already in years.
        Retained for API consistency.
        """
        return x

    # ── Data loading ────────────────────────────────────────────────
    @classmethod
    def load_coefficients(cls, filepath: str) -> pd.DataFrame:
        """
        Load the PhenoAge 513-CpG coefficient file.

        The expected format matches the supplementary CSV from Levine
        et al. (2018). The ``CpG`` column holds probe IDs (plus an
        ``"Intercept"`` row) and ``Weight`` holds the model coefficient.
        Extra columns (Chromosome, Map Info, etc.) are retained but
        ignored during prediction.

        Parameters
        ----------
        filepath : str
            Path to the CSV file.

        Returns
        -------
        pd.DataFrame
            513-row coefficient table (plus intercept row) ready for use
            in :meth:`predict`.
        """
        df = pd.read_csv(filepath)
        n_cpgs = (df[cls.CPG_COL] != cls.INTERCEPT_LABEL).sum()
        print(f"Loaded {n_cpgs} CpG coefficients + intercept.")
        return df

    # ── Prediction ──────────────────────────────────────────────────
    def predict(
        self,
        methylation: pd.DataFrame,
        coefficients: Optional[pd.DataFrame] = None,
        impute_missing: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Predict DNAm PhenoAge from a beta-value methylation matrix.

        Parameters
        ----------
        methylation : pd.DataFrame
            Beta-value matrix, shape (n_CpGs, n_samples). Index = CpG
            probe IDs, columns = sample IDs. Values in [0, 1].
        coefficients : pd.DataFrame, optional
            Overrides ``self.coefficients`` for this call.
        impute_missing : bool
            If True, CpGs absent from ``methylation`` are imputed with a
            population-mean beta of 0.5. If False, missing sites are
            skipped (zero contribution).
        verbose : bool
            Print a report of missing CpGs.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by ``SampleID`` with a ``DNAmAge`` column
            (PhenoAge estimate in years).
        """
        coefficients = coefficients if coefficients is not None else self.coefficients
        if coefficients is None:
            raise ValueError(
                "No coefficients available. Pass `coefficients_path` / "
                "`coefficients` to __init__, or `coefficients` to predict()."
            )

        return predict_methylation_age(
            methylation=methylation,
            coefficients=coefficients,
            cpg_col=self.CPG_COL,
            coef_col=self.COEF_COL,
            intercept_label=self.INTERCEPT_LABEL,
            impute_missing=impute_missing,
            impute_beta=0.5,
            anti_trafo_fn=self.anti_trafo,
            verbose=verbose,
        )

    # ── Metrics ─────────────────────────────────────────────────────
    def compute_metrics(
        self,
        chron_ages: np.ndarray,
        dnam_ages: np.ndarray,
        age_accel_method: str = "residual",
    ) -> pd.DataFrame:
        """
        Compute the three standard PhenoAge clock evaluation metrics:
        Cor(Age, DNAmPhenoAge), Median Absolute Error, Median Age
        Acceleration.

        See :func:`compute_epigenetic_age_metrics` for parameter details.
        """
        return compute_epigenetic_age_metrics(
            chron_ages=chron_ages,
            dnam_ages=dnam_ages,
            age_accel_method=age_accel_method,
            metric_names=self.METRIC_NAMES,
        )

    # ── Validation ──────────────────────────────────────────────────
    def validate(
        self,
        your_metrics: pd.DataFrame,
        paper_df: pd.DataFrame,
        tolerances: Optional[Dict[str, float]] = None,
        dataset_name: str = "Dataset",
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Compare computed PhenoAge metrics against published paper /
        supplementary values from Levine et al. (2018).

        See :func:`validate_clock_results` for parameter details.
        """
        return validate_clock_results(
            your_df=your_metrics,
            paper_df=paper_df,
            column_map=self.PAPER_COLUMN_MAP,
            tolerances=tolerances or self.DEFAULT_TOLERANCES,
            labels=self.METRIC_LABELS,
            dataset_name=dataset_name,
            verbose=verbose,
        )

    # ── Convenience: end-to-end pipeline ────────────────────────────
    def run(
        self,
        methylation: pd.DataFrame,
        chron_ages: Optional[pd.Series] = None,
        age_accel_method: str = "residual",
        impute_missing: bool = True,
        verbose: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Convenience wrapper: predict DNAm PhenoAge and, if
        ``chron_ages`` is provided, join it and compute evaluation
        metrics in one call.

        Returns
        -------
        dict
            ``{"predictions": DataFrame}`` and, if ``chron_ages`` was
            provided, additionally ``{"comparison": DataFrame,
            "metrics": DataFrame}``.
        """
        predictions = self.predict(
            methylation, impute_missing=impute_missing, verbose=verbose
        )
        out = {"predictions": predictions}

        if chron_ages is not None:
            comparison = predictions.join(chron_ages.rename("ChronAge"))
            comparison["AgeAcceleration"] = (
                comparison["DNAmAge"] - comparison["ChronAge"]
            )
            metrics = self.compute_metrics(
                comparison["ChronAge"],
                comparison["DNAmAge"],
                age_accel_method=age_accel_method,
            )
            out["comparison"] = comparison
            out["metrics"] = metrics

        return out


# =====================================================================
# GRIMAGE V2 — clock-specific implementation
# =====================================================================
class GrimAgeV2:
    """
    Lu et al. (2022) GrimAge Version 2 DNA methylation clock.

    Reference
    ---------
    Lu AT, et al. (2022) "DNA methylation GrimAge version 2."
    Aging, 14(23): 9484–9549.
    https://doi.org/10.18632/aging.204434

    Architecture
    ------------
    GrimAge V2 is a **two-stage** model:

    Stage 1 — Sub-component prediction:
        Ten protein/smoking proxy variables are each predicted from a
        separate linear combination of CpG beta-values plus, in some
        cases, chronological age::

            DNAmGDF15, DNAmB2M, DNAmCystatinC, DNAmTIMP1, DNAmADM,
            DNAmPAI1, DNAmLeptin, DNAmPACKYRS, DNAmlogCRP, DNAmlogA1C

    Stage 2 — COX linear predictor → GrimAge:
        A single Cox-regression linear predictor is formed from the ten
        stage-1 outputs plus chronological age and biological sex
        (Female = 1, Male = 0). The raw predictor is then standardised
        to a "biological age" in years via stored population mean/SD
        parameters::

            GrimAge = m_age + sd_age *
                      (cox_lp - m_cox) / sd_cox

    Coefficient file format
    -----------------------
    The single CSV supplied to this class contains all components
    stacked in "long" format with three columns::

        Y.pred | var | beta

    ``Y.pred`` identifies the target (one of the ten sub-components,
    ``"COX"``, or ``"transform"``). ``var`` is either ``"Intercept"``,
    ``"Age"``, ``"Female"``, a CpG probe ID (``cg…``), or a
    sub-component name. ``beta`` is the corresponding coefficient.

    The ``"transform"`` block holds four scalar parameters:
    ``m_age``, ``sd_age``, ``m_cox``, ``sd_cox``.

    Usage
    -----
    >>> clock = GrimAgeV2(coefficients_path="GrimAgeV2.csv")
    >>> dnam_ages = clock.predict(methylation, chron_ages, female_mask)
    >>> metrics = clock.compute_metrics(chron_ages, dnam_ages["DNAmAge"])
    >>> clock.validate(metrics, paper_results)

    Notes
    -----
    - Chronological age and biological sex are **required** at
      prediction time (unlike single-stage clocks).
    - The ``female_mask`` parameter is a boolean Series/array aligned
      with the methylation column order: True for female, False for male.
    - Trained on whole blood; tissue-type restrictions apply as for
      other blood-based clocks.
    """

    # ── Clock-specific constants ────────────────────────────────────
    YPRED_COL: str = "Y.pred"
    VAR_COL: str = "var"
    BETA_COL: str = "beta"
    INTERCEPT_LABEL: str = "Intercept"
    AGE_LABEL: str = "Age"
    FEMALE_LABEL: str = "Female"
    COX_COMPONENT: str = "COX"
    TRANSFORM_COMPONENT: str = "transform"

    # Sub-components that use Age as a covariate in stage 1
    _AGE_DEPENDENT_COMPONENTS = frozenset({
        "DNAmGDF15", "DNAmB2M", "DNAmCystatinC", "DNAmTIMP1",
        "DNAmADM", "DNAmPACKYRS", "DNAmlogA1C",
    })

    METRIC_NAMES: Dict[str, str] = {
        "correlation": "Cor(Age, DNAmGrimAge)",
        "median_error": "Median Absolute Error (years)",
        "median_age_accel": "Median Age Acceleration",
    }

    PAPER_COLUMN_MAP: Dict[str, str] = {
        "Cor(Age, DNAmGrimAge)": "Cor(Age, DNAmGrimAge)",
        "Median Absolute Error (years)": "Median Absolute Error (years)",
        "Median Age Acceleration": "Median Age Acceleration",
    }

    METRIC_LABELS: Dict[str, str] = {
        "Cor(Age, DNAmGrimAge)": "Cor(Age, DNAmGrimAge)",
        "Median Absolute Error (years)": "Median Absolute Error",
        "Median Age Acceleration": "Median Age Acceleration",
    }

    DEFAULT_TOLERANCES: Dict[str, float] = {
        "Cor(Age, DNAmGrimAge)": 0.05,
        "Median Absolute Error (years)": 0.5,
        "Median Age Acceleration": 0.5,
    }

    def __init__(
        self,
        coefficients_path: Optional[str] = None,
        coefficients: Optional[pd.DataFrame] = None,
    ):
        """
        Parameters
        ----------
        coefficients_path : str, optional
            Path to the GrimAge V2 long-format coefficient CSV with
            columns ``Y.pred``, ``var``, and ``beta``.
        coefficients : pd.DataFrame, optional
            Pre-loaded coefficient table (used instead of
            ``coefficients_path``).
        """
        if coefficients is not None:
            self._coef_raw = coefficients
        elif coefficients_path is not None:
            self._coef_raw = self.load_coefficients(coefficients_path)
        else:
            self._coef_raw = None

        # Parse into convenient internal structures on first access
        self._parsed: Optional[Dict[str, Any]] = None

    # ── Data loading ────────────────────────────────────────────────
    @classmethod
    def load_coefficients(cls, filepath: str) -> pd.DataFrame:
        """
        Load the GrimAge V2 long-format coefficient CSV.

        Expected columns: ``Y.pred``, ``var``, ``beta``. The file
        contains all sub-component models plus the COX predictor and
        transformation parameters stacked row-wise.

        Parameters
        ----------
        filepath : str
            Path to the CSV file.

        Returns
        -------
        pd.DataFrame
            Full long-format coefficient table.
        """
        df = pd.read_csv(filepath)
        n_components = df[cls.YPRED_COL].nunique() - 2  # exclude COX + transform
        n_cpgs = df[df[cls.VAR_COL].str.startswith("cg")][cls.VAR_COL].nunique()
        print(
            f"Loaded GrimAge V2: {n_components} sub-components, "
            f"{n_cpgs} unique CpG probes."
        )
        return df

    # ── Internal: parse coefficient table into structured form ───────
    def _parse_coefficients(self, coef_raw: pd.DataFrame) -> Dict[str, Any]:
        """
        Parse the raw long-format coefficient table into a dict of
        sub-component models, COX coefficients, and transform params.

        Returns
        -------
        dict with keys:
            ``sub_components`` : dict mapping component name →
                {intercept, age_coef (or None), cpg_coefs (Series)}
            ``cox`` : {intercept (0), component_coefs (dict), age_coef,
                female_coef}
            ``transform`` : {m_age, sd_age, m_cox, sd_cox}
        """
        parsed: Dict[str, Any] = {
            "sub_components": {},
            "cox": {},
            "transform": {},
        }

        # ── Transform parameters ────────────────────────────────────
        t_block = coef_raw[coef_raw[self.YPRED_COL] == self.TRANSFORM_COMPONENT]
        t_dict = dict(zip(t_block[self.VAR_COL], t_block[self.BETA_COL]))
        parsed["transform"] = t_dict  # m_age, sd_age, m_cox, sd_cox

        # ── COX block ───────────────────────────────────────────────
        cox_block = coef_raw[coef_raw[self.YPRED_COL] == self.COX_COMPONENT]
        cox_dict: Dict[str, float] = {}
        for _, row in cox_block.iterrows():
            cox_dict[row[self.VAR_COL]] = row[self.BETA_COL]
        parsed["cox"] = cox_dict

        # ── Sub-component blocks ─────────────────────────────────────
        excluded = {self.COX_COMPONENT, self.TRANSFORM_COMPONENT}
        sub_names = [
            c for c in coef_raw[self.YPRED_COL].unique() if c not in excluded
        ]

        for name in sub_names:
            block = coef_raw[coef_raw[self.YPRED_COL] == name]
            intercept_row = block[block[self.VAR_COL] == self.INTERCEPT_LABEL]
            age_row = block[block[self.VAR_COL] == self.AGE_LABEL]
            cpg_rows = block[block[self.VAR_COL].str.startswith("cg")]

            parsed["sub_components"][name] = {
                "intercept": float(intercept_row[self.BETA_COL].values[0]),
                "age_coef": float(age_row[self.BETA_COL].values[0])
                if len(age_row)
                else None,
                "cpg_coefs": cpg_rows.set_index(self.VAR_COL)[self.BETA_COL],
            }

        return parsed

    @property
    def coefficients(self) -> pd.DataFrame:
        """
        Return a two-column DataFrame of all CpG probes used across all
        sub-components, mirroring the flat {CPG_COL, COEF_COL} format
        exposed by single-stage clocks (Hannum, Horvath, PhenoAge).
        
        Probes appearing in multiple sub-components are deduplicated;
        coefficients are summed across sub-components for consistency,
        though the primary use-case is probe-set extraction for missingness
        analysis.
        """
        if self._coef_raw is None:
            raise ValueError("No coefficients loaded.")
        
        if self._parsed is None:
            self._parsed = self._parse_coefficients(self._coef_raw)
        
        # Collect all CpG rows across sub-components
        rows = []
        for sub in self._parsed["sub_components"].values():
            rows.append(sub["cpg_coefs"])
        
        combined = pd.concat(rows)
        # Deduplicate by summing (a probe shared across sub-components
        # has an aggregate contribution; for probe-set purposes only the
        # index matters)
        combined = combined.groupby(level=0).sum()
        combined.index.name = "CpG"
        
        return combined.reset_index().rename(columns={0: "Coefficient"})

    # ── Internal: predict a single sub-component ─────────────────────
    def _predict_sub_component(
        self,
        name: str,
        sub: Dict[str, Any],
        methylation: pd.DataFrame,
        chron_ages: np.ndarray,
        impute_missing: bool,
        impute_beta: float,
        verbose: bool,
    ) -> np.ndarray:
        """
        Predict one GrimAge V2 sub-component for all samples.

        Parameters
        ----------
        name : str
            Sub-component name (for reporting).
        sub : dict
            Parsed sub-component dict from :meth:`_parse_coefficients`.
        methylation : pd.DataFrame
            Beta-value matrix (n_CpGs × n_samples).
        chron_ages : np.ndarray
            Chronological ages aligned with methylation columns.
        impute_missing : bool
            Impute missing CpGs with ``impute_beta`` if True.
        impute_beta : float
            Value used for imputation.
        verbose : bool
            Warn about missing CpGs.

        Returns
        -------
        np.ndarray
            Predicted sub-component values, shape (n_samples,).
        """
        cpg_coefs: pd.Series = sub["cpg_coefs"]
        n_samples = methylation.shape[1]

        all_cpgs = set(cpg_coefs.index)
        present_cpgs = all_cpgs & set(methylation.index)
        missing_cpgs = all_cpgs - present_cpgs

        if missing_cpgs and verbose:
            pct = 100 * len(missing_cpgs) / max(len(all_cpgs), 1)
            print(
                f"\u26a0  [{name}] {len(missing_cpgs)} / {len(all_cpgs)} "
                f"CpGs ({pct:.1f}%) missing from methylation data."
            )

        # Start from intercept
        lp = np.full(n_samples, sub["intercept"], dtype=float)

        # Add CpG contributions
        if present_cpgs:
            present_coefs = cpg_coefs.loc[list(present_cpgs)]
            lp += present_coefs.values @ methylation.loc[list(present_cpgs)].values

        if impute_missing and missing_cpgs:
            lp += cpg_coefs.loc[list(missing_cpgs)].sum() * impute_beta

        # Add age contribution if applicable
        if sub["age_coef"] is not None:
            lp += sub["age_coef"] * chron_ages

        return lp

    # ── Prediction ──────────────────────────────────────────────────
    def predict(
        self,
        methylation: pd.DataFrame,
        chron_ages: "pd.Series | np.ndarray",
        female: "pd.Series | np.ndarray",
        coefficients: Optional[pd.DataFrame] = None,
        impute_missing: bool = True,
        impute_beta: float = 0.5,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Predict GrimAge V2 biological age.

        GrimAge requires two covariates in addition to the methylation
        matrix: chronological age and biological sex. Both must be
        aligned with the column order of ``methylation``.

        Parameters
        ----------
        methylation : pd.DataFrame
            Beta-value matrix, shape (n_CpGs, n_samples). Index = CpG
            probe IDs, columns = sample IDs. Values in [0, 1].
        chron_ages : array-like
            Chronological ages (years) aligned with ``methylation``
            columns.
        female : array-like of bool or int
            Biological sex indicator aligned with ``methylation``
            columns. ``True`` / ``1`` = female; ``False`` / ``0`` = male.
        coefficients : pd.DataFrame, optional
            Overrides the stored coefficient table for this call.
        impute_missing : bool
            If True, CpGs absent from ``methylation`` are imputed with
            ``impute_beta``. If False, missing sites are skipped.
        impute_beta : float
            Beta value used to impute missing CpGs (default 0.5).
        verbose : bool
            Print per-component reports of missing CpGs.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by ``SampleID`` with columns:

            - ``DNAmAge``   : GrimAge V2 biological age estimate (years)
            - ``COX_lp``    : Raw COX linear predictor (unstandardised)
            - One column per sub-component (e.g. ``DNAmGDF15``, …)
        """
        coef_raw = coefficients if coefficients is not None else self._coef_raw
        if coef_raw is None:
            raise ValueError(
                "No coefficients available. Pass `coefficients_path` / "
                "`coefficients` to __init__, or `coefficients` to predict()."
            )

        # Parse coefficients (cache after first call)
        if self._parsed is None or coefficients is not None:
            self._parsed = self._parse_coefficients(coef_raw)

        parsed = self._parsed
        transform = parsed["transform"]
        cox_dict = parsed["cox"]
        sub_components = parsed["sub_components"]

        chron_ages_arr = np.asarray(chron_ages, dtype=float)
        female_arr = np.asarray(female, dtype=float)
        n_samples = methylation.shape[1]

        if len(chron_ages_arr) != n_samples:
            raise ValueError(
                f"chron_ages length ({len(chron_ages_arr)}) must match "
                f"the number of samples ({n_samples})."
            )
        if len(female_arr) != n_samples:
            raise ValueError(
                f"female length ({len(female_arr)}) must match "
                f"the number of samples ({n_samples})."
            )

        # ── Stage 1: predict all sub-components ─────────────────────
        sub_preds: Dict[str, np.ndarray] = {}
        for name, sub in sub_components.items():
            if verbose:
                print(f"  → Predicting {name} …")
            sub_preds[name] = self._predict_sub_component(
                name=name,
                sub=sub,
                methylation=methylation,
                chron_ages=chron_ages_arr,
                impute_missing=impute_missing,
                impute_beta=impute_beta,
                verbose=verbose,
            )

        # ── Stage 2: COX linear predictor ───────────────────────────
        cox_lp = np.zeros(n_samples, dtype=float)

        for var, beta in cox_dict.items():
            if var == self.AGE_LABEL:
                cox_lp += beta * chron_ages_arr
            elif var == self.FEMALE_LABEL:
                cox_lp += beta * female_arr
            elif var in sub_preds:
                cox_lp += beta * sub_preds[var]
            # (no intercept in COX block by convention)

        # ── Stage 3: standardise to biological age in years ─────────
        m_age = transform["m_age"]
        sd_age = transform["sd_age"]
        m_cox = transform["m_cox"]
        sd_cox = transform["sd_cox"]

        grim_age = m_age + sd_age * (cox_lp - m_cox) / sd_cox

        # ── Assemble output DataFrame ────────────────────────────────
        sample_ids = methylation.columns
        result = pd.DataFrame(
            {"DNAmAge": grim_age, "COX_lp": cox_lp},
            index=pd.Index(sample_ids, name="SampleID"),
        )
        for name, vals in sub_preds.items():
            result[name] = vals

        return result

    # ── Metrics ─────────────────────────────────────────────────────
    def compute_metrics(
        self,
        chron_ages: np.ndarray,
        dnam_ages: np.ndarray,
        age_accel_method: str = "residual",
    ) -> pd.DataFrame:
        """
        Compute the three standard GrimAge V2 clock evaluation metrics:
        Cor(Age, DNAmGrimAge), Median Absolute Error, Median Age
        Acceleration.

        See :func:`compute_epigenetic_age_metrics` for parameter details.
        """
        return compute_epigenetic_age_metrics(
            chron_ages=chron_ages,
            dnam_ages=dnam_ages,
            age_accel_method=age_accel_method,
            metric_names=self.METRIC_NAMES,
        )

    # ── Validation ──────────────────────────────────────────────────
    def validate(
        self,
        your_metrics: pd.DataFrame,
        paper_df: pd.DataFrame,
        tolerances: Optional[Dict[str, float]] = None,
        dataset_name: str = "Dataset",
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Compare computed GrimAge V2 metrics against published paper /
        supplementary values from Lu et al. (2022).

        See :func:`validate_clock_results` for parameter details.
        """
        return validate_clock_results(
            your_df=your_metrics,
            paper_df=paper_df,
            column_map=self.PAPER_COLUMN_MAP,
            tolerances=tolerances or self.DEFAULT_TOLERANCES,
            labels=self.METRIC_LABELS,
            dataset_name=dataset_name,
            verbose=verbose,
        )

    # ── Convenience: end-to-end pipeline ────────────────────────────
    def run(
        self,
        methylation: pd.DataFrame,
        chron_ages: "pd.Series | np.ndarray",
        female: "pd.Series | np.ndarray",
        age_accel_method: str = "residual",
        impute_missing: bool = True,
        verbose: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Convenience wrapper: predict GrimAge V2 and compute evaluation
        metrics in one call.

        Unlike the single-stage clock ``run()`` methods, ``chron_ages``
        and ``female`` are **required** (not optional) because GrimAge
        uses them during prediction.

        Parameters
        ----------
        methylation : pd.DataFrame
            Beta-value matrix (n_CpGs × n_samples).
        chron_ages : array-like
            Chronological ages aligned with ``methylation`` columns.
        female : array-like of bool or int
            Biological sex (True/1 = female) aligned with columns.
        age_accel_method : {'residual', 'difference'}
            Method for computing age acceleration in metrics.
        impute_missing : bool
            Impute missing CpGs with beta = 0.5 if True.
        verbose : bool
            Print per-component progress.

        Returns
        -------
        dict
            ``{"predictions": DataFrame, "comparison": DataFrame,
            "metrics": DataFrame}``.
            ``predictions`` contains ``DNAmAge``, ``COX_lp``, and
            one column per sub-component. ``comparison`` adds
            ``ChronAge`` and ``AgeAcceleration``.
        """
        predictions = self.predict(
            methylation,
            chron_ages=chron_ages,
            female=female,
            impute_missing=impute_missing,
            verbose=verbose,
        )

        chron_series = pd.Series(
            np.asarray(chron_ages, dtype=float),
            index=predictions.index,
            name="ChronAge",
        )
        comparison = predictions.join(chron_series)
        comparison["AgeAcceleration"] = (
            comparison["DNAmAge"] - comparison["ChronAge"]
        )

        metrics = self.compute_metrics(
            comparison["ChronAge"],
            comparison["DNAmAge"],
            age_accel_method=age_accel_method,
        )

        return {
            "predictions": predictions,
            "comparison": comparison,
            "metrics": metrics,
        }


# =====================================================================
# GRIMAGE ADAPTER — bridges GrimAgeV2 into run_all_clocks
# =====================================================================
class GrimAgeAdapter:
    """
    Thin wrapper that makes :class:`GrimAgeV2` compatible with
    :func:`run_all_clocks`.

    ``run_all_clocks`` calls ``clock.predict(methylation, impute_missing,
    verbose)`` — a fixed two-argument signature. GrimAge additionally
    requires chronological age and biological sex, which vary per dataset.
    This adapter stores those mappings and resolves the correct arrays at
    call time using the methylation column order.

    Parameters
    ----------
    clock : GrimAgeV2
        An initialised (coefficients loaded) GrimAgeV2 instance.
    chron_ages : dict[str, pd.Series]
        ``{dataset_name: Series}`` mapping sample IDs → chronological age.
        The adapter matches samples by index against the methylation columns.
    female : dict[str, pd.Series]
        ``{dataset_name: Series}`` mapping sample IDs → boolean/int sex
        indicator (True / 1 = female). Same key set as ``chron_ages``.

    Examples
    --------
    >>> adapter = GrimAgeAdapter(
    ...     GrimAgeV2(coefficients_path="GrimAgeV2.csv"),
    ...     chron_ages={"GSE40279": ages_series, ...},
    ...     female={"GSE40279": female_series, ...},
    ... )
    >>> results = run_all_clocks(..., clocks={"grimage_age": adapter})
    """

    def __init__(
        self,
        clock: "GrimAgeV2",
        chron_ages: Dict[str, pd.Series],
        female: Dict[str, pd.Series],
    ):
        self._clock = clock
        self._chron_ages = chron_ages  # {dataset_name: Series indexed by sample ID}
        self._female = female          # {dataset_name: Series indexed by sample ID}

    def predict(
        self,
        methylation: pd.DataFrame,
        impute_missing: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Resolve the correct age/sex arrays for this methylation matrix and
        delegate to :meth:`GrimAgeV2.predict`.

        The dataset is identified by matching ``methylation.columns`` against
        the index of each stored Series. The dict key whose Series index has
        the most overlapping sample IDs is used; if none match at all a
        ``ValueError`` is raised.
        """
        sample_ids = list(methylation.columns)

        # ── Find the best-matching dataset ──────────────────────────
        best_key: Optional[str] = None
        best_overlap: int = -1

        for key, ages_series in self._chron_ages.items():
            overlap = len(set(sample_ids) & set(ages_series.index))
            if overlap > best_overlap:
                best_overlap = overlap
                best_key = key

        if best_key is None or best_overlap == 0:
            raise ValueError(
                "GrimAgeAdapter: none of the stored chron_ages Series share "
                "sample IDs with the supplied methylation matrix. "
                "Check that your chron_ages dict keys and Series indices "
                "match the methylation column names."
            )

        if best_overlap < len(sample_ids) and verbose:
            print(
                f"\u26a0  GrimAgeAdapter: matched dataset '{best_key}' "
                f"({best_overlap}/{len(sample_ids)} samples found in chron_ages). "
                "Missing samples will receive NaN ages."
            )

        ages_series = self._chron_ages[best_key]
        sex_series = self._female[best_key]

        # ── Align to methylation column order ───────────────────────
        chron_ages_aligned = ages_series.reindex(sample_ids).values
        female_aligned = sex_series.reindex(sample_ids).values.astype(float)

        result = self._clock.predict(
            methylation,
            chron_ages=chron_ages_aligned,
            female=female_aligned,
            impute_missing=impute_missing,
            verbose=verbose,
        )
        return result[["DNAmAge"]]

    @property
    def coefficients(self) -> pd.DataFrame:
        """Delegate to the underlying GrimAgeV2 instance."""
        return self._clock.coefficients


def run_all_clocks(
    datasets: Union[Dict[str, pd.DataFrame], List[pd.DataFrame]],
    save_path: Union[str, Path],
    clocks: Union[List[Any], Dict[str, Any]],
    *,
    chron_ages: Optional[Dict[str, pd.Series]] = None,
    impute_missing: bool = True,
    impute_within_cohort: bool = True,
    impute_reference: Optional[pd.Series] = None,
    save_summary: bool = True,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Run an arbitrary collection of epigenetic clocks across one or more
    methylation datasets and save per-dataset age prediction matrices.

    For each dataset a CSV is written with one column per clock plus
    chronological age::

        SampleID | horvath_age | hannum_age | phenoage_age | chron_age
        ─────────┼─────────────┼────────────┼──────────────┼──────────
        GSM001   |       62.3  |      58.7  |        60.1  |    65.0

    Output path: ``save_path/{dataset_name}/age_predictions.csv``

    Parameters
    ----------
    datasets : dict or list
        - dict : ``{dataset_name: methylation_df}``
        - list : ``[methylation_df, ...]`` — auto-named
                 ``"dataset_0"``, ``"dataset_1"``, …
        Each methylation DataFrame must be shaped (n_CpGs × n_samples),
        with CpG probe IDs as the index and sample IDs as columns.
    save_path : str or Path
        Root directory under which per-dataset sub-directories are
        created. Created automatically if it does not exist.
    clocks : list or dict
        - list  : ``[ClockInstance, ...]`` — column names are derived
                  from the class name, e.g. ``Horvath → horvath_age``.
                  Duplicate class names get a numeric suffix
                  (``hannum_age``, ``hannum_1_age``, …).
        - dict  : ``{"my_label": ClockInstance, ...}`` — keys are used
                  directly as column names. Useful when you need custom
                  labels or have multiple instances of the same class
                  trained on different data.
        Every clock instance must expose a ``predict(methylation,
        impute_missing, verbose)`` method that returns a DataFrame with
        a ``DNAmAge`` column indexed by ``SampleID``, matching the
        interface of :class:`Horvath` and :class:`Hannum`.
    chron_ages : dict, optional
        ``{dataset_name: pd.Series}`` mapping sample IDs (index) to
        chronological ages in years. If a dataset name is absent,
        ``chron_age`` is filled with NaN for all its samples.
    impute_missing : bool
        Passed through to each clock's ``predict`` call. Applies only to
        clock CpGs still missing after the per-CpG imputation below.
    impute_within_cohort : bool
        If True (default), missing values of a CpG are filled with that
        CpG's median across the dataset's other samples before any clock
        runs (see :func:`impute_per_cpg`).
    impute_reference : pd.Series, optional
        Per-CpG reference values (see :func:`build_cpg_reference`) used for
        CpGs missing in every sample of a dataset. Without it, such CpGs
        fall back to each clock's ``impute_missing`` behaviour (beta 0.5).
    save_summary : bool
        If True, also writes ``save_path/all_datasets_summary.csv``
        stacking all per-dataset matrices with a leading ``dataset``
        column.
    verbose : bool
        Print per-dataset progress, warnings, and save paths.

    Returns
    -------
    dict
        ``{dataset_name: age_predictions_df}`` — one entry per dataset.
        Each DataFrame has one column per clock plus ``chron_age``,
        indexed by ``SampleID``.

    Raises
    ------
    TypeError
        If ``clocks`` is not a list or dict.
    ValueError
        If ``clocks`` or ``datasets`` is empty.

    Examples
    --------
    >>> results = run_all_clocks(
    ...     datasets={"GSE40279": methy_blood, "GSE87571": methy_saliva},
    ...     save_path="outputs/predictions",
    ...     clocks=[
    ...         Horvath(coefficients_path="data/horvath.csv"),
    ...         Hannum(coefficients_path="data/hannum.csv", intercept=0.0),
    ...     ],
    ...     chron_ages={"GSE40279": blood_ages, "GSE87571": saliva_ages},
    ... )

    >>> # Custom column names via dict — useful for same-class variants
    >>> results = run_all_clocks(
    ...     datasets={"GSE40279": methy_blood},
    ...     save_path="outputs/predictions",
    ...     clocks={
    ...         "horvath_age"    : Horvath(coefficients_path="data/horvath.csv"),
    ...         "hannum_age"     : Hannum(coefficients_path="data/hannum.csv"),
    ...         "phenoage_age"   : PhenoAge(coefficients_path="data/phenoage.csv"),
    ...     },
    ... )

    .. note::
        :class:`GrimAgeV2` requires chronological age and biological sex
        at prediction time and cannot be passed to ``run_all_clocks``
        directly. Wrap it in a thin adapter that captures those arrays::

            class GrimAgeAdapter:
                def __init__(self, clock, chron_ages, female):
                    self._clock = clock
                    self._ages  = chron_ages
                    self._sex   = female

                def predict(self, methylation, impute_missing=True, verbose=True):
                    return self._clock.predict(
                        methylation, self._ages, self._sex,
                        impute_missing=impute_missing, verbose=verbose,
                    )[["DNAmAge"]]

            adapter = GrimAgeAdapter(
                GrimAgeV2(coefficients_path="data/GrimAgeV2.csv"),
                blood_ages, female_mask,
            )
            results = run_all_clocks(..., clocks={"grimage_age": adapter})
    """
    _W = 60

    # ── 0. Normalise inputs ─────────────────────────────────────────
    save_path = Path(save_path)
    chron_ages = chron_ages or {}

    if isinstance(datasets, list):
        datasets = {f"dataset_{i}": df for i, df in enumerate(datasets)}

    if not datasets:
        warnings.warn("datasets is empty — nothing to process.", UserWarning, stacklevel=2)
        return {}

    # ── 1. Normalise clocks to {column_name: clock_instance} ────────
    if isinstance(clocks, dict):
        # User supplies column names directly
        named_clocks: Dict[str, Any] = clocks

    elif isinstance(clocks, list):
        # Derive column names from class names; handle duplicates
        seen: Dict[str, int] = {}
        named_clocks = {}
        for clock in clocks:
            base = type(clock).__name__.lower()
            n = seen.get(base, 0)
            seen[base] = n + 1
            col = f"{base}_age" if n == 0 else f"{base}_{n}_age"
            named_clocks[col] = clock

    else:
        raise TypeError("`clocks` must be a list or dict of clock instances.")

    if not named_clocks:
        raise ValueError("`clocks` must not be empty.")

    if verbose:
        clock_labels = ", ".join(named_clocks.keys())
        print(f"\n  Clocks   : {clock_labels}")
        print(f"  Datasets : {', '.join(datasets.keys())}")

    results: Dict[str, pd.DataFrame] = {}

    # ── 2. Per-dataset loop ─────────────────────────────────────────
    for dataset_name, methylation in datasets.items():

        if verbose:
            print(f"\n{'=' * _W}")
            print(f"  Dataset : {dataset_name}")
            print(f"  Samples : {methylation.shape[1]}   CpGs : {methylation.shape[0]}")
            print(f"{'=' * _W}")

        # ── 2a. Restrict to clock CpGs, then per-CpG imputation ─────
        needed = clock_cpgs(named_clocks)
        if needed:
            # clocks only read their own CpGs; working on ~1k rows instead of
            # ~450–865k makes everything below fast
            methylation = methylation.loc[methylation.index.intersection(list(needed))]
        if verbose:
            if needed:
                in_data  = needed & set(methylation.index)
                all_nan  = methylation.loc[list(in_data)].isna().all(axis=1) if in_data else pd.Series(dtype=bool)
                any_nan  = methylation.loc[list(in_data)].isna().any(axis=1) if in_data else pd.Series(dtype=bool)
                n_whole  = len(needed - in_data) + int(all_nan.sum())
                n_part   = int((any_nan & ~all_nan).sum())
                n_ref    = 0
                if impute_reference is not None:
                    whole = (needed - in_data) | set(all_nan[all_nan].index)
                    n_ref = int(impute_reference.reindex(list(whole)).notna().sum())
                print(f"  Clock CpGs : {len(needed)} | partly missing: {n_part} "
                      f"(cohort median) | missing in all samples: {n_whole} "
                      f"({n_ref} from reference, {n_whole - n_ref} → clock fallback)")

        methylation_imp = impute_per_cpg(
            methylation,
            reference=impute_reference,
            within_cohort=impute_within_cohort,
        )
        if impute_reference is not None and needed:
            # CpGs absent from the matrix but available in the reference
            absent = impute_reference.index.difference(methylation_imp.index)
            add = [g for g in absent if g in needed and pd.notna(impute_reference[g])]
            if add:
                extra = pd.DataFrame(
                    np.repeat(impute_reference[add].to_numpy()[:, None],
                              methylation_imp.shape[1], axis=1),
                    index=add, columns=methylation_imp.columns,
                )
                methylation_imp = pd.concat([methylation_imp, extra])
        methylation_imp = methylation_imp.dropna(how="all")

        # ── 2b. Run each clock ──────────────────────────────────────
        clock_frames: List[pd.DataFrame] = []

        for col_name, clock in named_clocks.items():
            if verbose:
                print(f"\n  → Running {col_name} ({type(clock).__name__}) …")
            try:
                pred = (
                    clock
                    .predict(
                        methylation_imp,
                        impute_missing=impute_missing,
                        verbose=verbose,
                    )
                    .rename(columns={"DNAmAge": col_name})
                )
                clock_frames.append(pred)

            except Exception as exc:
                warnings.warn(
                    f"[{dataset_name}] {col_name} prediction failed: {exc}. "
                    f"Filling '{col_name}' with NaN.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                clock_frames.append(
                    pd.DataFrame(
                        {col_name: np.nan},
                        index=pd.Index(methylation.columns, name="SampleID"),
                    )
                )

        # ── 2c. Assemble age matrix ─────────────────────────────────
        age_matrix = (
            pd.concat(clock_frames, axis=1)
            if clock_frames
            else pd.DataFrame(
                index=pd.Index(methylation.columns, name="SampleID")
            )
        )

        # ── 2d. Attach chronological age ────────────────────────────
        if dataset_name in chron_ages:
            chron_series = chron_ages[dataset_name].rename("chron_age")
            chron_series.index.name = "SampleID"
            age_matrix = age_matrix.join(chron_series, how="left")
            missing_n = age_matrix["chron_age"].isna().sum()
            if missing_n and verbose:
                print(
                    f"\u26a0  {missing_n} sample(s) have no matching "
                    "chronological age."
                )
        else:
            age_matrix["chron_age"] = np.nan
            if verbose:
                print(
                    f"  \u2139  No chronological ages for '{dataset_name}'. "
                    "chron_age \u2192 NaN."
                )

        # ── 2e. Save ────────────────────────────────────────────────
        out_dir = save_path / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "age_predictions.csv"
        age_matrix.to_csv(out_path)

        results[dataset_name] = age_matrix

        if verbose:
            print(f"\n  Age prediction statistics:")
            print(
                age_matrix
                .describe()
                .loc[["count", "mean", "std", "min", "max"]]
                .round(2)
                .to_string()
            )
            print(f"\n  \u2713 Saved \u2192 {out_path}")

    # ── 3. Optional cross-dataset summary ───────────────────────────
    if save_summary and results:
        summary_df = pd.concat(
            [df.assign(dataset=name) for name, df in results.items()]
        )
        summary_df.insert(0, "dataset", summary_df.pop("dataset"))

        summary_path = save_path / "all_datasets_summary.csv"
        summary_df.to_csv(summary_path)

        if verbose:
            print(f"\n{'=' * _W}")
            print(f"  Cross-dataset summary")
            print(f"  Datasets : {len(results)}")
            print(f"  Samples  : {len(summary_df)} total")
            print(f"  \u2713 Saved  \u2192 {summary_path}")
            print(f"{'=' * _W}\n")

    return results