"""Generate a synthetic subscription-telecom customer base with churn labels.

Why synthetic: the project needs demographics, tenure, usage, support interactions and
billing history for the same customers, with a known ground truth, and no public dataset
carries all four cleanly. The generator is built so the modelling work is not trivial:

* Churn comes from a latent logistic model whose signal is deliberately diluted with
  noise, so a good model lands near ROC-AUC 0.85 rather than a meaningless 0.99.
* Six months of per-month usage, billing, ticket and login history are emitted as raw
  columns, so `features.py` has genuine trend/volatility features to construct.
* The raw CSV is intentionally messy -- blank `total_charges` for day-one customers,
  inconsistent `gender`/`region` spellings, structurally missing resolution times and a
  sprinkling of usage outliers -- so `preprocess.py` has real cleaning to do.

Run:  python -m src.generate_data  (from the project folder)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HISTORY_MONTHS = 6
DEFAULT_N_CUSTOMERS = 8000
DEFAULT_CHURN_RATE = 0.26
SEED = 42

REGIONS = ["North", "South", "East", "West", "Central"]
CONTRACTS = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHODS = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
INTERNET = ["DSL", "Fiber optic", "No"]


def _standardise(values: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance -- keeps the latent coefficients interpretable."""
    values = np.asarray(values, dtype=float)
    std = values.std()
    return (values - values.mean()) / std if std > 1e-9 else np.zeros_like(values)


def _solve_intercept(latent: np.ndarray, target_rate: float) -> float:
    """Bisect for the intercept that makes the mean churn probability hit the target."""
    low, high = -12.0, 12.0
    for _ in range(80):
        mid = (low + high) / 2
        rate = 1.0 / (1.0 + np.exp(-(latent + mid)))
        if rate.mean() > target_rate:
            high = mid
        else:
            low = mid
    return (low + high) / 2


def _build_customers(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Demographics, account terms and subscribed services."""
    tenure = np.clip(rng.gamma(shape=1.9, scale=17.0, size=n), 0, 72).round().astype(int)

    # Long-tenure customers skew towards longer contracts, as they do in reality.
    contract_bias = np.clip(tenure / 72.0, 0, 1)
    contract_probs = np.column_stack(
        [0.78 - 0.60 * contract_bias, 0.14 + 0.18 * contract_bias, 0.08 + 0.42 * contract_bias]
    )
    contract_probs /= contract_probs.sum(axis=1, keepdims=True)
    contract = np.array([rng.choice(CONTRACTS, p=p) for p in contract_probs])

    internet = rng.choice(INTERNET, size=n, p=[0.34, 0.44, 0.22])
    has_internet = internet != "No"

    def addon(base_rate: float) -> np.ndarray:
        """An add-on service, only ever sold alongside internet."""
        return np.where(has_internet & (rng.random(n) < base_rate), "Yes", "No")

    phone_service = np.where(rng.random(n) < 0.90, "Yes", "No")

    return pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:06d}" for i in range(1, n + 1)],
            "age": np.clip(rng.normal(46, 15, n), 18, 88).round().astype(int),
            "gender": rng.choice(["Female", "Male"], size=n),
            "senior_citizen": np.where(rng.random(n) < 0.16, 1, 0),
            "partner": rng.choice(["Yes", "No"], size=n, p=[0.48, 0.52]),
            "dependents": rng.choice(["Yes", "No"], size=n, p=[0.30, 0.70]),
            "region": rng.choice(REGIONS, size=n, p=[0.24, 0.22, 0.19, 0.21, 0.14]),
            "tenure_months": tenure,
            "contract": contract,
            "paperless_billing": rng.choice(["Yes", "No"], size=n, p=[0.59, 0.41]),
            "payment_method": rng.choice(PAYMENT_METHODS, size=n, p=[0.34, 0.23, 0.22, 0.21]),
            "autopay": np.where(rng.random(n) < 0.46, "Yes", "No"),
            "phone_service": phone_service,
            "multiple_lines": np.where(
                (phone_service == "Yes") & (rng.random(n) < 0.42), "Yes", "No"
            ),
            "internet_service": internet,
            "online_security": addon(0.37),
            "tech_support": addon(0.35),
            "device_protection": addon(0.39),
            "streaming_tv": addon(0.45),
            "streaming_movies": addon(0.44),
        }
    )


def _build_history(
    rng: np.random.Generator, customers: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Six months of usage, billing, support and login history per customer.

    Month 1 is the most recent. Each customer gets a latent monthly drift, and the
    customers drifting downwards are the ones the model should learn to flag.
    """
    n = len(customers)
    has_internet = (customers["internet_service"] != "No").to_numpy()

    # Monthly price: base fee plus per-service add-ons.
    base_price = np.where(customers["phone_service"].to_numpy() == "Yes", 20.0, 0.0)
    base_price += np.select(
        [
            customers["internet_service"].to_numpy() == "Fiber optic",
            customers["internet_service"].to_numpy() == "DSL",
        ],
        [52.0, 28.0],
        default=0.0,
    )
    for service, price in [
        ("multiple_lines", 9.0),
        ("online_security", 7.0),
        ("tech_support", 7.5),
        ("device_protection", 6.5),
        ("streaming_tv", 11.0),
        ("streaming_movies", 11.0),
    ]:
        base_price += np.where(customers[service].to_numpy() == "Yes", price, 0.0)
    base_price += rng.normal(0, 2.5, n)

    # A fifth of the base gets a mid-history price rise -- the retention team's nightmare.
    price_rise_pct = np.where(rng.random(n) < 0.21, rng.uniform(0.04, 0.22, n), 0.0)

    usage_level = np.where(has_internet, rng.lognormal(4.9, 0.62, n), rng.lognormal(2.2, 0.5, n))
    usage_drift = rng.normal(0.0, 0.085, n)          # per-month multiplicative drift
    login_level = np.clip(rng.normal(22, 9, n), 1, None)
    ticket_propensity = rng.gamma(1.5, 0.55, n)

    history: dict[str, np.ndarray] = {}
    for month in range(1, HISTORY_MONTHS + 1):
        months_back = month - 1  # month 1 = now, month 6 = five months ago

        usage = usage_level * (1.0 + usage_drift) ** (-months_back)
        usage *= rng.lognormal(0.0, 0.14, n)
        history[f"usage_gb_m{month}"] = np.round(np.clip(usage, 0.1, None), 2)

        # The price rise lands three months ago, so months 1-3 carry the higher charge.
        charge = base_price * (1.0 + np.where(months_back <= 2, price_rise_pct, 0.0))
        charge *= rng.lognormal(0.0, 0.02, n)
        history[f"charges_m{month}"] = np.round(np.clip(charge, 15.0, None), 2)

        # Tickets rise as customers grow unhappy (i.e. as usage drifts down).
        unhappiness = np.clip(-usage_drift * 7.0, 0, None)
        rate = ticket_propensity * (1.0 + unhappiness * (1.0 - months_back / HISTORY_MONTHS))
        history[f"tickets_m{month}"] = rng.poisson(np.clip(rate * 0.45, 0.02, None))

        logins = login_level * (1.0 + usage_drift * 0.8) ** (-months_back)
        history[f"logins_m{month}"] = np.clip(rng.normal(logins, 3.5), 0, None).round().astype(int)

    # Customers shorter than the history window have no data in the older months.
    tenure = customers["tenure_months"].to_numpy()
    for month in range(1, HISTORY_MONTHS + 1):
        too_new = tenure < month
        for prefix in ("usage_gb", "charges", "tickets", "logins"):
            column = history[f"{prefix}_m{month}"].astype(float)
            column[too_new] = np.nan
            history[f"{prefix}_m{month}"] = column

    drivers = {
        "usage_drift": usage_drift,
        "price_rise_pct": price_rise_pct,
        "ticket_propensity": ticket_propensity,
        "base_price": base_price,
    }
    return pd.DataFrame(history), drivers


def _build_billing(
    rng: np.random.Generator, customers: pd.DataFrame, history: pd.DataFrame, drivers: dict
) -> pd.DataFrame:
    """Current charges, lifetime value, payment behaviour and support quality."""
    n = len(customers)
    tenure = customers["tenure_months"].to_numpy()

    monthly_charges = np.where(
        np.isnan(history["charges_m1"].to_numpy()),
        drivers["base_price"],
        history["charges_m1"].to_numpy(),
    )
    total_charges = np.round(monthly_charges * tenure * rng.uniform(0.94, 1.03, n), 2)

    late_payments = rng.poisson(
        np.where(customers["autopay"].to_numpy() == "Yes", 0.22, 0.95), n
    )

    tickets_6m = history[[f"tickets_m{m}" for m in range(1, HISTORY_MONTHS + 1)]].sum(
        axis=1, skipna=True
    ).to_numpy()

    # Resolution time only exists for customers who actually raised a ticket.
    avg_resolution_hours = np.where(
        tickets_6m > 0, np.round(rng.gamma(2.2, 7.5, n), 1), np.nan
    )

    return pd.DataFrame(
        {
            "monthly_charges": np.round(monthly_charges, 2),
            "total_charges": total_charges,
            "late_payments_12m": late_payments,
            "support_tickets_6m": tickets_6m.astype(int),
            "escalations_6m": rng.binomial(np.maximum(tickets_6m.astype(int), 0), 0.17),
            "avg_resolution_hours": avg_resolution_hours,
            "satisfaction_score": np.clip(rng.normal(7.1, 1.8, n), 1, 10).round(1),
        }
    )


def _assign_churn(
    rng: np.random.Generator,
    customers: pd.DataFrame,
    billing: pd.DataFrame,
    drivers: dict,
    churn_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw churn from a latent logistic model over the true underlying drivers.

    The latent score has three parts:

    1. additive main effects -- the drivers any linear model can pick up,
    2. interaction and non-linear terms -- a price rise matters far more on a
       no-commitment contract than on a two-year term, and the age effect is U-shaped.
       These are what give the tree ensembles something a logistic regression cannot
       capture, so the model comparison in `train.py` is a fair fight,
    3. an irreducible noise term, sized so an excellent model reaches roughly 0.85
       ROC-AUC -- high enough to be useful, low enough to be honest.
    """
    n = len(customers)
    contract = customers["contract"].to_numpy()

    main_effects = (
        -0.95 * _standardise(np.log1p(customers["tenure_months"].to_numpy()))
        + 1.25 * (contract == "Month-to-month")
        - 0.55 * (contract == "Two year")
        - 0.90 * _standardise(drivers["usage_drift"])          # falling usage -> churn
        + 0.55 * _standardise(drivers["price_rise_pct"])
        + 0.50 * _standardise(np.log1p(billing["support_tickets_6m"].to_numpy()))
        + 0.30 * _standardise(billing["escalations_6m"].to_numpy())
        + 0.45 * _standardise(billing["late_payments_12m"].to_numpy())
        - 0.60 * _standardise(billing["satisfaction_score"].to_numpy())
        - 0.42 * (customers["tech_support"].to_numpy() == "Yes")
        - 0.34 * (customers["online_security"].to_numpy() == "Yes")
        + 0.38 * (customers["internet_service"].to_numpy() == "Fiber optic")
        - 0.45 * (customers["autopay"].to_numpy() == "Yes")
        + 0.30 * (customers["payment_method"].to_numpy() == "Electronic check")
        - 0.20 * _standardise(customers["age"].to_numpy())
        + 0.22 * (customers["dependents"].to_numpy() == "No")
        + 0.25 * _standardise(np.log1p(billing["monthly_charges"].to_numpy()))
    )

    # --- Interactions and non-linearities -------------------------------------------
    month_to_month = (contract == "Month-to-month").astype(float)
    locked_in = (contract != "Month-to-month").astype(float)
    short_tenure = (customers["tenure_months"].to_numpy() <= 12).astype(float)
    price_shock = (drivers["price_rise_pct"] > 0.10).astype(float)
    heavy_support = (billing["support_tickets_6m"].to_numpy() >= 4).astype(float)
    unhappy = (billing["satisfaction_score"].to_numpy() < 6.0).astype(float)
    no_addons = (
        (customers["tech_support"].to_numpy() == "No")
        & (customers["online_security"].to_numpy() == "No")
    ).astype(float)
    fiber = (customers["internet_service"].to_numpy() == "Fiber optic").astype(float)
    steep_decline = (drivers["usage_drift"] < -0.06).astype(float)

    interactions = (
        1.05 * month_to_month * price_shock      # price rise with nothing holding them
        + 0.90 * short_tenure * unhappy          # bad first year is rarely recovered
        + 0.75 * fiber * heavy_support           # premium product, poor experience
        + 0.70 * steep_decline * no_addons       # disengaging with no switching friction
        + 0.60 * month_to_month * steep_decline
        - 0.55 * locked_in * price_shock         # a contract absorbs the same price rise
    )

    # U-shaped age effect: the youngest and the oldest churn most, the middle is stable.
    age_z = _standardise(customers["age"].to_numpy())
    non_linear = 0.35 * (age_z**2 - 1.0)

    latent = main_effects + interactions + non_linear + rng.normal(0.0, 1.30, n)
    latent += _solve_intercept(latent, churn_rate)
    probability = 1.0 / (1.0 + np.exp(-latent))
    return rng.binomial(1, probability), probability


def _add_realistic_mess(rng: np.random.Generator, frame: pd.DataFrame) -> pd.DataFrame:
    """Dirty the frame the way a real CRM export is dirty."""
    frame = frame.copy()
    n = len(frame)

    # Inconsistent category spellings from multiple source systems.
    gender_variants = {"Female": ["Female", "female", "F"], "Male": ["Male", "male", "M"]}
    frame["gender"] = [
        rng.choice(gender_variants[value], p=[0.80, 0.14, 0.06]) for value in frame["gender"]
    ]
    frame["region"] = [
        value if rng.random() > 0.08 else rng.choice([value.upper(), f" {value} ", value.lower()])
        for value in frame["region"]
    ]

    # Day-one customers were never billed, and the export writes a blank string.
    frame["total_charges"] = frame["total_charges"].astype(object)
    frame.loc[frame["tenure_months"] == 0, "total_charges"] = " "

    # A few percent of usage readings are missing outright.
    for month in (1, 2, 3):
        column = f"usage_gb_m{month}"
        frame.loc[rng.random(n) < 0.025, column] = np.nan

    # Meter glitches: implausible usage spikes that preprocessing must cap.
    spike_rows = rng.random(n) < 0.012
    frame.loc[spike_rows, "usage_gb_m1"] = frame.loc[spike_rows, "usage_gb_m1"] * 11.0

    # A handful of duplicated CRM rows.
    duplicates = frame.sample(n=max(int(0.004 * n), 1), random_state=7)
    frame = pd.concat([frame, duplicates], ignore_index=True)

    return frame.sample(frac=1.0, random_state=11).reset_index(drop=True)


def generate(
    n_customers: int = DEFAULT_N_CUSTOMERS,
    churn_rate: float = DEFAULT_CHURN_RATE,
    seed: int = SEED,
    messy: bool = True,
) -> pd.DataFrame:
    """Build the full customer table, churn label included."""
    rng = np.random.default_rng(seed)

    customers = _build_customers(rng, n_customers)
    history, drivers = _build_history(rng, customers)
    billing = _build_billing(rng, customers, history, drivers)
    churned, probability = _assign_churn(rng, customers, billing, drivers, churn_rate)

    frame = pd.concat([customers, billing, history], axis=1)
    frame["churned"] = churned
    # Kept out of modelling; used in the report to sanity-check achievable performance.
    frame["_true_churn_probability"] = np.round(probability, 4)

    return _add_realistic_mess(rng, frame) if messy else frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic churn data.")
    parser.add_argument("--customers", type=int, default=DEFAULT_N_CUSTOMERS)
    parser.add_argument("--churn-rate", type=float, default=DEFAULT_CHURN_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).absolute().parent.parent / "data" / "raw" / "customers.csv",
    )
    args = parser.parse_args()

    frame = generate(args.customers, args.churn_rate, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    print(f"wrote {len(frame):,} rows x {frame.shape[1]} columns -> {args.out}")
    print(f"churn rate: {frame['churned'].mean():.1%}")
    print(f"missing cells: {int(frame.isna().sum().sum()):,}")


if __name__ == "__main__":
    main()
