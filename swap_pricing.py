from dataclasses import dataclass
from datetime import datetime
from dateutil.relativedelta import relativedelta
import math
from typing import List, Tuple

# ---------------- Day-counts & schedules ----------------

def year_fraction(d1: datetime, d2: datetime, day_count: str = "ACT/360") -> float:
    dc = day_count.upper()
    if dc in ("ACT/360", "ACT360", "A/360"):
        return (d2 - d1).days / 360.0
    if dc in ("ACT/365F", "ACT/365", "A/365F"):
        return (d2 - d1).days / 365.0
    if dc in ("30/360", "30E/360", "30_360"):
        d1y, d1m, d1d = d1.year, d1.month, min(d1.day, 30)
        d2y, d2m, d2d = d2.year, d2.month, min(d2.day, 30)
        return ((d2y - d1y) * 360 + (d2m - d1m) * 30 + (d2d - d1d)) / 360.0
    raise ValueError(f"Unsupported day count: {day_count}")

def generate_schedule(start: datetime, end: datetime, freq_per_year: int) -> List[datetime]:
    """Unadjusted schedule of period-end dates (no BD roll)."""
    months = 12 // freq_per_year
    dates, d = [], start
    while True:
        d = d + relativedelta(months=+months)
        if d >= end:
            dates.append(end)
            break
        dates.append(d)
    return dates

# ---------------- Discount curve (continuous zero) ----------------

@dataclass
class ZeroCurve:
    valuation_date: datetime
    pillars_years: List[float]   # e.g. [0.25, 0.5, 1, 2, 3, 5, 7, 10]
    zeros_cc: List[float]        # continuous zero rates, e.g. 0.045

    def _interp_zero(self, T: float) -> float:
        if T <= 0: return 0.0
        xs, ys = self.pillars_years, self.zeros_cc
        if T <= xs[0]: return ys[0]
        if T >= xs[-1]: return ys[-1]
        for i in range(1, len(xs)):
            if T <= xs[i]:
                x0, x1 = xs[i-1], xs[i]
                y0, y1 = ys[i-1], ys[i]
                w = (T - x0) / (x1 - x0)
                return y0 + w * (y1 - y0)
        return ys[-1]

    def df(self, T: float) -> float:
        z = self._interp_zero(T)
        return math.exp(-z * T)

    def year_frac_from_val(self, d: datetime) -> float:
        return (d - self.valuation_date).days / 365.0

    def forward_simple(self, T0: float, T1: float) -> float:
        """Simple-annualized forward over [T0,T1] derived from discount factors."""
        P0, P1 = self.df(T0), self.df(T1)
        alpha = T1 - T0
        return 0.0 if alpha <= 0 else (P0 / P1 - 1.0) / alpha

# ---------------- Swap valuation ----------------

@dataclass
class PlainVanillaSwap:
    notional: float
    fixed_rate: float              # deal fixed rate K (decimal)
    fixed_freq: int
    fixed_dc: str
    float_freq: int
    float_dc: str
    effective: datetime
    maturity: datetime
    valuation: datetime
    curve: ZeroCurve

    def _fixed_leg_annuity(self) -> Tuple[float, List[Tuple[datetime, float, float]]]:
        """Annuity A = Σ α_i P(0,T_i) and detail (date, alpha, DF)."""
        pay_dates = generate_schedule(self.effective, self.maturity, self.fixed_freq)
        A, detail, prev = 0.0, [], self.effective
        for d in pay_dates:
            if d <= self.valuation:
                prev = d
                continue
            alpha = year_fraction(prev, d, self.fixed_dc)
            T = self.curve.year_frac_from_val(d)
            df = self.curve.df(T)
            A += alpha * df
            detail.append((d, alpha, df))
            prev = d
        return A, detail

    def _float_leg_pv(self) -> Tuple[float, List[Tuple[datetime, float, float, float]]]:
        """PV of remaining floating coupons (no principal redemption).
           On a reset date: PV_float = N * (1 - P(0,T_n))."""
        if self.valuation == self.effective:
            Tn = self.curve.year_frac_from_val(self.maturity)
            return self.notional * (1.0 - self.curve.df(Tn)), []

        pay_dates = generate_schedule(self.effective, self.maturity, self.float_freq)
        pv, detail, prev = 0.0, [], self.effective
        for d in pay_dates:
            if d <= self.valuation:
                prev = d
                continue
            alpha = year_fraction(prev, d, self.float_dc)
            T0 = self.curve.year_frac_from_val(prev)
            T1 = self.curve.year_frac_from_val(d)
            fwd = self.curve.forward_simple(T0, T1)
            df  = self.curve.df(T1)
            pv += self.notional * fwd * alpha * df
            detail.append((d, alpha, fwd, df))
            prev = d
        return pv, detail

    def par_rate(self) -> float:
        """R* = PV_float / (N*A). On reset date this equals (1 - P(0,Tn)) / A."""
        A, _ = self._fixed_leg_annuity()
        pv_float, _ = self._float_leg_pv()
        return 0.0 if A == 0 else (pv_float / self.notional) / A

    def pv(self, payer: bool = True) -> float:
        """payer=True => pay fixed, receive float."""
        A, _ = self._fixed_leg_annuity()
        pv_float, _ = self._float_leg_pv()
        pv_fixed = self.notional * self.fixed_rate * A
        return (pv_float - pv_fixed) if payer else (pv_fixed - pv_float)

    def pv01(self) -> float:
        A, _ = self._fixed_leg_annuity()
        return self.notional * A * 1e-4

# ---------------- Example ----------------
if __name__ == "__main__":
    val_date = datetime(2025, 8, 31)
    eff_date = val_date                 # price on reset date (clean identity holds)
    mat_date = datetime(2030, 8, 31)

    # Toy OIS zero curve (continuous comp), for illustration only
    pillars = [0.25, 0.5, 1, 2, 3, 5, 7]
    zeros   = [0.048, 0.0475, 0.047, 0.046, 0.045, 0.0435, 0.043]

    curve = ZeroCurve(valuation_date=val_date, pillars_years=pillars, zeros_cc=zeros)

    swap = PlainVanillaSwap(
        notional=100_000_000,
        fixed_rate=0.0450,           # 4.50% deal rate
        fixed_freq=2, fixed_dc="30/360",
        float_freq=4, float_dc="ACT/360",
        effective=eff_date, maturity=mat_date, valuation=val_date,
        curve=curve
    )

    par = swap.par_rate()
    pv_payer = swap.pv(payer=True)
    pv01 = swap.pv01()
    pv_approx = (par - swap.fixed_rate) * pv01 / 1e-4  # (R*-K)*PV01

    print(f"Par swap rate (R*): {par*100:.3f}%")
    print(f"PV (payer, pay fixed @ 4.50%): {pv_payer:,.0f}")
    print(f"PV01: {pv01:,.0f}")
    print(f"Approx PV via (R*-K)*PV01: {pv_approx:,.0f}")
