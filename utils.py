import numpy as np
from scipy.optimize import newton
import datetime as dt
from typing import List, Tuple
from dateutil.relativedelta import relativedelta

# Generate coupon schedule back from maturity
from datetime import datetime
from dateutil.relativedelta import relativedelta
import calendar
from typing import List


def is_month_end(date: datetime) -> bool:
    return date.day == calendar.monthrange(date.year, date.month)[1]


def generate_coupon_schedule(maturity_date: datetime, freq: int = 2, tenor_years: int = 2) -> List[datetime]:
    """Generate coupon dates for a bond with given tenor, ending at maturity."""
    months = 12 // freq
    coupon_dates = []
    current = maturity_date
    eom = is_month_end(maturity_date)
    num_payments = tenor_years * freq

    for _ in range(num_payments):
        coupon_dates.insert(0, current)
        current -= relativedelta(months=months)
        if eom:
            last_day = calendar.monthrange(current.year, current.month)[1]
            current = current.replace(day=last_day)

    return coupon_dates


# Function to compute clean price
def calculate_clean_price(
        settlement_date: dt.datetime,
        coupon_dates: List[dt.datetime],
        coupon_rate: float,
        yield_rate: float,
        face_value: float,
        freq: int
) -> float:
    coupon_payment = face_value * coupon_rate / freq
    cash_flows = []
    times = []
    for d in coupon_dates:
        if d > settlement_date:
            cf = coupon_payment
            if d == coupon_dates[-1]:  # maturity
                cf += face_value
            t = (d - settlement_date).days / 365
            cash_flows.append(cf)
            times.append(t)
    clean_price = sum([cf / (1 + yield_rate / freq) ** (t * freq) for cf, t in zip(cash_flows, times)])
    return clean_price


# Function to compute accrued interest
def calculate_accrued_interest(
        settlement_date: dt.datetime,
        coupon_dates: List[dt.datetime],
        coupon_rate: float,
        face_value: float,
        freq: int
) -> Tuple[float, dt.datetime, dt.datetime, int]:
    last_coupon = max([d for d in coupon_dates if d <= settlement_date])
    next_coupon = min([d for d in coupon_dates if d > settlement_date])
    days_between = (next_coupon - last_coupon).days
    days_accrued = (settlement_date - last_coupon).days
    coupon_payment = face_value * coupon_rate / freq
    accrued = coupon_payment * days_accrued / days_between
    return accrued, last_coupon, next_coupon, days_accrued


# --- Dirty Price ---
def calculate_dirty_price(yield_rate, maturity, coupon_rate, maturity_date, settlement_date, freq=2,
                     face_value=100) -> float:
    coupon_dates = generate_coupon_schedule(maturity_date=maturity_date, freq=freq, tenor_years=maturity)
    clean_price = calculate_clean_price(settlement_date=settlement_date, coupon_dates=coupon_dates,
                                        coupon_rate=coupon_rate, yield_rate=yield_rate, face_value=face_value,
                                        freq=freq)
    accrued_interest, last_coupon, next_coupon, days_accrued = calculate_accrued_interest(
        settlement_date=settlement_date, coupon_dates=coupon_dates,
        coupon_rate=coupon_rate, face_value=face_value, freq=freq)
    return clean_price + accrued_interest


# --- Yield to Maturity ---
def bond_ytm(price, maturity, coupon, face=100, freq=2, guess=0.03):
    n = int(maturity * freq)
    c = coupon * face / freq

    def price_diff(y):
        y = y / freq
        pv_coupons = np.sum([c / (1 + y) ** t for t in range(1, n + 1)])
        pv_face = face / (1 + y) ** n
        return pv_coupons + pv_face - price

    return newton(price_diff, guess)


# Calculate Conversion Factor using 6% semiannual yield assumption
def calculate_cf(coupon_rate, settlement_date, coupon_dates, face_value=100, freq=2, yield_rate=0.06):
    clean_price = calculate_clean_price(settlement_date=settlement_date, coupon_dates=coupon_dates,
                                        coupon_rate=coupon_rate, yield_rate=yield_rate, face_value=face_value,
                                        freq=freq)
    return round(clean_price / 100, 4)


def calculate_implied_forward_rate(coupon_rate, coupon_dates, futures_price, settlement_date, tenor_year):
    cf = calculate_cf(coupon_rate, settlement_date, coupon_dates)
    forward_price = futures_price * cf
    implied_forward_yield = bond_ytm(forward_price, tenor_year, coupon_rate)

    return implied_forward_yield


def calculate_implied_repo_rate(
    futures_price: float,
    coupon_rate: float,
    coupon_dates: List[dt.datetime],
    settlement_date: dt.datetime,       # "today"
    delivery_date: dt.datetime,         # futures delivery/expiry date
    maturity_years: int,                # tenor used by your calculate_dirty_price
    maturity_date: dt.datetime,         # bond maturity
    yield_rate_for_dirty: float,        # yield used to build today's dirty price
    face_value: float = 100.0,
    freq: int = 2,
    day_count_base: int = 360           # annualization base (repo convention)
) -> float:
    """
    Implied repo rate (annualized) for delivering this bond into the futures.

    IRR = [ (F * CF + AI_delivery) - Dirty_today ] / Dirty_today * (base / days_to_delivery)

    Notes:
      - CF is evaluated at the delivery date by convention.
      - AI_delivery is accrued interest as of the delivery date.
      - Dirty_today is your current dirty price (built via your pricing functions here).
    """

    # 1) Conversion factor at DELIVERY date (pass delivery_date as 'settlement_date' to your CF helper)
    cf = calculate_cf(
        coupon_rate=coupon_rate,
        settlement_date=delivery_date,       # CF convention: evaluated for delivery
        coupon_dates=coupon_dates,
        face_value=face_value,
        freq=freq,
        yield_rate=0.06                      # per futures convention in your helper
    )

    # 2) Accrued interest AT DELIVERY (added to the invoice price)
    ai_delivery, _, _, _ = calculate_accrued_interest(
        settlement_date=delivery_date,
        coupon_dates=coupon_dates,
        coupon_rate=coupon_rate,
        face_value=face_value,
        freq=freq
    )

    # 3) Today's dirty price (you chose to build via yield; alternatively you could pass a market dirty price)
    dirty_today = calculate_dirty_price(
        yield_rate=yield_rate_for_dirty,
        maturity=maturity_years,
        coupon_rate=coupon_rate,
        maturity_date=maturity_date,
        settlement_date=settlement_date,
        freq=freq,
        face_value=face_value
    )

    # 4) Time to delivery (ACT)
    days_to_delivery = (delivery_date - settlement_date).days
    if days_to_delivery <= 0:
        raise ValueError("delivery_date must be after settlement_date")

    # 5) Invoice proceeds and IRR
    invoice_price = futures_price * cf + ai_delivery
    irr = (invoice_price - dirty_today) / dirty_today * (day_count_base / days_to_delivery)

    return irr

def calculate_gross_basis_clean(
    clean_price_today: float,
    futures_price: float,
    conversion_factor: float
) -> float:
    """Gross basis using clean price."""
    return clean_price_today - futures_price * conversion_factor


def calculate_gross_basis_dirty(
    dirty_price_today: float,
    futures_price: float,
    conversion_factor: float
) -> float:
    """Gross basis using dirty price."""
    return dirty_price_today - futures_price * conversion_factor


def calculate_net_basis(
    dirty_price_today: float,
    futures_price: float,
    conversion_factor: float,
    ai_delivery: float
) -> float:
    """Net basis = dirty today - invoice (F*CF + AI at delivery)."""
    invoice = futures_price * conversion_factor + ai_delivery
    return dirty_price_today - invoice


import datetime as dt
from typing import List, Dict

def calculate_carry_to_delivery(
    settlement_date: dt.datetime,
    delivery_date: dt.datetime,
    coupon_dates: List[dt.datetime],
    coupon_rate: float,
    repo_rate: float,
    dirty_today: float,
    face_value: float = 100.0,
    freq: int = 2,
    day_count_base: int = 360
) -> float:
    """
    Income carry to delivery (no roll):
      Carry = (AI_T - AI_0) + sum(coupons before delivery) - repo_rate * dirty_today * (T/day_count_base)
    """
    # Accrued today and at delivery
    ai_today, _, _, _ = calculate_accrued_interest(
        settlement_date=settlement_date,
        coupon_dates=coupon_dates,
        coupon_rate=coupon_rate,
        face_value=face_value,
        freq=freq
    )
    ai_deliv, _, _, _ = calculate_accrued_interest(
        settlement_date=delivery_date,
        coupon_dates=coupon_dates,
        coupon_rate=coupon_rate,
        face_value=face_value,
        freq=freq
    )
    # Coupons received before delivery
    c_semi = face_value * coupon_rate / freq
    coupons_received = sum(
        c_semi for d in coupon_dates if (settlement_date < d <= delivery_date and d != coupon_dates[-1])
    )
    # Financing cost
    T = (delivery_date - settlement_date).days
    financing_cost = repo_rate * dirty_today * (T / day_count_base)
    # Carry
    return (ai_deliv - ai_today) + coupons_received - financing_cost


def calculate_basis_net_of_carry(
    futures_price: float,
    coupon_rate: float,
    maturity_date: dt.datetime,
    settlement_date: dt.datetime,
    delivery_date: dt.datetime,
    maturity_years: int,
    yield_rate_for_dirty: float,
    repo_rate: float,
    face_value: float = 100.0,
    freq: int = 2,
    day_count_base: int = 360
) -> Dict[str, float]:
    """
    Returns net basis, carry to delivery, and basis net of carry.
    Uses your helpers: generate_coupon_schedule, calculate_clean_price, calculate_accrued_interest, calculate_dirty_price, calculate_cf.
    """
    # Coupon schedule
    coupon_dates: List[dt.datetime] = generate_coupon_schedule(
        maturity_date=maturity_date, freq=freq, tenor_years=maturity_years
    )

    # Today's prices
    clean_today = calculate_clean_price(
        settlement_date=settlement_date, coupon_dates=coupon_dates,
        coupon_rate=coupon_rate, yield_rate=yield_rate_for_dirty,
        face_value=face_value, freq=freq
    )
    dirty_today = calculate_dirty_price(
        yield_rate=yield_rate_for_dirty, maturity=maturity_years, coupon_rate=coupon_rate,
        maturity_date=maturity_date, settlement_date=settlement_date,
        freq=freq, face_value=face_value
    )

    # CF and AI at delivery
    cf = calculate_cf(
        coupon_rate=coupon_rate,
        settlement_date=delivery_date,     # CF convention: at delivery
        coupon_dates=coupon_dates,
        face_value=face_value, freq=freq, yield_rate=0.06
    )
    ai_deliv, _, _, _ = calculate_accrued_interest(
        settlement_date=delivery_date,
        coupon_dates=coupon_dates,
        coupon_rate=coupon_rate,
        face_value=face_value,
        freq=freq
    )

    # Net basis today (invoice-consistent)
    net_basis_today = dirty_today - (futures_price * cf + ai_deliv)

    # Carry to delivery
    carry_to_delivery = calculate_carry_to_delivery(
        settlement_date=settlement_date, delivery_date=delivery_date,
        coupon_dates=coupon_dates, coupon_rate=coupon_rate,
        repo_rate=repo_rate, dirty_today=dirty_today,
        face_value=face_value, freq=freq, day_count_base=day_count_base
    )

    # Basis net of carry
    bnc = net_basis_today - carry_to_delivery

    return {
        "clean_today": round(clean_today, 8),
        "dirty_today": round(dirty_today, 8),
        "conversion_factor": round(cf, 8),
        "ai_delivery": round(ai_deliv, 8),
        "net_basis_today": round(net_basis_today, 8),
        "carry_to_delivery": round(carry_to_delivery, 8),
        "basis_net_of_carry": round(bnc, 8),
    }

