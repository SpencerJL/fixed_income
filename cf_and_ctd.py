import datetime as dt
import pandas as pd
from utils import generate_coupon_schedule, calculate_clean_price, calculate_accrued_interest, calculate_dirty_price, \
    bond_ytm, calculate_cf, calculate_implied_forward_rate, calculate_implied_repo_rate

# Display the results
if __name__ == "__main__":
    maturity_date = dt.datetime(2026, 1, 1)
    settlement_date = dt.datetime.today()
    coupon_rate = 0.04
    yield_rate = 0.03
    face_value = 100
    tenor_year = 2
    freq = 2
    futures_price = 105
    delivery_date = dt.datetime(2025, 10, 1)
    coupon_dates = generate_coupon_schedule(maturity_date=maturity_date, freq=2, tenor_years=tenor_year)
    clean_price = calculate_clean_price(settlement_date=settlement_date, coupon_dates=coupon_dates,
                                        coupon_rate=coupon_rate, yield_rate=yield_rate, face_value=face_value,
                                        freq=freq)
    accrued_interest, last_coupon, next_coupon, days_accrued = calculate_accrued_interest(
        settlement_date=settlement_date,
        coupon_dates=coupon_dates,
        coupon_rate=coupon_rate,
        face_value=face_value, freq=freq)
    dirty_price = calculate_dirty_price(yield_rate, tenor_year, coupon_rate, maturity_date, settlement_date, freq=freq,
                                        face_value=face_value)
    ytm = bond_ytm(dirty_price, tenor_year, coupon_rate, face=face_value, freq=freq, guess=0.03)
    cf = calculate_cf(coupon_rate, settlement_date, coupon_dates, face_value=100, freq=2, yield_rate=0.06)
    implied_forward_rate = calculate_implied_forward_rate(coupon_rate, coupon_dates, futures_price, settlement_date,
                                                          tenor_year)
    implied_repo_rate = calculate_implied_repo_rate(futures_price=futures_price, coupon_rate=coupon_rate,
                                                    coupon_dates=coupon_dates, settlement_date=settlement_date,
                                                    delivery_date=delivery_date, maturity_years=tenor_year,
                                                    maturity_date=maturity_date, yield_rate_for_dirty=yield_rate)

    print(implied_repo_rate)
