import numpy as np
from scipy.optimize import newton
from datetime import date

# --- Clean Price ---
def bond_clean_price(yield_rate, maturity, coupon, face=100, freq=2):
    n = int(maturity * freq)
    y = yield_rate / freq
    c = coupon * face / freq
    pv_coupons = np.sum([c / (1 + y)**t for t in range(1, n + 1)])
    pv_face = face / (1 + y)**n
    return pv_coupons + pv_face

# --- Accrued Interest ---
def accrued_interest(coupon, last_coupon_date, settlement_date, freq=2):
    days_between = (settlement_date - last_coupon_date).days
    days_in_period = 365 // freq
    return coupon * 100 / freq * (days_between / days_in_period)

# --- Dirty Price ---
def bond_dirty_price(clean_price, coupon, last_coupon_date, settlement_date, freq=2):
    ai = accrued_interest(coupon, last_coupon_date, settlement_date, freq)
    return clean_price + ai

# --- Yield to Maturity ---
def bond_ytm(price, maturity, coupon, face=100, freq=2, guess=0.03):
    n = int(maturity * freq)
    c = coupon * face / freq

    def price_diff(y):
        y = y / freq
        pv_coupons = np.sum([c / (1 + y)**t for t in range(1, n + 1)])
        pv_face = face / (1 + y)**n
        return pv_coupons + pv_face - price

    return newton(price_diff, guess)

# --- Carry Calculation Function ---
# def calculate_carry(coupon_dates, today, carry_days, financing_rate, coupon_rate, face=100, price=100):
#     year = today.year
#     coupon_datetimes = [datetime.strptime(f"{day}-{year}", "%d-%m-%Y") for day in coupon_dates]
#     coupon_datetimes.sort()
#
#     last_coupon_date = max([d for d in coupon_datetimes if d <= today])
#     next_coupon_date = min([d for d in coupon_datetimes if d > today] + [d.replace(year=year + 1) for d in coupon_datetimes])
#
#     # Accrued interest
#     days_in_period = (next_coupon_date - last_coupon_date).days
#     days_accrued = (today - last_coupon_date).days
#     accrued_interest = (coupon_rate / 2) * face * (days_accrued / days_in_period)
#
#     # Carry calculations
#     carry_interest = price * financing_rate * (carry_days / 365)
#     expected_coupon = (coupon_rate / 2) * face if today + timedelta(days=carry_days) > next_coupon_date else 0
#     net_carry = expected_coupon - carry_interest
#
#     return {
#         "Last Coupon Date": last_coupon_date.date(),
#         "Next Coupon Date": next_coupon_date.date(),
#         "Days Accrued": days_accrued,
#         "Accrued Interest": round(accrued_interest, 4),
#         "Carry Interest Cost": round(carry_interest, 4),
#         "Expected Coupon in Period": round(expected_coupon, 4),
#         "Net Carry": round(net_carry, 4)
#     }

# --- Yield & Carry Calculation ---
# ytm = bond_yield(price, maturity, coupon)
# recalculated_price = bond_price(ytm / 100, maturity, coupon)
# carry_result = calculate_carry(coupon_dates, today, carry_days, financing_rate, coupon)
#
# print({
#     "Yield to Maturity (%)": round(ytm, 4),
#     "Recalculated Price": round(recalculated_price, 4),
#     **carry_result
# })

def main():
    print("=== Bond Pricing Examples ===\n")

    # 1️⃣ Case 1: 30Y Treasury issued 30-Jun-2010 (coupon 1-Jan and 1-Jul)
    issue_date = date(2010, 6, 30)
    settlement_date = date(2025, 8, 25)
    last_coupon = date(2025, 7, 1)
    maturity_date = date(2040, 6, 30)
    coupon = 0.04  # 4%
    freq = 2
    years_to_maturity = (maturity_date - settlement_date).days / 365

    # Assume market yield for similar bond is 4.2%
    ytm_guess = 0.042
    clean = bond_clean_price(ytm_guess, years_to_maturity, coupon, freq=freq)
    dirty = bond_dirty_price(clean, coupon, last_coupon, settlement_date, freq=freq)
    implied_ytm = bond_ytm(clean, years_to_maturity, coupon, freq=freq)

    print("▶️ 30Y Treasury issued 30-Jun-2010")
    print(f"Clean Price: {clean:.4f}")
    print(f"Dirty Price: {dirty:.4f}")
    print(f"Implied YTM: {implied_ytm:.4%}\n")

    # 2️⃣ Case 2: 2Y Treasury issued on 1-Jan-2025, pricing on 25-Nov-2024
    issue_date2 = date(2025, 1, 1)
    settlement_date2 = date(2024, 11, 25)  # 3 months before issue
    maturity_date2 = date(2027, 1, 1)
    coupon2 = 0.045  # 4.5%
    last_coupon2 = date(2024, 7, 1)
    years_to_maturity2 = (maturity_date2 - settlement_date2).days / 365

    ytm_guess2 = 0.044
    clean2 = bond_clean_price(ytm_guess2, years_to_maturity2, coupon2, freq=freq)
    dirty2 = bond_dirty_price(clean2, coupon2, last_coupon2, settlement_date2, freq=freq)
    implied_ytm2 = bond_ytm(clean2, years_to_maturity2, coupon2, freq=freq)

    print("▶️ 2Y Treasury issued 1-Jan-2025 (viewed on 25-Nov-2024)")
    print(f"Clean Price: {clean2:.4f}")
    print(f"Dirty Price: {dirty2:.4f}")
    print(f"Implied YTM: {implied_ytm2:.4%}\n")

    # 3️⃣ Extra Case: 10Y Zero Coupon Treasury Strip
    print("▶️ 10Y Zero Coupon Treasury Strip")
    zero_coupon = 0.00
    maturity3 = 10
    ytm3 = 0.045
    clean3 = bond_clean_price(ytm3, maturity3, zero_coupon)
    implied_ytm3 = bond_ytm(clean3, maturity3, zero_coupon)

    print(f"Clean Price: {clean3:.4f}")
    print(f"Dirty Price (same as clean): {clean3:.4f}")
    print(f"Implied YTM: {implied_ytm3:.4%}")

if __name__ == "__main__":
    main()



