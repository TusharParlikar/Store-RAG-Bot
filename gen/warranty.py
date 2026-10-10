"""Warranty check in plain Python. The LLM never does date maths."""
import calendar
from datetime import date, timedelta


def add_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    year, month = d.year + y, m + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def check(purchase: date, months: int, today: date | None = None) -> tuple[bool, date]:
    """(covered, last covered day). Bought 2024-01-15 with 12 months: covered through 2025-01-14."""
    last_day = add_months(purchase, months) - timedelta(days=1)
    return (today or date.today()) <= last_day, last_day


def describe(name: str, purchase: date, months: int, today: date | None = None) -> str:
    covered, last_day = check(purchase, months, today)
    state = "STILL COVERED" if covered else "EXPIRED"
    return (f"Warranty check (computed): {name}, bought {purchase:%d %b %Y}, "
            f"{months}-month warranty, last covered day {last_day:%d %b %Y}. Status: {state}.")


if __name__ == "__main__":
    bought = date(2024, 1, 15)
    assert check(bought, 12, date(2024, 6, 1)) == (True, date(2025, 1, 14))   # normal date
    assert check(bought, 12, date(2025, 1, 14))[0] is True                    # last day
    assert check(bought, 12, date(2025, 1, 15))[0] is False                   # day after
    assert check(date(2024, 1, 31), 1)[1] == date(2024, 2, 28)                # short month clamps
    print("warranty checks pass")
