"""Warranty check in plain Python. The LLM never does date maths.

check()     is a purchase still covered, and what is the last covered day?
report()    the answer for the customer, used by the warranty checker in the sidebar
describe()  the same as one plain sentence, for handing to the LLM as a fact

python -m gen.warranty    run the checks
"""

import calendar
from datetime import date, timedelta


def add_months(day: date, months: int) -> date:
    """The same day of the month, `months` later.

    When the target month is shorter, the day moves back to its last day:
    31 January + 1 month = 28 (or 29) February.
    """
    years_on, month_index = divmod(day.month - 1 + months, 12)
    year = day.year + years_on
    month = month_index + 1
    last_day_of_month = calendar.monthrange(year, month)[1]
    return date(year, month, min(day.day, last_day_of_month))


def check(purchase: date, months: int, today: date | None = None) -> tuple[bool, date]:
    """Returns (still covered?, last covered day).

    Bought 15 Jan 2024 with a 12-month warranty: covered through 14 Jan 2025.
    """
    last_day = add_months(purchase, months) - timedelta(days=1)
    today = today or date.today()
    return today <= last_day, last_day


def report(name: str, purchase: date, months: int, today: date | None = None) -> str:
    """The warranty answer shown to the customer."""
    covered, last_day = check(purchase, months, today)
    years = f" ({months // 12} years)" if months >= 12 and months % 12 == 0 else ""
    lines = [
        f"**{name}** has a {months}-month warranty{years}.",
        f"Bought on {purchase:%d %b %Y}, it is covered through **{last_day:%d %b %Y}**.",
    ]
    if covered:
        lines.append("Status: **still covered**.")
        lines.append("To make a claim, bring or send the order number and a photo of the problem.")
    else:
        lines.append("Status: **expired**.")
    return "\n\n".join(lines)


def describe(name: str, purchase: date, months: int, today: date | None = None) -> str:
    """The warranty check as one line the LLM can quote without doing any maths."""
    covered, last_day = check(purchase, months, today)
    status = "STILL COVERED" if covered else "EXPIRED"
    return (
        f"Warranty check (computed): {name}, bought {purchase:%d %b %Y}, "
        f"{months}-month warranty, last covered day {last_day:%d %b %Y}. Status: {status}."
    )


if __name__ == "__main__":
    bought = date(2024, 1, 15)

    # A normal date, inside the warranty.
    assert check(bought, 12, date(2024, 6, 1)) == (True, date(2025, 1, 14))
    # The last covered day still counts.
    assert check(bought, 12, date(2025, 1, 14))[0] is True
    # The day after does not.
    assert check(bought, 12, date(2025, 1, 15))[0] is False
    # 31 January + 1 month ends in February, a shorter month.
    assert check(date(2024, 1, 31), 1)[1] == date(2024, 2, 28)

    # The customer-facing report says covered or expired, with the last covered day.
    assert "still covered" in report("MARKUS", bought, 12, date(2024, 6, 1))
    expired = report("MARKUS", bought, 12, date(2026, 1, 1))
    assert "expired" in expired and "14 Jan 2025" in expired and "claim" not in expired

    print("warranty checks pass")
