"""Synthetic HR dataset generator for the anomaly detection pipeline.

Generates one row per employee covering the four required data domains --
payroll, attendance, leave, performance -- with deliberate, realistic
correlations:
  - pay scales with department base rate * level multiplier * tenure, plus
    small natural noise within a (department, level) cohort
  - leave balance respects a fixed annual allotment
On top of that "normal" population, a small known fraction of records are
shocked into payroll outliers / overtime breaches / leave abusers. Knowing
exactly which rows were injected lets us sanity-check the detectors'
recall against ground truth (see the summary printed at the bottom).

The dataset represents a single quarter (Q1), so `leave_taken_qtr` is the
same as leave taken year-to-date -- there's no multi-quarter history here.
"""

import csv
import random
from pathlib import Path

import numpy as np

SEED = 42
NUM_EMPLOYEES = 800

DEPARTMENTS = [
    "Engineering",
    "Sales",
    "Finance",
    "HR",
    "Operations",
    "Customer Support",
    "Product",
]

# arbitrary monthly currency units -- absolute scale doesn't matter since
# detection is relative (z-score within a cohort), only the ratios do.
DEPARTMENT_BASE_SALARY = {
    "Engineering": 9000,
    "Product": 8800,
    "Finance": 8200,
    "Sales": 7000,
    "HR": 6500,
    "Operations": 6200,
    "Customer Support": 5500,
}

TITLES_BY_DEPARTMENT = {
    "Engineering": ["Associate Engineer", "Engineer", "Senior Engineer", "Staff Engineer", "Principal Engineer"],
    "Product": ["Associate PM", "Product Manager", "Senior PM", "Group PM", "Director of Product"],
    "Finance": ["Finance Analyst", "Senior Analyst", "Finance Manager", "Senior Manager", "Finance Director"],
    "Sales": ["SDR", "Account Executive", "Senior AE", "Sales Manager", "Sales Director"],
    "HR": ["HR Associate", "HR Generalist", "Senior HR Generalist", "HR Manager", "HR Director"],
    "Operations": ["Ops Associate", "Ops Analyst", "Ops Manager", "Senior Ops Manager", "Ops Director"],
    "Customer Support": [
        "Support Associate",
        "Support Specialist",
        "Senior Specialist",
        "Support Lead",
        "Support Manager",
    ],
}

LEVELS = [1, 2, 3, 4, 5]
LEVEL_MULTIPLIER = {1: 1.0, 2: 1.3, 3: 1.7, 4: 2.2, 5: 2.8}
LEVEL_WEIGHTS = [0.30, 0.28, 0.20, 0.14, 0.08]  # seniority pyramid: most ICs, few directors

ANNUAL_LEAVE_ALLOTMENT_DAYS = 24

PAYROLL_OUTLIER_RATE = 0.05
OVERTIME_BREACH_RATE = 0.04
LEAVE_ABUSE_RATE = 0.06
MISSING_TRAINING_RATE = 0.08

FIRST_NAMES = [
    "Aisha", "Liam", "Noor", "Ravi", "Maya", "Omar", "Zara", "Kabir", "Lena", "Arjun",
    "Sofia", "Yusuf", "Priya", "Mateo", "Layla", "Dev", "Hana", "Karim", "Nadia", "Aarav",
    "Elena", "Tariq", "Mira", "Sami", "Anika", "Diego", "Farah", "Ivan", "Nisha", "Bilal",
]
LAST_NAMES = [
    "Khan", "Mehta", "Silva", "Patel", "Hassan", "Rao", "Costa", "Sheikh", "Iyer", "Naidu",
    "Farooq", "Gomes", "Reddy", "Malik", "Pinto", "Joshi", "Haddad", "Nair", "Qureshi", "Fonseca",
]


def generate_employees(n: int = NUM_EMPLOYEES, seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """Returns (employee_rows, injected_ground_truth). The ground truth list
    is only for self-checking detector recall -- it is not written to the
    dataset file, since a real system would never know this in advance.
    """
    rng = np.random.default_rng(seed)
    py_random = random.Random(seed)

    rows: list[dict] = []
    ground_truth: list[dict] = []

    for i in range(n):
        department = py_random.choice(DEPARTMENTS)
        level = py_random.choices(LEVELS, weights=LEVEL_WEIGHTS, k=1)[0]
        title = TITLES_BY_DEPARTMENT[department][level - 1]
        tenure_months = max(1, int(rng.normal(18 + level * 10, 12)))

        base = DEPARTMENT_BASE_SALARY[department] * LEVEL_MULTIPLIER[level]
        tenure_bonus = min(tenure_months / 6 * 0.005, 0.10)
        noise = rng.normal(1.0, 0.06)
        monthly_salary = base * (1 + tenure_bonus) * noise

        is_payroll_outlier = py_random.random() < PAYROLL_OUTLIER_RATE
        if is_payroll_outlier:
            shock = py_random.choice([rng.uniform(1.4, 1.8), rng.uniform(0.5, 0.7)])
            monthly_salary *= shock
        monthly_salary = round(float(monthly_salary), 2)

        overtime_hours_week = max(0.0, round(float(rng.normal(3.0, 2.5)), 1))
        is_overtime_breach = py_random.random() < OVERTIME_BREACH_RATE
        if is_overtime_breach:
            overtime_hours_week = round(float(rng.uniform(15, 26)), 1)

        mandatory_training_completed = py_random.random() > MISSING_TRAINING_RATE

        is_leave_abuser = py_random.random() < LEAVE_ABUSE_RATE
        if is_leave_abuser:
            leave_taken_qtr = py_random.randint(16, 22)
            leave_long_weekend_count = int(leave_taken_qtr * py_random.uniform(0.55, 0.85))
        else:
            leave_taken_qtr = max(0, int(rng.normal(5, 3)))
            leave_long_weekend_count = int(leave_taken_qtr * py_random.uniform(0.0, 0.3))
        leave_long_weekend_count = min(leave_long_weekend_count, leave_taken_qtr)
        leave_balance_days = max(0, ANNUAL_LEAVE_ALLOTMENT_DAYS - leave_taken_qtr)

        performance_rating = int(np.clip(round(rng.normal(3 + level * 0.15, 0.8)), 1, 5))
        attendance_rate = round(float(np.clip(rng.normal(0.96, 0.04), 0.5, 1.0)), 3)

        employee_id = f"EMP-{i + 1:04d}"
        rows.append(
            {
                "employee_id": employee_id,
                "name": f"{py_random.choice(FIRST_NAMES)} {py_random.choice(LAST_NAMES)}",
                "department": department,
                "level": level,
                "title": title,
                "tenure_months": tenure_months,
                "monthly_salary": monthly_salary,
                "overtime_hours_week": overtime_hours_week,
                "mandatory_training_completed": mandatory_training_completed,
                "leave_taken_qtr": leave_taken_qtr,
                "leave_long_weekend_count": leave_long_weekend_count,
                "leave_balance_days": leave_balance_days,
                "performance_rating": performance_rating,
                "attendance_rate": attendance_rate,
            }
        )
        ground_truth.append(
            {
                "employee_id": employee_id,
                "is_payroll_outlier": is_payroll_outlier,
                "is_overtime_breach": is_overtime_breach,
                "is_leave_abuser": is_leave_abuser,
                "is_missing_training": not mandatory_training_completed,
            }
        )

    return rows, ground_truth


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    employee_rows, truth = generate_employees()
    out_path = Path(__file__).parent / "employees.csv"
    write_csv(employee_rows, out_path)
    print(f"wrote {len(employee_rows)} employee records to {out_path}")

    injected_payroll = sum(t["is_payroll_outlier"] for t in truth)
    injected_overtime = sum(t["is_overtime_breach"] for t in truth)
    injected_leave = sum(t["is_leave_abuser"] for t in truth)
    injected_training = sum(t["is_missing_training"] for t in truth)
    print(
        f"injected ground truth -- payroll outliers: {injected_payroll}, "
        f"overtime breaches: {injected_overtime}, leave abusers: {injected_leave}, "
        f"missing training: {injected_training}"
    )
