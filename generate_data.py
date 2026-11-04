"""
Генерация синтетических данных для проекта «Telegram-бот для учёта расходов студента».

Создаёт два файла:
  students.csv  - «опрос» студентов (одна строка = один студент)
  expenses.csv  - отдельные траты за сентябрь и октябрь 2026 (одна строка = одна трата)

Все суммы в рублях. Данные синтетические (сгенерированы кодом), seed зафиксирован,
поэтому результат воспроизводим: python generate_data.py
"""
import numpy as np
import pandas as pd

SEED = 42
N_STUDENTS = 200
MONTHS = [(2026, 9), (2026, 10)]

rng = np.random.default_rng(SEED)

# ---------- 1. Студенты ----------
living = rng.choice(["общежитие", "снимает жильё", "с родителями"],
                    size=N_STUDENTS, p=[0.40, 0.25, 0.35])
course = rng.choice([1, 2, 3, 4], size=N_STUDENTS, p=[0.32, 0.26, 0.24, 0.18])
age = np.clip(17 + course + rng.integers(0, 2, N_STUDENTS), 17, 23)

# подработка чаще на старших курсах
p_job = np.array([0.12, 0.22, 0.38, 0.50])[course - 1]
has_job = rng.random(N_STUDENTS) < p_job

stipend = rng.choice([0, 2500, 4500, 7500], size=N_STUDENTS, p=[0.30, 0.35, 0.22, 0.13])

parents = np.where(
    living == "с родителями",
    rng.choice([0, 3000, 6000, 10000], size=N_STUDENTS, p=[0.35, 0.30, 0.20, 0.15]),
    np.where(
        living == "снимает жильё",
        rng.choice([8000, 12000, 18000, 25000], size=N_STUDENTS, p=[0.25, 0.35, 0.25, 0.15]),
        rng.choice([0, 3000, 6000, 10000, 15000], size=N_STUDENTS, p=[0.12, 0.28, 0.30, 0.20, 0.10]),
    ),
)

job_income = np.where(has_job, np.round(rng.normal(16000, 6000, N_STUDENTS), -2), 0)
job_income = np.clip(job_income, 0, None)

income = stipend + parents + job_income
# защита от слишком маленького дохода
income = np.where(income < 4000, 4000 + rng.integers(0, 3, N_STUDENTS) * 1000, income)

tracks = rng.random(N_STUDENTS) < 0.28  # ведут учёт расходов
overspend_trait = rng.random(N_STUDENTS) < np.where(tracks, 0.15, 0.34)  # склонность перерасходовать

students = pd.DataFrame({
    "student_id": np.arange(1, N_STUDENTS + 1),
    "age": age,
    "course": course,
    "living_type": living,
    "has_job": has_job.astype(int),
    "stipend": stipend,
    "parents_support": parents,
    "job_income": job_income.astype(int),
    "monthly_income": income.astype(int),
    "tracks_expenses": tracks.astype(int),
})

# ---------- 2. Траты ----------
CATEGORIES = ["еда", "транспорт", "жильё", "развлечения", "учёба",
              "здоровье", "одежда", "подписки", "прочее"]

# базовые доли «свободных» трат (без аренды/жилья)
BASE_SHARE = {"еда": 0.44, "транспорт": 0.10, "развлечения": 0.15, "учёба": 0.06,
              "здоровье": 0.05, "одежда": 0.08, "подписки": 0.04, "прочее": 0.08}

N_TX_RANGE = {"еда": (20, 40), "транспорт": (6, 20), "развлечения": (3, 8),
              "учёба": (1, 5), "здоровье": (0, 3), "одежда": (0, 3),
              "подписки": (1, 3), "прочее": (2, 6)}

COMMENTS = {
    "еда": ["столовая", "кофе", "продукты", "доставка", "перекус", "супермаркет", "обед"],
    "транспорт": ["автобус", "метро", "такси", "проездной"],
    "жильё": ["оплата общежития", "аренда комнаты", "коммуналка"],
    "развлечения": ["кино", "кафе с друзьями", "игры", "концерт", "боулинг"],
    "учёба": ["печать", "канцелярия", "учебник", "курс"],
    "здоровье": ["аптека", "врач"],
    "одежда": ["одежда", "обувь"],
    "подписки": ["музыка", "стриминг", "облако"],
    "прочее": ["подарок", "связь", "мелочи", "ремонт телефона"],
}


def day_weights(year, month, category):
    """Вес каждого дня месяца: траты выше в начале месяца и по выходным."""
    days = pd.date_range(f"{year}-{month:02d}-01", periods=pd.Period(f"{year}-{month:02d}").days_in_month)
    n = len(days)
    idx = np.arange(n)
    w = 1.5 - 0.9 * idx / (n - 1)           # от 1.5 в начале месяца до 0.6 в конце
    if category in ("развлечения", "одежда", "еда"):
        w = w * np.where(days.dayofweek >= 5, 1.6, 1.0)
    if category in ("учёба", "транспорт"):
        w = w * np.where(days.dayofweek >= 5, 0.4, 1.0)
    return days, w / w.sum()


rows = []
for i, s in students.iterrows():
    sid = int(s["student_id"])
    for (year, month) in MONTHS:
        # итоговые траты за месяц относительно дохода
        factor = rng.normal(1.12 if overspend_trait[i] else 0.88, 0.07)
        factor = np.clip(factor, 0.6, 1.5)
        total = s["monthly_income"] * factor

        # жильё
        if s["living_type"] == "общежитие":
            rent = rng.choice([1200, 1800, 2500])
        elif s["living_type"] == "снимает жильё":
            rent = rng.choice([9000, 12000, 15000, 18000])
            rent = min(rent, total * 0.55)
        else:
            rent = 0

        free = max(total - rent, 1500)

        shares = np.array([BASE_SHARE[c] for c in BASE_SHARE])
        shares = rng.dirichlet(shares * 60)
        cat_totals = dict(zip(BASE_SHARE.keys(), shares * free))

        if rent > 0:
            d = pd.Timestamp(year=year, month=month, day=int(rng.integers(1, 8)))
            rows.append((sid, d.date(), "жильё", round(float(rent), 0),
                         str(rng.choice(COMMENTS["жильё"]))))

        for cat, cat_total in cat_totals.items():
            lo, hi = N_TX_RANGE[cat]
            n_tx = int(rng.integers(lo, hi + 1))
            if n_tx == 0:
                continue
            parts = rng.lognormal(0, 0.6, n_tx)
            amounts = parts / parts.sum() * cat_total
            days, w = day_weights(year, month, cat)
            chosen = rng.choice(days, size=n_tx, p=w)
            for d, a in zip(chosen, amounts):
                amount = max(round(float(a) / 10) * 10, 30)
                rows.append((sid, pd.Timestamp(d).date(), cat, float(amount),
                             str(rng.choice(COMMENTS[cat]))))

expenses = pd.DataFrame(rows, columns=["student_id", "date", "category", "amount", "comment"])
expenses = expenses.sort_values(["date", "student_id"]).reset_index(drop=True)
expenses.insert(0, "expense_id", np.arange(1, len(expenses) + 1))

# ---------- 3. Производные поля «опроса» (согласованы с тратами) ----------
oct_ = expenses[pd.to_datetime(expenses["date"]).dt.month == 10]
spend_oct = oct_.groupby("student_id")["amount"].sum()
students["spend_last_month"] = students["student_id"].map(spend_oct).fillna(0).astype(int)
students["exceeded_budget"] = (students["spend_last_month"] > students["monthly_income"]).astype(int)

# день месяца, на который «заканчиваются деньги» (самооценка), только у тех, кто превысил бюджет
ratio = students["monthly_income"] / students["spend_last_month"].replace(0, np.nan)
run_out = np.where(ratio < 1, np.clip(np.round(31 * ratio + rng.normal(0, 1.5, N_STUDENTS)), 10, 30), np.nan)
students["money_runs_out_day"] = pd.Series(run_out).astype("Int64")

students["would_use_bot"] = np.where(
    students["tracks_expenses"] == 1,
    (rng.random(N_STUDENTS) < 0.55).astype(int),
    (rng.random(N_STUDENTS) < 0.72 + 0.15 * students["exceeded_budget"]).astype(int),
)
students["main_difficulty"] = rng.choice(
    ["лень записывать", "забываю вносить", "сложные приложения",
     "не вижу смысла", "нет проблем"],
    size=N_STUDENTS, p=[0.30, 0.28, 0.17, 0.10, 0.15])

students.to_csv("students.csv", index=False, encoding="utf-8-sig")
expenses.to_csv("expenses.csv", index=False, encoding="utf-8-sig")
print(f"students.csv: {len(students)} строк, expenses.csv: {len(expenses)} строк")
