"""
Анализ данных для проекта «Telegram-бот для учёта расходов студента».
Запуск: python analysis.py
Читает students.csv и expenses.csv, сохраняет графики в папку charts/
и печатает ключевые цифры для защиты.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("charts", exist_ok=True)
plt.rcParams.update({"figure.figsize": (8, 5), "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 11})

students = pd.read_csv("students.csv")
expenses = pd.read_csv("expenses.csv", parse_dates=["date"])

# ---------- Общие цифры ----------
n = len(students)
share_exceeded = students["exceeded_budget"].mean() * 100
share_tracks = students["tracks_expenses"].mean() * 100
share_bot = students["would_use_bot"].mean() * 100
print("=== Ключевые цифры ===")
print(f"Студентов: {n}, трат: {len(expenses)}")
print(f"Ведут учёт расходов: {share_tracks:.0f}%")
print(f"Потратили больше дохода в октябре: {share_exceeded:.0f}%")
print(f"Стали бы пользоваться ботом: {share_bot:.0f}%")

# ---------- 1. Структура расходов ----------
by_cat = expenses.groupby("category")["amount"].sum().sort_values(ascending=False)
fig, ax = plt.subplots()
ax.pie(by_cat, labels=by_cat.index, autopct="%1.0f%%", startangle=90, pctdistance=0.8)
ax.set_title("Структура расходов студентов (сентябрь-октябрь 2026)")
ax.grid(False)
fig.tight_layout()
fig.savefig("charts/1_structure.png", dpi=150)
plt.close(fig)
print("\nСтруктура расходов, %:")
print((by_cat / by_cat.sum() * 100).round(1).to_string())

# ---------- 2. Распределение дохода и трат ----------
fig, ax = plt.subplots()
ax.hist(students["monthly_income"], bins=20, alpha=0.65, label="Доход")
ax.hist(students["spend_last_month"], bins=20, alpha=0.65, label="Траты за октябрь")
ax.axvline(students["monthly_income"].median(), color="C0", ls="--", lw=1)
ax.axvline(students["spend_last_month"].median(), color="C1", ls="--", lw=1)
ax.set_xlabel("Рублей в месяц")
ax.set_ylabel("Число студентов")
ax.set_title("Распределение месячного дохода и трат")
ax.legend()
fig.tight_layout()
fig.savefig("charts/2_income_spend_hist.png", dpi=150)
plt.close(fig)
print(f"\nМедианный доход: {students['monthly_income'].median():.0f} руб., "
      f"медианные траты: {students['spend_last_month'].median():.0f} руб.")

# ---------- 3. Траты по дням месяца ----------
# жильё исключаем: аренда платится разово в первую неделю и искажает картину
oct_ = expenses[(expenses["date"].dt.month == 10) & (expenses["category"] != "жильё")].copy()
oct_["day"] = oct_["date"].dt.day
daily = oct_.groupby("day")["amount"].sum() / n  # средняя трата одного студента за день
first_half = daily[daily.index <= 15].sum()
second_half = daily[daily.index > 15].sum()
fig, ax = plt.subplots()
ax.plot(daily.index, daily.values, marker="o", ms=4)
ax.set_xlabel("День месяца (октябрь)")
ax.set_ylabel("Средние траты на студента, руб.")
ax.set_title("Повседневные траты по дням месяца (без жилья)")
fig.tight_layout()
fig.savefig("charts/3_daily.png", dpi=150)
plt.close(fig)
print(f"\nТраты в 1-15 числа: {first_half:.0f} руб. на студента, "
      f"в 16-31: {second_half:.0f} руб. (в {first_half / second_half:.1f} раза больше в первой половине)")

# ---------- 4. Перерасход: ведут учёт или нет ----------
grp = students.groupby("tracks_expenses")["exceeded_budget"].mean() * 100
labels = ["Не ведут учёт", "Ведут учёт"]
fig, ax = plt.subplots()
bars = ax.bar(labels, [grp[0], grp[1]], color=["#d9534f", "#5cb85c"])
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.0f}%", ha="center")
ax.set_ylabel("% студентов, превысивших бюджет")
ax.set_title("Перерасход бюджета: ведут ли студенты учёт расходов")
fig.tight_layout()
fig.savefig("charts/4_overspend_tracking.png", dpi=150)
plt.close(fig)
print(f"\nПревысили бюджет: не ведут учёт {grp[0]:.0f}%, ведут учёт {grp[1]:.0f}%")

# ---------- 5. Связь дохода и доли трат на еду ----------
food = expenses[(expenses["category"] == "еда") & (expenses["date"].dt.month == 10)] \
    .groupby("student_id")["amount"].sum()
df = students.set_index("student_id").join(food.rename("food"))
df["food_share"] = df["food"] / df["spend_last_month"] * 100
df = df.dropna(subset=["food_share"])
r = np.corrcoef(df["monthly_income"], df["food_share"])[0, 1]
k, b = np.polyfit(df["monthly_income"], df["food_share"], 1)
xs = np.linspace(df["monthly_income"].min(), df["monthly_income"].max(), 50)
fig, ax = plt.subplots()
ax.scatter(df["monthly_income"], df["food_share"], alpha=0.5)
ax.plot(xs, k * xs + b, color="red", label=f"тренд, r = {r:.2f}")
ax.set_xlabel("Месячный доход, руб.")
ax.set_ylabel("Доля трат на еду, %")
ax.set_title("Доход и доля расходов на еду")
ax.legend()
fig.tight_layout()
fig.savefig("charts/5_income_vs_food.png", dpi=150)
plt.close(fig)
print(f"\nКорреляция дохода и доли трат на еду: r = {r:.2f}")

# ---------- 6. Расходы по типу жилья ----------
by_living = students.groupby("living_type").agg(
    spend=("spend_last_month", "mean"), exceeded=("exceeded_budget", "mean"))
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
axes[0].bar(by_living.index, by_living["spend"], color="C0")
axes[0].set_title("Средние траты в месяц, руб.")
axes[1].bar(by_living.index, by_living["exceeded"] * 100, color="C3")
axes[1].set_title("Доля превысивших бюджет, %")
for ax in axes:
    ax.tick_params(axis="x", rotation=15)
fig.tight_layout()
fig.savefig("charts/6_by_living.png", dpi=150)
plt.close(fig)
print("\nПо типу жилья:")
print(by_living.assign(spend=by_living["spend"].round(0), exceeded=(by_living["exceeded"] * 100).round(0)).to_string())

# ---------- 7. Сложности и интерес к боту ----------
diff = students["main_difficulty"].value_counts()
fig, ax = plt.subplots()
ax.barh(diff.index[::-1], diff.values[::-1], color="C2")
ax.set_xlabel("Число студентов")
ax.set_title(f"Главная сложность с учётом расходов (бот нужен {share_bot:.0f}% студентов)")
fig.tight_layout()
fig.savefig("charts/7_difficulty.png", dpi=150)
plt.close(fig)

# когда заканчиваются деньги
ran_out = students["money_runs_out_day"].dropna()
print(f"\nУ {len(ran_out)} студентов деньги заканчиваются раньше конца месяца, "
      f"в среднем к {ran_out.mean():.0f}-му числу")
print("\nГрафики сохранены в папку charts/")
