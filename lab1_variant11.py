import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2

N = 3000
SIGMA = 1.0
A = 0.0
K = 20
ALPHA = 0.05

def cdf_variant11(x, sigma=SIGMA, a=A):
    z = (np.asarray(x, dtype=float) - a) / sigma
    z = np.clip(z, 0, 2)
    return 0.5 * (-1 + np.sqrt(1 + 4 * z))

def merge_bins(counts, edges, min_count=5):
    counts = counts.astype(int, copy=True)
    edges = edges.astype(float, copy=True)
    while len(counts) > 1:
        i = int(np.argmin(counts))
        if counts[i] >= min_count:
            break
        if i == len(counts) - 1:
            counts[i - 1] += counts[i]
            counts = np.delete(counts, i)
            edges = np.delete(edges, i)
        else:
            counts[i] += counts[i + 1]
            counts = np.delete(counts, i + 1)
            edges = np.delete(edges, i + 1)
    return counts, edges

np.random.seed(42)
xi = np.random.uniform(0, 1, N)
x = SIGMA * (xi + xi**2) + A

print("1) Згенеровано 3000 значень. Перші 10 значень x:")
print(x[:10])

mean = float(np.mean(x))
var = float(np.var(x, ddof=1))
p3_line1 = f"\n3) Середнє μ = {mean}"
p3_line2 = f"   Дисперсія σ^2 = {var}"

xmin, xmax = float(np.min(x)), float(np.max(x))
bins = np.linspace(xmin, xmax, K + 1)
counts, _ = np.histogram(x, bins=bins)
h = (xmax - xmin) / K if K > 0 else 0.0

plt.figure(figsize=(12, 6))
plt.bar(bins[:-1], counts, width=h, edgecolor="black", align="edge", alpha=0.7)
plt.xlabel("Значення x_i")
plt.ylabel("Частота n_i")
plt.title(f"Гістограма частот (n={N}, σ={SIGMA}, a={A})")
plt.grid(True, alpha=0.3)
plt.text(
    0.02,
    0.98,
    f"Середнє μ: {mean:.4f}\nДисперсія σ²: {var:.4f}",
    transform=plt.gca().transAxes,
    fontsize=10,
    va="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)
plt.tight_layout()
plt.savefig("histogram.png", dpi=300, bbox_inches="tight")
plt.close()
print("\n2) Побудова гістограми частот:")
print("   min =", xmin)
print("   max =", xmax)
print("   довжина інтервалу h =", h)
print("   масив меж інтервалів:", bins)
print("   частоти n_i:", counts)
print("   Гістограму збережено у файл histogram.png")

print(p3_line1)
print(p3_line2)

print("\n4) За видом гістограми: розподіл неперервний, обмежений на [a, a+2σ] та асиметричний.")
print("   Отже це не нормальний/рівномірний/експоненційний, а заданий закон варіанту 11.")

counts_m, edges_m = merge_bins(counts, bins, min_count=5)
expected = N * np.diff(cdf_variant11(edges_m, sigma=SIGMA, a=A))
mask = expected > 0
obs = counts_m[mask]
exp = expected[mask]
chi_sq = float(np.sum((obs - exp) ** 2 / exp))
df = max(1, len(obs) - 1)
chi_crit = float(chi2.ppf(1 - ALPHA, df))
p_value = float(1 - chi2.cdf(chi_sq, df))

print("\n5) Критерій χ² для закону варіанту 11 (α=0.05):")
print("   χ² =", chi_sq)
print("   df =", df)
print("   χ²кр =", chi_crit)
print("   p-value =", p_value)
print("   Висновок:", "χ² < χ²кр ⇒ вибірка узгоджується із законом" if chi_sq < chi_crit else "χ² ≥ χ²кр ⇒ гіпотеза відхиляється")
