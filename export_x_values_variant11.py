import csv

import numpy as np

N = 3000
SIGMA = 1.0
A = 0.0


def main() -> None:
    np.random.seed(42)
    xi = np.random.uniform(0, 1, N)
    x = SIGMA * (xi + xi**2) + A

    out_path = "x_values_3000.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["i", "x"])
        for i, val in enumerate(x, start=1):
            w.writerow([i, float(val)])

    print(f"Saved {N} values to {out_path}")


if __name__ == "__main__":
    main()

