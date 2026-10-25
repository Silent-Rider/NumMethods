import numpy as np

A = np.array([[65, -5, -11, -19],
              [1, 64, -11, -20],
              [5, -1, 57, -19],
              [11, 4, -5, 50]], dtype=float)

b = np.array([1, 3, 5, 7], dtype=float)
n = 4

A_inv = np.linalg.inv(A)
x_exact_inv = A_inv @ b
print("=== Точные методы ===")
print(f"Метод обратной матрицы: X = {x_exact_inv}")

det_A = np.linalg.det(A)
x_cramer = np.zeros(n)
for i in range(n):
    A_i = A.copy()
    A_i[:, i] = b
    x_cramer[i] = np.linalg.det(A_i) / det_A
print(f"Метод Крамера:          X = {x_cramer}")
print()

x0 = np.array([b[i] / A[i, i] for i in range(n)])
print(f"Начальное приближение: {x0}")
print()


def jacobi(A, b, x0, eps, max_iter=10000):
    n = len(b)
    x = x0.copy()
    iterations = 0
    while iterations < max_iter:
        x_new = np.zeros(n)
        for i in range(n):
            s = b[i]
            for j in range(n):
                if i != j:
                    s -= A[i, j] * x[j]
            x_new[i] = s / A[i, i]

        max_diff = max(abs(x_new - x))
        x = x_new
        iterations += 1
        if max_diff < eps:
            break
    return x, iterations, max_diff


def gauss_seidel(A, b, x0, eps, max_iter=10000):
    n = len(b)
    x = x0.copy()
    iterations = 0
    while iterations < max_iter:
        x_new = x.copy()
        for i in range(n):
            s = b[i]
            for j in range(n):
                if i != j:
                    s -= A[i, j] * x_new[j]
            x_new[i] = s / A[i, i]

        max_diff = max(abs(x_new - x))
        x = x_new
        iterations += 1
        if max_diff < eps:
            break
    return x, iterations, max_diff


epsilons = [0.01, 0.001, 0.0001]

print("=" * 100)
print(f"{'Метод':<25} {'x1':<12} {'x2':<12} {'x3':<12} {'x4':<12} {'max':<10} {'ε':<8} {'Итерации':<10}")
print("=" * 100)

for eps in epsilons:
    if eps == 0.01:
        decimals = 2
    elif eps == 0.001:
        decimals = 3
    elif eps == 0.0001:
        decimals = 4

    fmt = f"{{:.{decimals}f}}"

    x_jac, iter_jac, max_jac = jacobi(A, b, x0, eps)
    print(
        f"{'Простые итерации':<25} {fmt.format(x_jac[0]):<12} {fmt.format(x_jac[1]):<12} {fmt.format(x_jac[2]):<12} {fmt.format(x_jac[3]):<12} {fmt.format(max_jac):<10} {eps:<8} {iter_jac:<10}")

for eps in epsilons:
    if eps == 0.01:
        decimals = 2
    elif eps == 0.001:
        decimals = 3
    elif eps == 0.0001:
        decimals = 4

    fmt = f"{{:.{decimals}f}}"

    x_gs, iter_gs, max_gs = gauss_seidel(A, b, x0, eps)
    print(
        f"{'Гаусс-Зейдель':<25} {fmt.format(x_gs[0]):<12} {fmt.format(x_gs[1]):<12} {fmt.format(x_gs[2]):<12} {fmt.format(x_gs[3]):<12} {fmt.format(max_gs):<10} {eps:<8} {iter_gs:<10}")

print("=" * 100)

print(f"\nТочное решение: {x_exact_inv}")
print(f"Погрешность Простых итераций (ε=0.0001):    {max(abs(x_jac - x_exact_inv)):.6f}")
print(f"Погрешность Гаусса-Зейделя (ε=0.0001): {max(abs(x_gs - x_exact_inv)):.6f}")