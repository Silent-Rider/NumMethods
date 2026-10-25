import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x ** 3 + 4.5 * x ** 2 - 9 * x - 4.5


def f_prime(x):
    return 3 * x ** 2 + 9 * x - 9


def f_double_prime(x):
    return 6 * x + 9


x_vals = np.linspace(-7, 3, 400)
y_vals = f(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, 'b-', linewidth=2)
plt.axhline(y=0, color='k', linewidth=0.8)
plt.axvline(x=0, color='k', linewidth=0.8)
plt.grid(True, alpha=0.3)
plt.title('f(x) = x³ + 4.5x² − 9x − 4.5')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.xticks(range(-7, 4))
plt.savefig('graph1.png', dpi=150)
plt.show()

intervals = [(-6, -5), (-1, 0), (1, 2)]


def bisection(a, b, eps, max_iter=10000):
    iterations = 0
    while (b - a) / 2 > eps and iterations < max_iter:
        c = (a + b) / 2
        if f(c) == 0:
            break
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
        iterations += 1
    return (a + b) / 2, iterations


def newton(x0, eps, max_iter=10000):
    x = x0
    iterations = 0
    while iterations < max_iter:
        fp = f_prime(x)
        if abs(fp) < 1e-15:
            break
        x_new = x - f(x) / fp
        if abs(x_new - x) < eps:
            x = x_new
            iterations += 1
            break
        x = x_new
        iterations += 1
    return x, iterations


def simple_iteration(a, b, eps, max_iter=10000):
    mid = (a + b) / 2
    lam = 1.0 / f_prime(mid)

    x = mid
    iterations = 0
    while iterations < max_iter:
        x_new = x - lam * f(x)
        if abs(x_new - x) < eps:
            return x_new, iterations + 1
        x = x_new
        iterations += 1
    return x, iterations


epsilons = [0.01, 0.001, 0.0001]

print("=" * 85)
print(f"{'Метод':<20} {'Точность':<10} {'Интервал':<12} {'Корень':<15} {'Итерации':<10}")
print("=" * 85)

for (a, b) in intervals:
    x0_newton = b if f(b) * f_double_prime(b) > 0 else a

    for eps in epsilons:
        root_bis, iter_bis = bisection(a, b, eps)
        root_new, iter_new = newton(x0_newton, eps)
        root_si, iter_si = simple_iteration(a, b, eps)

        if eps == 0.01:
            decimals = 2
        elif eps == 0.001:
            decimals = 3
        elif eps == 0.0001:
            decimals = 4

        fmt = f"{{:.{decimals}f}}"

        print(f"{'Бисекция':<20} {eps:<10} [{a},{b}]      {fmt.format(root_bis):<15} {iter_bis:<10}")
        print(f"{'Ньютон':<20} {eps:<10} [{a},{b}]      {fmt.format(root_new):<15} {iter_new:<10}")
        print(f"{'Простые итерации':<20} {eps:<10} [{a},{b}]      {fmt.format(root_si):<15} {iter_si:<10}")
        print("-" * 85)