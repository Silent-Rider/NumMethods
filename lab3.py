import numpy as np
import matplotlib.pyplot as plt

# Часть 1. Интерполирование функций

def g(x):
    return 1 / (np.cos(x) ** 2)

a = -np.pi / 4
b = np.pi / 4
z = 0.3
g_z = g(z)

def get_nodes(N):
    h = (b - a) / N
    x_nodes = np.array([a + h * i for i in range(N + 1)])
    y_nodes = g(x_nodes)
    return x_nodes, y_nodes

def piecewise_const_left(x_nodes, y_nodes, z):
    idx = np.searchsorted(x_nodes, z) - 1
    idx = max(0, idx)
    return y_nodes[idx]

def piecewise_const_right(x_nodes, y_nodes, z):
    idx = np.searchsorted(x_nodes, z)
    idx = min(len(x_nodes) - 1, idx)
    return y_nodes[idx]

def piecewise_linear(x_nodes, y_nodes, z):
    idx = np.searchsorted(x_nodes, z) - 1
    idx = max(0, idx)
    idx2 = min(len(x_nodes) - 1, idx + 1)
    x1, y1 = x_nodes[idx], y_nodes[idx]
    x2, y2 = x_nodes[idx2], y_nodes[idx2]
    k = (y2 - y1) / (x2 - x1)
    l_val = y1 - k * x1
    return k * z + l_val, k, l_val

def lagrange(x_nodes, y_nodes, z):
    n = len(x_nodes)
    result = 0.0
    for i in range(n):
        L_i = 1.0
        for j in range(n):
            if i != j:
                L_i *= (z - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += y_nodes[i] * L_i
    return result

x_nodes_10, y_nodes_10 = get_nodes(10)
x_nodes_20, y_nodes_20 = get_nodes(20)

print("Узлы интерполяции (N=10):")
print("x:", [f"{x:.4f}" for x in x_nodes_10])
print("y:", [f"{y:.6f}" for y in y_nodes_10])
print()

x_plot = np.linspace(a, b, 400)
y_plot = g(x_plot)

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='g(x) = 1/cos²(x)')
plt.plot(x_nodes_10, y_nodes_10, 'ro', markersize=8, label='Узлы (N=10)')
plt.axvline(x=z, color='green', linestyle='--', label=f'z = {z}')
plt.grid(True, alpha=0.3)
plt.title('График функции g(x) = 1/cos²(x)')
plt.xlabel('x')
plt.ylabel('g(x)')
plt.legend()
plt.xticks(np.arange(a, b + 0.1, 0.2))
plt.savefig('graph_part1.png', dpi=150)
plt.show()

F_left = piecewise_const_left(x_nodes_10, y_nodes_10, z)
F_right = piecewise_const_right(x_nodes_10, y_nodes_10, z)
F_lin, k, l_val = piecewise_linear(x_nodes_10, y_nodes_10, z)
F_lag_10 = lagrange(x_nodes_10, y_nodes_10, z)
F_lag_20 = lagrange(x_nodes_20, y_nodes_20, z)

print(f"Точка z = {z}")
print(f"Точное значение g(z) = {g_z:.6f}")
print()
print(f"Уравнение прямой для кусочно-линейной: F(x) = {k:.6f}·x + {l_val:.6f}")
print()

results = [
    ("Кусочно-постоянная слева", F_left),
    ("Кусочно-постоянная справа", F_right),
    ("Кусочно-линейная", F_lin),
    ("Полином Лагранжа, N=10", F_lag_10),
    ("Полином Лагранжа, N=20", F_lag_20),
]

print("=" * 80)
print(f"{'Метод':<30} {'z':<8} {'g(z)':<12} {'F(z)':<12} {'|g(z)-F(z)|':<12}")
print("=" * 80)
for name, F in results:
    error = abs(g_z - F)
    print(f"{name:<30} {z:<8.4f} {g_z:<12.6f} {F:<12.6f} {error:<12.6f}")
print("=" * 80)


# Часть 2. Аппроксимация функций

x_data = np.array([-10, -8.3, -6.7, -5, -3, -1.67, 0, 1.67, 3.33, 5, 6.7, 8.33, 10], dtype=float)
y_data = np.array([86, 60, 39, 22, 10, 3.36, 1, 3.36, 10.4, 22, 39, 60, 86], dtype=float)
n = len(x_data)

S = [np.sum(x_data ** k) for k in range(7)]
T = [np.sum((x_data ** k) * y_data) for k in range(4)]

A1 = np.array([[S[0], S[1]], [S[1], S[2]]])
b1 = np.array([T[0], T[1]])
coeffs1 = np.linalg.solve(A1, b1)

A2 = np.array([[S[0], S[1], S[2]], [S[1], S[2], S[3]], [S[2], S[3], S[4]]])
b2 = np.array([T[0], T[1], T[2]])
coeffs2 = np.linalg.solve(A2, b2)

A3 = np.array([[S[0], S[1], S[2], S[3]], [S[1], S[2], S[3], S[4]], [S[2], S[3], S[4], S[5]], [S[3], S[4], S[5], S[6]]])
b3 = np.array([T[0], T[1], T[2], T[3]])
coeffs3 = np.linalg.solve(A3, b3)

def P1(x):
    return coeffs1[0] + coeffs1[1] * x

def P2(x):
    return coeffs2[0] + coeffs2[1] * x + coeffs2[2] * x ** 2

def P3(x):
    return coeffs3[0] + coeffs3[1] * x + coeffs3[2] * x ** 2 + coeffs3[3] * x ** 3

def sum_squared_errors(P_func):
    return np.sum((y_data - P_func(x_data)) ** 2)

S1 = sum_squared_errors(P1)
S2 = sum_squared_errors(P2)
S3 = sum_squared_errors(P3)

output = f"""
Системы нормальных уравнений:

Для P1(x) = a0 + a1*x:
  {S[0]:.4f}·a0 + {S[1]:.4f}·a1 = {T[0]:.4f}
  {S[1]:.4f}·a0 + {S[2]:.4f}·a1 = {T[1]:.4f}
  Решение: a0 = {coeffs1[0]:.6f}, a1 = {coeffs1[1]:.6f}

Для P2(x) = a0 + a1*x + a2*x^2:
  {S[0]:.4f}·a0 + {S[1]:.4f}·a1 + {S[2]:.4f}·a2 = {T[0]:.4f}
  {S[1]:.4f}·a0 + {S[2]:.4f}·a1 + {S[3]:.4f}·a2 = {T[1]:.4f}
  {S[2]:.4f}·a0 + {S[3]:.4f}·a1 + {S[4]:.4f}·a2 = {T[2]:.4f}
  Решение: a0 = {coeffs2[0]:.6f}, a1 = {coeffs2[1]:.6f}, a2 = {coeffs2[2]:.6f}

Для P3(x) = a0 + a1*x + a2*x^2 + a3*x^3:
  {S[0]:.4f}·a0 + {S[1]:.4f}·a1 + {S[2]:.4f}·a2 + {S[3]:.4f}·a3 = {T[0]:.4f}
  {S[1]:.4f}·a0 + {S[2]:.4f}·a1 + {S[3]:.4f}·a2 + {S[4]:.4f}·a3 = {T[1]:.4f}
  {S[2]:.4f}·a0 + {S[3]:.4f}·a1 + {S[4]:.4f}·a2 + {S[5]:.4f}·a3 = {T[2]:.4f}
  {S[3]:.4f}·a0 + {S[4]:.4f}·a1 + {S[5]:.4f}·a2 + {S[6]:.4f}·a3 = {T[3]:.4f}
  Решение: a0 = {coeffs3[0]:.6f}, a1 = {coeffs3[1]:.6f}, a2 = {coeffs3[2]:.6f}, a3 = {coeffs3[3]:.6f}

Сумма квадратов отклонений:
  P1(x): S = {S1:.6f}
  P2(x): S = {S2:.6f}
  P3(x): S = {S3:.6f}
"""
print(output)

best = min([(S1, "P1"), (S2, "P2"), (S3, "P3")], key=lambda x: x[0])
print(f"Лучший полином: {best[1]}(x) с S = {best[0]:.6f}")
print()

x_plot = np.linspace(-10, 10, 400)

plt.figure(figsize=(12, 7))
plt.scatter(x_data, y_data, color='red', zorder=5, label='Исходные данные')
plt.plot(x_plot, P1(x_plot), 'b-', linewidth=2, label=f'P1(x) = {coeffs1[0]:.4f} + {coeffs1[1]:.4f}·x')
plt.plot(x_plot, P3(x_plot), 'm-', linewidth=2, label=f'P3(x) = {coeffs3[0]:.4f} + {coeffs3[1]:.4f}·x + {coeffs3[2]:.4f}·x² + {coeffs3[3]:.4f}·x³')
plt.plot(x_plot, P2(x_plot), 'y--', dashes=(5, 10), linewidth=2, label=f'P2(x) = {coeffs2[0]:.4f} + {coeffs2[1]:.4f}·x + {coeffs2[2]:.4f}·x²')
plt.grid(True, alpha=0.3)
plt.title('Аппроксимация данных полиномами')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('graph_part2.png', dpi=150)
plt.show()