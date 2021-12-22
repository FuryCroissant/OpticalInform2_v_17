import matplotlib.pyplot as plt
import math
import numpy as np

### ОДНОМЕРНАЯ ЗАДАЧА ######
# Одномерный гауссов пучок
def gauss(x):
    return np.exp(-x ** 2)
# Дискретизация отрезка координат и входной функции
def disc(a, n, f):
    h = 2 * a / n
    return h, np.array([-a + i * h for i in range(n)]), np.array([f(-a + i * h) for i in range(n)])
# Интегрирование функции по дискретизированному отрезку
def finit_fourier(x, u, h):
    result = 0
    for i in range(len(x)):
        result += np.exp(-2 * np.pi * 1j * x[i] * u) * gauss(x[i])
    return result * h
#БПФ
def FFT(f, h, n, m):
    # дополнение нулями, разбиение на две части и их обмен
    zeros = np.zeros(int((m-n)/2))
    f = np.concatenate((zeros, f, zeros), axis=None)
    # Свап частей вектора:
    middle = int(len(f) / 2)
    f = np.concatenate((f[middle:], f[:middle]))
    # БПФ
    F = np.fft.fft(f, m)*h
    # Свап частей вектора:
    middle = int(len(F) / 2)
    F = np.concatenate((F[middle:], F[:middle]))
    # Выделение центральных N отсчетов:
    F = F[int((m - n) / 2): int((m - n) / 2 + n)]
    return F
#входное поле согласно варианту
def f(x):
    return (x**2)*np.exp(-x ** 2)
#аналитическое решение
def f_a(u):
    C = math.sqrt(math.pi)/2
    mpi = -((math.pi*u)**2)
    return C*np.exp(mpi)*(1+2*mpi)
#отрисовка одномерных графиков
def plot(x, y,label, y2 = None, label2=None ):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x, abs(y), label=label)
    if y2 is not None:
        plt.plot(x, abs(y2), label=label2, linestyle='--')
    plt.title("Амплитуда")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(x, np.angle(y), label=label)
    if y2 is not None:
        plt.plot(x, np.angle(y2), label=label2, linestyle='--')
    plt.title("Фаза")
    plt.grid()
    plt.legend()
    plt.show()

### ДВУМЕРНАЯ ЗАДАЧА ######
# Двумерный гауссов пучок
def gauss2d(x, y):
    return np.exp(-(x ** 2 + y ** 2))
# Дискретизация двумерной функции
def disc2d(a, n, function):
    h = 2 * a / n
    return h, np.array([[function(-a + i * h, -a + j * h) for i in range(n)] for j in range(n)])
#Двумерное БПФ
def fft2d(func, h, n, m):
    fft = np.zeros((n, n), dtype=complex)
    temp = np.zeros((n, n), dtype=complex)
    # проход по строкам
    for i in range(n):
        temp[i] = FFT(func[i], h, n, m)
    temp = temp.T
    # проход по столбцам
    for i in range(n):
        fft[i] = FFT(temp[i], h, n, m)
    return fft.T
#входное двумерное поле согласно варианту
def f2d(x,y):
    return (x**2)*(y**2)*np.exp(-(x ** 2+y**2))
#аналитическое решение
def f_a2d(u, v):
    C = math.sqrt(math.pi)/2
    mpi = -((math.pi*u)**2)
    mpi2 = -((math.pi*v)**2)
    return C*np.exp(mpi)*np.exp(mpi2)*(1+2*mpi)*(1+2*mpi2)
# Отрисовка двумерных графиков
def plot2d(x, y, field, label):
    my_map = plt.get_cmap("plasma")
    plt.set_cmap(my_map)
    extent = [x[0], x[-1], y[0], y[-1]]

    plt.subplot(1, 2, 1)
    plt.imshow(abs(field), extent=extent)
    plt.title("Амплитуда " + label)
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(np.angle(field), extent=extent, vmin=-np.pi, vmax=np.pi)
    plt.title("Фаза " + label)
    plt.colorbar()
    plt.show()

N = 200
M = 512
a = 5
b = N ** 2 / (4 * a * M)

#ОДНОМЕРНОЕ РЕШЕНИЕ
# получение шага, дискретизация x и входной ф-ии
h, x, Gauss_func = disc(a, N, gauss)
# Амплитуда и фаза моды Гаусса
plot(x, Gauss_func, " моды Гаусса")
# БПФ гауссового пучка
GaussFFT = FFT(Gauss_func, h, N, M)
u = disc(b, N, gauss)[1]
plot(u, GaussFFT, " БПФ моды Гаусса")
# преобразование Фурье численным интегрированием
GaussIntegr = np.array([finit_fourier(x, u[i], h) for i in range(len(u))])
plot(u, GaussIntegr, "ПФ ЧМ моды Гаусса")
# БПФ и ПФ ЧМ гауссового пучка на одном графике
plot(u, GaussFFT, "БПФ моды Гаусса", GaussIntegr, "ПФ ЧМ моды Гаусса")
# теперь входное поле
# получаем отрезки x, вектор входного поля
h, x, Inp_func = disc(a, N, f)
# Амплитуда и фаза входной функции
plot(x, Inp_func, "f(x) = x²exp(-x²)")
# БПФ входной функции
InpFFT = FFT(Inp_func, h, N, M)
u = disc(b, N, f)[1]
plot(u, InpFFT, "БПФ f(x) = x²exp(-x²)")
# Аналитическое решение F_a(u)
Analytic = np.array([f_a(u[i]) for i in range(len(u))])
plot(u, Analytic, "f_a(ε) =exp(-π²ε²)*(1-2π²ε²)*sqrt(π)/2")
# БПФ и аналитическое решение входного поля на одном графике
plot(u, InpFFT, "БПФ f(x) = x²exp(-x²)", Analytic, "f_a(ε) = exp(-π²ε²)*(1-2π²ε²)*sqrt(π)/2")

#ДВУМЕРНОЕ РЕШЕНИЕ
# Получаем двумерную моду Гаусса
Gauss2D = disc2d(a, N, gauss2d)[1]
plot2d(x, x, Gauss2D, "моды Гаусса")
# БПФ двумерной моды Гаусса
GaussFFT2d = fft2d(Gauss2D, h, N, M)
plot2d(u, u, GaussFFT2d, "БПФ моды Гаусса")
# Двумерное входное поле
Inp_func2d = disc2d(a, N, f2d)[1]
plot2d(x, x, Inp_func2d, "входного поля")
# БПФ двумерного входного поля
InpFFT2d = fft2d(Inp_func2d, h, N, M)
plot2d(u, u, InpFFT2d, "БПФ входного поля")
# Двумерное аналитическое решение
Analytic2d = np.array([[f_a2d(u[i], u[j]) for i in range(len(u))] for j in range(len(u))])
plot2d(u, u, Analytic2d, "аналитического решения")
