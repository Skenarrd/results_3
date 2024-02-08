import numpy as np
from numpy.fft import fft, fftshift
import math
import tools
import matplotlib.pyplot as plt
import scipy.constants as sycon

class Sampler:
    def __init__(self, discrete: float):
        self.discrete = discrete

    def sample(self, x: float) -> int:
        return math.floor(x / self.discrete + 0.5)

class GaussianMod:
    ''' Класс с уравнением плоской волны для модулированного гауссова сигнала в дискретном виде
    dg - коэффициент, задающий начальную задержку гауссова импульса;
    wg - коэффициент, задающий ширину гауссова импульса.
    Nl - количество ячеек на длину волны.
    Sc - число Куранта.
    eps - относительная диэлектрическая проницаемость среды, в которой расположен источник.
    mu - относительная магнитная проницаемость среды, в которой расположен источник.
    '''

    def __init__(self, dg, wg, Nl, Sc=1.0, eps=1.0, mu=1.0):
        self.dg = dg
        self.wg = wg
        self.Nl = Nl
        self.Sc = Sc
        self.eps = eps
        self.mu = mu

    def getE(self, m, q):
        '''
        Расчет поля E в дискретной точке пространства m
        в дискретный момент времени q
        '''
        return (np.sin(2 * np.pi / self.Nl * (q * self.Sc - m * np.sqrt(self.eps * self.mu))) *
                np.exp(-(((q - m * np.sqrt(self.eps * self.mu) / self.Sc) - self.dg) / self.wg) ** 2))


if __name__ == '__main__':
    # Волновое сопротивление свободного пространства
    W0 = 120.0 * np.pi

    # Число Куранта
    Sc = 1.0

    # Скорость света
    c = sycon.c

    # Время расчета в отсчетах
    maxTime = 1500

    # Размер области моделирования вдоль оси X в метрах
    X = 1.5
    
    #Размер ячейки разбиения
    dx = 4e-3

    # Размер области моделирования в отсчетах
    maxSize = int(X / dx)

    #Шаг дискретизации по времени
    dt = Sc * dx / c

    # Положение источника в отсчетах
    sourcePos = int(maxSize / 2)


    # Параметры среды
    # Диэлектрическая проницаемость
    eps = np.ones(maxSize)
    eps[:] = 4.0

    # Магнитная проницаемость
    mu = np.ones(maxSize - 1)

    Ez = np.zeros(maxSize)
    Hy = np.zeros(maxSize - 1)


    # Пробники для регистрации поля
    probesPos = [sourcePos + 100]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    # Скорость обновления графика поля
    speed_refresh = 25
    

    source = GaussianMod(250, 100, 200, eps[sourcePos], mu[sourcePos])

    # Ez[1] в предыдущий момент времени
    oldEzLeft = Ez[1]

    # Ez[-2] в предыдущий момент времени
    oldEzRight = Ez[-2]

    # Расчет коэффициентов для граничных условий
    tempLeft = Sc / np.sqrt(mu[0] * eps[0])
    koeffABCLeft = (tempLeft - 1) / (tempLeft + 1)

    tempRight = Sc / np.sqrt(mu[-1] * eps[-1])
    koeffABCRight = (tempRight - 1) / (tempRight + 1)

    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -1.5
    display_ymax = 1.5

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel, dx)

    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])


    for q in range(maxTime):
        
        # Расчет компоненты поля H
        Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)

        # Расчет компоненты поля E
        Hy_shift = Hy[:-1]
        Ez[1:-1] = Ez[1:-1] + (Hy[1:] - Hy_shift) * Sc * W0 / eps[1:-1]

        # Источник возбуждения с использованием метода
        Ez[sourcePos] += (Sc / (np.sqrt(eps[sourcePos] * mu[sourcePos])) *
                          source.getE(-0.5, q + 0.5))

        # Граничные условия ABC первой степени
        Ez[0] = oldEzLeft + koeffABCLeft * (Ez[1] - Ez[0])
        oldEzLeft = Ez[1]

        Ez[-1] = oldEzRight + koeffABCRight * (Ez[-2] - Ez[-1])
        oldEzRight = Ez[-2]
        # Регистрация поля в пробниках
        for probe in probes:
            probe.addData(Ez, Hy)

        if q % speed_refresh == 0:
            display.updateData(display_field, q)

    display.stop()

    # Расчёт спектра сигнала
    sp = fftshift(np.abs(fft(probe.E)))
    # Расчёт шага частоты
    df = 1.0 / (maxTime * dt)
    # Расчёт частотной сетки
    freq = np.arange(-maxTime/ 2 , maxTime / 2 ) * df
    # Оформление сетки
    tlist = np.arange(0, maxTime * dt, dt) 

    # Вывод сигнала и спектра зарегистрированых в пробнике
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_xlim(0, 20e-9)
    ax1.set_ylim(-1.4, 1.4)
    ax1.set_xlabel('t, с')
    ax1.set_ylabel('Ez, В/м')
    ax1.plot(tlist, probe.E)
    ax1.minorticks_on()
    ax1.grid()
    ax2.set_xlim(0, 1e10)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlabel('f, Гц')
    ax2.set_ylabel('|S| / |Smax|')
    ax2.plot(freq, sp / np.max(sp))
    ax2.minorticks_on()
    ax2.grid()
    plt.show()
