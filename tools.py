import pylab
import numpy as np
from typing import List
import numpy.fft as fft


class Probe:
    '''
    Класс для хранения временного сигнала в пробнике.
    '''
    def __init__(self, position: int, maxTime: int):
        '''
        position - положение пробника (номер ячейки).
        maxTime - максимально количество временных шагов для хранения в пробнике.
        '''
        self.position = position

        # Временные сигналы для полей E и H
        self.E = np.zeros(maxTime)
        self.H = np.zeros(maxTime)

        # Номер временного шага для cохранения полей
        self._time = 0

    def addData(self, E: List[float], H: List[float]):
        '''
        Добавить данные по полям E и H в пробник.
        '''
        self.E[self._time] = E[self.position]
        self.H[self._time] = H[self.position]
        self._time += 1


class AnimateFieldDisplay:
    '''
    Класс для отображения анимации распространения ЭМ волны в пространстве
    '''

    def __init__(self,
                 maxXSize: int,
                 minYSize: float, maxYSize: float,
                 yLabel: str, dx: float):
        '''
        maxXSize - размер области моделирования в отсчетах.
        minYSize, maxYSize - интервал отображения графика по оси Y.
        yLabel - метка для оси Y.
        dx - размер ячейки разбиения.
        '''
        self.dx = dx
        self.maxXSize = maxXSize
        self.minYSize = minYSize
        self.maxYSize = maxYSize
        self._xList = None
        self._line = None
        self._xlabel = 'x, м'
        self._ylabel = yLabel
        self._probeStyle = 'xr'
        self._sourceStyle = 'ok'

    def activate(self):
        '''
        Инициализировать окно с анимацией
        '''
        self._xList = np.arange(self.maxXSize) * self.dx

        # Включить интерактивный режим для анимации
        pylab.ion()

        # Создание окна для графика
        self._fig, self._ax = pylab.subplots()

        # Установка отображаемых интервалов по осям
        self._ax.set_xlim(0, self.maxXSize * self.dx)
        self._ax.set_ylim(self.minYSize, self.maxYSize)

        # Установка меток по осям
        self._ax.set_xlabel(self._xlabel)
        self._ax.set_ylabel(self._ylabel)

        # Включить сетку на графике
        self._ax.grid()

        # Отобразить поле в начальный момент времени
        self._line, = self._ax.plot(self._xList, np.zeros(self.maxXSize))

    def drawProbes(self, probesPos: List[int]):
        '''
        Нарисовать пробники.

        probesPos - список координат пробников для регистрации временных
        сигналов (в отсчетах).
        '''
        # Отобразить положение пробников
        self._ax.plot([i * self.dx for i in probesPos], [0] * len(probesPos), self._probeStyle)

    def drawSources(self, sourcesPos: List[int]):
        '''
        Нарисовать источники.

        sourcesPos - список координат источников (в отсчетах).
        '''
        # Отобразить положение источников
        self._ax.plot([i * self.dx for i in sourcesPos], [0] * len(sourcesPos),
                      self._sourceStyle)

    def drawBoundary(self, position: int):
        '''
        Нарисовать границу в области моделирования.

        position - координата X границы (в отсчетах).
        '''
        self._ax.plot([position * self.dx, position * self.dx],
                      [self.minYSize, self.maxYSize],
                      '--k')

    def stop(self):
        '''
        Остановить анимацию
        '''
        pylab.ioff()

    def updateData(self, data: List[float], timeCount: int):
        '''
        Обновить данные с распределением поля в пространстве
        '''
        self._line.set_ydata(data)
        self._ax.set_title(str(timeCount))
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()


def showProbeSignals(probes: List[Probe], minYSize: float, maxYSize: float,
                     dt: float):
    '''
    Показать графики сигналов, зарегистрированых в пробниках.

    probes - список экземпляров класса Probe.
    minYSize, maxYSize - интервал отображения графика по оси Y.
    dt - шаг дискретизации по времени.
    '''
    # Создание окна с графиков
    fig, ax = pylab.subplots()

    # Настройка внешнего вида графиков
    ax.set_xlim(0, len(probes[0].E) * dt)
    ax.set_ylim(minYSize, maxYSize)
    ax.set_xlabel('t, c')
    ax.set_ylabel('Ez, В/м')
    ax.grid()
    t = np.arange(0, len(probes[0].E)) * dt

    # Вывод сигналов в окно
    for probe in probes:
        ax.plot(t, probe.E)

    # Создание и отображение легенды на графике
    legend = ['Probe x = {}'.format(probe.position) for probe in probes]
    ax.legend(legend)

    # Показать окно с графиками
    pylab.show()


class FFT:
    '''
        Класс для получения спектра преобразованием Фурье.
    '''

    def __init__(self, signal: float, dt: float):
        '''
        size - размер массива БПФ.
        signal - сигнал, подвергаемый БПФ.
        dt - шаг дискретизации по времени
        '''
        self.size = 2 ** 18
        self.signal = signal
        self.dt = dt

    def spectr(self):
        '''
        Функция получения ПФ сигнала из пробника.
        '''
        # Взятие преобразование Фурье
        self._z = fft.fft(self.signal, self.size)
        self.z = np.abs(fft.fftshift(self._z))
        # Определние шага по частоте
        self.df = 1 / (self.size * self.dt)
        self._freq = np.arange(-self.size / 2 *
                                  self.df, self.size / 2 * self.df, self.df)
        # Построение графика
        fig, ax = pylab.subplots()
        # Настройка внешнего вида графиков
        ax.plot(self._freq, self.z / np.max(self.z))
        ax.set_xlim(0, 9e9)
        ax.set_xlabel('Частота, Гц')
        ax.set_ylabel('|S| / |Smax|')
        ax.grid()
        pylab.show()