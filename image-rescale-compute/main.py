import argparse

import numpy as np
from PySide6 import QtGui, QtWidgets
import pyqtgraph as pg

from ImageRescaleRhiWidget import ImageRescaleRhiWidget


class Widget(QtWidgets.QWidget):
    def __init__(self, image, *, api=None, debug=False):
        super().__init__()
        self._rhi_widget = None
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        view = ImageRescaleRhiWidget()
        if api is not None:
            view.setApi(api)
        self.layout.addWidget(view, 1)
        self._rhi_widget = view

        imgitem = pg.ImageItem(image, axisOrder='row-major')
        hist = pg.HistogramLUTWidget(gradientPosition="left")
        hist.sigLookupTableChanged.connect(self.onLutChanged)
        hist.sigLevelsChanged.connect(self.onLevelsChanged)
        hist.setImageItem(imgitem)
        self.layout.addWidget(hist, 0)

        view.setData(image)
        view.setLevels(hist.getLevels())

    def closeEvent(self, e):
        # top-level widgets get this event
        print('closeEvent')
        if self._rhi_widget is not None:
            self._rhi_widget.releaseResources()
        e.accept()

    def onLutChanged(self, hist):
        self._rhi_widget.setLut(hist.getLookupTable(n=256, alpha=True))

    def onLevelsChanged(self, hist):
        self._rhi_widget.setLevels(hist.getLevels())


def parse_api(api : str | None) -> QtWidgets.QRhiWidget.Api | None:
    match api:
        case None:
            out = None
        case 'opengl':
            out = QtWidgets.QRhiWidget.Api.OpenGL
        case 'vulkan':
            out = QtWidgets.QRhiWidget.Api.Vulkan
        case _:
            raise ValueError(f"unrecognized api: {api}")
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='?')
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--api', choices=['opengl', 'vulkan'])
    parser.add_argument('--debug', action='store_true')
    ARGS = parser.parse_args()

    if ARGS.filename is not None:
        qimage = QtGui.QImage(ARGS.filename)
        print(qimage.size())
        print(qimage.format())
        qimage.convertTo(QtGui.QImage.Format.Format_Grayscale8)
        img = pg.functions.ndarray_from_qimage(qimage).astype(np.float32)
    else:
        rng = np.random.default_rng()
        img = rng.rayleigh(size=(ARGS.size, ARGS.size)).astype(np.float32)

    app = pg.mkQApp()
    win = Widget(img, api=parse_api(ARGS.api), debug=ARGS.debug)
    win.resize(800, 600)
    win.show()
    app.exec()

if __name__ == '__main__':
    main()
