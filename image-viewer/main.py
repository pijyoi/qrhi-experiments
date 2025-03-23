import argparse

from PySide6 import QtGui, QtWidgets

from ImageRhiWidget import ImageRhiWidget

def parse_api(api : str | None) -> QtWidgets.QRhiWidget.Api:
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

class Widget(QtWidgets.QWidget):
    def __init__(self, parent=None, *, api=None, debug=False):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._rhi_widget = ImageRhiWidget(self, api=api, debug=debug)
        layout.addWidget(self._rhi_widget)

    def closeEvent(self, e):
        # top-level widgets get this event
        print('closeEvent')
        self._rhi_widget.releaseResources()
        e.accept()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='?')
    parser.add_argument('--api', choices=['opengl', 'vulkan'])
    parser.add_argument('--debug', action='store_true')
    ARGS = parser.parse_args()

    qimage = QtGui.QImage(ARGS.filename)
    qimage.convertTo(QtGui.QImage.Format.Format_RGBA8888)
    app = QtWidgets.QApplication([])
    win = Widget(api=parse_api(ARGS.api), debug=ARGS.debug)
    win.resize(640, 480)
    win._rhi_widget.setData(qimage)
    win.show()
    app.exec()
