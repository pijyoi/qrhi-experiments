import sys

from PySide6 import QtGui, QtWidgets

from ImageRhiWidget import ImageRhiWidget

class Widget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._rhi_widget = ImageRhiWidget(self)
        layout.addWidget(self._rhi_widget)

    def closeEvent(self, e):
        # top-level widgets get this event
        print('closeEvent')
        self._rhi_widget.releaseResources()
        e.accept()

if __name__ == '__main__':
    qimage = QtGui.QImage(sys.argv[1])
    qimage.convertTo(QtGui.QImage.Format.Format_RGBA8888)
    app = QtWidgets.QApplication([])
    win = Widget()
    win.resize(640, 480)
    win._rhi_widget.setData(qimage)
    win.show()
    app.exec()
