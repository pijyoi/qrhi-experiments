import sys

import trimesh
from PySide6 import QtWidgets

from MeshRhiWidget import MeshRhiWidget

class Widget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._rhi_widget = MeshRhiWidget(self)
        layout.addWidget(self._rhi_widget)

    def closeEvent(self, e):
        # top-level widgets get this event
        print('closeEvent')
        self._rhi_widget.releaseResources()
        e.accept()

if __name__ == '__main__':
    mesh = trimesh.load_mesh(sys.argv[1])
    app = QtWidgets.QApplication([])
    win = Widget()
    win.resize(640, 480)
    win._rhi_widget.setData(mesh)
    win.show()
    app.exec()
