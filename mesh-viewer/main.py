import argparse

import trimesh
from PySide6 import QtWidgets

from MeshRhiWidget import MeshRhiWidget

def parse_api(api : str) -> QtWidgets.QRhiWidget.Api:
    match api:
        case 'opengl':
            out = QtWidgets.QRhiWidget.Api.OpenGL
        case 'vulkan':
            out = QtWidgets.QRhiWidget.Api.Vulkan
        case _:
            raise ValueError(f"unrecognized api: {api}")
    return out

class Widget(QtWidgets.QWidget):
    def __init__(self, api = None):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._rhi_widget = MeshRhiWidget(self)
        if api is not None:
            self._rhi_widget.setApi(parse_api(api))
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
    ARGS = parser.parse_args()

    mesh = trimesh.load_mesh(ARGS.filename)
    app = QtWidgets.QApplication([])
    win = Widget(ARGS.api)
    win.resize(640, 480)
    win._rhi_widget.setData(mesh)
    win.show()
    app.exec()
