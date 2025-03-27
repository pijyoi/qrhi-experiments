import argparse

import trimesh
from PySide6 import QtWidgets

from MeshRhiWidget import MeshRhiWidget

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

class Widget(QtWidgets.QWidget):
    def __init__(self, *, api=None, debug=False):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._rhi_widget = MeshRhiWidget(self, api=api, debug=debug)
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

    app = QtWidgets.QApplication([])
    win = Widget(api=parse_api(ARGS.api), debug=ARGS.debug)
    if ARGS.filename is not None:
        mesh = trimesh.load_mesh(ARGS.filename)
        win._rhi_widget.setData(mesh)
    win.resize(640, 480)
    win.show()
    app.exec()
