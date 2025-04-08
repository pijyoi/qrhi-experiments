import logging
import pathlib
import struct
from PySide6 import QtCore, QtGui, QtWidgets

def load_shader(filename):
    pathname = pathlib.Path(__file__).parent / filename
    return QtGui.QShader.fromSerialized(pathname.read_bytes())

class ImageLoader(QtCore.QObject):
    sigLoaded = QtCore.Signal(object)

    def __init__(self, pathname):
        super().__init__()
        self.pathname = pathname

    def __call__(self):
        try:
            qimage = QtGui.QImage(self.pathname)
            qimage.convertTo(QtGui.QImage.Format.Format_RGBA8888)
        except Exception as e:
            print(e)
        else:
            self.sigLoaded.emit(qimage)

class ImageRhiWidget(QtWidgets.QRhiWidget):

    def __init__(self, parent=None, *, api=None, debug=False):
        super().__init__(parent)

        if isinstance(api, QtWidgets.QRhiWidget.Api):
            self.setApi(api)

        if debug:
            self.setDebugLayerEnabled(debug)

        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setAcceptDrops(True)

        self.m_rhi = None
        self.m_srb = None
        self.m_ubuf = None
        self.m_texture = None
        self.m_sampler = None
        self.m_pipeline = None

        self.vert_shader = load_shader("texture.vert.qsb")
        self.frag_shader = load_shader("texture.frag.qsb")

        self.setData(None)
        self.resetZoom()

    def releaseResources(self):
        logging.debug("releaseResources")

        if self.m_pipeline is None:
            return

        self.m_rhi = None
        self.uploaded = False

        self.m_pipeline.destroy()
        self.m_pipeline = None
        self.m_sampler.destroy()
        self.m_sampler = None
        self.m_texture.destroy()
        self.m_texture = None
        self.m_ubuf.destroy()
        self.m_ubuf = None
        self.m_srb.destroy()
        self.m_srb = None

    def resetZoom(self):
        self.pan_dx = 0
        self.pan_dy = 0
        self.zoom_dz = 0

    def keyReleaseEvent(self, ev):
        match ev.key():
            case QtCore.Qt.Key.Key_Q:
                QtWidgets.QApplication.instance().quit()
            case QtCore.Qt.Key.Key_Z:
                self.resetZoom()
                self.update()
            case _:
                super().keyReleaseEvent(ev)

    def mousePressEvent(self, ev):
        self.mousePos = ev.position()

    def mouseMoveEvent(self, ev):
        lpos = ev.position()
        diff = lpos - self.mousePos
        self.mousePos = lpos

        if ev.buttons() == QtCore.Qt.MouseButton.LeftButton:
            self.pan_dx += diff.x()
            self.pan_dy += diff.y()
            self.update()

    def wheelEvent(self, ev):
        delta = ev.angleDelta().x() or ev.angleDelta().y()
        self.zoom_dz += delta
        self.update()

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        if not ev.mimeData().hasUrls():
            return

        ev.acceptProposedAction()

        links = [url.toLocalFile() for url in ev.mimeData().urls()]
        self.loadImage(links[0])

    def loadImage(self, pathname):
        runner = ImageLoader(pathname)
        runner.sigLoaded.connect(self.setData)
        QtCore.QThreadPool.globalInstance().start(runner)

    @QtCore.Slot(object)
    def setData(self, qimage):
        if qimage is None:
            qimage = QtGui.QImage(1, 1, QtGui.QImage.Format.Format_RGBA8888)
            qimage.fill(QtCore.Qt.GlobalColor.transparent)

        self.qimage = qimage
        self.uploaded = False
        self.update()

    def initialize(self, cb):
        if self.m_rhi != self.rhi():
            self.m_pipeline = None
            self.m_rhi = self.rhi()

        if self.m_pipeline is not None:
            return

        logging.debug("initialize")

        self.m_ubuf = self.m_rhi.newBuffer(QtGui.QRhiBuffer.Dynamic, QtGui.QRhiBuffer.UniformBuffer, 64)
        self.m_ubuf.create()

        self.m_texture = self.m_rhi.newTexture(QtGui.QRhiTexture.Format.RGBA8, self.qimage.size())
        self.m_texture.create()
        FI = QtGui.QRhiSampler.Filter
        AM = QtGui.QRhiSampler.AddressMode
        self.m_sampler = self.m_rhi.newSampler(FI.Nearest, FI.Nearest, FI.None_, AM.ClampToEdge, AM.ClampToEdge)
        self.m_sampler.create()

        self.m_srb = self.m_rhi.newShaderResourceBindings()
        SRB = QtGui.QRhiShaderResourceBinding
        self.m_srb.setBindings([
            SRB.uniformBuffer(0, SRB.VertexStage, self.m_ubuf),
            SRB.sampledTexture(1, SRB.FragmentStage, self.m_texture, self.m_sampler)
        ])
        self.m_srb.create()

        self.m_pipeline = self.m_rhi.newGraphicsPipeline()

        # set up for non-premultiplied alpha
        BF = QtGui.QRhiGraphicsPipeline.BlendFactor
        blend = QtGui.QRhiGraphicsPipeline.TargetBlend()
        blend.enable = True
        blend.srcColor = BF.SrcAlpha
        self.m_pipeline.setTargetBlends([blend])

        self.m_pipeline.setTopology(QtGui.QRhiGraphicsPipeline.Topology.TriangleStrip)
        self.m_pipeline.setShaderStages([
            QtGui.QRhiShaderStage(QtGui.QRhiShaderStage.Vertex, self.vert_shader),
            QtGui.QRhiShaderStage(QtGui.QRhiShaderStage.Fragment, self.frag_shader),
        ])
        self.m_pipeline.setShaderResourceBindings(self.m_srb)
        self.m_pipeline.setRenderPassDescriptor(self.renderTarget().renderPassDescriptor())
        self.m_pipeline.create()

    def render(self, cb):
        if self.m_texture.pixelSize() != self.qimage.size():
            self.m_texture.setPixelSize(self.qimage.size())
            self.m_texture.create()

        resourceUpdates = self.m_rhi.nextResourceUpdateBatch()

        if not self.uploaded:
            resourceUpdates.uploadTexture(self.m_texture, self.qimage)
            self.uploaded = True

        view = self.m_rhi.clipSpaceCorrMatrix()
        view.scale(1, -1)   # y-flip
        view.translate(self.pan_dx * 2 / self.width(), self.pan_dy * 2 / self.height())
        view.scale(0.999 ** -self.zoom_dz)
        ubuf_data = struct.pack('16f', *view.data())
        resourceUpdates.updateDynamicBuffer(self.m_ubuf, 0, len(ubuf_data), ubuf_data)

        clearColor = QtGui.QColor.fromRgbF(0.0, 0.0, 0.0, 1.0)
        cv = QtGui.QRhiDepthStencilClearValue(1.0, 0)
        cb.beginPass(self.renderTarget(), clearColor, cv, resourceUpdates)

        outputSize = self.renderTarget().pixelSize()
        cb.setGraphicsPipeline(self.m_pipeline)
        cb.setViewport(QtGui.QRhiViewport(0, 0, outputSize.width(), outputSize.height()))
        cb.setShaderResources()
        cb.draw(4)

        cb.endPass()
