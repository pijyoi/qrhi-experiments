import logging
import pathlib
from PySide6 import QtCore, QtGui, QtWidgets
import numpy as np

def load_shader(filename):
    pathname = pathlib.Path(__file__).parent / filename
    return QtGui.QShader.fromSerialized(pathname.read_bytes())

class ImageRescaleRhiWidget(QtWidgets.QRhiWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        self.m_rhi = None

        self.m_ubuf = None
        self.m_texture = None

        self.m_texture_lut = None
        self.m_sampler_lut = None

        self.m_srb = None
        self.m_sampler = None
        self.m_pipeline = None

        self.vert_shader = load_shader("rescale.vert.qsb")
        self.frag_shader = load_shader("rescale.frag.qsb")

        lut = np.zeros((256, 4), dtype=np.uint8)
        lut[:, :3] = np.arange(256)[:, None]
        lut[:, 3] = 255

        self.setData(None)
        self.setLevels((0.0, 1.0))
        self.setLut(lut)
        self.resetZoom()

    def releaseResources(self):
        logging.debug("releaseResources")

        if self.m_pipeline is None:
            return

        self.data_uploaded = False
        self.lut_uploaded = False
        self.img_rendered = False

        self.m_rhi = None

        self.m_pipeline.destroy()
        self.m_pipeline = None
        self.m_sampler.destroy()
        self.m_sampler = None
        self.m_srb.destroy()
        self.m_srb = None

        self.m_sampler_lut.destroy()
        self.m_sampler_lut = None
        self.m_texture_lut.destroy()
        self.m_texture_lut = None

        self.m_texture.destroy()
        self.m_texture = None
        self.m_ubuf.destroy()
        self.m_ubuf = None

    def resetZoom(self):
        self.pan_dx = 0
        self.pan_dy = 0
        self.zoom_dz = 0
        self.update()

    def keyReleaseEvent(self, ev):
        if ev.key() == QtCore.Qt.Key.Key_Home:
            self.resetZoom()
        else:
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

    def setData(self, image):
        if image is None:
            image = np.zeros((1, 1), dtype=np.float32)

        self.image = image
        self.data_uploaded = False
        self.img_rendered = False
        self.update()

    def setLevels(self, levels):
        self.levels = levels
        self.img_rendered = False
        self.update()

    def setLut(self, lut):
        self.lut = lut
        self.lut_uploaded = False
        self.img_rendered = False
        self.update()

    def initialize(self, cb):
        if self.m_rhi != self.rhi():
            self.m_pipeline = None
            self.m_rhi = self.rhi()

        if self.m_pipeline is not None:
            return

        logging.debug("initialize")

        self.ubuf_data = np.zeros((5, 4), dtype=np.float32)
        self.m_ubuf = self.m_rhi.newBuffer(QtGui.QRhiBuffer.Dynamic, QtGui.QRhiBuffer.UniformBuffer, self.ubuf_data.nbytes)
        self.m_ubuf.create()

        # lut resources

        self.m_texture_lut = self.m_rhi.newTexture(QtGui.QRhiTexture.Format.RGBA8,
                                                   QtCore.QSize(self.lut.shape[0], 1))
        self.m_texture_lut.create()

        FI = QtGui.QRhiSampler.Filter
        AM = QtGui.QRhiSampler.AddressMode
        self.m_sampler_lut = self.m_rhi.newSampler(FI.Nearest, FI.Nearest, FI.None_, AM.ClampToEdge, AM.ClampToEdge)
        self.m_sampler_lut.create()

        # texture blitter resources

        h, w = self.image.shape
        self.m_texture = self.m_rhi.newTexture(QtGui.QRhiTexture.Format.R32F,
                                               QtCore.QSize(w, h))
        self.m_texture.create()

        FI = QtGui.QRhiSampler.Filter
        AM = QtGui.QRhiSampler.AddressMode
        self.m_sampler = self.m_rhi.newSampler(FI.Nearest, FI.Nearest, FI.None_, AM.ClampToEdge, AM.ClampToEdge)
        self.m_sampler.create()

        self.m_srb = self.m_rhi.newShaderResourceBindings()
        SRB = QtGui.QRhiShaderResourceBinding
        self.m_srb.setBindings([
            SRB.uniformBuffer(0, SRB.VertexStage | SRB.FragmentStage, self.m_ubuf),
            SRB.sampledTexture(1, SRB.FragmentStage, self.m_texture_lut, self.m_sampler_lut),
            SRB.sampledTexture(2, SRB.FragmentStage, self.m_texture, self.m_sampler),
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
        h, w = self.image.shape
        if (size := QtCore.QSize(w, h)) != self.m_texture.pixelSize():
            self.m_texture.setPixelSize(size)
            self.m_texture.create()

        resourceUpdates = self.m_rhi.nextResourceUpdateBatch()

        if not self.data_uploaded:
            desc = QtGui.QRhiTextureSubresourceUploadDescription(self.image, self.image.nbytes)
            resourceUpdates.uploadTexture(self.m_texture, QtGui.QRhiTextureUploadEntry(0, 0, desc))
            self.data_uploaded = True

        if not self.lut_uploaded:
            qimage = QtGui.QImage(self.lut, self.lut.shape[0], 1, QtGui.QImage.Format.Format_RGBA8888)
            resourceUpdates.uploadTexture(self.m_texture_lut, qimage)
            self.lut_uploaded = True

        view = self.m_rhi.clipSpaceCorrMatrix()
        view.scale(1, -1)   # y-flip
        view.translate(self.pan_dx * 2 / self.width(), self.pan_dy * 2 / self.height())
        view.scale(0.999 ** -self.zoom_dz)
        self.ubuf_data[0:4, 0:4] = np.array(view.data()).reshape((4, 4))
        self.ubuf_data[4, 0:2] = np.array(self.levels)
        resourceUpdates.updateDynamicBuffer(self.m_ubuf, 0, self.ubuf_data.nbytes, self.ubuf_data)

        clearColor = QtGui.QColor.fromRgbF(0.0, 0.0, 0.0, 1.0)
        cv = QtGui.QRhiDepthStencilClearValue(1.0, 0)
        cb.beginPass(self.renderTarget(), clearColor, cv, resourceUpdates)

        outputSize = self.renderTarget().pixelSize()
        cb.setGraphicsPipeline(self.m_pipeline)
        cb.setViewport(QtGui.QRhiViewport(0, 0, outputSize.width(), outputSize.height()))
        cb.setShaderResources()
        cb.draw(4)

        cb.endPass()
