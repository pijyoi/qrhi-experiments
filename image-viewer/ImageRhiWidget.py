import struct
from PySide6 import QtCore, QtGui, QtWidgets

class ImageRhiWidget(QtWidgets.QRhiWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.m_rhi = None
        self.m_srb = None
        self.m_ubuf = None
        self.m_texture = None
        self.m_sampler = None
        self.m_pipeline = None

        self.vert_shader = QtGui.QShader.fromSerialized(open("texture.vert.qsb", "rb").read())
        self.frag_shader = QtGui.QShader.fromSerialized(open("texture.frag.qsb", "rb").read())

        self.setData(None)

    def releaseResources(self):
        print('releaseResources')

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

        print("initialize")

        self.m_ubuf = self.m_rhi.newBuffer(QtGui.QRhiBuffer.Dynamic, QtGui.QRhiBuffer.UniformBuffer, 4)
        self.m_ubuf.create()

        resourceUpdates = self.m_rhi.nextResourceUpdateBatch()
        yscale = -1.0 if self.m_rhi.isYUpInNDC() else 1.0
        resourceUpdates.updateDynamicBuffer(self.m_ubuf, 0, 4, struct.pack('f', yscale))
        cb.resourceUpdate(resourceUpdates)

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
            SRB.sampledTexture(0, SRB.FragmentStage, self.m_texture, self.m_sampler)
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

        resourceUpdates = None
        if not self.uploaded:
            resourceUpdates = self.m_rhi.nextResourceUpdateBatch()
            resourceUpdates.uploadTexture(self.m_texture, self.qimage)
            self.uploaded = True

        clearColor = QtGui.QColor.fromRgbF(0.0, 0.0, 0.0, 1.0)
        cv = QtGui.QRhiDepthStencilClearValue(1.0, 0)
        cb.beginPass(self.renderTarget(), clearColor, cv, resourceUpdates)

        outputSize = self.renderTarget().pixelSize()
        cb.setGraphicsPipeline(self.m_pipeline)
        cb.setViewport(QtGui.QRhiViewport(0, 0, outputSize.width(), outputSize.height()))
        cb.setShaderResources()
        cb.draw(4)

        cb.endPass()
