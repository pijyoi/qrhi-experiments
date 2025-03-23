import pathlib
from PySide6 import QtCore, QtGui, QtWidgets
import numpy as np

def load_shader(filename):
    pathname = pathlib.Path(__file__).parent / filename
    return QtGui.QShader.fromSerialized(pathname.read_bytes())

class ImageRescaleRhiWidget(QtWidgets.QRhiWidget):

    def __init__(self, parent=None, *, api=None, debug=False):
        super().__init__(parent)

        if isinstance(api, QtWidgets.QRhiWidget.Api):
            self.setApi(api)

        if debug:
            self.setDebugLayerEnabled(debug)

        self.m_rhi = None

        self.m_ubuf = None
        self.m_texture = None

        self.m_compute_srb = None
        self.m_texture_lut = None
        self.m_sampler_lut = None
        self.m_ssbo = None
        self.m_compute_pipeline = None

        self.m_blitter_srb = None
        self.m_sampler = None
        self.m_blitter_pipeline = None

        self.comp_shader = load_shader("rescale.comp.qsb")
        self.vert_shader = load_shader("texture.vert.qsb")
        self.frag_shader = load_shader("texture.frag.qsb")

        lut = np.zeros((256, 4), dtype=np.uint8)
        lut[:, :3] = np.arange(256)[:, None]
        lut[:, 3] = 255

        self.setData(None)
        self.setLevels((0.0, 1.0))
        self.setLut(lut)

    def releaseResources(self):
        print('releaseResources')

        if self.m_blitter_pipeline is None:
            return

        self.data_uploaded = False
        self.lut_uploaded = False
        self.img_rendered = False

        self.m_rhi = None

        self.m_blitter_pipeline.destroy()
        self.m_blitter_pipeline = None
        self.m_sampler.destroy()
        self.m_sampler = None
        self.m_blitter_srb.destroy()
        self.m_blitter_srb = None

        self.m_compute_pipeline.destroy()
        self.m_compute_pipeline = None
        self.m_ssbo.destroy()
        self.m_ssbo = None
        self.m_sampler_lut.destroy()
        self.m_sampler_lut = None
        self.m_texture_lut.destroy()
        self.m_texture_lut = None
        self.m_compute_srb.destroy()
        self.m_compute_srb = None

        self.m_texture.destroy()
        self.m_texture = None
        self.m_ubuf.destroy()
        self.m_ubuf = None

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
            self.m_compute_pipeline = None
            self.m_blitter_pipeline = None
            self.m_rhi = self.rhi()

        if self.m_blitter_pipeline is not None:
            return

        print("initialize")

        # small lookup tables

        self.ubuf_data = np.zeros((1, 4), dtype=np.float32)
        self.m_ubuf = self.m_rhi.newBuffer(QtGui.QRhiBuffer.Dynamic, QtGui.QRhiBuffer.UniformBuffer, self.ubuf_data.nbytes)
        self.m_ubuf.create()

        self.m_texture_lut = self.m_rhi.newTexture(QtGui.QRhiTexture.Format.RGBA8,
                                                   QtCore.QSize(self.lut.shape[0], 1))
        self.m_texture_lut.create()

        # resources common to both pipelines

        h, w = self.image.shape
        self.m_texture = self.m_rhi.newTexture(QtGui.QRhiTexture.Format.RGBA8,
                                               QtCore.QSize(w, h),
                                               1, QtGui.QRhiTexture.Flag.UsedWithLoadStore)
        self.m_texture.create()

        # compute shader resources

        self.m_ssbo = self.m_rhi.newBuffer(QtGui.QRhiBuffer.Static, QtGui.QRhiBuffer.StorageBuffer, 0)

        FI = QtGui.QRhiSampler.Filter
        AM = QtGui.QRhiSampler.AddressMode
        self.m_sampler_lut = self.m_rhi.newSampler(FI.Nearest, FI.Nearest, FI.None_, AM.ClampToEdge, AM.ClampToEdge)
        self.m_sampler_lut.create()

        self.m_compute_srb = self.m_rhi.newShaderResourceBindings()
        SRB = QtGui.QRhiShaderResourceBinding
        self.m_compute_srb.setBindings([
            SRB.uniformBuffer(0, SRB.ComputeStage, self.m_ubuf),
            SRB.sampledTexture(1, SRB.ComputeStage, self.m_texture_lut, self.m_sampler_lut),
            SRB.bufferLoad(2, SRB.ComputeStage, self.m_ssbo),
            SRB.imageStore(3, SRB.ComputeStage, self.m_texture, 0),
        ])
        self.m_compute_srb.create()

        self.m_compute_pipeline = self.m_rhi.newComputePipeline()
        self.m_compute_pipeline.setShaderStage(
            QtGui.QRhiShaderStage(QtGui.QRhiShaderStage.Compute, self.comp_shader)
        )
        self.m_compute_pipeline.setShaderResourceBindings(self.m_compute_srb)
        self.m_compute_pipeline.create()

        # texture blitter resources

        FI = QtGui.QRhiSampler.Filter
        AM = QtGui.QRhiSampler.AddressMode
        self.m_sampler = self.m_rhi.newSampler(FI.Nearest, FI.Nearest, FI.None_, AM.ClampToEdge, AM.ClampToEdge)
        self.m_sampler.create()

        self.m_blitter_srb = self.m_rhi.newShaderResourceBindings()
        SRB = QtGui.QRhiShaderResourceBinding
        self.m_blitter_srb.setBindings([
            SRB.uniformBuffer(0, SRB.VertexStage, self.m_ubuf),
            SRB.sampledTexture(1, SRB.FragmentStage, self.m_texture, self.m_sampler)
        ])
        self.m_blitter_srb.create()

        self.m_blitter_pipeline = self.m_rhi.newGraphicsPipeline()

        # set up for non-premultiplied alpha
        BF = QtGui.QRhiGraphicsPipeline.BlendFactor
        blend = QtGui.QRhiGraphicsPipeline.TargetBlend()
        blend.enable = True
        blend.srcColor = BF.SrcAlpha
        self.m_blitter_pipeline.setTargetBlends([blend])

        self.m_blitter_pipeline.setTopology(QtGui.QRhiGraphicsPipeline.Topology.TriangleStrip)
        self.m_blitter_pipeline.setShaderStages([
            QtGui.QRhiShaderStage(QtGui.QRhiShaderStage.Vertex, self.vert_shader),
            QtGui.QRhiShaderStage(QtGui.QRhiShaderStage.Fragment, self.frag_shader),
        ])
        self.m_blitter_pipeline.setShaderResourceBindings(self.m_blitter_srb)
        self.m_blitter_pipeline.setRenderPassDescriptor(self.renderTarget().renderPassDescriptor())
        self.m_blitter_pipeline.create()

    def render(self, cb):
        if self.m_ssbo.size() < self.image.nbytes:
            self.m_ssbo.setSize(self.image.nbytes)
            self.m_ssbo.create()

        h, w = self.image.shape
        if (size := QtCore.QSize(w, h)) != self.m_texture.pixelSize():
            self.m_texture.setPixelSize(size)
            self.m_texture.create()

        resourceUpdates = self.m_rhi.nextResourceUpdateBatch()

        if not self.data_uploaded:
            resourceUpdates.uploadStaticBuffer(self.m_ssbo, self.image)
            self.data_uploaded = True

        if not self.lut_uploaded:
            qimage = QtGui.QImage(self.lut, self.lut.shape[0], 1, QtGui.QImage.Format.Format_RGBA8888)
            resourceUpdates.uploadTexture(self.m_texture_lut, qimage)
            self.lut_uploaded = True

        yscale = -1.0 if self.m_rhi.isYUpInNDC() else 1.0
        self.ubuf_data[0, 0:2] = np.array(self.levels)
        self.ubuf_data[0, 2] = yscale
        resourceUpdates.updateDynamicBuffer(self.m_ubuf, 0, self.ubuf_data.nbytes, self.ubuf_data)

        if not self.img_rendered:
            cb.beginComputePass(resourceUpdates)
            resourceUpdates = None
            cb.setComputePipeline(self.m_compute_pipeline)
            cb.setShaderResources()
            x_groups = (self.image.shape[1] + 31) // 32
            y_groups = (self.image.shape[0] + 7) // 8
            cb.dispatch(x_groups, y_groups, 1)
            cb.endComputePass()
            self.img_rendered = True

        clearColor = QtGui.QColor.fromRgbF(0.0, 0.0, 0.0, 1.0)
        cv = QtGui.QRhiDepthStencilClearValue(1.0, 0)
        cb.beginPass(self.renderTarget(), clearColor, cv, resourceUpdates)

        outputSize = self.renderTarget().pixelSize()
        cb.setGraphicsPipeline(self.m_blitter_pipeline)
        cb.setViewport(QtGui.QRhiViewport(0, 0, outputSize.width(), outputSize.height()))
        cb.setShaderResources()
        cb.draw(4)

        cb.endPass()
