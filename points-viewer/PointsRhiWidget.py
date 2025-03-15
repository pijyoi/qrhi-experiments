import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

class PointsRhiWidget(QtWidgets.QRhiWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)

        self.m_rhi = None
        self.m_vbuf = None
        self.m_ubuf = None
        self.m_srb = None
        self.m_pipeline = None

        self.xaxis_rotate = 0.0
        self.yaxis_rotate = 0.0
        self.distance = 0
        self.need_upload = None     # None means no data

        self.vert_shader = QtGui.QShader.fromSerialized(open("points.vert.qsb", "rb").read())
        self.frag_shader = QtGui.QShader.fromSerialized(open("points.frag.qsb", "rb").read())

    def releaseResources(self):
        print('releaseResources')

        if self.m_pipeline is None:
            return

        self.m_rhi = None

        self.m_pipeline.destroy()
        self.m_pipeline = None
        self.m_srb.destroy()
        self.m_srb = None
        self.m_ubuf.destroy()
        self.m_ubuf = None
        self.m_vbuf.destroy()
        self.m_vbuf = None

    def keyReleaseEvent(self, ev):
        super().keyReleaseEvent(ev)

    def mousePressEvent(self, ev):
        self.mousePos = ev.position()

    def mouseMoveEvent(self, ev):
        lpos = ev.position()
        diff = lpos - self.mousePos
        self.mousePos = lpos
        
        if ev.buttons() == QtCore.Qt.MouseButton.LeftButton:
            self.xaxis_rotate += diff.y()
            self.yaxis_rotate += diff.x()
            self.update()

    def wheelEvent(self, ev):
        delta = ev.angleDelta().x() or ev.angleDelta().y()
        self.distance *= 0.999**delta
        self.update()

    def setData(self, mesh):
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
        self.distance = (mesh.extents**2).sum()**0.5

        self.vertex_data = np.empty((len(mesh.vertices), 6), dtype=np.float32)
        self.vertex_data[:, 0:3] = mesh.vertices
        self.vertex_data[:, 3:6] = mesh.vertex_normals
        self.need_upload = True
        self.update()

    def initialize(self, cb):
        if self.m_rhi != self.rhi():
            self.m_pipeline = None
            self.m_rhi = self.rhi()

        if self.m_pipeline is not None:
            return
        
        print("initialize")

        # don't call create on the buffers yet
        self.m_vbuf = self.m_rhi.newBuffer(QtGui.QRhiBuffer.Immutable, QtGui.QRhiBuffer.VertexBuffer, 0)
        if self.need_upload is False:
            # need to re-upload to the new buffers
            self.need_upload = True

        self.ubuf_data = np.zeros((8, 4), dtype=np.float32)
        self.m_ubuf = self.m_rhi.newBuffer(QtGui.QRhiBuffer.Dynamic, QtGui.QRhiBuffer.UniformBuffer, self.ubuf_data.nbytes)
        self.m_ubuf.create()

        self.m_srb = self.m_rhi.newShaderResourceBindings()
        SRB = QtGui.QRhiShaderResourceBinding
        self.m_srb.setBindings([
            SRB.uniformBuffer(0, SRB.VertexStage, self.m_ubuf),
        ])
        self.m_srb.create()

        self.m_pipeline = self.m_rhi.newGraphicsPipeline()

        # set up for non-premultiplied alpha
        BF = QtGui.QRhiGraphicsPipeline.BlendFactor
        blend = QtGui.QRhiGraphicsPipeline.TargetBlend()
        blend.enable = True
        blend.srcColor = BF.SrcAlpha
        self.m_pipeline.setTargetBlends([blend])

        self.m_pipeline.setTopology(QtGui.QRhiGraphicsPipeline.Topology.Points)
        self.m_pipeline.setDepthTest(True)
        self.m_pipeline.setDepthWrite(True)
        self.m_pipeline.setShaderStages([
            QtGui.QRhiShaderStage(QtGui.QRhiShaderStage.Vertex, self.vert_shader),
            QtGui.QRhiShaderStage(QtGui.QRhiShaderStage.Fragment, self.frag_shader),
        ])
        inputLayout = QtGui.QRhiVertexInputLayout()
        inputLayout.setBindings([
            QtGui.QRhiVertexInputBinding(6 * 4)
        ])
        inputLayout.setAttributes([
            QtGui.QRhiVertexInputAttribute(0, 0, QtGui.QRhiVertexInputAttribute.Float3, 0),
            QtGui.QRhiVertexInputAttribute(0, 1, QtGui.QRhiVertexInputAttribute.Float3, 3 * 4)
        ])
        self.m_pipeline.setVertexInputLayout(inputLayout)
        self.m_pipeline.setShaderResourceBindings(self.m_srb)
        self.m_pipeline.setRenderPassDescriptor(self.renderTarget().renderPassDescriptor())
        self.m_pipeline.create()

    def render(self, cb):
        if self.need_upload is None:
            return

        mat_view = QtGui.QMatrix4x4()
        mat_view.translate(0, 0, -self.distance)
        mat_view.rotate(QtGui.QQuaternion.fromEulerAngles(self.xaxis_rotate, self.yaxis_rotate, 0))

        mat_normal = mat_view.normalMatrix()

        outputSize = self.renderTarget().pixelSize()
        mat_mvp = self.m_rhi.clipSpaceCorrMatrix()
        r = outputSize.width() / outputSize.height()
        mat_mvp.perspective(45.0, r, 0.01, 1000.0)

        mat_mvp *= mat_view

        # pack to uniform buffer alignment requirements
        ubuf_data = np.zeros_like(self.ubuf_data)
        ubuf_data[0:4, 0:4] = np.array(mat_mvp.data()).reshape((4, 4))
        ubuf_data[4:7, 0:3] = np.array(mat_normal.data()).reshape((3, 3))
        ubuf_data[7, 0] = 100.0      # size scale

        resourceUpdates = self.m_rhi.nextResourceUpdateBatch()

        if self.need_upload:
            if self.m_vbuf.size() < self.vertex_data.nbytes:
                self.m_vbuf.setSize(self.vertex_data.nbytes)
                self.m_vbuf.create()
            resourceUpdates.uploadStaticBuffer(self.m_vbuf, self.vertex_data)

            self.need_upload = False

        resourceUpdates.updateDynamicBuffer(self.m_ubuf, 0, ubuf_data.nbytes, ubuf_data)

        clearColor = QtGui.QColor.fromRgbF(0.0, 0.0, 0.0, 1.0)
        ds = QtGui.QRhiDepthStencilClearValue(1.0, 0)
        cb.beginPass(self.renderTarget(), clearColor, ds, resourceUpdates)

        cb.setGraphicsPipeline(self.m_pipeline)
        cb.setViewport(QtGui.QRhiViewport(0, 0, outputSize.width(), outputSize.height()))
        cb.setShaderResources()
        cb.setVertexInput(0, [(self.m_vbuf, 0)])
        cb.draw(len(self.vertex_data))

        cb.endPass()
