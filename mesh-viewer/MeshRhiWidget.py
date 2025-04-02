import logging
import pathlib
import numpy as np
import trimesh
from PySide6 import QtCore, QtGui, QtWidgets

def load_shader(filename):
    pathname = pathlib.Path(__file__).parent / filename
    return QtGui.QShader.fromSerialized(pathname.read_bytes())

class MeshLoader(QtCore.QObject):
    sigLoaded = QtCore.Signal(object)

    def __init__(self, pathname):
        super().__init__()
        self.pathname = pathname

    def __call__(self):
        mesh = trimesh.load_mesh(self.pathname)

        if len(mesh.vertices) == 0:
            return

        # access vertex_normals property to trigger its
        # potentially expensive computation
        mesh.vertex_normals

        self.sigLoaded.emit(mesh)

class MeshRhiWidget(QtWidgets.QRhiWidget):

    def __init__(self, parent=None, *, api=None, debug=False):
        super().__init__(parent)

        if isinstance(api, QtWidgets.QRhiWidget.Api):
            self.setApi(api)

        if debug:
            self.setDebugLayerEnabled(debug)

        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setAcceptDrops(True)

        self.m_rhi = None
        self.m_vbuf = None
        self.m_ibuf = None
        self.m_ubuf = None
        self.m_texture = None
        self.m_sampler = None
        self.m_srb = None
        self.m_pipeline = None

        self.vert_shader = load_shader("shaded.vert.qsb")
        self.frag_shader = load_shader("shaded.frag.qsb")

        self.model_center = [0, 0, 0]
        self.distance = 0
        self.need_upload = None     # None means no data
        self.background_white = False
        self.wireframe_toggle = False
        self.cullface_toggle = False
        self.resetView()

    def releaseResources(self):
        logging.debug("releaseResources")

        if self.m_pipeline is None:
            return

        self.m_rhi = None

        self.m_pipeline.destroy()
        self.m_pipeline = None
        self.m_srb.destroy()
        self.m_srb = None
        self.m_sampler.destroy()
        self.m_sampler = None
        self.m_texture.destroy()
        self.m_texture = None
        self.m_ubuf.destroy()
        self.m_ubuf = None
        self.m_ibuf.destroy()
        self.m_ibuf = None
        self.m_vbuf.destroy()
        self.m_vbuf = None

    def resetView(self):
        self.rotation = QtGui.QQuaternion()
        self.zaxis_zoom = 0.0
        self.xaxis_pan = 0.0
        self.yaxis_pan = 0.0

    def keyReleaseEvent(self, ev):
        match ev.key():
            case QtCore.Qt.Key.Key_Home:
                self.resetView()
                self.update()
            case QtCore.Qt.Key.Key_B:
                self.background_white = not self.background_white
                self.update()
            case QtCore.Qt.Key.Key_C:
                self.cullface_toggle = True
                self.update()
            case QtCore.Qt.Key.Key_W:
                self.wireframe_toggle = True
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
            if ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
                self.xaxis_pan += diff.x()
                self.yaxis_pan -= diff.y()
            else:
                delta = QtGui.QQuaternion.fromEulerAngles(diff.y(), diff.x(), 0)
                self.rotation = delta * self.rotation
            self.update()

    def wheelEvent(self, ev):
        delta = ev.angleDelta().x() or ev.angleDelta().y()
        self.zaxis_zoom += delta
        self.update()

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        if not ev.mimeData().hasUrls():
            return

        ev.acceptProposedAction()

        links = [url.toLocalFile() for url in ev.mimeData().urls()]
        self.loadMesh(links[0])

    def loadMesh(self, pathname):
        runner = MeshLoader(pathname)
        runner.sigLoaded.connect(self.setData)
        QtCore.QThreadPool.globalInstance().start(runner)

    @QtCore.Slot(object)
    def setData(self, mesh):
        self.model_center = mesh.center_mass.tolist()
        self.distance = (mesh.extents**2).sum()**0.5

        self.vertex_data = np.zeros((len(mesh.vertices), 8), dtype=np.float32)
        self.vertex_data[:, 0:3] = mesh.vertices
        self.vertex_data[:, 3:6] = mesh.vertex_normals

        dtype = np.uint16 if len(self.vertex_data) <= 65535 else np.uint32
        self.faces_data = np.ascontiguousarray(mesh.faces, dtype=dtype)

        # create a single-pixeled texture to use as constant color
        self.image_data = QtGui.QImage(1, 1, QtGui.QImage.Format.Format_RGBA8888)
        self.image_data.fill(QtCore.Qt.GlobalColor.white)

        visual = mesh.visual

        if visual.kind == 'vertex':
            visual = visual.to_texture()

        if visual.kind == 'texture':
            self.vertex_data[:, 6:8] = visual.uv
            material = visual.material
            if not hasattr(material, 'image'):
                material = material.to_simple()
            if material.image is not None:
                self.image_data = material.image.toqimage().mirrored()
                self.image_data.convertTo(QtGui.QImage.Format.Format_RGBA8888)
            else:
                self.image_data.fill(QtGui.QColor(*material.main_color.tolist()))

        self.need_upload = True
        self.update()

    def initialize(self, cb):
        if self.m_rhi != self.rhi():
            self.m_pipeline = None
            self.m_rhi = self.rhi()

        if self.m_pipeline is not None:
            return
        
        logging.debug("initialize")

        # don't call create on the buffers yet
        self.m_vbuf = self.m_rhi.newBuffer(QtGui.QRhiBuffer.Immutable, QtGui.QRhiBuffer.VertexBuffer, 0)
        self.m_ibuf = self.m_rhi.newBuffer(QtGui.QRhiBuffer.Immutable, QtGui.QRhiBuffer.IndexBuffer, 0)
        if self.need_upload is False:
            # need to re-upload to the new buffers
            self.need_upload = True

        self.ubuf_data = np.zeros((8, 4), dtype=np.float32)
        self.m_ubuf = self.m_rhi.newBuffer(QtGui.QRhiBuffer.Dynamic, QtGui.QRhiBuffer.UniformBuffer, self.ubuf_data.nbytes)
        self.m_ubuf.create()

        self.m_texture = self.m_rhi.newTexture(QtGui.QRhiTexture.Format.RGBA8, QtCore.QSize(1, 1))
        self.m_texture.create()
        FI = QtGui.QRhiSampler.Filter
        AM = QtGui.QRhiSampler.AddressMode
        self.m_sampler = self.m_rhi.newSampler(FI.Nearest, FI.Nearest, FI.None_, AM.ClampToEdge, AM.ClampToEdge)
        self.m_sampler.create()

        self.m_srb = self.m_rhi.newShaderResourceBindings()
        SRB = QtGui.QRhiShaderResourceBinding
        self.m_srb.setBindings([
            SRB.uniformBuffer(0, SRB.VertexStage | SRB.FragmentStage, self.m_ubuf),
            SRB.sampledTexture(1, SRB.FragmentStage, self.m_texture, self.m_sampler),
        ])
        self.m_srb.create()

        self.m_pipeline = self.m_rhi.newGraphicsPipeline()
        blend = QtGui.QRhiGraphicsPipeline.TargetBlend()
        self.m_pipeline.setTargetBlends([blend])
        self.m_pipeline.setTopology(QtGui.QRhiGraphicsPipeline.Topology.Triangles)
        self.m_pipeline.setDepthTest(True)
        self.m_pipeline.setDepthWrite(True)
        # self.m_pipeline.setCullMode(QtGui.QRhiGraphicsPipeline.CullMode.Back)
        self.m_pipeline.setShaderStages([
            QtGui.QRhiShaderStage(QtGui.QRhiShaderStage.Vertex, self.vert_shader),
            QtGui.QRhiShaderStage(QtGui.QRhiShaderStage.Fragment, self.frag_shader),
        ])
        inputLayout = QtGui.QRhiVertexInputLayout()
        inputLayout.setBindings([
            QtGui.QRhiVertexInputBinding(8 * 4)
        ])
        inputLayout.setAttributes([
            QtGui.QRhiVertexInputAttribute(0, 0, QtGui.QRhiVertexInputAttribute.Float3, 0),
            QtGui.QRhiVertexInputAttribute(0, 1, QtGui.QRhiVertexInputAttribute.Float3, 3 * 4),
            QtGui.QRhiVertexInputAttribute(0, 2, QtGui.QRhiVertexInputAttribute.Float2, 6 * 4),
        ])
        self.m_pipeline.setVertexInputLayout(inputLayout)
        self.m_pipeline.setShaderResourceBindings(self.m_srb)
        self.m_pipeline.setRenderPassDescriptor(self.renderTarget().renderPassDescriptor())
        self.m_pipeline.create()

    def render(self, cb):
        if self.need_upload is None:
            return

        distance = self.distance * 0.999**self.zaxis_zoom

        mat_model = QtGui.QMatrix4x4()
        mat_model.translate(*[-x for x in self.model_center])

        xpan = self.xaxis_pan * 2 / self.width() * distance
        ypan = self.yaxis_pan * 2 / self.height() * distance

        mat_view = QtGui.QMatrix4x4()
        mat_view.translate(xpan, ypan, -distance)
        mat_view.rotate(self.rotation)

        mat_normal = mat_view.normalMatrix()

        outputSize = self.renderTarget().pixelSize()
        mat_mvp = self.m_rhi.clipSpaceCorrMatrix()
        r = outputSize.width() / outputSize.height()
        mat_mvp.perspective(45.0, r, 0.01 * distance, 1000.0 * distance)

        mat_mvp *= mat_view
        mat_mvp *= mat_model

        # pack to uniform buffer alignment requirements
        ubuf_data = np.zeros_like(self.ubuf_data)
        ubuf_data[0:4, 0:4] = np.array(mat_mvp.data()).reshape((4, 4))
        ubuf_data[4:7, 0:3] = np.array(mat_normal.data()).reshape((3, 3))
        ubuf_data[7:8, 0:3] = np.array([-1.0, 1.0, 1.0])  # light direction

        resourceUpdates = self.m_rhi.nextResourceUpdateBatch()

        if self.need_upload:
            if self.m_vbuf.size() < self.vertex_data.nbytes:
                self.m_vbuf.setSize(self.vertex_data.nbytes)
                self.m_vbuf.create()
            if self.m_ibuf.size() < self.faces_data.nbytes:
                self.m_ibuf.setSize(self.faces_data.nbytes)
                self.m_ibuf.create()
            resourceUpdates.uploadStaticBuffer(self.m_vbuf, 0, self.vertex_data.nbytes, self.vertex_data)
            resourceUpdates.uploadStaticBuffer(self.m_ibuf, 0, self.faces_data.nbytes, self.faces_data)

            if self.m_texture.pixelSize() != self.image_data.size():
                self.m_texture.setPixelSize(self.image_data.size())
                self.m_texture.create()
            resourceUpdates.uploadTexture(self.m_texture, self.image_data)

            self.need_upload = False

        resourceUpdates.updateDynamicBuffer(self.m_ubuf, 0, ubuf_data.nbytes, ubuf_data)

        pipeline_dirty = False

        if self.cullface_toggle:
            self.cullface_toggle = False
            CM = self.m_pipeline.CullMode
            old_mode = self.m_pipeline.cullMode()
            new_mode = CM.Back if old_mode == CM.None_ else CM.None_
            self.m_pipeline.setCullMode(new_mode)
            pipeline_dirty = True

        if self.wireframe_toggle:
            self.wireframe_toggle = False
            PM = self.m_pipeline.PolygonMode
            old_mode = self.m_pipeline.polygonMode()
            new_mode = PM.Line if old_mode == PM.Fill else PM.Fill
            self.m_pipeline.setPolygonMode(new_mode)
            pipeline_dirty = True

        if pipeline_dirty:
            self.m_pipeline.create()

        v = 1.0 if self.background_white else 0.0
        clearColor = QtGui.QColor.fromRgbF(v, v, v)
        ds = QtGui.QRhiDepthStencilClearValue(1.0, 0)
        cb.beginPass(self.renderTarget(), clearColor, ds, resourceUpdates)

        cb.setGraphicsPipeline(self.m_pipeline)
        cb.setViewport(QtGui.QRhiViewport(0, 0, outputSize.width(), outputSize.height()))
        cb.setShaderResources()
        IF = QtGui.QRhiCommandBuffer.IndexFormat
        ibuf_dtype = IF.IndexUInt16 if self.faces_data.dtype == np.uint16 else IF.IndexUInt32
        cb.setVertexInput(0, [(self.m_vbuf, 0)],
                          self.m_ibuf, 0, ibuf_dtype)
        cb.drawIndexed(self.faces_data.size)

        cb.endPass()
