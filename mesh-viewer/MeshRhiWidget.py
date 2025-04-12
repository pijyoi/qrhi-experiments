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
        try:
            mesh = trimesh.load_mesh(self.pathname)

            if len(mesh.vertices) == 0:
                raise ValueError("mesh with no vertices")

            print(f'{mesh.visual.kind=} {mesh.vertices.shape=}')

            # access vertex_normals property to trigger its
            # potentially expensive computation
            mesh.vertex_normals
        except Exception as e:
            print(e)
        else:
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
        self.m_srb_texture = None
        self.m_pipeline_texture = None
        self.m_srb_vertex = None
        self.m_pipeline_vertex = None

        self.visual_kind = 'texture'

        self.vert_texture_shader = load_shader("shaded.vert.qsb")
        self.frag_texture_shader = load_shader("shaded.frag.qsb")
        self.vert_color_shader = load_shader("color.vert.qsb")
        self.frag_color_shader = load_shader("color.frag.qsb")

        self.model_center = [0, 0, 0]
        self.distance = 0
        self.need_upload = None     # None means no data
        self.background_white = False
        self.wireframe_toggle = False
        self.cullface_toggle = False
        self.resetView()

    def releaseResources(self):
        logging.debug("releaseResources")

        if self.m_pipeline_texture is None:
            return

        self.m_rhi = None

        names = [
            'pipeline_vertex', 'srb_vertex',
            'pipeline_texture', 'srb_texture',
            'sampler', 'texture',
            'ubuf', 'ibuf', 'vbuf'
        ]

        for name in names:
            fullname = 'm_' + name
            obj = getattr(self, fullname)
            if obj is not None:
                obj.destroy()
                setattr(self, fullname, None)

    def resetView(self):
        self.rotation = QtGui.QQuaternion()
        self.zaxis_zoom = 0.0
        self.xaxis_pan = 0.0
        self.yaxis_pan = 0.0

    def keyReleaseEvent(self, ev):
        match ev.key():
            case QtCore.Qt.Key.Key_B:
                self.background_white = not self.background_white
                self.update()
            case QtCore.Qt.Key.Key_C:
                self.cullface_toggle = True
                self.update()
            case QtCore.Qt.Key.Key_W:
                self.wireframe_toggle = True
                self.update()
            case QtCore.Qt.Key.Key_Q:
                QtWidgets.QApplication.instance().quit()
            case QtCore.Qt.Key.Key_Z:
                self.resetView()
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

        self.visual_kind = visual.kind

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

        elif visual.kind == 'vertex':
            # this kind is actually quite rare
            byte_view = self.vertex_data.view(np.uint8).reshape((-1, 8, 4))
            byte_view[:, 6, :] = visual.vertex_colors

        else:
            # fill with white
            byte_view = self.vertex_data.view(np.uint8).reshape((-1, 8, 4))
            byte_view[:, 6, :] = (255, 255, 255, 255)
            self.visual_kind = 'vertex'

        self.need_upload = True
        self.update()

    def initialize(self, cb):
        if self.m_rhi != self.rhi():
            self.m_pipeline_texture = None
            self.m_rhi = self.rhi()

        if self.m_pipeline_texture is not None:
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

        self.m_pipeline_texture, self.m_srb_texture = self.create_pipeline('texture')
        self.m_pipeline_vertex, self.m_srb_vertex = self.create_pipeline('vertex')

    def create_pipeline(self, kind : str):
        SS = QtGui.QRhiShaderStage
        SRB = QtGui.QRhiShaderResourceBinding
        VIA = QtGui.QRhiVertexInputAttribute
        if kind == 'texture':
            shader_stages = [
                SS(SS.Vertex, self.vert_texture_shader),
                SS(SS.Fragment, self.frag_texture_shader),
            ]
            bindings = [
                SRB.uniformBuffer(0, SRB.VertexStage | SRB.FragmentStage, self.m_ubuf),
                SRB.sampledTexture(1, SRB.FragmentStage, self.m_texture, self.m_sampler),
            ]
            input_attributes = [
                VIA(0, 0, VIA.Float3, 0),
                VIA(0, 1, VIA.Float3, 3 * 4),
                VIA(0, 2, VIA.Float2, 6 * 4),
            ]
        else:
            shader_stages = [
                SS(SS.Vertex, self.vert_color_shader),
                SS(SS.Fragment, self.frag_color_shader),
            ]
            bindings = [
                SRB.uniformBuffer(0, SRB.VertexStage | SRB.FragmentStage, self.m_ubuf),
            ]
            input_attributes = [
                VIA(0, 0, VIA.Float3, 0),
                VIA(0, 1, VIA.Float3, 3 * 4),
                VIA(0, 2, VIA.UNormByte4, 6 * 4),
            ]

        srb = self.m_rhi.newShaderResourceBindings()
        SRB = QtGui.QRhiShaderResourceBinding
        srb.setBindings(bindings)
        srb.create()

        pipeline = self.m_rhi.newGraphicsPipeline()
        blend = QtGui.QRhiGraphicsPipeline.TargetBlend()
        pipeline.setTargetBlends([blend])
        pipeline.setTopology(QtGui.QRhiGraphicsPipeline.Topology.Triangles)
        pipeline.setDepthTest(True)
        pipeline.setDepthWrite(True)
        pipeline.setShaderStages(shader_stages)
        inputLayout = QtGui.QRhiVertexInputLayout()
        inputLayout.setBindings([QtGui.QRhiVertexInputBinding(8 * 4)])
        inputLayout.setAttributes(input_attributes)
        pipeline.setVertexInputLayout(inputLayout)
        pipeline.setShaderResourceBindings(srb)
        pipeline.setRenderPassDescriptor(self.renderTarget().renderPassDescriptor())
        pipeline.create()

        return pipeline, srb

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

        if self.visual_kind == 'texture':
            pipeline = self.m_pipeline_texture
        else:
            pipeline = self.m_pipeline_vertex

        pipeline_dirty = False

        if self.cullface_toggle:
            self.cullface_toggle = False
            CM = pipeline.CullMode
            old_mode = pipeline.cullMode()
            new_mode = CM.Back if old_mode == CM.None_ else CM.None_
            pipeline.setCullMode(new_mode)
            pipeline_dirty = True

        if self.wireframe_toggle:
            self.wireframe_toggle = False
            PM = pipeline.PolygonMode
            old_mode = pipeline.polygonMode()
            new_mode = PM.Line if old_mode == PM.Fill else PM.Fill
            pipeline.setPolygonMode(new_mode)
            pipeline_dirty = True

        if pipeline_dirty:
            pipeline.create()

        v = 1.0 if self.background_white else 0.0
        clearColor = QtGui.QColor.fromRgbF(v, v, v)
        ds = QtGui.QRhiDepthStencilClearValue(1.0, 0)
        cb.beginPass(self.renderTarget(), clearColor, ds, resourceUpdates)

        cb.setGraphicsPipeline(pipeline)
        cb.setViewport(QtGui.QRhiViewport(0, 0, outputSize.width(), outputSize.height()))
        cb.setShaderResources()
        IF = QtGui.QRhiCommandBuffer.IndexFormat
        ibuf_dtype = IF.IndexUInt16 if self.faces_data.dtype == np.uint16 else IF.IndexUInt32
        cb.setVertexInput(0, [(self.m_vbuf, 0)],
                          self.m_ibuf, 0, ibuf_dtype)
        cb.drawIndexed(self.faces_data.size)

        cb.endPass()
