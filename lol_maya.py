from maya.OpenMaya import *
from maya.OpenMayaAnim import *
from maya.OpenMayaMPx import *
from maya import cmds as cmds

from struct import pack, unpack
from random import choice, uniform

# plugin register


class SKNTranslator(MPxFileTranslator):
    typeName = 'League of Legends: SKN'
    extension = 'skn'

    def __init__(self):
        MPxFileTranslator.__init__(self)

    def haveReadMethod(self):
        return True

    def defaultExtension(self):
        return self.extension

    def filter(self):
        return f'*.{self.extension}'

    @classmethod
    def creator(cls):
        return asMPxPtr(cls())

    def reader(self, fileObject, optionString, accessMode):
        skn = SKN()
        skn.read(fileObject.expandedFullName())
        skn.load()


class SKLTranslator(MPxFileTranslator):
    typeName = 'League of Legends: SKL'
    extension = 'skl'

    def __init__(self):
        MPxFileTranslator.__init__(self)

    def haveReadMethod(self):
        return True

    def defaultExtension(self):
        return self.extension

    def filter(self):
        return f'*.{self.extension}'

    @classmethod
    def creator(cls):
        return asMPxPtr(cls())

    def reader(self, fileObject, optionString, accessMode):
        skl = SKL()
        skl.read(fileObject.expandedFullName())
        skl.load()


class SkinTranslator(MPxFileTranslator):
    typeName = 'League of Legends: SKN + SKL'
    extension = 'skn'

    def __init__(self):
        MPxFileTranslator.__init__(self)

    def haveWriteMethod(self):
        return True

    def defaultExtension(self):
        return self.extension

    def filter(self):
        return f'*.{self.extension}'

    @classmethod
    def creator(cls):
        return asMPxPtr(cls())

    def writer(self, fileObject, optionString, accessMode):
        if accessMode == MPxFileTranslator.kExportActiveAccessMode:
            skl = SKL()
            skn = SKN()

            # dump from scene
            skl.dump()
            skn.dump(skl)

            skl.flip()
            skn.flip()

            path = fileObject.rawFullName()
            if not path.endswith('.skn'):
                path += '.skn'
            skl.write(path.split('.skn')[0] + '.skl')
            skn.write(path)
        else:
            error('stop! u violated the law, use "export selection" or i violate u')


def initializePlugin(obj):
    # totally not copied code
    plugin = MFnPlugin(obj)
    try:
        plugin.registerFileTranslator(
            SKNTranslator.typeName,
            None,
            SKNTranslator.creator,
            None,
            None,
            True
        )
    except Exception as e:
        cmds.warning(
            f'Can\'t register plug-in node {SKNTranslator.typeName}: {e}')

    try:
        plugin.registerFileTranslator(
            SKLTranslator.typeName,
            None,
            SKLTranslator.creator,
            None,
            None,
            True
        )
    except Exception as e:
        cmds.warning(
            f'Can\'t register plug-in node {SKLTranslator.typeName}: {e}')

    try:
        plugin.registerFileTranslator(
            SkinTranslator.typeName,
            None,
            SkinTranslator.creator,
            None,
            None,
            True
        )
    except Exception as e:
        cmds.warning(
            f'Can\'t register plug-in node {SkinTranslator.typeName}: {e}')


def uninitializePlugin(obj):
    plugin = MFnPlugin(obj)
    try:
        plugin.deregisterFileTranslator(
            SKNTranslator.typeName
        )
    except Exception as e:
        cmds.warning(
            f'Can\'t deregister plug-in node {SKNTranslator.typeName}: {e}')

    try:
        plugin.deregisterFileTranslator(
            SKLTranslator.typeName
        )
    except Exception as e:
        cmds.warning(
            f'Can\'t deregister plug-in node {SKLTranslator.typeName}: {e}')

    try:
        plugin.deregisterFileTranslator(
            SkinTranslator.typeName
        )
    except Exception as e:
        cmds.warning(
            f'Can\'t deregister plug-in node {SkinTranslator.typeName}: {e}')


class BinaryStream:
    def __init__(self, f):
        self.stream = f

    def read_byte(self):
        return self.stream.read(1)

    def read_bytes(self, length):
        return self.stream.read(length)

    def read_int16(self):
        return unpack('h', self.stream.read(2))[0]

    def read_uint16(self):
        return unpack('H', self.stream.read(2))[0]

    def read_int32(self):
        return unpack('i', self.stream.read(4))[0]

    def read_uint32(self):
        return unpack('I', self.stream.read(4))[0]

    def read_float(self):
        return unpack('f', self.stream.read(4))[0]

    def read_char(self):
        return unpack('b', self.stream.read(1))[0]

    def read_zero_terminated_string(self):
        res = ''
        while True:
            c = self.read_char()
            if c == 0:
                break
            res += chr(c)
        return res

    def read_padded_string(self, length):
        return bytes(filter(lambda b: b != 0, self.stream.read(length))).decode('ascii')

    def read_vec2(self):
        return MVector(self.read_float(), self.read_float())

    def read_vec3(self):
        return MVector(self.read_float(), self.read_float(), self.read_float())

    def read_quat(self):
        return MQuaternion(self.read_float(), self.read_float(), self.read_float(), self.read_float())

    def write_bytes(self, value):
        self.stream.write(value)

    def write_int16(self, value):
        self.write_bytes(pack('h', value))

    def write_uint16(self, value):
        self.write_bytes(pack('H', value))

    def write_int32(self, value):
        self.write_bytes(pack('i', value))

    def write_uint32(self, value):
        self.write_bytes(pack('I', value))

    def write_float(self, value):
        self.write_bytes(pack('f', value))

    def write_padded_string(self, length, value):
        while len(value) < length:
            value += '\u0000'
        self.write_bytes(value.encode('ascii'))

    def write_vec2(self, vec2):
        self.write_float(vec2.x)
        self.write_float(vec2.y)

    def write_vec3(self, vec3):
        self.write_float(vec3.x)
        self.write_float(vec3.y)
        self.write_float(vec3.z)

    def write_quat(self, quat):
        self.write_float(quat.x)
        self.write_float(quat.y)
        self.write_float(quat.z)
        self.write_float(quat.w)


def error(message):
    cmds.confirmDialog(
        message=message,
        title='haha random RGB error message go skrt skrttttt~~',
        backgroundColor=[uniform(0.0, 1.0), uniform(
            0.0, 1.0), uniform(0.0, 1.0)],
        button=choice(
            ['UwU', '<(")', 'ok boomer', 'funny man', 'jesus', 'bruh',
             'stop', 'get some help', 'haha', 'lmao', 'ay yo', 'SUS']
        )
    )
    return -1


class SKLJoint:
    def __init__(self):
        self.name = None
        self.parent = None  # just id, not actual parent, especially not asian parent
        # self.local_transform = None # for update skl
        self.global_transform = None
        self.radius = 0.1  # or 2.1?

    def write(self, bs):
        bs.write_padded_string(32, self.name)
        bs.write_int32(self.parent)  # -1, cant be uint
        bs.write_float(self.radius)
        for i in range(0, 3):
            for j in range(0, 4):
                bs.write_float(self.global_transform.asMatrix()(j, i))


class SKL:
    def __init__(self):
        self.joints = []
        self.dag_paths = []
        #self.influences = []

        # for dumping
        self.dag_paths = None

    def dump(self):
        self.dag_paths = MDagPathArray()
        dag_path = MDagPath()
        ik_joint = MFnIkJoint()

        iterator = MItDag(MItDag.kDepthFirst, MFn.kJoint)
        iterator.reset()
        while not iterator.isDone():
            iterator.getPath(dag_path)
            self.dag_paths.append(dag_path)  # identify joint by DAG path
            ik_joint.setObject(dag_path)  # to get the joint transform

            rotation = MQuaternion()
            ik_joint.getRotation(rotation, MSpace.kWorld)
            rotation = ik_joint.rotateOrientation(MSpace.kTransform) * rotation
            translation = ik_joint.getTranslation(MSpace.kWorld)
            transform = MTransformationMatrix()
            transform.setRotationQuaternion(
                rotation.x, rotation.y, rotation.z, rotation.w, MSpace.kWorld)
            transform.setTranslation(translation, MSpace.kWorld)
            # am i high or those code just create a same transform matrix of ik_joint transform matrix
            joint = SKLJoint()
            joint.name = ik_joint.name()
            joint.global_transform = transform  # local for update skl
            self.joints.append(joint)

            iterator.next()

        parent_joint = MFnIkJoint()
        parent_dag_path = MDagPath()
        len1 = len(self.joints)
        len2 = self.dag_paths.length()
        for i in range(0, len1):
            ik_joint.setObject(self.dag_paths[i])
            if ik_joint.parentCount() == 1 and ik_joint.parent(0).apiType() == MFn.kJoint:
                # if ik_joint has parent (same as if joint has parent)

                # get ik_parent through ik_joint instance of current joint
                parent_joint.setObject(ik_joint.parent(0))
                parent_joint.getPath(parent_dag_path)

                # find parent of current joint in joints list by ik_parent
                for id in range(0, len2):
                    if self.dag_paths[id] == parent_dag_path:
                        self.joints[i].parent = id
                        break
            else:
                # must be batman
                self.joints[i].parent = -1

        print('dumped skl')

    def flip(self):
        # flip the L with R: #https://youtu.be/2yzMUs3badc
        for joint in self.joints:
            transform = joint.global_transform
            rotation = transform.rotation()
            transform.setRotationQuaternion(
                rotation.x, -rotation.y, -rotation.z, rotation. w)
            matrix = transform.asMatrix()
            # i cant replace matrix[3][0] with -matrix[3][0] because of this cursed api
            py_list = []
            for i in range(0, 4):
                for j in range(0, 4):
                    py_list.append(matrix(i, j))
            py_list[12] = -py_list[12]
            matrix = MMatrix()
            # create the matrix bacc from the list
            MScriptUtil.createMatrixFromList(py_list, matrix)
            joint.global_transform = MTransformationMatrix(matrix)

    def write(self, path):
        with open(path, 'wb') as f:
            bs = BinaryStream(f)

            bs.write_bytes('r3d2sklt'.encode('ascii'))  # magic
            bs.write_uint32(1)  # version
            bs.write_uint32(0)  # padding

            bs.write_uint32(len(self.joints))
            for joint in self.joints:
                joint.write(bs)


class SKNVertex:
    def __init__(self):
        self.position = None
        self.bones_indices = None
        self.weights = None
        self.normal = None
        self.uv = None

        # for dumping
        self.uv_index = None
        self.data_index = None

    def write(self, bs):
        bs.write_vec3(self.position)
        for byte in self.bones_indices:
            bs.write_bytes(byte)
        for weight in self.weights:
            bs.write_float(weight)
        bs.write_vec3(self.normal)
        bs.write_vec2(self.uv)


class SKNSubmesh:
    def __init__(self):
        self.name = None
        self.vertex_start = None
        self.vertex_count = None
        self.index_start = None
        self.index_count = None

    def write(self, bs):
        bs.write_padded_string(64, self.name)
        bs.write_uint32(self.vertex_start)
        bs.write_uint32(self.vertex_count)
        bs.write_uint32(self.index_start)
        bs.write_uint32(self.index_count)


class SKN:
    def __init__(self):
        self.indices = []
        self.vertices = []
        self.submeshes = []

    def dump(self, skl):
        # get mesh in selections
        selections = MSelectionList()
        MGlobal.getActiveSelectionList(selections)
        iterator = MItSelectionList(selections, MFn.kMesh)
        iterator.reset()
        if iterator.isDone():
            return error('no mesh selected, please select a mesh')
        mesh_dag_path = MDagPath()
        iterator.getDagPath(mesh_dag_path)  # get first mesh
        iterator.next()
        if not iterator.isDone():
            return error('more than 1 meshes selected\nin case u don\'t know: u must combine all mesh into 1')
        mesh = MFnMesh(mesh_dag_path)

        # find skin cluster
        in_mesh = mesh.findPlug("inMesh")
        in_mesh_connections = MPlugArray()
        in_mesh.connectedTo(in_mesh_connections, True, False)
        if in_mesh_connections.length() == 0:
            return error('failed to find skin cluster, make sure u binded the skin')
        output_geometry = in_mesh_connections[0]
        skin_cluster = MFnSkinCluster(output_geometry.node())
        influence_dag_paths = MDagPathArray()
        influence_count = skin_cluster.influenceObjects(influence_dag_paths)

        # idk what this is, used for SKNVertex.bones_indices
        mask_influence = MIntArray(influence_count)
        for i in range(0, influence_count):
            for j in range(0, skl.dag_paths.length()):
                if influence_dag_paths[i] == skl.dag_paths[j]:
                    mask_influence[i] = j
                    break

        # get shaders
        shaders = MObjectArray()
        poly_shaders = MIntArray()
        instance_num = mesh_dag_path.instanceNumber() if mesh_dag_path.isInstanced() else 0
        mesh.getConnectedShaders(instance_num, shaders, poly_shaders)
        shader_count = shaders.length()
        vertices_num = mesh.numVertices()

        # check some holes
        hole_info = MIntArray()
        hole_vertex = MIntArray()
        mesh.getHoles(hole_info, hole_vertex)
        if hole_info.length() != 0:
            return error('mesh contains holes, idk how to fix just get rid of the holes')

        # check non-triangulated polygons
        # check polygon has multiple shaders
        vertex_shaders = MIntArray(vertices_num, -1)
        iterator = MItMeshPolygon(mesh_dag_path)
        iterator.reset()
        while not iterator.isDone():
            if not iterator.hasValidTriangulation():
                return error(
                    'mesh contains a non-triangulated polygon, use Mesh -> Triangualate maybe?')

            index = iterator.index()
            shader_index = poly_shaders[index]
            if shader_index == -1:
                return error('mesh contains a face with no shader, no idea how')

            vertices = MIntArray()
            iterator.getVertices(vertices)
            len69 = vertices.length()
            for i in range(0, len69):
                if shader_count > 1 and vertex_shaders[vertices[i]] not in [-1, shader_index]:
                    return error('mesh contains a vertex with multiple shaders')
                vertex_shaders[vertices[i]] = shader_index

            iterator.next()

        # get weights
        vcomponent = MFnSingleIndexedComponent()
        vertex_component = vcomponent.create(MFn.kMeshVertComponent)
        temp_list = list(range(0, vertices_num))
        group_vertex_indices = MIntArray()
        MScriptUtil.createIntArrayFromList(temp_list, group_vertex_indices)
        vcomponent.addElements(group_vertex_indices)
        weights = MDoubleArray()
        util = MScriptUtil()  # cursed api
        ptr = util.asUintPtr()
        skin_cluster.getWeights(mesh_dag_path, vertex_component, weights, ptr)
        # is this different from influence_count?
        weight_influence_count = util.getUint(ptr)
        #  weight stuffs
        for i in range(0, vertices_num):
            # if vertices don't have more than 4 influences
            temp_count = 0
            weight_sum = 0.0
            for j in range(0, weight_influence_count):
                weight = weights[i * weight_influence_count + j]
                if weight != 0:
                    temp_count += 1
                    weight_sum += weight
            if temp_count > 4:
                return error('ay yo, the 4 influences incident, prune weight 0.05 or set max influences to 4 idk')

            # normalize weights
            for j in range(0, weight_influence_count):
                weights[i * weight_influence_count + j] /= weight_sum

        # init some important thing
        shader_vertex_indices = []
        shader_vertices = []
        shader_indices = []
        for i in range(0, shader_count):
            shader_vertex_indices.append(MIntArray())
            shader_vertices.append([])
            shader_indices.append(MIntArray())

        # SKNVertex
        iterator = MItMeshVertex(mesh_dag_path)
        iterator.reset()
        while not iterator.isDone():
            index = iterator.index()
            shader = vertex_shaders[index]
            if shader == -1:
                cmds.warning('mesh contains a vertex with no shader')
                continue

            # position
            temp_position = iterator.position(MSpace.kWorld)

            # bones_indcies and weights
            temp_bones_indices = [bytes([0]), bytes(
                [0]), bytes([0]), bytes([0])]
            temp_weights = [0.0, 0.0, 0.0, 0.0]
            f = 0
            j = 0
            while j < weight_influence_count and f < 4:
                weight = weights[index * weight_influence_count + j]
                if weight != 0:
                    temp_bones_indices[f] = bytes([mask_influence[j]])
                    temp_weights[f] = float(weight)
                    f += 1
                j += 1

            # normal - also normalized
            normals = MVectorArray()
            iterator.getNormals(normals)
            temp_normal = MVector(0.0, 0.0, 0.0)
            len123 = normals.length()
            for i in range(0, len123):
                temp_normal.x += normals[i].x
                temp_normal.y += normals[i].y
                temp_normal.z += normals[i].z
            temp_normal.x /= len123
            temp_normal.y /= len123
            temp_normal.z /= len123

            # unique uv
            uv_indices = MIntArray()
            iterator.getUVIndices(uv_indices)
            len555 = uv_indices.length()
            if len555 > 0:
                seen = []
                for j in range(0, len555):
                    uv_index = uv_indices[j]
                    temp_uv_index = uv_index
                    if not uv_index in seen:
                        u_util = MScriptUtil()  # lay trua tren cao, turn down for this
                        u_ptr = u_util.asFloatPtr()
                        v_util = MScriptUtil()
                        v_ptr = v_util.asFloatPtr()
                        mesh.getUV(uv_index, u_ptr, v_ptr)
                        temp_uv = MVector(
                            u_util.getFloat(u_ptr),
                            1.0 - v_util.getFloat(v_ptr)
                        )

                        # create SKNVertex
                        vertex = SKNVertex()
                        vertex.position = temp_position
                        vertex.bones_indices = temp_bones_indices
                        vertex.weights = temp_weights
                        vertex.normal = temp_normal
                        vertex.uv = temp_uv
                        vertex.uv_index = temp_uv_index

                        shader_vertices[shader].append(vertex)
                        shader_vertex_indices[shader].append(index)

                        seen.append(uv_index)
            else:
                return error('mesh contains a vertex with no UVs')
            iterator.next()

        # idk what this block does, must be not important
        current_index = 0
        data_indices = MIntArray(vertices_num, -1)
        for i in range(0, shader_count):
            len101 = shader_vertex_indices[i].length()
            for j in range(0, len101):
                index = shader_vertex_indices[i][j]
                if data_indices[index] == -1:
                    data_indices[index] = current_index
                    shader_vertices[i][j].data_index = current_index
                else:
                    shader_vertices[i][j].data_index = data_indices[index]
                current_index += 1
            # total vertices
            self.vertices += shader_vertices[i]

        # i only get that we got shader_indices after this block
        iterator = MItMeshPolygon(mesh_dag_path)
        iterator.reset()
        while not iterator.isDone():
            index = iterator.index()
            shader_index = poly_shaders[index]

            indices = MIntArray()
            points = MPointArray()
            iterator.getTriangles(points, indices)
            lenDOGE = indices.length()
            if iterator.hasUVs():
                new_indices = MIntArray(lenDOGE, -1)
                vertices = MIntArray()
                iterator.getVertices(vertices)
                len666 = vertices.length()
                for i in range(0, len666):
                    util = MScriptUtil()  # god pls
                    ptr = util.asIntPtr()
                    iterator.getUVIndex(i, ptr)
                    uv_index = util.getInt(ptr)

                    data_index = data_indices[vertices[i]]
                    if data_index == -1 or data_index >= len(self.vertices):
                        return error('data index out of range')

                    for j in range(data_index, len(self.vertices)):
                        if self.vertices[j].data_index != data_index:
                            return error('can\'t find corresponding face vertex in data')
                        elif self.vertices[j].uv_index == uv_index:
                            for k in range(0, lenDOGE):
                                if indices[k] == vertices[i]:
                                    new_indices[k] = j
                            break

                lenMEOW = new_indices.length()
                for i in range(0, lenMEOW):
                    shader_indices[shader_index].append(new_indices[i])
            else:
                for i in range(0, lenDOGE):
                    shader_indices[shader_index].append(
                        data_indices[indices[i]])

            iterator.next()

        # SKNSubmesh
        index_start = 0
        vertex_start = 0
        for i in range(0, shader_count):
            surface_shaders = MFnDependencyNode(
                shaders[i]).findPlug("surfaceShader")
            plug_array = MPlugArray()
            surface_shaders.connectedTo(plug_array, True, False)
            surface_shader = MFnDependencyNode(plug_array[0].node())

            index_count = shader_indices[i].length()
            vertex_count = shader_vertex_indices[i].length()

            # total indices
            for j in range(0, index_count):
                self.indices.append(shader_indices[i][j])

            submesh = SKNSubmesh()
            submesh.name = surface_shader.name()
            submesh.vertex_start = vertex_start
            submesh.vertex_count = vertex_count
            submesh.index_start = index_start
            submesh.index_count = index_count
            self.submeshes.append(submesh)

            index_start += index_count
            vertex_start += vertex_count

        print('dumped skn')
        # ay yo finally

    def flip(self):
        # read SKL.flip()
        for vertex in self.vertices:
            vertex.position.x = -vertex.position.x
            vertex.normal.y = -vertex.normal.y
            vertex.normal.z = -vertex.normal.z

    def write(self, path):
        with open(path, 'wb') as f:
            bs = BinaryStream(f)

            bs.write_uint32(0x00112233)  # magic
            bs.write_uint16(1)  # major
            bs.write_uint16(1)  # minor

            bs.write_uint32(len(self.submeshes))
            for submesh in self.submeshes:
                submesh.write(bs)

            bs.write_uint32(len(self.indices))
            bs.write_uint32(len(self.vertices))

            for index in self.indices:
                bs.write_uint16(index)
            for vertex in self.vertices:
                vertex.write(bs)
