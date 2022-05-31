from random import choice, uniform
from struct import pack, unpack

from maya import cmds as cmds
from maya.OpenMayaMPx import *
from maya.OpenMayaAnim import *
from maya.OpenMaya import *

# plugin register


class SKNTranslator(MPxFileTranslator):
    typeName = 'League of Legends: SKN'
    extension = 'skn'

    def __init__(self):
        MPxFileTranslator.__init__(self)

    def haveReadMethod(self):
        return True

    def canBeOpened(self):
        return False

    def defaultExtension(self):
        return self.extension

    def filter(self):
        return f'*.{self.extension}'

    @classmethod
    def creator(cls):
        return asMPxPtr(cls())

    def reader(self, fileObject, optionString, accessMode):
        skn = SKN()
        path = fileObject.expandedFullName()
        if not path.endswith('.skn'):
            path += '.skn'

        skn.read(path)
        name = path.split('/')[-1].split('.')[0]
        if optionString.split('=')[1] == '1':
            skl = SKL()
            skl.read(path.split('.skn')[0] + '.skl')

            skl.flip()
            skn.flip()

            skl.load()
            skn.load(name=name, skl=skl)
        else:
            skn.flip()
            skn.load(name=name, skl=None)
        return True


class SKLTranslator(MPxFileTranslator):
    typeName = 'League of Legends: SKL'
    extension = 'skl'

    def __init__(self):
        MPxFileTranslator.__init__(self)

    def haveReadMethod(self):
        return True

    def canBeOpened(self):
        return False

    def defaultExtension(self):
        return self.extension

    def filter(self):
        return f'*.{self.extension}'

    @classmethod
    def creator(cls):
        return asMPxPtr(cls())

    def reader(self, fileObject, optionString, accessMode):
        skl = SKL()
        path = fileObject.expandedFullName()
        if not path.endswith('.skl'):
            path += '.skl'
        skl.read(path)
        skl.flip()
        skl.load()
        return True


class SkinTranslator(MPxFileTranslator):
    typeName = 'League of Legends: SKN + SKL'
    extension = 'skn'

    def __init__(self):
        MPxFileTranslator.__init__(self)

    def haveWriteMethod(self):
        return True

    def canBeOpened(self):
        return False

    def defaultExtension(self):
        return self.extension

    def filter(self):
        return f'*.{self.extension}'

    @classmethod
    def creator(cls):
        return asMPxPtr(cls())

    def writer(self, fileObject, optionString, accessMode):
        if accessMode != MPxFileTranslator.kExportActiveAccessMode:
            raise RuntimeError(error(
                f'[SkinTranslator.writer()] Stop! u violated the law, use "Export Selection" or i violate u UwU.'))

        skl = SKL()
        skn = SKN()

        # dump from scene
        skl.dump()
        skn.dump(skl)
        # ay yo, do a flip!
        skl.flip()
        skn.flip()

        path = fileObject.rawFullName()
        # fix for file with mutiple '.', this api is just meh
        if not path.endswith('.skn'):
            path += '.skn'
        skl.write(path.split('.skn')[0] + '.skl')
        skn.write(path)
        return True


def initializePlugin(obj):
    # totally not copied code
    plugin = MFnPlugin(obj, 'tarngaina', '1.0')
    try:
        plugin.registerFileTranslator(
            SKNTranslator.typeName,
            None,
            SKNTranslator.creator,
            "SKNTranslatorOpts",
            "",  # idk wtf wrong with defaul options
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


# helper funcs and structures
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
        title='Error',
        backgroundColor=[uniform(0.0, 1.0), uniform(
            0.0, 1.0), uniform(0.0, 1.0)],
        button=choice(
            ['UwU', '<(")', 'ok boomer', 'funny man', 'jesus', 'bruh',
             'stop', 'get some help', 'haha', 'lmao', 'ay yo', 'SUS']
        )
    )
    return message


# skl


class SKLJoint:
    def __init__(self):
        self.name = None
        self.parent = None  # just id, not actual parent, especially not asian parent
        # fuck transform matrix
        self.local_translation = None
        self.local_scale = None
        self.local_rotation = None
        # yeah its actually inversed global, not global
        self.iglobal_translation = None
        self.iglobal_scale = None
        self.iglobal_rotation = None


class SKL:
    def decompose(transform, space):
        # get translation, scale and rotation (quaternion) out of transformation matrix

        # get scale by cursed api
        util = MScriptUtil()
        util.createFromDouble(0.0, 0.0, 0.0)
        ptr = util.asDoublePtr()
        transform.getScale(ptr, space)

        # get roration in quaternion by cursed api
        util_x = MScriptUtil()
        ptr_x = util_x.asDoublePtr()
        util_y = MScriptUtil()
        ptr_y = util_y.asDoublePtr()
        util_z = MScriptUtil()
        ptr_z = util_z.asDoublePtr()
        util_w = MScriptUtil()
        ptr_w = util_w.asDoublePtr()
        transform.getRotationQuaternion(ptr_x, ptr_y, ptr_z, ptr_w, space)

        # (translation, scale, rotation)
        return (
            transform.getTranslation(space),
            MVector(
                util.getDoubleArrayItem(ptr, 0),
                util.getDoubleArrayItem(ptr, 1),
                util.getDoubleArrayItem(ptr, 2)
            ),
            MQuaternion(
                util_x.getDouble(ptr_x),
                util_y.getDouble(ptr_y),
                util_z.getDouble(ptr_z),
                util_w.getDouble(ptr_w)
            )
        )

    def compose(translation, scale, rotation, space):
        # set translation, scale and rotation (quaternion) on a transformation matrix

        transform = MTransformationMatrix()

        # translation
        transform.setTranslation(
            MVector(translation.x, translation.y, translation.z), space)

        # cursed scale
        util = MScriptUtil()
        util.createFromDouble(scale.x, scale.y, scale.z)
        ptr = util.asDoublePtr()
        transform.setScale(ptr, space)

        # easy rotation (quaternion)
        transform.setRotationQuaternion(
            rotation.x, rotation.y, rotation.z, rotation.w, space)
        return transform

    def __init__(self):
        self.joints = []
        self.dag_paths = []

        # for dumping
        self.dag_paths = None

        # for loading
        self.legacy = None
        self.influences = []  # for load both skn + skl as skincluster

    def read(self, path):
        with open(path, 'rb') as f:
            bs = BinaryStream(f)

            bs.read_uint32()  # file size - pad
            magic = bs.read_uint32()
            if magic == 0x22FD4FC3:
                # new skl data
                self.legacy = False

                version = bs.read_uint32()
                if version != 0:
                    raise RuntimeError(
                        error(f'[SKL.read({path})] Unsupported file version: {version}.'))

                bs.read_uint16()  # flags - pad
                joint_count = bs.read_uint16()
                influences_count = bs.read_uint32()

                joints_offset = bs.read_uint32()
                joint_indices_offset = bs.read_uint32()
                influences_offset = bs.read_uint32()
                bs.read_uint32()  # name offset - pad
                bs.read_uint32()  # asset name offset - pad
                joint_names_offset = bs.read_uint32()

                bs.read_uint32()  # reserved offset - pad
                bs.read_uint32()
                bs.read_uint32()
                bs.read_uint32()
                bs.read_uint32()

                # read joints
                if joints_offset > 0 and joint_count > 0:
                    bs.stream.seek(joints_offset)
                    for i in range(0, joint_count):
                        joint = SKLJoint()
                        # read each joint
                        bs.read_uint16()  # flags - pad
                        bs.read_uint16()  # id - pad
                        joint.parent = bs.read_int16()  # cant be uint
                        bs.read_uint16()  # pad
                        bs.read_uint32()  # joint name's hash - pad
                        bs.read_float()  # radius/scale - pad
                        # local
                        joint.local_translation = bs.read_vec3()
                        joint.local_scale = bs.read_vec3()
                        joint.local_rotation = bs.read_quat()
                        # inversed global
                        joint.iglobal_translation = bs.read_vec3()
                        joint.iglobal_scale = bs.read_vec3()
                        joint.iglobal_rotation = bs.read_quat()
                        # name
                        joint_name_offset = bs.read_uint32()  # joint name offset - pad
                        return_offset = bs.stream.tell()
                        bs.stream.seek(return_offset - 4 + joint_name_offset)
                        joint.name = bs.read_zero_terminated_string()
                        bs.stream.seek(return_offset)

                        self.joints.append(joint)

                # read influences
                if influences_offset > 0 and influences_count > 0:
                    bs.stream.seek(influences_offset)
                    for i in range(0, influences_count):
                        self.influences.append(bs.read_uint16())

                # i think that is all we need, reading joint_indices_offset, name and asset name doesnt help anything

            else:
                # legacy
                self.legacy = True

                # because signature in old skl is first 8bytes
                # need to go back pos 0 to read 8 bytes again
                bs.stream.seek(0)
                magic = bs.read_bytes(8).decode('ascii')
                if magic != 'r3d2sklt':
                    raise RuntimeError(
                        error(f'[SKL.read({path})] Wrong file signature: {magic}.'))

                version = bs.read_uint32()
                if version not in [1, 2]:
                    raise RuntimeError(
                        error(f'[SKL.read({path})] Unsupported file version: {version}.'))

                bs.read_uint32()  # designer id or skl id - pad this

                # read joints
                joint_count = bs.read_uint32()
                for i in range(0, joint_count):
                    joint = SKLJoint()
                    # dont need id
                    joint.name = bs.read_padded_string(32)
                    joint.parent = bs.read_int32()  # -1, cant be uint
                    bs.read_float()  # radius/scale - pad
                    py_list = list([0.0] * 16)
                    for i in range(0, 3):
                        for j in range(0, 4):
                            py_list[j*4+i] = bs.read_float()
                    py_list[15] = 1.0
                    matrix = MMatrix()
                    MScriptUtil.createMatrixFromList(py_list, matrix)
                    joint.iglobal_translation, joint.iglobal_scale, joint.iglobal_rotation = SKL.decompose(
                        MTransformationMatrix(matrix),
                        MSpace.kTransform
                    )
                    self.joints.append(joint)

                # read influences
                if version == 2:
                    influences_count = bs.read_uint32()
                    for i in range(0, influences_count):
                        self.influences.append(bs.read_uint32())
                if version == 1:
                    self.influences = list(range(0, len(self.joints)))

    def load(self):
        self.dag_paths = MDagPathArray()
        dag_path = MDagPath()
        for joint in self.joints:
            # create ik joint
            ik_joint = MFnIkJoint()
            ik_joint.create()

            # get dag path
            ik_joint.getPath(dag_path)
            self.dag_paths.append(dag_path)

            # name
            ik_joint.setName(joint.name)

            # transform
            if self.legacy:
                ik_joint.set(SKL.compose(
                    joint.iglobal_translation, joint.iglobal_scale, joint.iglobal_rotation, MSpace.kTransform))
            else:
                ik_joint.set(SKL.compose(
                    joint.local_translation, joint.local_scale, joint.local_rotation, MSpace.kWorld
                ))

        # link parent
        for i in range(0, len(self.joints)):
            joint = self.joints[i]
            if joint.parent != -1:  # skip our ancestor
                ik_parent = MFnIkJoint(self.dag_paths[joint.parent])
                ik_joint = MFnIkJoint(self.dag_paths[i])

                ik_parent.addChild(ik_joint.object())

                if self.legacy:
                    # probably local transform thing here
                    translation = ik_joint.getTranslation(MSpace.kTransform)
                    rotation = MQuaternion()
                    ik_joint.getRotation(rotation, MSpace.kWorld)

                    ik_joint.setTranslation(translation, MSpace.kWorld)
                    ik_joint.setRotation(rotation, MSpace.kWorld)

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

            joint = SKLJoint()
            joint.name = ik_joint.name()
            # mama mia
            joint.local_translation, joint.local_scale, joint.local_rotation = SKL.decompose(
                MTransformationMatrix(ik_joint.transformationMatrix()),
                MSpace.kTransform
            )
            joint.iglobal_translation, joint.iglobal_scale, joint.iglobal_rotation = SKL.decompose(
                MTransformationMatrix(dag_path.inclusiveMatrixInverse()),
                MSpace.kWorld
            )
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

    def flip(self):
        # flip the L with R: https://youtu.be/2yzMUs3badc
        for joint in self.joints:
            # local
            if joint.local_translation:  # check when reading, legacy doesnt have local
                joint.local_translation.x = -joint.local_translation.x
                joint.local_rotation.y = -joint.local_rotation.y
                joint.local_rotation.z = -joint.local_rotation.z
            # inversed global
            joint.iglobal_translation.x = -joint.iglobal_translation.x
            joint.iglobal_rotation.y = -joint.iglobal_rotation.y
            joint.iglobal_rotation.z = -joint.iglobal_rotation.z

    def write(self, path):
        # ay yo check out this elf: https://i.imgur.com/Cvl8PFu.png
        def elf_hash(s):
            s = s.lower()
            h = 0
            for c in s:
                h = (h << 4) + ord(c)
                t = (h & 0xF0000000)
                if t != 0:
                    h ^= (t >> 24)
                h &= ~t
            return h

        with open(path, 'wb') as f:
            bs = BinaryStream(f)

            bs.write_uint32(0)
            bs.write_uint32(0x22FD4FC3)  # magic
            bs.write_uint32(0)  # version

            bs.write_uint16(0)  # flags
            bs.write_uint16(len(self.joints))
            bs.write_uint32(len(self.joints))  # influences

            joints_offset = 64
            joint_indices_offset = joints_offset + len(self.joints) * 100
            influences_offset = joint_indices_offset + len(self.joints) * 8
            joint_names_offset = influences_offset + len(self.joints) * 2

            bs.write_uint32(joints_offset)
            bs.write_uint32(joint_indices_offset)
            bs.write_uint32(influences_offset)
            bs.write_uint32(0)  # name
            bs.write_uint32(0)  # asset name
            bs.write_uint32(joint_names_offset)

            bs.write_uint32(0xFFFFFFFF)  # reserved offset field
            bs.write_uint32(0xFFFFFFFF)
            bs.write_uint32(0xFFFFFFFF)
            bs.write_uint32(0xFFFFFFFF)
            bs.write_uint32(0xFFFFFFFF)

            joint_offset = {}
            bs.stream.seek(joint_names_offset)
            for i in range(0, len(self.joints)):
                joint_offset[i] = bs.stream.tell()
                bs.write_bytes(self.joints[i].name.encode('ascii'))
                bs.write_bytes(bytes([0]))  # pad

            bs.stream.seek(joints_offset)
            for i in range(0, len(self.joints)):
                joint = self.joints[i]
                # write skljoint in this func
                bs.write_uint16(0)  # flags
                bs.write_uint16(i)  # id
                bs.write_int16(joint.parent)  # -1, cant be uint
                bs.write_uint16(0)  # pad
                bs.write_uint32(elf_hash(joint.name))
                bs.write_float(2.1)  # scale
                # local
                bs.write_vec3(joint.local_translation)
                bs.write_vec3(joint.local_scale)
                bs.write_quat(joint.local_rotation)
                # inversed global
                bs.write_vec3(joint.iglobal_translation)
                bs.write_vec3(joint.iglobal_scale)
                bs.write_quat(joint.iglobal_rotation)

                bs.write_uint32(joint_offset[i] - bs.stream.tell())

            bs.stream.seek(influences_offset)
            for i in range(0, len(self.joints)):
                bs.write_uint16(i)

            bs.stream.seek(joint_indices_offset)
            for i in range(0, len(self.joints)):
                bs.write_uint32(elf_hash(joint.name))
                bs.write_uint16(0)  # pad
                bs.write_uint16(i)

            bs.stream.seek(0, 2)
            fsize = bs.stream.tell()
            bs.stream.seek(0)
            bs.write_uint32(fsize)


# skn


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


class SKNSubmesh:
    def __init__(self):
        self.name = None
        self.vertex_start = None
        self.vertex_count = None
        self.index_start = None
        self.index_count = None


class SKN:
    def __init__(self):
        self.indices = []
        self.vertices = []
        self.submeshes = []

    def read(self, path):
        with open(path, 'rb') as f:
            bs = BinaryStream(f)

            magic = bs.read_uint32()
            if magic != 0x00112233:
                raise RuntimeError(
                    error(f'[SKN.read({path})]: Wrong signature file: {magic}.'))

            major = bs.read_uint16()
            minor = bs.read_uint16()
            if major not in [0, 2, 4] and minor != 1:
                raise RuntimeError(
                    error(f'[SKN.read({path})]: Unsupported file version: {major}.{minor}.'))

            if major == 0:
                # version 0 doesn't have submesh wrote in file
                index_count = bs.read_uint32()
                vertex_count = bs.read_uint32()
            else:
                # read submeshes
                submesh_count = bs.read_uint32()
                for i in range(0, submesh_count):
                    submesh = SKNSubmesh()
                    submesh.name = bs.read_padded_string(64)
                    submesh.vertex_start = bs.read_uint32()
                    submesh.vertex_count = bs.read_uint32()
                    submesh.index_start = bs.read_uint32()
                    submesh.index_count = bs.read_uint32()
                    self.submeshes.append(submesh)

                #  junk stuff from version 4
                if major == 4:
                    bs.read_uint32()  # pad flags

                index_count = bs.read_uint32()
                vertex_count = bs.read_uint32()

                # junk stuff from version 4
                vertex_type = 0
                if major == 4:  # pad all this, cause we dont need?
                    bs.read_uint32()  # vertex size
                    vertex_type = bs.read_uint32()  # vertex type, only need this for padding later
                    # bouding box
                    bs.read_vec3()
                    bs.read_vec3()
                    # bouding sphere
                    bs.read_vec3()
                    bs.read_float()

                # read indices
                for i in range(0, index_count):
                    self.indices.append(bs.read_uint16())

                # read vertices
                for i in range(0, vertex_count):
                    vertex = SKNVertex()
                    vertex.position = bs.read_vec3()
                    vertex.bones_indices = [
                        bs.read_byte(), bs.read_byte(), bs.read_byte(), bs.read_byte()]
                    vertex.weights = [
                        bs.read_float(), bs.read_float(), bs.read_float(), bs.read_float()]
                    vertex.normal = bs.read_vec3()
                    vertex.uv = bs.read_vec2()
                    # if vertex has color
                    if vertex_type == 1:
                        # pad all color (4 byte = r,g,b,a) since we dont need it
                        bs.read_byte()
                        bs.read_byte()
                        bs.read_byte()
                        bs.read_byte()
                    self.vertices.append(vertex)

                if major == 0:
                    # again version 1 doesnt have submesh wrote in file
                    # so we need to give it a submesh
                    submesh = SKNSubmesh
                    submesh.name = 'Base'
                    submesh.vertex_start = 0
                    submesh.vertex_count = len(self.vertices)
                    submesh.index_start = 0
                    submesh.index_count = len(self.indices)
                    self.submeshes.append(submesh)

    def load(self, name='noname', skl=None):
        mesh = MFnMesh()
        vertices_count = len(self.vertices)
        indices_count = len(self.indices)

        # create mesh with vertices, indices
        vertices = MFloatPointArray()
        for i in range(0, vertices_count):
            vertex = self.vertices[i]
            vertices.append(MFloatPoint(
                vertex.position.x, vertex.position.y, vertex.position.z))
        poly_index_count = MIntArray(indices_count // 3, 3)
        poly_indices = MIntArray()
        MScriptUtil.createIntArrayFromList(self.indices, poly_indices)
        mesh.create(
            vertices_count,
            indices_count // 3,
            vertices,
            poly_index_count,
            poly_indices,
        )

        # assign uv
        u_values = MFloatArray(vertices_count)
        v_values = MFloatArray(vertices_count)
        for i in range(0, vertices_count):
            vertex = self.vertices[i]
            u_values[i] = vertex.uv.x
            v_values[i] = 1.0 - vertex.uv.y
        mesh.setUVs(
            u_values, v_values
        )
        mesh.assignUVs(
            poly_index_count, poly_indices
        )

        # normal
        normals = MVectorArray()
        normal_indices = MIntArray(vertices_count)
        for i in range(0, vertices_count):
            vertex = self.vertices[i]
            normals.append(
                MVector(vertex.normal.x, vertex.normal.y, vertex.normal.z))
            normal_indices[i] = i
        mesh.setVertexNormals(
            normals,
            normal_indices
        )

        # dag_path and name
        mesh_dag_path = MDagPath()
        mesh.getPath(mesh_dag_path)
        mesh.setName(name)
        transform_node = MFnTransform(mesh.parent(0))
        transform_node.setName(f'mesh_{name}')

        # find render partition
        render_partition = MFnPartition()
        found_rp = False
        iterator = MItDependencyNodes(MFn.kPartition)
        while not iterator.isDone():
            render_partition.setObject(iterator.thisNode())
            if render_partition.name() == 'renderPartition' and render_partition.isRenderPartition():
                found_rp = True
                break
            iterator.next()
        # materials
        modifier = MDGModifier()
        set = MFnSet()
        for submesh in self.submeshes:
            # create lambert
            lambert = MFnLambertShader()
            lambert.create(True)

            lambert.setName(submesh.name)
            # some shader stuffs
            dependency_node = MFnDependencyNode()
            shading_engine = dependency_node.create(
                'shadingEngine', f'{submesh.name}_SG')
            material_info = dependency_node.create(
                'materialInfo', f'{submesh.name}_MaterialInfo')
            if found_rp:
                partition = MFnDependencyNode(
                    shading_engine).findPlug('partition')

                sets = render_partition.findPlug("sets")
                the_plug_we_need = None
                count = 0
                while True:
                    the_plug_we_need = sets.elementByLogicalIndex(count)
                    if not the_plug_we_need.isConnected():  # find the one that not connected
                        break
                    count += 1

                modifier.connect(partition, the_plug_we_need)

            # connect node
            out_color = lambert.findPlug('outColor')
            surface_shader = MFnDependencyNode(
                shading_engine).findPlug('surfaceShader')
            modifier.connect(out_color, surface_shader)

            message = MFnDependencyNode(shading_engine).findPlug('message')
            shading_group = MFnDependencyNode(
                material_info).findPlug('shadingGroup')
            modifier.connect(message, shading_group)

            modifier.doIt()

            # assign face to material
            component = MFnSingleIndexedComponent()
            face_component = component.create(MFn.kMeshPolygonComponent)
            group_poly_indices = MIntArray()
            for index in range(submesh.index_start // 3, (submesh.index_start + submesh.index_count) // 3):
                group_poly_indices.append(index)
            component.addElements(group_poly_indices)

            set.setObject(shading_engine)
            set.addMember(mesh_dag_path, face_component)

        if skl:
            influences_count = len(skl.influences)
            vertices_count = len(self.vertices)

            # select mesh + joint
            selections = MSelectionList()
            selections.add(mesh_dag_path)
            for i in range(0, influences_count):
                #joint = skl.joints[skl.influences[i]]
                selections.add(skl.dag_paths[skl.influences[i]])

            # bind selections
            MGlobal.selectCommand(selections)
            MGlobal.executeCommand(
                f'skinCluster -mi 4 -tsb -n skinCluster_{name}')

            # get skin cluster
            in_mesh = mesh.findPlug('inMesh')
            in_mesh_connections = MPlugArray()
            in_mesh.connectedTo(in_mesh_connections, True, False)
            if in_mesh_connections.length() == 0:
                raise RuntimeError(
                    error('SKN.load(skl): failed to find the created skin cluster'))
            skin_cluster = MFnSkinCluster(in_mesh_connections[0].node())

            # some mask
            influences_dag_paths = MDagPathArray()
            skin_cluster.influenceObjects(influences_dag_paths)
            influences_indices = MIntArray(influences_count)
            for i in range(0, influences_count):
                influence_dag_path = skl.dag_paths[skl.influences[i]]

                for j in range(0, influences_count):
                    if influence_dag_path == influences_dag_paths[j]:
                        influences_indices[i] = j
                        break

            # random things
            component = MFnSingleIndexedComponent()
            vertex_component = component.create(MFn.kMeshVertComponent)
            group_vertex_indices = MIntArray()
            py_list = list(range(0, vertices_count))
            MScriptUtil.createIntArrayFromList(py_list, group_vertex_indices)
            component.addElements(group_vertex_indices)

            MGlobal.executeCommand(
                f"setAttr {skin_cluster.name()}.normalizeWeights 0")

            # set weights
            weights = MDoubleArray(vertices_count * influences_count)
            for i in range(0, vertices_count):
                vertex = self.vertices[i]
                for j in range(0, 4):
                    weight = vertex.weights[j]
                    # treate bytes as a list, element with index 0 of [byte] is byte
                    influence = vertex.bones_indices[j][0]
                    if weight != 0:
                        weights[i * influences_count + influence] = weight
            skin_cluster.setWeights(
                mesh_dag_path, vertex_component, influences_indices, weights, False)

            # random things
            MGlobal.executeCommand(
                f"setAttr {skin_cluster.name()}.normalizeWeights 1")
            MGlobal.executeCommand(
                f"skinPercent -normalize true {skin_cluster.name()} {mesh.name()}")

        # shud be final line
        mesh.updateSurface()

    def flip(self):
        # read SKL.flip()
        for vertex in self.vertices:
            vertex.position.x = -vertex.position.x
            vertex.normal.y = -vertex.normal.y
            vertex.normal.z = -vertex.normal.z

    def dump(self, skl):
        # get mesh in selections
        selections = MSelectionList()
        MGlobal.getActiveSelectionList(selections)
        iterator = MItSelectionList(selections, MFn.kMesh)
        iterator.reset()
        if iterator.isDone():
            raise RuntimeError(error(f'[SKL.dump()] Please select a mesh.'))
        mesh_dag_path = MDagPath()
        iterator.getDagPath(mesh_dag_path)  # get first mesh
        iterator.next()
        if not iterator.isDone():
            raise RuntimeError(error(
                f'[SKL.dump()] More than 1 mesh selected., combine all meshes if you have mutiple meshes.'))
        mesh = MFnMesh(mesh_dag_path)

        # find skin cluster
        in_mesh = mesh.findPlug("inMesh")
        in_mesh_connections = MPlugArray()
        in_mesh.connectedTo(in_mesh_connections, True, False)
        if in_mesh_connections.length() == 0:
            raise RuntimeError(error(
                f'[SKL.dump({mesh.name()})] Failed to find skin cluster, make sure you binded the skin.'))
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
            raise RuntimeError(
                error(f'[SKL.dump({mesh.name()})] Mesh contains holes.'))

        # check non-triangulated polygons
        # check vertex has multiple shaders
        vertex_shaders = MIntArray(vertices_num, -1)
        iterator = MItMeshPolygon(mesh_dag_path)
        iterator.reset()
        while not iterator.isDone():
            if not iterator.hasValidTriangulation():
                raise RuntimeError(error(
                    f'[SKL.dump({mesh.name})] Mesh contains a non-triangulated polygon, try Mesh -> Triangulate.'))

            index = iterator.index()
            shader_index = poly_shaders[index]
            if shader_index == -1:
                raise RuntimeError(error(
                    f'[SKL.dump({mesh.name})] Mesh contains a face with no shader.'))

            vertices = MIntArray()
            iterator.getVertices(vertices)
            len69 = vertices.length()
            for i in range(0, len69):
                if shader_count > 1 and vertex_shaders[vertices[i]] not in [-1, shader_index]:
                    raise RuntimeError(error(
                        f'[SKL.dump({mesh.name})] Mesh contains a vertex with multiple shaders.'))
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
                raise RuntimeError(error(
                    f'[SKL.dump({mesh.name})] Mesh contains a vertex with more than 4 influences, try to:\n1. Rebind the skin with setting max 4 influences.\n2. Prune weights by 0.05.'))

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
                raise RuntimeError(error(
                    f'[SKL.dump({mesh.name})] Mesh contains a vertex with no UVs.'))
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
                        raise RuntimeError(error(
                            f'[SKL.dump({mesh.name})] data_index out of range.'))

                    for j in range(data_index, len(self.vertices)):
                        if self.vertices[j].data_index != data_index:
                            raise RuntimeError(error(
                                f'[SKL.dump({mesh.name})] Could not find corresponding face vertex.'))
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

        # fix same name of joint and material, example: Yone's Fish
        # if the name of node is already in maya, it adds "1" in the second name (and 2, 3, 4 ...)
        # -> remove "1" for either joint or material that has same name
        for joint in skl.joints:
            for submesh in self.submeshes:
                if joint.name == submesh.name+'1':
                    submesh.name = joint.name
                elif submesh.name == joint.name+'1':
                    joint.name = submesh.name

        # ay yo finally

    def write(self, path):
        with open(path, 'wb') as f:
            bs = BinaryStream(f)

            bs.write_uint32(0x00112233)  # magic
            bs.write_uint16(1)  # major
            bs.write_uint16(1)  # minor

            bs.write_uint32(len(self.submeshes))
            for submesh in self.submeshes:
                bs.write_padded_string(64, submesh.name)
                bs.write_uint32(submesh.vertex_start)
                bs.write_uint32(submesh.vertex_count)
                bs.write_uint32(submesh.index_start)
                bs.write_uint32(submesh.index_count)

            bs.write_uint32(len(self.indices))
            bs.write_uint32(len(self.vertices))

            for index in self.indices:
                bs.write_uint16(index)
            for vertex in self.vertices:
                bs.write_vec3(vertex.position)
                for byte in vertex.bones_indices:  # must have for, this is a list of bytes, not an array of byte
                    bs.write_bytes(byte)
                for weight in vertex.weights:
                    bs.write_float(weight)
                bs.write_vec3(vertex.normal)
                bs.write_vec2(vertex.uv)
