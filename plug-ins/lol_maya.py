from random import choice, uniform
from struct import pack, unpack
from math import sqrt, isclose


from maya import cmds
from maya.OpenMayaMPx import *
from maya.OpenMayaAnim import *
from maya.OpenMaya import *

# really need to clean things up


# maya file translator set up
class SKNTranslator(MPxFileTranslator):
    name = 'League of Legends: SKN'
    ext = 'skn'

    def __init__(self):
        MPxFileTranslator.__init__(self)

    def haveReadMethod(self):
        return True

    def canBeOpened(self):
        return False

    def defaultExtension(self):
        return self.ext

    def filter(self):
        return f'*.{self.ext}'

    @classmethod
    def creator(cls):
        return asMPxPtr(cls())

    def reader(self, file, options, acces):
        skn = SKN()
        path = file.expandedFullName()
        if not path.endswith('.skn'):
            path += '.skn'

        skn.read(path)
        name = path.split('/')[-1].split('.')[0]
        if options.split('=')[1] == '1':
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
    name = 'League of Legends: SKL'
    ext = 'skl'

    def __init__(self):
        MPxFileTranslator.__init__(self)

    def haveReadMethod(self):
        return True

    def canBeOpened(self):
        return False

    def defaultExtension(self):
        return self.ext

    def filter(self):
        return f'*.{self.ext}'

    @classmethod
    def creator(cls):
        return asMPxPtr(cls())

    def reader(self, file, options, access):
        skl = SKL()
        path = file.expandedFullName()
        if not path.endswith('.skl'):
            path += '.skl'
        skl.read(path)
        skl.flip()
        skl.load()
        return True


class SkinTranslator(MPxFileTranslator):
    name = 'League of Legends: SKN + SKL'
    ext = 'skn'

    def __init__(self):
        MPxFileTranslator.__init__(self)

    def haveWriteMethod(self):
        return True

    def canBeOpened(self):
        return False

    def defaultExtension(self):
        return self.ext

    def filter(self):
        return f'*.{self.ext}'

    @classmethod
    def creator(cls):
        return asMPxPtr(cls())

    def writer(self, file, options, access):
        if access != MPxFileTranslator.kExportActiveAccessMode:
            raise FunnyError(
                f'[SkinTranslator.writer()]: Stop! u violated the law, use "Export Selection" or i violate u UwU.')

        skl = SKL()
        skn = SKN()

        # dump from scene
        skl.dump()
        skn.dump(skl)
        # ay yo, do a flip!
        skl.flip()
        skn.flip()

        path = file.rawFullName()
        # fix for file with mutiple '.', this api is just meh
        if not path.endswith('.skn'):
            path += '.skn'
        skl.write(path.split('.skn')[0] + '.skl')
        skn.write(path)
        return True


class ANMTranslator(MPxFileTranslator):
    name = 'League of Legends: ANM'
    ext = 'anm'

    def __init__(self):
        MPxFileTranslator.__init__(self)

    def haveReadMethod(self):
        return True

    def haveWriteMethod(self):
        return True

    def canBeOpened(self):
        return False

    def defaultExtension(self):
        return self.ext

    def filter(self):
        return f'*.{self.ext}'

    @classmethod
    def creator(cls):
        return asMPxPtr(cls())

    def reader(self, file, options, access):
        anm = ANM()
        path = file.expandedFullName()
        if not path.endswith('.anm'):
            path += '.anm'

        anm.read(path)
        anm.flip()
        anm.load()
        return True

    def writer(self, file, options, access):
        if access != MPxFileTranslator.kExportAccessMode:
            raise FunnyError(
                f'[ANMTranslator.writer()]: Stop! u violated the law, use "Export All" or i violate u UwU.')

        anm = ANM()
        anm.dump()
        anm.flip()
        path = file.expandedFullName()
        if not path.endswith('.anm'):
            path += '.anm'
        anm.write(path)
        return True


# plugin register
def initializePlugin(obj):
    # totally not copied code
    plugin = MFnPlugin(obj, 'tarngaina', '1.0')
    try:
        plugin.registerFileTranslator(
            SKNTranslator.name,
            None,
            SKNTranslator.creator,
            "SKNTranslatorOpts",
            "",  # idk wtf wrong with defaul options
            True
        )
    except Exception as e:
        cmds.warning(
            f'Couldn\'t register SKNTranslator: [{e}]: {e.message}')

    try:
        plugin.registerFileTranslator(
            SKLTranslator.name,
            None,
            SKLTranslator.creator,
            None,
            None,
            True
        )
    except Exception as e:
        cmds.warning(
            f'Couldn\'t register SKLTranslator: [{e}]: {e.message}')

    try:
        plugin.registerFileTranslator(
            SkinTranslator.name,
            None,
            SkinTranslator.creator,
            None,
            None,
            True
        )
    except Exception as e:
        cmds.warning(
            f'Couldn\'t register SkinTranslator: [{e}]: {e.message}')

    try:
        plugin.registerFileTranslator(
            ANMTranslator.name,
            None,
            ANMTranslator.creator,
            None,
            None,
            True
        )
    except Exception as e:
        cmds.warning(
            f'Couldn\'t register ANMTranslator: [{e}]: {e.message}')


def uninitializePlugin(obj):
    plugin = MFnPlugin(obj)
    try:
        plugin.deregisterFileTranslator(
            SKNTranslator.name
        )
    except Exception as e:
        cmds.warning(
            f'Couldn\'t deregister SKNTranslator: [{e}]: {e.message}')

    try:
        plugin.deregisterFileTranslator(
            SKLTranslator.name
        )
    except Exception as e:
        cmds.warning(
            f'Couldn\'t deregister SKLTranslator: [{e}]: {e.message}')

    try:
        plugin.deregisterFileTranslator(
            SkinTranslator.name
        )
    except Exception as e:
        cmds.warning(
            f'Couldn\'t deregister SkinTranslator: [{e}]: {e.message}')

    try:
        plugin.deregisterFileTranslator(
            ANMTranslator.name
        )
    except Exception as e:
        cmds.warning(
            f'Couldn\'t deregister ANMTranslator: [{e}]: {e.message}')


# helper funcs and structures
class BinaryStream:
    # totally not copied code
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


class FunnyError(Exception):
    def __init__(self, message):
        self.show_message(message)

    def show_message(self, message):
        title = 'Error:'
        if ']: ' in message:
            temp = message.split(']: ')
            title = temp[0][1:] + ':'
            message = temp[1]
        cmds.confirmDialog(
            message=message,
            title=title,
            backgroundColor=[uniform(0.0, 1.0), uniform(
                0.0, 1.0), uniform(0.0, 1.0)],
            button=choice(
                ['UwU', '<(")', 'ok boomer', 'funny man', 'jesus', 'bruh',
                 'stop', 'get some help', 'haha', 'lmao', 'ay yo', 'SUS']
            )
        )
        return message


# for convert anm/skl joint name to elf hash
class Hash:
    # ay yo check out this elf: https://i.imgur.com/Cvl8PFu.png
    def elf(s):
        s = s.lower()
        h = 0
        for c in s:
            h = (h << 4) + ord(c)
            t = (h & 0xF0000000)
            if t != 0:
                h ^= (t >> 24)
            h &= ~t
        return h


# for v5 anm decompress transform properties
class CTransform:
    class Quat:
        def decompress(bytes):
            first = bytes[0] | (bytes[1] << 8)
            second = bytes[2] | (bytes[3] << 8)
            third = bytes[4] | (bytes[5] << 8)

            bits = first | second << 16 | third << 32

            max_index = (bits >> 45) & 3

            v_a = (bits >> 30) & 32767
            v_b = (bits >> 15) & 32767
            v_c = bits & 32767

            sqrt2 = 1.41421356237
            a = (v_a / 32767.0) * sqrt2 - 1 / sqrt2
            b = (v_b / 32767.0) * sqrt2 - 1 / sqrt2
            c = (v_c / 32767.0) * sqrt2 - 1 / sqrt2
            d = sqrt(max(0.0, 1.0 - (a * a + b * b + c * c)))

            if max_index == 0:
                return MQuaternion(d, a, b, c)
            elif max_index == 1:
                return MQuaternion(a, d, b, c)
            elif max_index == 2:
                return MQuaternion(a, b, d, c)
            else:
                return MQuaternion(a, b, c, d)

    class Vec:
        def decompress(min, max, bytes):
            res = max - min

            cx = bytes[0] | bytes[1] << 8
            cy = bytes[2] | bytes[3] << 8
            cz = bytes[4] | bytes[5] << 8
            res.x *= (cx / 65535.0)
            res.y *= (cy / 65535.0)
            res.z *= (cz / 65535.0)

            res += min
            return MVector(res.x, res.y, res.z)


# for anm/skl joint set transform - transformation matrix
class MTransform():
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
    def __init__(self):
        self.joints = []

        # for dumping
        self.dag_paths = None

        # for loading
        self.legacy = None
        self.influences = []  # for load both skn + skl as skincluster

    def flip(self):
        # flip the L with R: https://youtu.be/2yzMUs3badc
        for joint in self.joints:
            # local
            if joint.local_translation:  # check when reading, legacy doesnt have local
                joint.local_translation.x *= -1.0
                joint.local_rotation.y *= -1.0
                joint.local_rotation.z *= -1.0
            # inversed global
            joint.iglobal_translation.x *= -1.0
            joint.iglobal_rotation.y *= -1.0
            joint.iglobal_rotation.z *= -1.0

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
                    raise FunnyError(
                        f'[SKL.read({path})]: Unsupported file version: {version}')

                bs.read_uint16()  # flags - pad
                joint_count = bs.read_uint16()
                influences_count = bs.read_uint32()

                joints_offset = bs.read_int32()
                bs.read_int32()  # joint indices offset
                influences_offset = bs.read_int32()
                bs.read_uint32()  # name offset - pad
                bs.read_uint32()  # asset name offset - pad
                bs.read_int32()  # joint names offset

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
                        joint_name_offset = bs.read_int32()  # joint name offset - pad
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
                    raise FunnyError(
                        f'[SKL.read({path})]: Wrong file signature: {magic}')

                version = bs.read_uint32()
                if version not in [1, 2]:
                    raise FunnyError(
                        f'[SKL.read({path})]: Unsupported file version: {version}')

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
                    joint.iglobal_translation, joint.iglobal_scale, joint.iglobal_rotation = MTransform.decompose(
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
                    self.influences = list(range(0, joint_count))

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
                ik_joint.set(MTransform.compose(
                    joint.iglobal_translation, joint.iglobal_scale, joint.iglobal_rotation, MSpace.kTransform))
            else:
                ik_joint.set(MTransform.compose(
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
        while not iterator.isDone():
            iterator.getPath(dag_path)
            self.dag_paths.append(dag_path)  # identify joint by DAG path
            ik_joint.setObject(dag_path)  # to get the joint transform

            joint = SKLJoint()
            joint.name = ik_joint.name()
            # mama mia
            joint.local_translation, joint.local_scale, joint.local_rotation = MTransform.decompose(
                MTransformationMatrix(ik_joint.transformationMatrix()),
                MSpace.kTransform
            )
            joint.iglobal_translation, joint.iglobal_scale, joint.iglobal_rotation = MTransform.decompose(
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

    def write(self, path):
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

            bs.write_int32(joints_offset)
            bs.write_int32(joint_indices_offset)
            bs.write_int32(influences_offset)
            bs.write_int32(0)  # name
            bs.write_int32(0)  # asset name
            bs.write_int32(joint_names_offset)

            bs.write_uint32(0xFFFFFFFF)  # reserved offset field
            bs.write_uint32(0xFFFFFFFF)
            bs.write_uint32(0xFFFFFFFF)
            bs.write_uint32(0xFFFFFFFF)
            bs.write_uint32(0xFFFFFFFF)

            joint_offset = {}
            bs.stream.seek(joint_names_offset)
            len1235 = len(self.joints)
            for i in range(0, len1235):
                joint_offset[i] = bs.stream.tell()
                bs.write_bytes(self.joints[i].name.encode('ascii'))
                bs.write_bytes(bytes([0]))  # pad

            bs.stream.seek(joints_offset)
            for i in range(0, len1235):
                joint = self.joints[i]
                # write skljoint in this func
                bs.write_uint16(0)  # flags
                bs.write_uint16(i)  # id
                bs.write_int16(joint.parent)  # -1, cant be uint
                bs.write_uint16(0)  # pad
                bs.write_uint32(Hash.elf(joint.name))
                bs.write_float(2.1)  # scale
                # local
                bs.write_vec3(joint.local_translation)
                bs.write_vec3(joint.local_scale)
                bs.write_quat(joint.local_rotation)
                # inversed global
                bs.write_vec3(joint.iglobal_translation)
                bs.write_vec3(joint.iglobal_scale)
                bs.write_quat(joint.iglobal_rotation)

                bs.write_int32(joint_offset[i] - bs.stream.tell())

            bs.stream.seek(influences_offset)
            for i in range(0, len1235):
                bs.write_uint16(i)

            bs.stream.seek(joint_indices_offset)
            for i in range(0, len1235):
                bs.write_uint32(Hash.elf(joint.name))
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

    def flip(self):
        # read SKL.flip()
        for vertex in self.vertices:
            vertex.position.x *= -1.0
            vertex.normal.y *= -1.0
            vertex.normal.z *= -1.0

    def read(self, path):
        with open(path, 'rb') as f:
            bs = BinaryStream(f)

            magic = bs.read_uint32()
            if magic != 0x00112233:
                raise FunnyError(
                    f'[SKN.read({path})]: Wrong signature file: {magic}')

            major = bs.read_uint16()
            minor = bs.read_uint16()
            if major not in [0, 2, 4] and minor != 1:
                raise FunnyError(
                    f'[SKN.read({path})]: Unsupported file version: {major}.{minor}')

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
                    # again version 0 doesnt have submesh wrote in file
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
                # joint = skl.joints[skl.influences[i]]
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
                raise FunnyError(
                    f'[SKN.load({name}, skl)]: Failed to find created skin cluster.')
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

    def dump(self, skl):
        # get mesh in selections
        selections = MSelectionList()
        MGlobal.getActiveSelectionList(selections)
        iterator = MItSelectionList(selections, MFn.kMesh)
        if iterator.isDone():
            raise FunnyError(f'[SKN.dump()]: Please select a mesh.')
        mesh_dag_path = MDagPath()
        iterator.getDagPath(mesh_dag_path)  # get first mesh
        iterator.next()
        if not iterator.isDone():
            raise FunnyError(
                f'[SKN.dump()]: More than 1 mesh selected., combine all meshes if you have mutiple meshes.')
        mesh = MFnMesh(mesh_dag_path)

        # find skin cluster
        in_mesh = mesh.findPlug("inMesh")
        in_mesh_connections = MPlugArray()
        in_mesh.connectedTo(in_mesh_connections, True, False)
        if in_mesh_connections.length() == 0:
            raise FunnyError(
                f'[SKN.dump({mesh.name()})]: Failed to find skin cluster, make sure you binded the skin.')
        skin_cluster = MFnSkinCluster(in_mesh_connections[0].node())
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
            raise FunnyError(
                f'[SKN.dump({mesh.name()})]: Mesh contains holes: {hole_info.length()}')

        # check non-triangulated polygons
        # check vertex has multiple shaders
        vertex_shaders = MIntArray(vertices_num, -1)
        iterator = MItMeshPolygon(mesh_dag_path)
        iterator.reset()
        while not iterator.isDone():
            if not iterator.hasValidTriangulation():
                raise FunnyError(
                    f'[SKN.dump({mesh.name()})]: Mesh contains a non-triangulated polygon, try Mesh -> Triangulate.')

            index = iterator.index()
            shader_index = poly_shaders[index]
            if shader_index == -1:
                raise FunnyError(
                    f'[SKN.dump({mesh.name()})]: Mesh contains a face with no shader, make sure you assigned a material to all faces.')

            vertices = MIntArray()
            iterator.getVertices(vertices)
            len69 = vertices.length()
            for i in range(0, len69):
                if shader_count > 1 and vertex_shaders[vertices[i]] not in [-1, shader_index]:
                    raise FunnyError(
                        f'[SKN.dump({mesh.name()})]: Mesh contains a vertex with multiple shaders, try re-assign a lambert material to this mesh.')
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
                raise FunnyError(
                    f'[SKN.dump({mesh.name()})]: Mesh contains a vertex have weight afftected more than 4 influences, try:\n1. Rebind the skin with max 4 influences setting.\n2. Prune weights by 0.05.')

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

                        # create SKNVertex - recreate pointer for safe
                        vertex = SKNVertex()
                        vertex.position = MVector(
                            temp_position.x, temp_position.y, temp_position.z)
                        vertex.bones_indices = temp_bones_indices
                        vertex.weights = temp_weights
                        vertex.normal = MVector(
                            temp_normal.x, temp_normal.y, temp_normal.z)
                        vertex.uv = MVector(temp_uv.x, temp_uv.y)
                        vertex.uv_index = temp_uv_index

                        shader_vertices[shader].append(vertex)
                        shader_vertex_indices[shader].append(index)

                        seen.append(uv_index)
            else:
                raise FunnyError(
                    f'[SKN.dump({mesh.name()})]: Mesh contains a vertex with no UVs.')
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
                        raise FunnyError(
                            f'[SKN.dump({mesh.name()})]: Data index out of range.')

                    for j in range(data_index, len(self.vertices)):
                        if self.vertices[j].data_index != data_index:
                            raise FunnyError(
                                f'[SKN.dump({mesh.name()})]: Could not find corresponding face vertex.')
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


# anm
class ANMPose:
    def __init__(self):
        self.time = None

        self.translation = None
        self.scale = None
        self.rotation = None

        # for dumping
        self.translation_index = None
        self.scale_index = None
        self.rotation_index = None


class ANMTrack:
    def __init__(self):
        self.joint_hash = None

        self.poses = []

        # for loading
        self.joint_name = None
        self.dag_path = None


class ANM:
    def __init__(self):
        self.duration = None

        self.tracks = []

        # for loading
        self.compressed = None

    def flip(self):
        # DO A FLIP!
        for track in self.tracks:
            for pose in track.poses:
                if pose.translation != None:
                    pose.translation.x *= -1.0

                if pose.rotation != None:
                    pose.rotation.y *= -1.0
                    pose.rotation.z *= -1.0

    def read(self, path):
        with open(path, 'rb') as f:
            bs = BinaryStream(f)

            magic = bs.read_bytes(8).decode('ascii')
            version = bs.read_uint32()

            if magic == 'r3d2canm':
                self.compressed = True
                # compressed

                bs.read_uint32()  # resource_size
                bs.read_uint32()  # format token
                bs.read_uint32()  # flags

                joint_count = bs.read_uint32()
                frames_count = bs.read_uint32()
                bs.read_uint32()  # jump cache count
                duration = bs.read_float()
                bs.read_float()  # fps
                for i in range(0, 6):  # pad some random things
                    bs.read_float()

                translation_min = bs.read_vec3()
                translation_max = bs.read_vec3()

                scale_min = bs.read_vec3()
                scale_max = bs.read_vec3()

                frames_offset = bs.read_int32()
                bs.read_int32()  # jump caches offset
                joint_hashes_offset = bs.read_int32()

                if frames_offset <= 0:
                    raise FunnyError(
                        f'[ANM.read({path})]: File does not contain frames.'
                    )
                if joint_hashes_offset <= 0:
                    raise FunnyError(
                        f'[ANM.read({path})]: File does not contain joint hashes.'
                    )

                # read joint hashes
                bs.stream.seek(joint_hashes_offset + 12)
                joint_hashes = []
                for i in range(0, joint_count):
                    joint_hashes.append(bs.read_uint32())

                # create tracks
                self.duration = duration * 30.0 + 1
                for i in range(0, joint_count):
                    track = ANMTrack()
                    track.joint_hash = joint_hashes[i]
                    self.tracks.append(track)

                bs.stream.seek(frames_offset + 12)
                for i in range(0, frames_count):
                    compressed_time = bs.read_uint16()
                    bits = bs.read_uint16()
                    compressed_transform = bs.read_bytes(6)

                    # find track by joint hash
                    joint_hash = joint_hashes[bits & 16383]
                    for track in self.tracks:
                        if track.joint_hash == joint_hash:
                            break

                    # find pose existed with time
                    time = compressed_time / 65535.0 * duration * 30.0
                    pose = None
                    for pose in track.poses:
                        if isclose(pose.time, time, rel_tol=0, abs_tol=0.01):
                            break
                        else:
                            pose = None

                    # no pose found, create new
                    if pose == None:
                        pose = ANMPose()
                        pose.time = time
                        track.poses.append(pose)

                    # decompress data and add to pose
                    transform_type = bits >> 14
                    if transform_type == 0:
                        rotation = CTransform.Quat.decompress(
                            compressed_transform)
                        pose.rotation = MQuaternion(
                            rotation.x, rotation.y, rotation.z, rotation.w)
                    elif transform_type == 1:
                        translation = CTransform.Vec.decompress(
                            translation_min, translation_max, compressed_transform)
                        pose.translation = MVector(
                            translation.x, translation.y, translation.z)
                    elif transform_type == 2:
                        scale = CTransform.Vec.decompress(
                            scale_min, scale_max, compressed_transform)
                        pose.scale = MVector(scale.x, scale.y, scale.z)
                    else:
                        raise FunnyError(
                            f'[ANM.read({path})]: Unknown compressed transform type: {transform_type}.'
                        )

            elif magic == 'r3d2anmd':
                self.compressed = False

                if version == 5:
                    # v5

                    bs.read_uint32()  # resource_size
                    bs.read_uint32()  # format_token
                    bs.read_uint32()  # version??
                    bs.read_uint32()  # flags

                    track_count = bs.read_uint32()
                    frame_count = bs.read_uint32()
                    self.duration = frame_count
                    bs.read_float()  # frame_duration

                    joint_hashes_offset = bs.read_int32()
                    bs.read_int32()  # asset name offset
                    bs.read_int32()  # time offset
                    vecs_offset = bs.read_int32()
                    quats_offset = bs.read_int32()
                    frames_offset = bs.read_int32()

                    if joint_hashes_offset <= 0:
                        raise FunnyError(
                            f'[ANM.read({path})]: File does not contain joint hashes.'
                        )
                    if vecs_offset <= 0:
                        raise FunnyError(
                            f'[ANM.read({path})]: File does not contain vectors.'
                        )
                    if quats_offset <= 0:
                        raise FunnyError(
                            f'[ANM.read({path})]: File does not contain quaternion.'
                        )
                    if frames_offset <= 0:
                        raise FunnyError(
                            f'[ANM.read({path})]: File does not contain frames.'
                        )

                    joint_hahses_count = (
                        frames_offset - joint_hashes_offset) // 4
                    vecs_count = (quats_offset - vecs_offset) // 12
                    quats_count = (joint_hashes_offset - quats_offset) // 6

                    # read joint hashes
                    joint_hashes = []
                    bs.stream.seek(joint_hashes_offset + 12)
                    for i in range(0, joint_hahses_count):
                        joint_hashes.append(bs.read_uint32())

                    # read vecs
                    vecs = []
                    bs.stream.seek(vecs_offset + 12)
                    for i in range(0, vecs_count):
                        vecs.append(bs.read_vec3())  # unique vec

                    # read quats
                    quats = []
                    bs.stream.seek(quats_offset + 12)
                    for i in range(0, quats_count):
                        quats.append(
                            CTransform.Quat.decompress(bs.read_bytes(6)))  # unique compressed quat

                    # read frames
                    frames = []
                    bs.stream.seek(frames_offset + 12)
                    for i in range(0, frame_count * track_count):
                        frames.append((
                            bs.read_uint16(),  # translation index
                            bs.read_uint16(),  # scale index
                            bs.read_uint16()  # rotation index
                        ))

                    # create tracks
                    for i in range(0, track_count):
                        track = ANMTrack()
                        track.joint_hash = joint_hashes[i]
                        self.tracks.append(track)

                    for t in range(0, track_count):
                        track = self.tracks[t]
                        for f in range(0, frame_count):
                            translation_index, scale_index, rotation_index = frames[f * track_count + t]

                            # create pointer, read v4
                            pose = ANMPose()
                            pose.time = f
                            pose.translation = MVector(
                                vecs[translation_index].x,
                                vecs[translation_index].y,
                                vecs[translation_index].z
                            )
                            pose.scale = MVector(
                                vecs[scale_index].x,
                                vecs[scale_index].y,
                                vecs[scale_index].z
                            )
                            pose.rotation = MQuaternion(
                                quats[rotation_index].x,
                                quats[rotation_index].y,
                                quats[rotation_index].z,
                                quats[rotation_index].w
                            )
                            track.poses.append(pose)

                elif version == 4:
                    # v4

                    bs.read_uint32()  # resource_size
                    bs.read_uint32()  # format_token
                    bs.read_uint32()  # version??
                    bs.read_uint32()  # flags

                    track_count = bs.read_uint32()
                    frame_count = bs.read_uint32()
                    self.duration = frame_count
                    bs.read_float()  # frame_duration

                    bs.read_int32()  # tracks offset
                    bs.read_int32()  # asset name offset
                    bs.read_int32()  # time offset
                    vecs_offset = bs.read_int32()
                    quats_offset = bs.read_int32()
                    frames_offset = bs.read_int32()

                    if vecs_offset <= 0:
                        raise FunnyError(
                            f'[ANM.read({path})]: File does not contain vectors.'
                        )
                    if quats_offset <= 0:
                        raise FunnyError(
                            f'[ANM.read({path})]: File does not contain quaternion.'
                        )
                    if frames_offset <= 0:
                        raise FunnyError(
                            f'[ANM.read({path})]: File does not contain frames.'
                        )

                    vecs_count = (quats_offset - vecs_offset) // 12
                    quats_count = (frames_offset - quats_offset) // 16

                    bs.stream.seek(vecs_offset + 12)
                    vecs = []
                    for i in range(0, vecs_count):
                        vecs.append(bs.read_vec3())  # unique vec

                    bs.stream.seek(quats_offset + 12)
                    quats = []
                    for i in range(0, quats_count):
                        quats.append(bs.read_quat())  # unique quat
                    bs.stream.seek(frames_offset + 12)
                    frames = []
                    for i in range(0, frame_count * track_count):
                        frames.append((
                            bs.read_uint32(),  # joint hash
                            bs.read_uint16(),  # translation index
                            bs.read_uint16(),  # scale index
                            bs.read_uint16()  # rotation index
                        ))
                        bs.read_uint16()  # pad

                    # parse data from frames
                    for joint_hash, translation_index, scale_index, rotation_index in frames:
                        # apparently, those index can be same in total vecs/squats (i guess they want to compress it)
                        # we need to recreate pointer
                        pose = ANMPose()
                        pose.translation = MVector(
                            vecs[translation_index].x,
                            vecs[translation_index].y,
                            vecs[translation_index].z
                        )
                        pose.scale = MVector(
                            vecs[scale_index].x,
                            vecs[scale_index].y,
                            vecs[scale_index].z
                        )
                        pose.rotation = MQuaternion(
                            quats[rotation_index].x,
                            quats[rotation_index].y,
                            quats[rotation_index].z,
                            quats[rotation_index].w
                        )

                        # find the track that already in total tracks with joint hash
                        track = None
                        for t in self.tracks:
                            if t.joint_hash == joint_hash:
                                track = t
                                break
                            else:
                                track = None

                        # couldnt found track that has joint hash, create new
                        if track == None:
                            track = ANMTrack()
                            track.joint_hash = joint_hash
                            self.tracks.append(track)

                        # time = index
                        pose.time = len(track.poses)
                        track.poses.append(pose)

                else:
                    # legacy

                    bs.read_uint32()  # skeletion id
                    track_count = bs.read_uint32()
                    frame_count = bs.read_uint32()  # need this to index by frame and stuffs
                    self.duration = frame_count
                    bs.read_uint32()  # fps
                    for i in range(0, track_count):
                        track = ANMTrack()
                        track.joint_hash = Hash.elf(bs.read_padded_string(
                            32))  # joint name -> joint hash
                        bs.read_uint32()  # flags
                        for j in range(0, frame_count):
                            pose = ANMPose()
                            pose.time = j
                            pose.rotation = bs.read_quat()
                            pose.translation = bs.read_vec3()
                            pose.scale = MVector(
                                1.0, 1.0, 1.0)  # legacy not support scaling
                            track.poses.append(pose)
                        self.tracks.append(track)

            else:
                raise FunnyError(
                    f'[ANM.read({path})]: Wrong signature file: {magic}')

    def load(self):
        # actual tracks (track of joints that found in scene)
        actual_tracks = []
        joint_dag_path = MDagPath()
        for i in range(0, len(self.tracks)):
            track = self.tracks[i]

            # loop through all ik joint in scenes
            iterator = MItDag(MItDag.kDepthFirst, MFn.kJoint)
            while not iterator.isDone():
                iterator.getPath(joint_dag_path)
                ik_joint = MFnIkJoint(joint_dag_path)
                joint_name = ik_joint.name()
                # compare ik joint's hash vs track joint's hash
                if track.joint_hash == Hash.elf(joint_name):
                    # found joint in scene
                    track.joint_name = joint_name
                    track.dag_path = MDagPath(
                        joint_dag_path)  # recreate pointer
                    actual_tracks.append(track)
                    break
                iterator.next()

            if iterator.isDone():
                # the loop isnt broken, mean we havent found the joint
                # so its not in actual tracks, only in our imagination
                cmds.warning(
                    f'[ANM.load()]: No joint named found: {track.joint_hash}')

        if len(actual_tracks) == 0:
            raise FunnyError(
                '[ANM.load()]: No data joints found in scene, please import SKL if joints are not in scene.')

        # delete all channel data
        MGlobal.executeCommand('delete -all -c')

        # ensure 30fps scene
        # im pretty sure all lol's anms are in 30fps, or i can change this later idk
        # this only ensure the "import scene", not the existing scene in maya, to make this work:
        # select "Override to Math Source" for both Framerate % Animation Range in Maya's import options panel
        MGlobal.executeCommand('currentUnit -time ntsc')

        # adjust animation range
        MGlobal.executeCommand(
            f'playbackOptions -e -min 0 -max {self.duration-1} -animationStartTime 0 -animationEndTime {self.duration-1} -playbackSpeed 1')

        if self.compressed:
            # slow but safe load for compressed anm
            for track in actual_tracks:
                ik_joint = MFnIkJoint(track.dag_path)

                for pose in track.poses:
                    # this can be float too
                    MGlobal.executeCommand(f'currentTime {pose.time}')

                    setKeyFrame = 'setKeyframe -breakdown 0 -hierarchy none -controlPoints 0 -shape 0'
                    modified = False  # check if we actually need to set key frame
                    # translation
                    if pose.translation != None:
                        translation = pose.translation
                        ik_joint.setTranslation(
                            MVector(translation.x, translation.y, translation.z), MSpace.kTransform)
                        setKeyFrame += ' -at translateX -at translateY -at translateZ'
                        modified = True
                    # scale
                    if pose.scale != None:
                        scale = pose.scale
                        util = MScriptUtil()
                        util.createFromDouble(scale.x, scale.y, scale.z)
                        ptr = util.asDoublePtr()
                        ik_joint.setScale(ptr)
                        setKeyFrame += ' -at scaleX -at scaleY -at scaleZ'
                        modified = True
                    # rotation
                    if pose.rotation != None:
                        rotation = pose.rotation
                        rotation = MQuaternion(
                            rotation.x, rotation.y, rotation.z, rotation.w)  # recreate pointer
                        orient = MQuaternion()
                        ik_joint.getOrientation(orient)
                        axe = ik_joint.rotateOrientation(MSpace.kTransform)
                        rotation = axe.inverse() * rotation * orient.inverse()
                        ik_joint.setRotation(rotation, MSpace.kTransform)
                        setKeyFrame += ' -at rotateX -at rotateY -at rotateZ'
                        modified = True

                    if modified:
                        setKeyFrame += f' {track.joint_name}'
                        MGlobal.executeCommand(setKeyFrame)

            # slerp all quaternions - EULER SUCKS!
            for track in actual_tracks:
                MGlobal.executeCommand(
                    f'rotationInterpolation -c quaternionSlerp {track.joint_name}.rotateX {track.joint_name}.rotateY {track.joint_name}.rotateZ'
                )
        else:
            # fast load decompressed animation
            joint_names = [track.joint_name for track in actual_tracks]
            for time in range(0, self.duration):
                MGlobal.executeCommand(f'currentTime {time}')
                for track in actual_tracks:
                    pose = track.poses[time]
                    ik_joint = MFnIkJoint(track.dag_path)

                    # translation
                    translation = pose.translation
                    ik_joint.setTranslation(translation, MSpace.kTransform)
                    # scale
                    scale = pose.scale
                    util = MScriptUtil()
                    util.createFromDouble(scale.x, scale.y, scale.z)
                    ptr = util.asDoublePtr()
                    ik_joint.setScale(ptr)
                    # rotation
                    rotation = pose.rotation
                    orient = MQuaternion()
                    ik_joint.getOrientation(orient)
                    axe = ik_joint.rotateOrientation(MSpace.kTransform)
                    rotation = axe.inverse() * rotation * orient.inverse()
                    ik_joint.setRotation(rotation, MSpace.kTransform)

                setKeyFrame = 'setKeyframe -breakdown 0 -hierarchy none -controlPoints 0 -shape 0 -at translateX -at translateY -at translateZ -at scaleX -at scaleY -at scaleZ -at rotateX -at rotateY -at rotateZ '
                setKeyFrame += ' '.join(joint_names)
                MGlobal.executeCommand(setKeyFrame)

    def dump(self):
        # get joint in scene
        dag_path = MDagPath()
        iterator = MItDag(MItDag.kDepthFirst, MFn.kJoint)
        while not iterator.isDone():
            iterator.getPath(dag_path)
            ik_joint = MFnIkJoint(dag_path)

            track = ANMTrack()
            track.dag_path = MDagPath(dag_path)
            track.joint_name = ik_joint.name()
            track.joint_hash = Hash.elf(track.joint_name)
            self.tracks.append(track)
            iterator.next()

        # ensure 30 fps, read ANM.load()
        MGlobal.executeCommand('currentUnit -time ntsc')

        # assume that animation start time always at 0
        # if its not then well, its the ppl fault, not mine. haha suckers
        start_util = MScriptUtil()
        start_ptr = start_util.asDoublePtr()
        MGlobal.executeCommand(
            "playbackOptions -q -animationStartTime", start_ptr)
        start = start_util.getDouble(start_ptr)
        if int(start) != 0:
            raise FunnyError(
                f'[ANM.dump()]: Animation start time is not at 0: {start}, check Time slider, make sure animation start at 0.')

        # get duration with cursed api
        end_util = MScriptUtil()
        end_ptr = end_util.asDoublePtr()
        MGlobal.executeCommand("playbackOptions -q -animationEndTime", end_ptr)
        end = end_util.getDouble(end_ptr)
        self.duration = int(round(end)) + 1

        for time in range(0, self.duration):
            MGlobal.executeCommand(f'currentTime {time}')

            for track in self.tracks:
                ik_joint = MFnIkJoint(track.dag_path)

                pose = ANMPose()
                pose.time = time
                # translation
                translation = ik_joint.getTranslation(MSpace.kTransform)
                pose.translation = MVector(
                    translation.x, translation.y, translation.z)
                # scale
                scale_util = MScriptUtil()
                scale_util.createFromDouble(0.0, 0.0, 0.0)
                scale_ptr = scale_util.asDoublePtr()
                ik_joint.getScale(scale_ptr)
                pose.scale = MVector(
                    scale_util.getDoubleArrayItem(scale_ptr, 0),
                    scale_util.getDoubleArrayItem(scale_ptr, 1),
                    scale_util.getDoubleArrayItem(scale_ptr, 2)
                )
                # rotation
                orient = MQuaternion()
                ik_joint.getOrientation(orient)
                axe = ik_joint.rotateOrientation(MSpace.kTransform)
                rotation = MQuaternion()
                ik_joint.getRotation(rotation, MSpace.kTransform)
                rotation = axe * rotation * orient
                pose.rotation = MQuaternion(
                    rotation.x, rotation.y, rotation.z, rotation.w
                )
                track.poses.append(pose)

    def write(self, path):
        # build unique vecs + quats
        uni_vecs = []
        uni_quats = []
        for track in self.tracks:
            for pose in track.poses:
                for i in range(0, len(uni_vecs)):
                    # find pose translation
                    if pose.translation == uni_vecs[i]:
                        pose.translation_index = i

                    # find pose scale
                    if pose.scale == uni_vecs[i]:
                        pose.scale_index = i

                    if pose.translation_index != None and pose.scale_index != None:
                        # if found both in unique vecs then break loop
                        break

                if pose.translation_index == None:
                    # add new unique translation
                    pose.translation_index = len(uni_vecs)
                    uni_vecs.append(pose.translation)
                if pose.scale_index == None:
                    # also check if scale = translation
                    if pose.scale == pose.translation:
                        pose.scale_index = pose.translation_index
                    else:
                        # add new unique scale
                        pose.scale_index = len(uni_vecs)
                        uni_vecs.append(pose.scale)

                for i in range(0, len(uni_quats)):
                    if pose.rotation == uni_quats[i]:
                        pose.rotation_index = i

                    if pose.rotation_index != None:
                        break

                if pose.rotation_index == None:
                    pose.rotation_index = len(uni_quats)
                    uni_quats.append(pose.rotation)

        with open(path, 'wb') as f:
            bs = BinaryStream(f)

            bs.write_bytes('r3d2anmd'.encode('ascii'))  # magic
            bs.write_uint32(4)  # ver 4

            bs.write_uint32(0)  # size - later
            bs.write_uint32(0xBE0794D3)  # format token
            bs.write_uint32(0)  # version?
            bs.write_uint32(0)  # flags

            bs.write_uint32(len(self.tracks))  # track count
            bs.write_uint32(self.duration)  # frame count
            bs.write_float(1.0 / 30.0)  # frame duration = 1 / 30fps

            bs.write_int32(0)  # tracks offset
            bs.write_int32(0)  # asset name offset
            bs.write_int32(0)  # time offset

            bs.write_int32(64)  # vecs offset
            quats_offset_offset = bs.stream.tell()
            bs.write_int32(0)   # quats offset - later
            bs.write_int32(0)   # frames offset - later

            bs.stream.seek(12, 1)  # pad 12 empty bytes

            # vecs
            for vec in uni_vecs:
                bs.write_vec3(vec)

            quats_offset = bs.stream.tell()
            for quat in uni_quats:
                bs.write_quat(quat)

            frames_offset = bs.stream.tell()
            for time in range(0, self.duration):
                for track in self.tracks:
                    bs.write_uint32(track.joint_hash)
                    bs.write_uint16(track.poses[time].translation_index)
                    bs.write_uint16(track.poses[time].scale_index)
                    bs.write_uint16(track.poses[time].rotation_index)
                    bs.write_uint16(0)

            # quats offset and frames offset
            bs.stream.seek(quats_offset_offset)
            bs.write_int32(quats_offset - 12)  # need to minus 12 bytes back
            bs.write_int32(frames_offset - 12)

            # resource size
            bs.stream.seek(0, 2)
            fsize = bs.stream.tell()
            bs.stream.seek(12)
            bs.write_uint32(fsize)


# static object
class SOVertex:
    def __init__(self):
        self.position = None
        self.uv = None


class SOSubmesh:
    def __init__(self):
        self.name = None
        self.vertex_start = None
        self.vertex_count = None
        self.index_start = None
        self.index_count = None


class SOFace:
    def __init__(self):
        self.indices = []
        self.material = None
        self.uvs = []


class SOData:
    def __init__(self):
        self.name = None

        # this shud be joint?
        self.central = None
        self.pivot = None

        self.vertices = []
        self.faces = []

    def read_sco(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
            lines = [line[:-1] for line in lines]

            magic = lines[0]
            if magic != '[ObjectBegin]':
                raise FunnyError(
                    f'[SOData.read({path})]: Wrong file signature: {magic}')

            index = 1  # skip first line
            len1234 = len(lines)
            while index < len1234:
                inp = lines[index].split()
                if len(inp) == 0:  # cant split, must be a random line
                    index += 1
                    continue

                if inp[0] == 'Name=':
                    self.name = inp[1]
                    if ':' in self.name:
                        self.name = self.name.split(':')[1]

                elif inp[0] == 'CentralPoint=':
                    self.central = MVector(
                        float(inp[1]), float(inp[2]), float(inp[3]))
                    self.pivot = MVector(self.central)

                elif inp[0] == 'PivotPoint=':
                    self.pivot = MVector(
                        float(inp[1]), float(inp[2]), float(inp[3]))

                elif inp[0] == 'Verts=':
                    vertex_count = int(inp[1])
                    for i in range(index+1, index+1 + vertex_count):
                        inp2 = lines[i].split()
                        vertex = MVector(
                            float(inp2[0]), float(inp2[1]), float(inp2[2]))
                        self.vertices.append(vertex)
                    index = i+1
                    continue

                elif inp[0] == 'Faces=':
                    face_count = int(inp[1])
                    for i in range(index+1, index+1 + face_count):
                        inp2 = lines[i].replace('\t', ' ').split()
                        face = SOFace()
                        face.indices = [int(inp2[1]), int(
                            inp2[2]), int(inp2[3])]
                        face.material = inp2[4]
                        face.uvs = [
                            MVector(float(inp2[5]), float(inp2[8])),
                            MVector(float(inp2[6]), float(inp2[9])),
                            MVector(float(inp2[7]), float(inp2[10]))
                        ]
                        self.faces.append(face)
                    index = i+1
                    continue

                index += 1

    def read_scb(self, path):
        with open(path, 'rb') as f:
            bs = BinaryStream(f)

            magic = bs.read_bytes(8).decode('ascii')
            if magic != 'r3d2Mesh':
                raise FunnyError(
                    f'[SOData.read({path})]: Wrong file signature: {magic}')

            major = bs.read_uint16()
            minor = bs.read_uint16()
            if major not in [3, 2] and minor != 1:
                raise FunnyError(
                    f'[SOData.read({path})]: Unsupported file version: {major}.{minor}')

            self.name = bs.read_padded_string(128)

            vertex_count = bs.read_uint32()
            face_count = bs.read_uint32()

            flags = bs.read_uint32()

            bs.read_vec3()  # bouding box
            bs.read_vec3()

            vertex_color = 0
            if major == 3 and minor == 2:
                vertex_color = bs.read_uint32()  # for padding

            for i in range(0, vertex_count):
                self.vertices.append(bs.read_vec3())

            if vertex_color == 1:
                for i in range(0, vertex_count):
                    bs.read_byte()  # pad all color
                    bs.read_byte()
                    bs.read_byte()
                    bs.read_byte()

            self.central = bs.read_vec3()
            self.pivot = MVector(self.central)

            for i in range(0, face_count):
                face = SOFace()
                face.indices = [bs.read_uint32(), bs.read_uint32(),
                                bs.read_uint32()]
                face.material = bs.read_padded_string(64)

                uvs = [bs.read_float(), bs.read_float(), bs.read_float(),
                       bs.read_float(), bs.read_float(), bs.read_float()]
                face.uvs = [
                    MVector(uvs[0], uvs[3]),
                    MVector(uvs[1], uvs[4]),
                    MVector(uvs[2], uvs[5])
                ]
                self.faces.append(face)


class SO:
    def __init__(self):
        self.submeshes = []
        self.indices = []
        self.vertices = []

    def read(self, data):
        self.name = data.name
        if self.name == '':
            self.name = 'noname'

        material_faces = {}  # faces by material
        for face in data.faces:
            if face.material not in material_faces:
                material_faces[face.material] = []
            material_faces[face.material].append(face)

        for material in material_faces:
            uvs = {}
            indices = []

            # incides for this submesh + build uv maps
            for face in material_faces[material]:
                for i in range(0, 3):
                    index = face.indices[i]
                    indices.append(index)
                    uvs[index] = face.uvs[i]

            # vertex range
            min_vertex = min(indices)
            max_vertex = max(indices)

            # vertices for this submesh
            vertices = []
            for i in range(min_vertex, max_vertex+1):
                vertex = SOVertex()
                vertex.position = data.vertices[i]
                vertex.uv = uvs[i]
                vertices.append(vertex)

            # normalize indices
            for i in range(0, i < len(indices)):
                indices[i] -= min_vertex

            # build SOSubmesh
            submesh = SOSubmesh()
            submesh.name = material
            submesh.vertex_start = len(self.vertices)
            submesh.vertex_count = len(vertices)
            self.vertices += vertices
            submesh.index_start = len(self.indices)
            submesh.index_count = len(indices)
            self.indices += indices
            self.submeshes.append(submesh)

    def load(self):
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

        # dag_path and name
        mesh_dag_path = MDagPath()
        mesh.getPath(mesh_dag_path)
        mesh.setName(self.name)
        transform_node = MFnTransform(mesh.parent(0))
        transform_node.setName(f'mesh_{self.name}')

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

        mesh.updateSurface()


def db():
    # uv not working
    so_data = SOData()
    so_data.read_scb('D:\\katarina_base_blade.scb')
    so = SO()
    so.read(so_data)
    so.load()
    return
    so_data = SOData()
    so_data.read_sco('D:\\katarina_blade.sco')
    so2 = SO()
    so2.read(so_data)
    so2.load()
