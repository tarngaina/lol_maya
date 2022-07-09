from maya.OpenMaya import *
from maya.OpenMayaAnim import *
from maya.OpenMayaMPx import *
from math import sqrt
from struct import pack, unpack
from random import choice


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

    def reader(self, file, options, access):
        path = file.expandedFullName()
        if not path.endswith('.skn'):
            path += '.skn'

        skn = SKN()
        skn.read(path)
        skn.flip()

        # check options to read skl
        skl = None
        if 'skl=0' not in options:
            mfo = MFileObject()
            mfo.setRawFullName(path.split('.skn')[0] + '.skl')
            if mfo.exists():
                skl = SKL()
                skl.read(mfo.rawFullName())
                skl.flip()
                skl.load()

        skn.load(skl)
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
        path = file.expandedFullName()
        if not path.endswith('.skl'):
            path += '.skl'

        skl = SKL()
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
                f'[SkinTranslator.writer()]: Stop! u violated the law, use Export Selection or i violate u UwU.')

        path = file.rawFullName()
        if not path.endswith('.skn'):
            path += '.skn'

        # read riot.skl
        riot = None
        mfo = MFileObject()
        mfo.setRawFullName('/'.join(path.split('/')[:-1]+['riot.skl']))
        if mfo.exists():
            riot = SKL()
            riot.read(mfo.rawFullName())

        skl = SKL()
        skl.dump(riot)
        skl.flip()
        skl.write(path.split('.skn')[0] + '.skl')

        skn = SKN()
        skn.dump(skl)
        skn.flip()
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
        path = file.expandedFullName()
        if not path.endswith('.anm'):
            path += '.anm'
        anm = ANM()
        anm.read(path)
        anm.flip()
        anm.load()
        return True

    def writer(self, file, options, access):
        if access != MPxFileTranslator.kExportAccessMode:
            raise FunnyError(
                f'[ANMTranslator.writer()]: Stop! u violated the law, use Export All or i violate u UwU.')

        path = file.expandedFullName()
        if not path.endswith('.anm'):
            path += '.anm'

        anm = ANM()
        anm.dump()
        anm.flip()
        anm.write(path)
        return True


class SCOTranslator(MPxFileTranslator):
    name = 'League of Legends: SCO'
    ext = 'sco'

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
        path = file.expandedFullName()
        if not path.endswith('.sco'):
            path += '.sco'

        so = SO()
        so.read_sco(path)
        so.flip()
        so.load()
        return True

    def writer(self, file, options, access):
        if access != MPxFileTranslator.kExportActiveAccessMode:
            raise FunnyError(
                f'[SCO.writer()]: Stop! u violated the law, use Export Selection or i violate u UwU.')

        path = file.expandedFullName()
        if not path.endswith('.sco'):
            path += '.sco'

        so = SO()
        so.dump()
        so.flip()
        so.write_sco(path)
        return True


class SCBTranslator(MPxFileTranslator):
    name = 'League of Legends: SCB'
    ext = 'scb'

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
        path = file.expandedFullName()
        if not path.endswith('.scb'):
            path += '.scb'

        so = SO()
        so.read_scb(path)
        so.flip()
        so.load()
        return True

    def writer(self, file, options, access):
        if access != MPxFileTranslator.kExportActiveAccessMode:
            raise FunnyError(
                f'[SCB.writer()]: Stop! u violated the law, use Export Selection or i violate u UwU.')

        path = file.expandedFullName()
        if not path.endswith('.scb'):
            path += '.scb'

        so = SO()
        so.dump()
        so.flip()
        so.write_scb(path)
        return True


class MAPGEOTranslator(MPxFileTranslator):
    name = 'League of Legends: MAPGEO'
    ext = 'mapgeo'

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
        path = file.expandedFullName()
        if not path.endswith('.mapgeo'):
            path += '.mapgeo'

        # if there is a .py to control loading textures, attemp to read that .py
        # the assets folder must be in same place
        mgbin = None
        mfo = MFileObject()
        mfo.setRawFullName(path.split('.mapgeo')[0] + '.materials.py')
        if mfo.exists():
            mgbin = MAPGEOBin()
            mgbin.read(mfo.rawFullName())

        mg = MAPGEO()
        mg.read(path)
        mg.flip()
        mg.load(mgbin)
        return True

    def writer(self, file, options, access):
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
            'SKNTranslatorOpts',
            'skl=1',
            True
        )
    except Exception as e:
        MGlobal.displayWarning(
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
        MGlobal.displayWarning(
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
        MGlobal.displayWarning(
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
        MGlobal.displayWarning(
            f'Couldn\'t register ANMTranslator: [{e}]: {e.message}')

    try:
        plugin.registerFileTranslator(
            SCOTranslator.name,
            None,
            SCOTranslator.creator,
            None,
            None,
            True
        )
    except Exception as e:
        MGlobal.displayWarning(
            f'Couldn\'t register SCOTranslator: [{e}]: {e.message}')

    try:
        plugin.registerFileTranslator(
            SCBTranslator.name,
            None,
            SCBTranslator.creator,
            None,
            None,
            True
        )
    except Exception as e:
        MGlobal.displayWarning(
            f'Couldn\'t register SCBTranslator: [{e}]: {e.message}')

    try:
        plugin.registerFileTranslator(
            MAPGEOTranslator.name,
            None,
            MAPGEOTranslator.creator,
            None,
            None,
            True
        )
    except Exception as e:
        MGlobal.displayWarning(
            f'Couldn\'t register MAPGEOTranslator: [{e}]: {e.message}')


def uninitializePlugin(obj):
    plugin = MFnPlugin(obj)
    try:
        plugin.deregisterFileTranslator(
            SKNTranslator.name
        )
    except Exception as e:
        MGlobal.displayWarning(
            f'Couldn\'t deregister SKNTranslator: [{e}]: {e.message}')

    try:
        plugin.deregisterFileTranslator(
            SKLTranslator.name
        )
    except Exception as e:
        MGlobal.displayWarning(
            f'Couldn\'t deregister SKLTranslator: [{e}]: {e.message}')

    try:
        plugin.deregisterFileTranslator(
            SkinTranslator.name
        )
    except Exception as e:
        MGlobal.displayWarning(
            f'Couldn\'t deregister SkinTranslator: [{e}]: {e.message}')

    try:
        plugin.deregisterFileTranslator(
            ANMTranslator.name
        )
    except Exception as e:
        MGlobal.displayWarning(
            f'Couldn\'t deregister ANMTranslator: [{e}]: {e.message}')

    try:
        plugin.deregisterFileTranslator(
            SCOTranslator.name
        )
    except Exception as e:
        MGlobal.displayWarning(
            f'Couldn\'t deregister SCOTranslator: [{e}]: {e.message}')

    try:
        plugin.deregisterFileTranslator(
            SCBTranslator.name
        )
    except Exception as e:
        MGlobal.displayWarning(
            f'Couldn\'t deregister SCBTranslator: [{e}]: {e.message}')

    try:
        plugin.deregisterFileTranslator(
            MAPGEOTranslator.name
        )
    except Exception as e:
        MGlobal.displayWarning(
            f'Couldn\'t deregister MAPGEOTranslator: [{e}]: {e.message}')


# helper functions and structures
class FunnyError(Exception):
    ignore_button = 'Ignore'

    def __init__(self, message, consider=None):
        self.consider = consider
        self.cmd = self.show(message)

    def show(self, message):
        title = 'Error:'
        if ']: ' in message:
            temp = message.split(']: ')
            title = temp[0][1:] + ':'
            message = temp[1]
        message = repr(message)[1:-1]
        button = choice(
            ['UwU', '<(\")', 'ok boomer', 'funny man', 'jesus', 'bruh',
             'stop', 'get some help', 'haha', 'lmao', 'ay yo', 'SUS', 'sOcIEtY.'])
        dialog = f'confirmDialog -title "{title}" -message "{message}" -button "{button}" -icon "critical"  -defaultButton "{button}"'
        if self.consider:
            dialog += f' -button "{FunnyError.ignore_button}"'

        return MGlobal.executeCommandStringResult(dialog)


# for read/write file in a binary way
# totally not copied code
class BinaryStream:
    def __init__(self, f):
        self.stream = f

    def seek(self, pos, mode=0):
        self.stream.seek(pos, mode)

    def tell(self):
        return self.stream.tell()

    def pad(self, length):
        self.stream.seek(length, 1)

    def read_byte(self):
        return self.stream.read(1)

    def read_bytes(self, length):
        return self.stream.read(length)

    def read_bool(self):
        return unpack('?', self.stream.read(1))[0]

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


# for decompress v5 anm decompress vecs / quats
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
            return MVector(res)

# for set skl joint transform (transformation matrix)


class MTransform():
    def decompose(transform, space):
        # get translation, scale and rotation (quaternion) out of transformation matrix

        util = MScriptUtil()
        util.createFromDouble(0.0, 0.0, 0.0)
        ptr = util.asDoublePtr()
        transform.getScale(ptr, space)

        util_x = MScriptUtil()
        ptr_x = util_x.asDoublePtr()
        util_y = MScriptUtil()
        ptr_y = util_y.asDoublePtr()
        util_z = MScriptUtil()
        ptr_z = util_z.asDoublePtr()
        util_w = MScriptUtil()
        ptr_w = util_w.asDoublePtr()
        transform.getRotationQuaternion(ptr_x, ptr_y, ptr_z, ptr_w, space)

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
        transform.setTranslation(translation, space)

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

        # just id, not actual parent, especially not asian parent
        self.parent = None

        # fuck transform matrix
        self.local_translation = None
        self.local_scale = None
        self.local_rotation = None

        # yeah its actually inversed global, not global
        # for dumping only
        self.iglobal_translation = None
        self.iglobal_scale = None
        self.iglobal_rotation = None

        # for converting legacy skl
        self.global_matrix = None

        # for finding joint
        self.dagpath = None


class SKL:
    def __init__(self):
        self.joints = []

        # for loading as skincluster
        self.influences = []

    def flip(self):
        # flip the L with R: https://youtu.be/2yzMUs3badc
        for joint in self.joints:
            # local
            joint.local_translation.x *= -1.0
            joint.local_rotation.y *= -1.0
            joint.local_rotation.z *= -1.0
            # inversed global
            if joint.iglobal_translation:
                joint.iglobal_translation.x *= -1.0
                joint.iglobal_rotation.y *= -1.0
                joint.iglobal_rotation.z *= -1.0

    def read(self, path):
        with open(path, 'rb') as f:
            bs = BinaryStream(f)
            bs.pad(4)  # resource size
            magic = bs.read_uint32()
            if magic == 0x22FD4FC3:
                # new skl data
                version = bs.read_uint32()
                if version != 0:
                    raise FunnyError(
                        f'[SKL.read()]: Unsupported file version: {version}')

                bs.pad(2)  # flags
                joint_count = bs.read_uint16()
                influence_count = bs.read_uint32()
                joints_offset = bs.read_int32()
                bs.pad(4)  # joint indices offset
                influences_offset = bs.read_int32()
                # name offset, asset name offset, joint names offset, 5 reserved offset
                bs.pad(4*8)

                # read joints
                if joints_offset > 0 and joint_count > 0:
                    bs.seek(joints_offset)
                    for i in range(0, joint_count):
                        joint = SKLJoint()

                        bs.pad(2+2)  # flag and id
                        joint.parent = bs.read_int16()  # cant be uint
                        bs.pad(2)  # flags
                        joint_hash = bs.read_uint32()
                        bs.pad(4)  # radius

                        # local
                        joint.local_translation = bs.read_vec3()
                        joint.local_scale = bs.read_vec3()
                        joint.local_rotation = bs.read_quat()

                        # inversed global - no need when readinh
                        # translation, scale, rotation (quat))
                        bs.pad(12+12+16)

                        # name
                        joint_name_offset = bs.read_int32()
                        return_offset = bs.tell()
                        bs.seek(return_offset - 4 + joint_name_offset)
                        joint.name = bs.read_zero_terminated_string()

                        # skl convert 0.1 fix before return
                        # (2 empty bytes asset name override on first joint)
                        if i == 0 and joint.name == '':
                            # pad 1 more
                            bs.pad(1)
                            # read the rest
                            joint.name = bs.read_zero_terminated_string()

                            # brute force unhash 2 letters
                            founds = []
                            table = '_abcdefighjklmnopqrstuvwxyz'
                            names = [
                                a+b+joint.name for a in table for b in table]
                            for name in names:
                                if Hash.elf(name) == joint_hash:
                                    founds.append(name.capitalize())

                            if len(founds) == 1:
                                joint.name = founds[0]
                            else:
                                msg = ' Sugest name: ' + \
                                    ', '.join(founds) if len(
                                        founds) > 1 else ''
                                MGlobal.displayWarning(
                                    f'[SKL.load()]: {joint.name} is bad joint name, please rename it.{msg}')

                        bs.seek(return_offset)
                        self.joints.append(joint)

                # read influences
                if influences_offset > 0 and influence_count > 0:
                    bs.seek(influences_offset)
                    for i in range(0, influence_count):
                        self.influences.append(bs.read_uint16())

                # i think that is all we need, reading joint_indices_offset, name and asset name doesnt help anything
            else:
                # legacy
                # because signature in old skl is first 8bytes
                # need to go back pos 0 to read 8bytes again
                bs.seek(0)

                magic = bs.read_bytes(8).decode('ascii')
                if magic != 'r3d2sklt':
                    raise FunnyError(
                        f'[SKL.read()]: Wrong file signature: {magic}')

                version = bs.read_uint32()
                if version not in [1, 2]:
                    raise FunnyError(
                        f'[SKL.read()]: Unsupported file version: {version}')

                bs.pad(4)  # designer id or skl id

                joint_count = bs.read_uint32()
                for i in range(0, joint_count):
                    joint = SKLJoint()

                    joint.name = bs.read_padded_string(32)
                    joint.parent = bs.read_int32()  # -1, cant be uint
                    bs.pad(4)  # radius/scale - pad
                    py_list = list([0.0] * 16)
                    for i in range(0, 3):
                        for j in range(0, 4):
                            py_list[j*4+i] = bs.read_float()
                    py_list[15] = 1.0
                    matrix = MMatrix()
                    MScriptUtil.createMatrixFromList(py_list, matrix)
                    joint.global_matrix = matrix

                    self.joints.append(joint)

                # read influences
                if version == 1:
                    self.influences = list(range(0, joint_count))

                if version == 2:
                    influence_count = bs.read_uint32()
                    for i in range(0, influence_count):
                        self.influences.append(bs.read_uint32())

                # convert old skl to new skl
                for joint in self.joints:
                    if joint.parent == -1:
                        joint.local_translation, joint.local_scale, joint.local_rotation = MTransform.decompose(
                            MTransformationMatrix(joint.global_matrix),
                            MSpace.kWorld
                        )
                    else:
                        joint.local_translation, joint.local_scale, joint.local_rotation = MTransform.decompose(
                            MTransformationMatrix(
                                joint.global_matrix * self.joints[joint.parent].global_matrix.inverse()),
                            MSpace.kWorld
                        )

    def load(self):
        # find joint existed in scene
        iterator = MItDag(MItDag.kDepthFirst, MFn.kJoint)
        while not iterator.isDone():
            dagpath = MDagPath()
            iterator.getPath(dagpath)
            ik_joint = MFnIkJoint(dagpath)
            joint_name = ik_joint.name()
            for joint in self.joints:
                if joint_name == joint.name:
                    joint.dagpath = dagpath
                    break
            iterator.next()

        # create joint if not existed
        # set transform
        for joint in self.joints:
            if joint.dagpath:
                ik_joint = MFnIkJoint(joint.dagpath)
            else:
                ik_joint = MFnIkJoint()
                ik_joint.create()
                ik_joint.setName(joint.name)
                # dagpath
                joint.dagpath = MDagPath()
                ik_joint.getPath(joint.dagpath)

            ik_joint.set(MTransform.compose(
                joint.local_translation, joint.local_scale, joint.local_rotation, MSpace.kWorld
            ))

        for joint in self.joints:
            if joint.parent > -1:
                parent_node = MFnIkJoint(self.joints[joint.parent].dagpath)
                child_node = MFnIkJoint(joint.dagpath)
                if not parent_node.isParentOf(child_node.object()):
                    parent_node.addChild(child_node.object())

    def dump(self, riot=None):
        # dump ik joint
        iterator = MItDag(MItDag.kDepthFirst, MFn.kJoint)
        while not iterator.isDone():
            joint = SKLJoint()

            # dagpath
            joint.dagpath = MDagPath()
            iterator.getPath(joint.dagpath)

            # name + transform
            ik_joint = MFnIkJoint(joint.dagpath)
            joint.name = ik_joint.name()
            joint.local_translation, joint.local_scale, joint.local_rotation = MTransform.decompose(
                MTransformationMatrix(ik_joint.transformationMatrix()),
                MSpace.kTransform
            )
            joint.iglobal_translation, joint.iglobal_scale, joint.iglobal_rotation = MTransform.decompose(
                MTransformationMatrix(joint.dagpath.inclusiveMatrixInverse()),
                MSpace.kWorld
            )

            self.joints.append(joint)
            iterator.next()

        # sort joints to match riot.skl joints order
        if riot:
            MGlobal.displayInfo(
                '[SKL.dump(riot.skl)]: Found riot.skl, sorting joints...')

            new_joints = []
            joint_count = len(self.joints)
            riot_joint_count = len(riot.joints)
            # for adding extra joint at the end of list
            flags = list([True] * joint_count)

            # find riot joint in scene
            for riot_joint in riot.joints:
                riot_joint_name = riot_joint.name.lower()
                found = False
                for i in range(0, joint_count):
                    if flags[i] and self.joints[i].name.lower() == riot_joint_name:
                        new_joints.append(self.joints[i])
                        flags[i] = False
                        found = True
                        break

                # if not found riot join in current scene -> not enough joints to match riot joints -> bad
                if not found:
                    MGlobal.displayWarning(
                        f'[SKL.dump(riot.skl)]: Missing riot joint: {riot_joint.name}')

            # joint in scene = riot joint: good
            # joint in scene < riot joint: bad, might not work
            new_joint_count = len(new_joints)
            if new_joint_count < riot_joint_count:
                MGlobal.displayWarning(
                    f'[SKL.dump(riot.skl)]: Missing {riot_joint_count - new_joint_count} joints compared to riot.skl, joints order might be wrong.')
            else:
                MGlobal.displayInfo(
                    f'[SKL.dump(riot.skl)]: Successfully matched {new_joint_count} joints with riot.skl.')

            # add extra/addtional joints to the end of list
            # animation layer weight value for those joint will auto be 1.0
            for i in range(0, joint_count):
                if flags[i]:
                    new_joints.append(self.joints[i])
                    flags[i] = False
                    MGlobal.displayInfo(
                        f'[SKL.dump(riot.skl)]: New joints: {self.joints[i].name}')

            # assign new list
            self.joints = new_joints

        # link parent
        joint_count = len(self.joints)
        for joint in self.joints:
            ik_joint = MFnIkJoint(joint.dagpath)
            if ik_joint.parentCount() == 1 and ik_joint.parent(0).apiType() == MFn.kJoint:
                # get parent dagpath of this joint node
                parent_dagpath = MDagPath()
                MFnIkJoint(ik_joint.parent(0)).getPath(parent_dagpath)

                # find parent id by parent dagpath
                for i in range(joint_count):
                    if self.joints[i].dagpath == parent_dagpath:
                        joint.parent = i
                        break
            else:
                # must be batman
                joint.parent = -1

        # check limit joint
        joint_count = len(self.joints)
        if joint_count > 256:
            raise FunnyError(
                f'[SKL.dump()]: Too many joints found: {joint_count}, max allowed: 256 joints.')

        # remove namespace
        for joint in self.joints:
            if ':' in joint.name:
                joint.name = joint.name.split(':')[-1]

    def write(self, path):
        with open(path, 'wb') as f:
            bs = BinaryStream(f)

            bs.write_uint32(0)  # resource size - later
            bs.write_uint32(0x22FD4FC3)  # magic
            bs.write_uint32(0)  # version

            len1235 = len(self.joints)

            bs.write_uint16(0)  # flags
            bs.write_uint16(len1235)  # joints
            bs.write_uint32(len1235)  # influences

            joints_offset = 64
            joint_indices_offset = joints_offset + len1235 * 100
            influences_offset = joint_indices_offset + len1235 * 8
            joint_names_offset = influences_offset + len1235 * 2

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
            bs.seek(joint_names_offset)
            for i in range(0, len1235):
                joint_offset[i] = bs.tell()
                bs.write_bytes(self.joints[i].name.encode('ascii'))
                bs.write_bytes(bytes([0]))  # pad

            bs.seek(joints_offset)
            for i in range(0, len1235):
                joint = self.joints[i]

                bs.write_uint16(0)  # flags
                bs.write_uint16(i)  # id
                bs.write_int16(joint.parent)  # -1, cant be uint
                bs.write_uint16(0)  # pad
                bs.write_uint32(Hash.elf(joint.name))
                bs.write_float(2.1)  # radius/scale
                # local
                bs.write_vec3(joint.local_translation)
                bs.write_vec3(joint.local_scale)
                bs.write_quat(joint.local_rotation)
                # inversed global
                bs.write_vec3(joint.iglobal_translation)
                bs.write_vec3(joint.iglobal_scale)
                bs.write_quat(joint.iglobal_rotation)

                bs.write_int32(joint_offset[i] - bs.tell())

            # influences v1: 0, 1, 2... -> len(joints)
            bs.seek(influences_offset)
            for i in range(0, len1235):
                bs.write_uint16(i)

            # joint indices
            bs.seek(joint_indices_offset)
            for i in range(0, len1235):
                bs.write_uint16(i)
                bs.write_uint16(0)  # pad
                bs.write_uint32(Hash.elf(joint.name))

            # resource size
            bs.seek(0, 2)
            fsize = bs.tell()
            bs.seek(0)
            bs.write_uint32(fsize)


# skn
class SKNVertex:
    def __init__(self):
        self.position = None
        self.influences = None
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

        # for loading
        self.name = None

    def flip(self):
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
                    f'[SKN.read()]: Wrong signature file: {magic}')

            major = bs.read_uint16()
            minor = bs.read_uint16()
            if major not in [0, 2, 4] and minor != 1:
                raise FunnyError(
                    f'[SKN.read()]: Unsupported file version: {major}.{minor}')

            self.name = path.split('/')[-1].split('.')[0]
            vertex_type = 0
            if major == 0:
                # version 0 doesn't have submesh data
                index_count = bs.read_uint32()
                vertex_count = bs.read_uint32()

                submesh = SKNSubmesh()
                submesh.name = 'Base'
                submesh.vertex_start = 0
                submesh.vertex_count = vertex_count
                submesh.index_start = 0
                submesh.index_count = index_count
                self.submeshes.append(submesh)
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

                if major == 4:
                    bs.pad(4)  # flags

                index_count = bs.read_uint32()
                vertex_count = bs.read_uint32()

                # junk stuff from version 4
                if major == 4:  # pad all this, cause we dont need?
                    bs.pad(4)  # vertex size
                    vertex_type = bs.read_uint32()
                    bs.pad(24)  # bouding box: 2 vec3 min-max
                    # bouding sphere: vec3 central + float radius
                    bs.pad(12 + 4)

            if index_count % 3 > 0:
                raise FunnyError(
                    f'[SKN.read()]: Bad indices data: {index_count}')

            # read indices by face
            face_count = index_count // 3
            for i in range(0, face_count):
                face = (bs.read_uint16(), bs.read_uint16(),
                        bs.read_uint16())
                # check dupe index in a face
                if not (face[0] == face[1] or face[1] == face[2] or face[2] == face[0]):
                    self.indices += face

            # read vertices
            for i in range(0, vertex_count):
                vertex = SKNVertex()
                vertex.position = bs.read_vec3()
                vertex.influences = [
                    bs.read_byte(), bs.read_byte(), bs.read_byte(), bs.read_byte()]
                vertex.weights = [
                    bs.read_float(), bs.read_float(), bs.read_float(), bs.read_float()]
                vertex.normal = bs.read_vec3()
                vertex.uv = bs.read_vec2()
                # 0: basic, 1: color, 2: color and tangent
                if vertex_type >= 1:
                    # pad 4 byte color
                    bs.pad(4)
                    if vertex_type == 2:
                        # pad vec4 tangent
                        bs.pad(16)
                self.vertices.append(vertex)

    def load(self, skl=None):
        vertex_count = len(self.vertices)
        index_count = len(self.indices)
        face_count = index_count // 3

        # create mesh
        vertices = MFloatPointArray()
        u_values = MFloatArray()
        v_values = MFloatArray()
        normals = MVectorArray()
        for vertex in self.vertices:
            vertices.append(MFloatPoint(
                vertex.position.x, vertex.position.y, vertex.position.z))
            u_values.append(vertex.uv.x)
            v_values.append(1.0 - vertex.uv.y)
            normals.append(vertex.normal)

        poly_count = MIntArray(face_count, 3)
        poly_indices = MIntArray()
        MScriptUtil.createIntArrayFromList(self.indices, poly_indices)
        normal_indices = MIntArray()
        MScriptUtil.createIntArrayFromList(
            list(range(len(self.vertices))), normal_indices)

        mesh = MFnMesh()
        mesh.create(
            vertex_count,
            face_count,
            vertices,
            poly_count,
            poly_indices,
            u_values,
            v_values
        )
        mesh.assignUVs(
            poly_count, poly_indices
        )
        mesh.setVertexNormals(
            normals,
            normal_indices
        )

        # name
        mesh.setName(self.name)
        mesh_name = mesh.name()
        MFnTransform(mesh.parent(0)).setName(f'mesh_{self.name}')

        # materials
        for submesh in self.submeshes:
            # check duplicate name node
            if skl:
                for joint in skl.joints:
                    if joint.name == submesh.name:
                        submesh.name = submesh.name.lower()
                        break

            # lambert material
            lambert = MFnLambertShader()
            lambert.create()
            lambert.setName(submesh.name)
            lambert_name = lambert.name()
            # shading group
            face_start = submesh.index_start // 3
            face_end = (submesh.index_start + submesh.index_count) // 3
            MGlobal.executeCommand((
                # create renderable, independent shading group
                f'sets -renderable true -noSurfaceShader true -empty -name "{lambert_name}_SG";'
                # add submesh faces to shading group
                f'sets -e -forceElement "{lambert_name}_SG" {mesh_name}.f[{face_start}:{face_end}];'
            ))
            # connect lambert to shading group
            MGlobal.executeCommand(
                f'connectAttr -f {lambert_name}.outColor {lambert_name}_SG.surfaceShader;')

        if skl:
            influence_count = len(skl.influences)
            mesh_dagpath = MDagPath()
            mesh.getPath(mesh_dagpath)

            # select mesh + joint
            selections = MSelectionList()
            selections.add(mesh_dagpath)
            for influence in skl.influences:
                selections.add(skl.joints[influence].dagpath)
            MGlobal.selectCommand(selections)

            # bind selections
            MGlobal.executeCommand(
                f'skinCluster -mi 4 -tsb -n {self.name}_skinCluster')

            # get skin cluster
            in_mesh = mesh.findPlug('inMesh')
            in_mesh_connections = MPlugArray()
            in_mesh.connectedTo(in_mesh_connections, True, False)
            skin_cluster = MFnSkinCluster(in_mesh_connections[0].node())
            skin_cluster_name = skin_cluster.name()

            # mask influence
            influences_dagpath = MDagPathArray()
            skin_cluster.influenceObjects(influences_dagpath)
            mask_influence = MIntArray(influence_count)
            for i in range(0, influence_count):
                dagpath = skl.joints[skl.influences[i]].dagpath

                for j in range(0, influence_count):
                    if dagpath == influences_dagpath[j]:
                        mask_influence[i] = j
                        break

            # weights
            MGlobal.executeCommand(
                f'setAttr {skin_cluster_name}.normalizeWeights 0')

            component = MFnSingleIndexedComponent()
            vertex_component = component.create(MFn.kMeshVertComponent)
            vertex_indices = MIntArray()
            MScriptUtil.createIntArrayFromList(
                list(range(0, vertex_count)), vertex_indices)
            component.addElements(vertex_indices)

            weights = MDoubleArray(vertex_count * influence_count)
            for i in range(0, vertex_count):
                vertex = self.vertices[i]
                for j in range(0, 4):
                    weight = vertex.weights[j]
                    influence = vertex.influences[j][0]
                    if weight > 0:
                        weights[i * influence_count + influence] = weight
            skin_cluster.setWeights(
                mesh_dagpath, vertex_component, mask_influence, weights, False)

            MGlobal.executeCommand(
                f'setAttr {skin_cluster_name}.normalizeWeights 1')
            MGlobal.executeCommand(
                f'skinPercent -normalize true {skin_cluster_name} {mesh_name}')

        # shud be final line
        mesh.updateSurface()

    def dump(self, skl):
        # get mesh in selections
        selections = MSelectionList()
        MGlobal.getActiveSelectionList(selections)
        iterator = MItSelectionList(selections, MFn.kMesh)
        if iterator.isDone():
            raise FunnyError(f'[SKN.dump()]: Please select a mesh.')
        mesh_dagpath = MDagPath()
        iterator.getDagPath(mesh_dagpath)  # get first mesh
        iterator.next()
        if not iterator.isDone():
            raise FunnyError(
                f'[SKN.dump()]: More than 1 mesh selected, combine all meshes if you have mutiple meshes.')
        mesh = MFnMesh(mesh_dagpath)

        # get shaders
        shaders = MObjectArray()
        poly_shaders = MIntArray()
        instance_num = mesh_dagpath.instanceNumber() if mesh_dagpath.isInstanced() else 0
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

        # check non-triangulated polygon
        # check face has multiple shaders
        vertex_shaders = MIntArray(vertices_num, -1)
        iterator = MItMeshPolygon(mesh_dagpath)
        iterator.reset()
        while not iterator.isDone():
            if not iterator.hasValidTriangulation():
                raise FunnyError(
                    f'[SKN.dump({mesh.name()})]: Mesh contains a invalid triangulation polygon.')

            index = iterator.index()
            shader_index = poly_shaders[index]
            if shader_index == -1:
                raise FunnyError(
                    f'[SKN.dump({mesh.name()})]: Mesh contains a face with no material assigned, make sure you assigned material to all faces.')

            vertices = MIntArray()
            iterator.getVertices(vertices)
            len69 = vertices.length()
            for i in range(0, len69):
                if shader_count > 1 and vertex_shaders[vertices[i]] not in [-1, shader_index]:
                    raise FunnyError(
                        f'[SKN.dump({mesh.name()})]: Mesh contains a vertex with multiple materials assigned.')
                vertex_shaders[vertices[i]] = shader_index

            iterator.next()

        # find skin cluster
        in_mesh = mesh.findPlug('inMesh')
        in_mesh_connections = MPlugArray()
        in_mesh.connectedTo(in_mesh_connections, True, False)
        if in_mesh_connections.length() > 0:
            if in_mesh_connections[0].node().apiType() != MFn.kSkinClusterFilter:
                raise FunnyError((
                    f'[SKN.dump({mesh.name()})]: History changed on the skin cluster.\n'
                    'Try one of following methods to fix it:\n'
                    '1. Delete non-Deformer history.\n'
                    '2. Export as FBX -> New scene -> Import FBX in -> Export as SKN\n'
                    '3. Save scene -> Unbind skin -> Delete all history -> Rebind skin -> Copy weight back.'
                ))
        else:
            raise FunnyError(
                f'[SKN.dump({mesh.name()})]: No skin cluster found, make sure you bound the skin.')
        skin_cluster = MFnSkinCluster(in_mesh_connections[0].node())

        # mask influence
        influences_dagpath = MDagPathArray()
        influence_count = skin_cluster.influenceObjects(influences_dagpath)
        mask_influence = MIntArray(influence_count)
        joint_count = len(skl.joints)
        for i in range(0, influence_count):
            dagpath = influences_dagpath[i]
            for j in range(0, joint_count):
                if dagpath == skl.joints[j].dagpath:
                    mask_influence[i] = j
                    break

        # get weights
        component = MFnSingleIndexedComponent()
        vertex_component = component.create(MFn.kMeshVertComponent)
        vertex_indices = MIntArray()
        MScriptUtil.createIntArrayFromList(
            list(range(0, vertices_num)), vertex_indices)
        component.addElements(vertex_indices)
        weights = MDoubleArray()
        util = MScriptUtil()
        ptr = util.asUintPtr()
        skin_cluster.getWeights(mesh_dagpath, vertex_component, weights, ptr)
        weight_influence_count = util.getUint(ptr)

        #  weight stuffs
        bad_vertices = MIntArray()
        for i in range(0, vertices_num):
            # count influences
            inf_count = 0
            weight_sum = 0.0
            found_weights = []
            for j in range(0, weight_influence_count):
                weight = weights[i * weight_influence_count + j]
                if weight > 0:
                    inf_count += 1
                    weight_sum += weight
                    found_weights.append((weight, j))

            # 4+ influences fix
            if inf_count > 4:
                bad_vertices.append(i)

                # get sorted weights + influence
                found_weights = sorted(
                    found_weights, key=lambda f: f[0], reverse=True)

                # 4 highest weights
                high_weights = found_weights[:4]
                # the rest small weights
                low_weights = found_weights[4:]

                # sum all low weights and remove them
                low_sum = 0.0
                for weight, influence in low_weights:
                    low_sum += weight
                    weights[i * weight_influence_count + influence] = 0.0

                # distributed low weights to high weights + re calculate weight sum for normalize
                low_sum /= 4
                weight_sum = 0.0
                for weight, influence in high_weights:
                    weights[i * weight_influence_count + influence] += low_sum
                    weight_sum += weights[i *
                                          weight_influence_count + influence]

            # normalize weights
            if weight_sum > 0:
                for j in range(0, weight_influence_count):
                    weights[i * weight_influence_count + j] /= weight_sum

        if bad_vertices.length() > 0:
            e = FunnyError(
                (
                    f'[SKN.dump({mesh.name()})]: Mesh contains {bad_vertices.length()} vertices that have weight on 4+ influences, those vertices will be selected in scene.\n'
                    'Try one of following methods to fix it:\n'
                    '1. Repaint weight on those vertices\n'
                    '2. Prune small weights.\n'
                    '3. Ignore = auto fit all weights to 4 influences. (not recommended)'
                ),
                consider=True
            )
            if e.cmd != FunnyError.ignore_button:
                # select 4+ influences vertices
                component = MFnSingleIndexedComponent()
                temp_component = component.create(
                    MFn.kMeshVertComponent)
                component.addElements(bad_vertices)

                temp_selections = MSelectionList()
                temp_selections.add(mesh_dagpath, temp_component)
                MGlobal.selectCommand(temp_selections)
                raise e
            else:
                MGlobal.displayWarning(
                    f'[SKN.dump({mesh.name()})]: All vertices weights will be re-calculated to fit 4 influences.')

        # init some important thing
        shader_vertex_indices = []
        shader_vertices = []
        shader_indices = []
        for i in range(0, shader_count):
            shader_vertex_indices.append(MIntArray())
            shader_vertices.append([])
            shader_indices.append(MIntArray())

        # dump vertices
        bad_vertices2 = MIntArray()  # for no material vertex
        iterator = MItMeshVertex(mesh_dagpath)
        iterator.reset()
        while not iterator.isDone():
            index = iterator.index()
            shader = vertex_shaders[index]
            if shader == -1:
                if index not in bad_vertices2:
                    bad_vertices2.append(index)
                iterator.next()
                continue

            # position
            position = iterator.position(MSpace.kWorld)

            # influences and weights
            influences = [bytes([0]), bytes(
                [0]), bytes([0]), bytes([0])]
            temp_weights = [0.0, 0.0, 0.0, 0.0]
            f = 0
            j = 0
            while j < weight_influence_count and f < 4:
                weight = weights[index * weight_influence_count + j]
                if weight > 0:
                    influences[f] = bytes([mask_influence[j]])
                    temp_weights[f] = float(weight)
                    f += 1
                j += 1

            # normal - also normalized
            normals = MVectorArray()
            iterator.getNormals(normals)
            normal = MVector(0.0, 0.0, 0.0)
            len123 = normals.length()
            for i in range(0, len123):
                normal.x += normals[i].x
                normal.y += normals[i].y
                normal.z += normals[i].z
            normal.x /= len123
            normal.y /= len123
            normal.z /= len123

            # unique uv
            uv_indices = MIntArray()
            iterator.getUVIndices(uv_indices)
            len555 = uv_indices.length()

            if len555 > 0:
                seen = []
                for j in range(len555):
                    uv_index = uv_indices[j]
                    if not uv_index in seen:
                        if uv_index == -1:
                            raise FunnyError(
                                f'[SKN.dump({mesh.name()})]: Mesh contains an index with no UV assigned.')
                        else:
                            u_util = MScriptUtil()  # lay trua tren cao, turn down for this
                            u_ptr = u_util.asFloatPtr()
                            v_util = MScriptUtil()
                            v_ptr = v_util.asFloatPtr()
                            mesh.getUV(uv_index, u_ptr, v_ptr)
                            uv = MVector(
                                u_util.getFloat(u_ptr),
                                1.0 - v_util.getFloat(v_ptr)
                            )

                        # create SKNVertex - recreate pointer for safe
                        vertex = SKNVertex()
                        vertex.position = MVector(position)
                        vertex.influences = influences
                        vertex.weights = temp_weights
                        vertex.normal = MVector(normal)
                        vertex.uv = MVector(uv)
                        vertex.uv_index = uv_index

                        shader_vertices[shader].append(vertex)
                        shader_vertex_indices[shader].append(index)

                        seen.append(uv_index)
            else:
                raise FunnyError(
                    f'[SKN.dump({mesh.name()})]: Mesh contains a vertex with no UVs.')
            iterator.next()

        # show vertex with no material assigned
        if bad_vertices2.length() > 0:
            e = FunnyError(
                (
                    f'[SKN.dump({mesh.name()})]: Mesh contains {bad_vertices2.length()} vertices with no material assigned, those vertices will be selected in scene.\n'
                    'Try assign material to them or ignore (not recommended).'
                ),
                consider=True
            )
            if e.cmd != FunnyError.ignore_button:
                # select vertices with no material
                component = MFnSingleIndexedComponent()
                temp_component = component.create(MFn.kMeshVertComponent)
                component.addElements(bad_vertices2)

                temp_selections = MSelectionList()
                temp_selections.add(
                    mesh_dagpath, temp_component)
                MGlobal.selectCommand(temp_selections)
                raise e
            else:
                MGlobal.displayWarning(
                    f'[SKN.dump({mesh.name()})]: All vertices with no material assigned will be ignored.')

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
        iterator = MItMeshPolygon(mesh_dagpath)
        iterator.reset()
        len333 = len(self.vertices)
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
                    if data_index == -1 or data_index >= len333:
                        raise FunnyError(
                            f'[SKN.dump({mesh.name()})]: Data index out of range.')

                    for j in range(data_index, len333):
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

        # dump submeshes
        index_start = 0
        vertex_start = 0
        for i in range(0, shader_count):
            surface_shaders = MFnDependencyNode(
                shaders[i]).findPlug('surfaceShader')
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

        # check limit vertices
        vertices_count = max(self.indices)+1
        if vertices_count > 65536:
            raise FunnyError(
                f'[SKN.dump({mesh.name()})]: Too many vertices found: {vertices_count}, max allowed: 65536 vertices.')

        # remove namespace
        for submesh in self.submeshes:
            if ':' in submesh.name:
                submesh.name = submesh.name.split(':')[-1]

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
                for byte in vertex.influences:
                    bs.write_bytes(byte)
                for weight in vertex.weights:
                    bs.write_float(weight)
                bs.write_vec3(vertex.normal)
                bs.write_vec2(vertex.uv)


# anm
class ANMPose:
    def __init__(self):
        self.translation = None
        self.scale = None
        self.rotation = None

        # for dumping v4
        self.translation_index = None
        self.scale_index = None
        self.rotation_index = None


class ANMTrack:
    def __init__(self):
        self.joint_hash = None
        self.poses = {}

        # for loading
        self.joint_name = None
        self.dagpath = None


class ANM:
    def __init__(self):
        self.tracks = []
        self.fps = None
        self.duration = None  # normalized

        # for loading
        self.compressed = None

    def flip(self):
        # DO A FLIP!
        for track in self.tracks:
            for time in track.poses:
                pose = track.poses[time]
                if pose.translation:
                    pose.translation.x *= -1.0

                if pose.rotation:
                    pose.rotation.y *= -1.0
                    pose.rotation.z *= -1.0

    def read(self, path):
        with open(path, 'rb') as f:
            bs = BinaryStream(f)

            magic = bs.read_bytes(8).decode('ascii')
            version = bs.read_uint32()

            if magic == 'r3d2canm':
                bs.pad(4*3)  # resource size, format token, flags

                joint_count = bs.read_uint32()
                frame_count = bs.read_uint32()
                bs.pad(4)  # jump cache count

                max_time = bs.read_float()
                self.fps = bs.read_float()
                self.duration = max_time + 1 / self.fps

                bs.pad(24)  # 24 float of transform quantization properties

                translation_min = bs.read_vec3()
                translation_max = bs.read_vec3()
                scale_min = bs.read_vec3()
                scale_max = bs.read_vec3()

                frames_offset = bs.read_int32()
                bs.pad(4)  # jump caches offset
                joint_hashes_offset = bs.read_int32()

                if frames_offset <= 0:
                    raise FunnyError(
                        f'[ANM.read()]: File does not contain frames.'
                    )
                if joint_hashes_offset <= 0:
                    raise FunnyError(
                        f'[ANM.read()]: File does not contain joint hashes.'
                    )

                # read joint hashes
                bs.seek(joint_hashes_offset + 12)
                joint_hashes = []
                for i in range(0, joint_count):
                    joint_hashes.append(bs.read_uint32())

                # create tracks
                for i in range(0, joint_count):
                    track = ANMTrack()
                    track.joint_hash = joint_hashes[i]
                    self.tracks.append(track)

                bs.seek(frames_offset + 12)
                for i in range(0, frame_count):
                    compressed_time = bs.read_uint16()
                    bits = bs.read_uint16()
                    compressed_transform = bs.read_bytes(6)

                    # find track by joint hash
                    joint_hash = joint_hashes[bits & 16383]
                    for track in self.tracks:
                        if track.joint_hash == joint_hash:
                            break

                    # set/get pose at time
                    time = compressed_time / 65535.0 * max_time
                    if time not in track.poses:
                        pose = ANMPose()
                        track.poses[time] = pose
                    else:
                        pose = track.poses[time]

                    # decompress data and add to pose
                    transform_type = bits >> 14
                    if transform_type == 0:
                        pose.rotation = CTransform.Quat.decompress(
                            compressed_transform)
                    elif transform_type == 1:
                        pose.translation = CTransform.Vec.decompress(
                            translation_min, translation_max, compressed_transform)
                    elif transform_type == 2:
                        pose.scale = CTransform.Vec.decompress(
                            scale_min, scale_max, compressed_transform)
                    else:
                        raise FunnyError(
                            f'[ANM.read()]: Unknown compressed transform type: {transform_type}.'
                        )

            elif magic == 'r3d2anmd':
                if version == 5:
                    # v5

                    bs.pad(4*4)  # resource size, format token, version, flags

                    track_count = bs.read_uint32()
                    frame_count = bs.read_uint32()

                    self.fps = 1 / bs.read_float()  # frame duration
                    self.duration = frame_count / self.fps

                    joint_hashes_offset = bs.read_int32()
                    bs.pad(4+4)  # asset name offset, time offset
                    vecs_offset = bs.read_int32()
                    quats_offset = bs.read_int32()
                    frames_offset = bs.read_int32()

                    if joint_hashes_offset <= 0:
                        raise FunnyError(
                            f'[ANM.read()]: File does not contain joint hashes data.'
                        )
                    if vecs_offset <= 0:
                        raise FunnyError(
                            f'[ANM.read()]: File does not contain unique vectors data.'
                        )
                    if quats_offset <= 0:
                        raise FunnyError(
                            f'[ANM.read()]: File does not contain unique quaternions data.'
                        )
                    if frames_offset <= 0:
                        raise FunnyError(
                            f'[ANM.read()]: File does not contain frames data.'
                        )

                    joint_hash_count = (
                        frames_offset - joint_hashes_offset) // 4
                    vec_count = (quats_offset - vecs_offset) // 12
                    quat_count = (joint_hashes_offset - quats_offset) // 6

                    # read joint hashes
                    joint_hashes = []
                    bs.seek(joint_hashes_offset + 12)
                    for i in range(0, joint_hash_count):
                        joint_hashes.append(bs.read_uint32())

                    # read vecs
                    uni_vecs = []
                    bs.seek(vecs_offset + 12)
                    for i in range(0, vec_count):
                        uni_vecs.append(bs.read_vec3())

                    # read quats
                    uni_quats = []
                    bs.seek(quats_offset + 12)
                    for i in range(0, quat_count):
                        uni_quats.append(
                            CTransform.Quat.decompress(bs.read_bytes(6)))

                    # read frames
                    frames = []
                    bs.seek(frames_offset + 12)
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

                            # recreate pointer
                            pose = ANMPose()
                            pose.time = f
                            pose.translation = MVector(
                                uni_vecs[translation_index])
                            pose.scale = MVector(uni_vecs[scale_index])
                            pose.rotation = MQuaternion(
                                uni_quats[rotation_index])

                            # time = index / fps
                            index = f
                            track.poses[index / self.fps] = pose

                elif version == 4:
                    # v4

                    bs.pad(4*4)  # resource size, format token, version, flags

                    track_count = bs.read_uint32()
                    frame_count = bs.read_uint32()
                    self.fps = 1 / bs.read_float()  # frame duration
                    self.duration = frame_count / self.fps

                    # tracks offset, asset name offset, time offset
                    bs.pad(4 * 3)
                    vecs_offset = bs.read_int32()
                    quats_offset = bs.read_int32()
                    frames_offset = bs.read_int32()

                    if vecs_offset <= 0:
                        raise FunnyError(
                            f'[ANM.read()]: File does not contain unique vectors data.'
                        )
                    if quats_offset <= 0:
                        raise FunnyError(
                            f'[ANM.read()]: File does not contain unique quaternions data.'
                        )
                    if frames_offset <= 0:
                        raise FunnyError(
                            f'[ANM.read()]: File does not contain frames data.'
                        )

                    vec_count = (quats_offset - vecs_offset) // 12
                    quat_count = (frames_offset - quats_offset) // 16

                    bs.seek(vecs_offset + 12)
                    uni_vecs = []
                    for i in range(0, vec_count):
                        uni_vecs.append(bs.read_vec3())

                    bs.seek(quats_offset + 12)
                    uni_quats = []
                    for i in range(0, quat_count):
                        uni_quats.append(bs.read_quat())
                    bs.seek(frames_offset + 12)

                    frames = []
                    for i in range(0, frame_count * track_count):
                        frames.append((
                            bs.read_uint32(),  # joint hash
                            bs.read_uint16(),  # translation index
                            bs.read_uint16(),  # scale index
                            bs.read_uint16()  # rotation index
                        ))
                        bs.pad(2)  # pad

                    # parse data from frames
                    for joint_hash, translation_index, scale_index, rotation_index in frames:
                        # recreate pointer
                        pose = ANMPose()
                        pose.translation = MVector(uni_vecs[translation_index])
                        pose.scale = MVector(uni_vecs[scale_index])
                        pose.rotation = MQuaternion(uni_quats[rotation_index])

                        # find existed track with joint hash
                        track = None
                        for t in self.tracks:
                            if t.joint_hash == joint_hash:
                                track = t
                                break

                        # couldnt found track that has joint hash, create new
                        if not track:
                            track = ANMTrack()
                            track.joint_hash = joint_hash
                            self.tracks.append(track)

                        # time = index / fps
                        index = len(track.poses)
                        track.poses[index / self.fps] = pose

                else:
                    # legacy

                    bs.pad(4)  # skl id
                    track_count = bs.read_uint32()
                    frame_count = bs.read_uint32()

                    self.fps = bs.read_uint32()
                    self.duration = frame_count / self.fps

                    for i in range(0, track_count):
                        track = ANMTrack()
                        track.joint_hash = Hash.elf(bs.read_padded_string(32))
                        bs.pad(4)  # flags
                        for index in range(0, frame_count):
                            pose = ANMPose()
                            pose.rotation = bs.read_quat()
                            pose.translation = bs.read_vec3()
                            # legacy not support scaling
                            pose.scale = MVector(1.0, 1.0, 1.0)

                            # time = index / fps
                            track.poses[index / self.fps] = pose
                        self.tracks.append(track)
            else:
                raise FunnyError(
                    f'[ANM.read()]: Wrong signature file: {magic}')

    def load(self):
        # track of data joints that found in scene
        scene_tracks = []

        # loop through all ik joint in scenes
        iterator = MItDag(MItDag.kDepthFirst, MFn.kJoint)
        while not iterator.isDone():
            dagpath = MDagPath()
            iterator.getPath(dagpath)
            ik_joint = MFnIkJoint(dagpath)
            joint_name = ik_joint.name()

            # find joint in tracks data
            found = False
            for track in self.tracks:
                if track.joint_hash == Hash.elf(joint_name):
                    track.joint_name = joint_name
                    track.dagpath = MDagPath(dagpath)
                    scene_tracks.append(track)
                    found = True
                    break

            # ignore data joint that not found in scene
            if not found:
                MGlobal.displayWarning(
                    f'[ANM.load()]: No joint hash found in scene: {track.joint_hash}')

            iterator.next()

        if len(scene_tracks) == 0:
            raise FunnyError(
                '[ANM.load()]: No data joints found in scene, please import SKL if joints are not in scene.')

        # delete all channel data
        MGlobal.executeCommand('delete -all -c')

        # ensure scene fps
        # this only ensure the "import scene", not the "opening/existing scene" in maya, to make this work:
        # select "Override to Math Source" for both Framerate % Animation Range in Maya's import options panel
        if self.fps > 59:
            MGlobal.executeCommand('currentUnit -time ntscf')
        else:
            MGlobal.executeCommand('currentUnit -time ntsc')

        # adjust animation range
        end = self.duration * self.fps
        MGlobal.executeCommand(
            f'playbackOptions -e -min 0 -max {end} -animationStartTime 0 -animationEndTime {end} -playbackSpeed 1')

        # bind current pose to frame 0 - very helpful if its bind pose
        MGlobal.executeCommand(f'currentTime 0')
        joint_names = ' '.join(
            [track.joint_name for track in scene_tracks])
        MGlobal.executeCommand(
            f'setKeyframe -breakdown 0 -hierarchy none -controlPoints 0 -shape 0 -at translateX -at translateY -at translateZ -at scaleX -at scaleY -at scaleZ -at rotateX -at rotateY -at rotateZ {joint_names}')

        # sync global times
        times = []
        for track in scene_tracks:
            for time in track.poses:
                if time not in times:
                    times.append(time)
        for track in scene_tracks:
            for time in times:
                if time not in track.poses:
                    track.poses[time] = None

        for time in times:
            # anm will start from frame 1
            MGlobal.executeCommand(f'currentTime {time * self.fps + 1}')

            setKeyFrame = f'setKeyframe -breakdown 0 -hierarchy none -controlPoints 0 -shape 0'
            ekf = True  # empty keyframe
            for track in scene_tracks:
                pose = track.poses[time]
                if pose:
                    ik_joint = MFnIkJoint(track.dagpath)
                    # translation
                    if pose.translation:
                        ik_joint.setTranslation(
                            pose.translation, MSpace.kTransform)
                        setKeyFrame += f' {track.joint_name}.translateX {track.joint_name}.translateY {track.joint_name}.translateZ'
                        ekf = True
                    # scale
                    if pose.scale:
                        scale = pose.scale
                        util = MScriptUtil()
                        util.createFromDouble(scale.x, scale.y, scale.z)
                        ptr = util.asDoublePtr()
                        ik_joint.setScale(ptr)
                        setKeyFrame += f' {track.joint_name}.scaleX {track.joint_name}.scaleY {track.joint_name}.scaleZ'
                        ekf = True
                    # rotation
                    if pose.rotation:
                        orient = MQuaternion()
                        ik_joint.getOrientation(orient)
                        axe = ik_joint.rotateOrientation(MSpace.kTransform)
                        ik_joint.setRotation(
                            axe.inverse() * pose.rotation * orient.inverse(), MSpace.kTransform)
                        setKeyFrame += f' {track.joint_name}.rotateX {track.joint_name}.rotateY {track.joint_name}.rotateZ'
                        ekf = True

            if ekf:
                MGlobal.executeCommand(setKeyFrame)

        # slerp all quaternions - EULER SUCKS!
        rotationInterpolation = 'rotationInterpolation -c quaternionSlerp'
        for track in scene_tracks:
            rotationInterpolation += f' {track.joint_name}.rotateX {track.joint_name}.rotateY {track.joint_name}.rotateZ'
        MGlobal.executeCommand(rotationInterpolation)

    def dump(self):
        # get joint in scene
        iterator = MItDag(MItDag.kDepthFirst, MFn.kJoint)
        while not iterator.isDone():
            # dag path + transform
            dagpath = MDagPath()
            iterator.getPath(dagpath)
            ik_joint = MFnIkJoint(dagpath)

            # track data
            track = ANMTrack()
            track.dagpath = dagpath
            track.joint_name = ik_joint.name()
            track.joint_hash = Hash.elf(track.joint_name)
            self.tracks.append(track)

            iterator.next()

        # dump fps
        util = MScriptUtil()
        ptr = util.asDoublePtr()
        MGlobal.executeCommand('currentTimeUnitToFPS', ptr)
        fps = util.getDouble(ptr)
        if fps > 59:
            self.fps = 60.0
        else:
            self.fps = 30.0

        # dump from frame 1 to frame end
        # if its not then well, its the ppl fault, not mine. haha suckers
        util = MScriptUtil()
        ptr = util.asDoublePtr()
        MGlobal.executeCommand('playbackOptions -q -animationStartTime', ptr)
        start = util.getDouble(ptr)
        if start < 0:
            raise FunnyError(
                f'[ANM.dump()]: Animation start time must be greater or equal 0: {start}')
        util = MScriptUtil()
        ptr = util.asDoublePtr()
        MGlobal.executeCommand('playbackOptions -q -animationEndTime', ptr)
        end = util.getDouble(ptr)
        if end < 1:
            raise FunnyError(
                f'[ANM.dump()]: Animation end time must be greater than 1: {end}')
        self.duration = int(end)

        for time in range(1, self.duration+1):
            MGlobal.executeCommand(f'currentTime {time}')

            for track in self.tracks:
                ik_joint = MFnIkJoint(track.dagpath)

                pose = ANMPose()
                # translation
                pose.translation = ik_joint.getTranslation(MSpace.kTransform)
                # scale
                util = MScriptUtil()
                util.createFromDouble(0.0, 0.0, 0.0)
                ptr = util.asDoublePtr()
                ik_joint.getScale(ptr)
                pose.scale = MVector(
                    util.getDoubleArrayItem(ptr, 0),
                    util.getDoubleArrayItem(ptr, 1),
                    util.getDoubleArrayItem(ptr, 2)
                )
                # rotation
                orient = MQuaternion()
                ik_joint.getOrientation(orient)
                axe = ik_joint.rotateOrientation(MSpace.kTransform)
                rotation = MQuaternion()
                ik_joint.getRotation(rotation, MSpace.kTransform)
                pose.rotation = axe * rotation * orient
                track.poses[time] = pose

    def write(self, path):
        # build unique vecs + quats
        uni_vecs = {}
        uni_quats = {}

        vec_index = 0
        quat_index = 0
        for time in range(1, self.duration+1):
            for track in self.tracks:
                pose = track.poses[time]
                translation_key = f'{pose.translation.x:.4f} {pose.translation.y:.4f} {pose.translation.z:.4f}'
                scale_key = f'{pose.scale.x:.4f} {pose.scale.y:.4f} {pose.scale.z:.4f}'
                rotation_key = f'{pose.rotation.x:.4f} {pose.rotation.y:.4f} {pose.rotation.z:.4f} {pose.rotation.w:.4f}'
                if translation_key not in uni_vecs:
                    uni_vecs[translation_key] = vec_index
                    pose.translation_index = vec_index
                    vec_index += 1
                else:
                    pose.translation_index = uni_vecs[translation_key]
                if scale_key not in uni_vecs:
                    uni_vecs[scale_key] = vec_index
                    pose.scale_index = vec_index
                    vec_index += 1
                else:
                    pose.scale_index = uni_vecs[scale_key]
                if rotation_key not in uni_quats:
                    uni_quats[rotation_key] = quat_index
                    pose.rotation_index = quat_index
                    quat_index += 1
                else:
                    pose.rotation_index = uni_quats[rotation_key]

        with open(path, 'wb') as f:
            bs = BinaryStream(f)

            bs.write_bytes('r3d2anmd'.encode('ascii'))  # magic
            bs.write_uint32(4)  # ver 4

            bs.write_uint32(0)  # resource size - later
            bs.write_uint32(0xBE0794D3)  # format token
            bs.write_uint32(0)  # version?
            bs.write_uint32(0)  # flags

            bs.write_uint32(len(self.tracks))  # track count
            bs.write_uint32(self.duration)  # frame count
            bs.write_float(1.0 / self.fps)  # frame duration = 1 / fps

            bs.write_int32(0)  # tracks offset
            bs.write_int32(0)  # asset name offset
            bs.write_int32(0)  # time offset

            bs.write_int32(64)  # vecs offset
            quats_offset_offset = bs.tell()
            bs.write_int32(0)   # quats offset - later
            bs.write_int32(0)   # frames offset - later

            # pad 12 empty bytes
            bs.write_int32(0)
            bs.write_int32(0)
            bs.write_int32(0)

            # uni vecs
            for vec_key in uni_vecs:
                vec = vec_key.split()
                for i in range(0, 3):
                    bs.write_float(float(vec[i]))

            # uni quats
            quats_offset = bs.tell()
            for quat_key in uni_quats:
                quat = quat_key.split()
                for i in range(0, 4):
                    bs.write_float(float(quat[i]))

            # frames
            frames_offset = bs.tell()
            for time in range(1, self.duration+1):
                for track in self.tracks:
                    bs.write_uint32(track.joint_hash)
                    bs.write_uint16(track.poses[time].translation_index)
                    bs.write_uint16(track.poses[time].scale_index)
                    bs.write_uint16(track.poses[time].rotation_index)
                    bs.write_uint16(0)

            # quats offset and frames offset
            bs.seek(quats_offset_offset)
            # need to minus 12 padded bytes
            bs.write_int32(quats_offset - 12)
            bs.write_int32(frames_offset - 12)

            # resource size
            bs.seek(0, 2)
            fsize = bs.tell()
            bs.seek(12)
            bs.write_uint32(fsize)


class SO:
    def __init__(self):
        self.name = None
        self.central = None
        self.pivot = None

        # assume sco/scb only have 1 material
        self.material = None
        self.indices = []
        # important: uv can be different at each index, can not map this uv data by vertex
        self.uvs = []
        # not actual vertex, its a position of vertex, no reason to create a class
        self.vertices = []

    def flip(self):
        for vertex in self.vertices:
            vertex.x *= -1.0
        self.central.x *= -1.0
        if self.pivot:
            self.pivot.x *= -1.0

    def read_sco(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
            lines = [line[:-1] for line in lines]

            magic = lines[0]
            if magic != '[ObjectBegin]':
                raise FunnyError(
                    f'[SO.read_sco()]: Wrong file signature: {magic}')

            # temporary use file name, not name inside file
            self.name = path.split('/')[-1].split('.')[0]

            index = 1  # skip magic
            len1234 = len(lines)
            while index < len1234:
                inp = lines[index].split()
                if len(inp) == 0:  # cant split, definitely not voldemort
                    index += 1
                    continue

                if inp[0] == 'CentralPoint=':
                    self.central = MVector(
                        float(inp[1]), float(inp[2]), float(inp[3]))

                elif inp[0] == 'PivotPoint=':
                    self.pivot = MVector(
                        float(inp[1]), float(inp[2]), float(inp[3]))

                elif inp[0] == 'Verts=':
                    vertex_count = int(inp[1])
                    for i in range(index+1, index+1 + vertex_count):
                        inp2 = lines[i].split()
                        self.vertices.append(MVector(
                            float(inp2[0]), float(inp2[1]), float(inp2[2])))
                    index = i+1
                    continue

                elif inp[0] == 'Faces=':
                    face_count = int(inp[1])
                    for i in range(index+1, index+1 + face_count):
                        inp2 = lines[i].replace('\t', ' ').split()

                        # skip bad faces
                        face = [int(inp2[1]), int(inp2[2]), int(inp2[3])]
                        if face[0] == face[1] or face[1] == face[2] or face[2] == face[0]:
                            continue
                        self.indices += face

                        self.material = inp2[4]

                        # u v, u v, u v
                        self.uvs.append(
                            MVector(float(inp2[5]), float(inp2[6])))
                        self.uvs.append(
                            MVector(float(inp2[7]), float(inp2[8])))
                        self.uvs.append(
                            MVector(float(inp2[9]), float(inp2[10])))

                    index = i+1
                    continue

                index += 1

    def read_scb(self, path):
        with open(path, 'rb') as f:
            bs = BinaryStream(f)

            magic = bs.read_bytes(8).decode('ascii')
            if magic != 'r3d2Mesh':
                raise FunnyError(
                    f'[SO.read_scb()]: Wrong file signature: {magic}')

            major = bs.read_uint16()
            minor = bs.read_uint16()
            if major not in [3, 2] and minor != 1:
                raise FunnyError(
                    f'[SO.read_scb()]: Unsupported file version: {major}.{minor}')

            # now im trying to use name from path
            # so i will pad name inside file
            bs.pad(128)
            self.name = path.split('/')[-1].split('.')[0]

            vertex_count = bs.read_uint32()
            face_count = bs.read_uint32()

            bs.pad(4 + 24)  # flags, bouding box

            vertex_type = 0  # for padding colors later
            if major == 3 and minor == 2:
                vertex_type = bs.read_uint32()

            for i in range(0, vertex_count):
                self.vertices.append(bs.read_vec3())

            if vertex_type == 1:
                bs.pad(4 * vertex_count)  # pad all vertex colors

            self.central = bs.read_vec3()
            # no pivot in scb

            for i in range(0, face_count):
                face = [bs.read_uint32(), bs.read_uint32(), bs.read_uint32()]
                if face[0] == face[1] or face[1] == face[2] or face[2] == face[0]:
                    continue
                self.indices += face

                self.material = bs.read_padded_string(64)

                uvs = [bs.read_float(), bs.read_float(), bs.read_float(),
                       bs.read_float(), bs.read_float(), bs.read_float()]

                # u u u, v v v
                self.uvs.append(MVector(uvs[0], uvs[3]))
                self.uvs.append(MVector(uvs[1], uvs[4]))
                self.uvs.append(MVector(uvs[2], uvs[5]))

    def load(self):
        vertex_count = len(self.vertices)
        index_count = len(self.indices)
        face_count = index_count // 3

        # create mesh
        vertices = MFloatPointArray()
        for vertex in self.vertices:
            position = vertex - self.central
            vertices.append(MFloatPoint(
                position.x, position.y, position.z))

        u_values = MFloatArray()
        v_values = MFloatArray()
        for i in range(0, index_count):
            uv = self.uvs[i]
            u_values.append(uv.x)
            v_values.append(1.0 - uv.y)

        poly_count = MIntArray(face_count, 3)
        poly_indices = MIntArray()
        MScriptUtil.createIntArrayFromList(self.indices, poly_indices)
        uv_indices = MIntArray()
        MScriptUtil.createIntArrayFromList(
            list(range(0, index_count)), uv_indices)

        mesh = MFnMesh()
        mesh.create(
            vertex_count,
            face_count,
            vertices,
            poly_count,
            poly_indices,
            u_values,
            v_values
        )
        mesh.assignUVs(
            poly_count, uv_indices
        )

        # name + central
        mesh.setName(self.name)
        mesh_name = mesh.name()
        transform = MFnTransform(mesh.parent(0))
        transform.setName(f'mesh_{self.name}')
        transform.setTranslation(self.central, MSpace.kTransform)

        # material
        # lambert material
        lambert = MFnLambertShader()
        lambert.create()
        lambert.setName(self.material)
        lambert_name = lambert.name()
        # shading group
        MGlobal.executeCommand((
            # create renderable, independent shading group
            f'sets -renderable true -noSurfaceShader true -empty -name "{lambert_name}_SG";'
            # add submesh faces to shading group
            f'sets -e -forceElement "{lambert_name}_SG" {mesh_name}.f[0:{face_count}];'
        ))
        # connect lambert to shading group
        MGlobal.executeCommand(
            f'connectAttr -f {lambert_name}.outColor {lambert_name}_SG.surfaceShader;')

        # use a joint for pivot
        if self.pivot:
            ik_joint = MFnIkJoint()
            ik_joint.create()
            ik_joint.setName(f'pivot_{self.name}')
            ik_joint.setTranslation(
                self.central - self.pivot, MSpace.kTransform)

            # bind pivot with mesh
            pivot_dagpath = MDagPath()
            ik_joint.getPath(pivot_dagpath)
            mesh_dagpath = MDagPath()
            mesh.getPath(mesh_dagpath)
            selections = MSelectionList()
            selections.add(mesh_dagpath)
            selections.add(pivot_dagpath)
            MGlobal.selectCommand(selections)
            MGlobal.executeCommand(
                f'skinCluster -mi 1 -tsb -n skinCluster_{self.name}')

        mesh.updateSurface()

    def dump(self):
        # get mesh
        selections = MSelectionList()
        MGlobal.getActiveSelectionList(selections)
        iterator = MItSelectionList(selections, MFn.kMesh)
        if iterator.isDone():
            raise FunnyError(
                f'[SO.dump()]: Please select a mesh.')
        mesh_dagpath = MDagPath()
        iterator.getDagPath(mesh_dagpath)
        iterator.next()
        if not iterator.isDone():
            raise FunnyError(
                f'[SO.dump()]: More than 1 mesh selected.')
        mesh = MFnMesh(mesh_dagpath)

        # name
        self.name = mesh.name()
        transform = MFnTransform(mesh.parent(0))
        self.central = transform.getTranslation(MSpace.kTransform)

        # check hole
        hole_info = MIntArray()
        hole_vertex = MIntArray()
        mesh.getHoles(hole_info, hole_vertex)
        if hole_info.length() > 0:
            raise FunnyError(f'[SO.dump()]: Mesh contains holes.')

        # check valid triangulation
        iterator = MItMeshPolygon(mesh_dagpath)
        iterator.reset()
        while not iterator.isDone():
            if not iterator.hasValidTriangulation():
                raise FunnyError(
                    f'[SO.dump()]: Mesh contains a invalid triangulation polygon.')
            iterator.next()

        # vertices
        vertex_count = mesh.numVertices()
        vertices = MFloatPointArray()
        mesh.getPoints(vertices, MSpace.kWorld)
        for i in range(0, vertex_count):
            position = MVector(vertices[i].x, vertices[i].y, vertices[i].z)
            self.vertices.append(MVector(position - self.central))

        # find pivot through skin cluster
        in_mesh = mesh.findPlug('inMesh')
        in_mesh_connections = MPlugArray()
        in_mesh.connectedTo(in_mesh_connections, True, False)
        if in_mesh_connections.length() > 0:
            if in_mesh_connections[0].node().apiType() == MFn.kSkinClusterFilter:
                skin_cluster = MFnSkinCluster(in_mesh_connections[0].node())
                influences_dagpath = MDagPathArray()
                influence_count = skin_cluster.influenceObjects(
                    influences_dagpath)
                if influence_count > 1:
                    raise FunnyError(
                        f'[SO.dump()]: There is more than 1 joint bound with mesh.')

                ik_joint = MFnTransform(influences_dagpath[0])
                translation = ik_joint.getTranslation(MSpace.kTransform)
                self.pivot = self.central - translation

        # uvs
        u_values = MFloatArray()
        v_values = MFloatArray()
        mesh.getUVs(u_values, v_values)
        uv_count = MIntArray()
        uv_indices = MIntArray()
        mesh.getAssignedUVs(uv_count, uv_indices)

        # faces
        face_count = MIntArray()
        face_vertices = MIntArray()
        mesh.getTriangles(face_count, face_vertices)

        # check triangulated: 1 tri per face = gud
        for triangle_count in face_count:
            if triangle_count > 1:
                raise FunnyError(
                    f'[SO.dump()]: Mesh contains a non-triangulated face, please Mesh -> Triangulate.')

        len666 = mesh.numPolygons()
        index = 0
        for i in range(0, len666):
            for j in range(0, face_count[i] * 3):
                self.indices.append(face_vertices[index])
                u = u_values[uv_indices[index]]
                v = v_values[uv_indices[index]]
                self.uvs.append(MVector(u, 1.0-v))
                index += 1

        # material name
        instance = 0
        if mesh_dagpath.isInstanced():
            instance = mesh_dagpath.instanceNumber()
        shaders = MObjectArray()
        shader_indices = MIntArray()
        mesh.getConnectedShaders(instance, shaders, shader_indices)
        if shaders.length() > 1:
            raise FunnyError(
                '[SO.dump()]: There are more than 1 material assigned to this mesh.')

        if shaders.length() > 0:
            surface_shaders = MFnDependencyNode(
                shaders[0]).findPlug('surfaceShader')
            plug_array = MPlugArray()
            surface_shaders.connectedTo(plug_array, True, False)
            surface_shader = MFnDependencyNode(plug_array[0].node())
            self.material = surface_shader.name()
        else:
            # its only allow 1 material anyway
            self.material = 'lambert'

    def write_sco(self, path):
        with open(path, 'w') as f:
            f.write('[ObjectBegin]\n')  # magic

            f.write(f'Name= {self.name}\n')
            f.write(
                f'CentralPoint= {self.central.x:.4f} {self.central.y:.4f} {self.central.z:.4f}\n')
            if self.pivot != None:
                f.write(
                    f'PivotPoint= {self.pivot.x:.4f} {self.pivot.y:.4f} {self.pivot.z:.4f}\n')

            # vertices
            f.write(f'Verts= {len(self.vertices)}\n')
            for position in self.vertices:
                f.write(f'{position.x:.4f} {position.y:.4f} {position.z:.4f}\n')

            # faces
            face_count = len(self.indices) // 3
            f.write(f'Faces= {face_count}\n')
            for i in range(0, face_count):
                index = i * 3
                f.write('3\t')
                f.write(f' {self.indices[index]:>5}')
                f.write(f' {self.indices[index+1]:>5}')
                f.write(f' {self.indices[index+2]:>5}')
                f.write(f'\t{self.material:>20}\t')
                f.write(f'{self.uvs[index].x:.12f} {self.uvs[index].y:.12f} ')
                f.write(
                    f'{self.uvs[index+1].x:.12f} {self.uvs[index+2].y:.12f} ')
                f.write(
                    f'{self.uvs[index+2].x:.12f} {self.uvs[index+2].y:.12f}\n')

            f.write('[ObjectEnd]')

    def write_scb(self, path):
        # dump bb before flip
        def get_bounding_box():
            # totally not copied code
            min = MVector(float('inf'), float('inf'), float('inf'))
            max = MVector(float('-inf'), float('-inf'), float('-inf'))
            for vertex in self.vertices:
                if min.x > vertex.x:
                    min.x = vertex.x
                if min.y > vertex.y:
                    min.y = vertex.y
                if min.z > vertex.z:
                    min.z = vertex.z
                if max.x < vertex.x:
                    max.x = vertex.x
                if max.y < vertex.y:
                    max.y = vertex.y
                if max.z < vertex.z:
                    max.z = vertex.z
            return min, max

        with open(path, 'wb') as f:
            bs = BinaryStream(f)

            bs.write_bytes('r3d2Mesh'.encode('ascii'))  # magic
            bs.write_uint16(3)  # major
            bs.write_uint16(2)  # minor

            bs.write_padded_string(128, '')  # well

            face_count = len(self.indices) // 3
            bs.write_uint32(len(self.vertices))
            bs.write_uint32(face_count)

            # flags:
            # 1 - vertex color
            # 2 - local origin locator and pivot
            # why 2? idk ¯\_(ツ)_/¯
            bs.write_uint32(2)

            # bounding box
            bb = get_bounding_box()
            bs.write_vec3(bb[0])
            bs.write_vec3(bb[1])

            bs.write_uint32(0)  # vertex color

            # vertices
            for vertex in self.vertices:
                bs.write_vec3(vertex)

            # central
            bs.write_vec3(self.central)

            # faces - easy peasy squeezy last part
            for i in range(0, face_count):
                index = i * 3
                bs.write_uint32(self.indices[index])
                bs.write_uint32(self.indices[index+1])
                bs.write_uint32(self.indices[index+2])

                bs.write_padded_string(64, self.material)

                # u u u
                bs.write_float(self.uvs[index].x)
                bs.write_float(self.uvs[index+1].x)
                bs.write_float(self.uvs[index+2].x)
                # v v v
                bs.write_float(self.uvs[index].y)
                bs.write_float(self.uvs[index+1].y)
                bs.write_float(self.uvs[index+2].y)


# test map geo
class MAPGEOVertexElement:
    def __init__(self):
        """
            Position,
            BlendWeight,
            Normal,
            PrimaryColor,
            SecondaryColor,
            FogCoordinate,
            BlendIndex,
            DiffuseUV,
            Texcoord1,
            Texcoord2,
            Texcoord3,
            Texcoord4,
            Texcoord5,
            Texcoord6,
            LightmapUV,
            StreamIndexCount
        """
        self.name = None

        """
            X_Float32,
            XY_Float32,
            XYZ_Float32,
            XYZW_Float32,
            BGRA_Packed8888,
            ZYXW_Packed8888,
            RGBA_Packed8888,
            XYZW_Packed8888
        """
        # self.format = None


class MAPGEOVertexElementGroup:
    def __init__(self):
        # 0 - static, 1 - dynamic, 2 - stream
        # self.usage = None
        self.vertex_elements = []


class MAPGEOVertex:
    def __init__(self):
        self.position = None
        self.normal = None
        self.diffuse_uv = None
        self.lightmap_uv = None
        # self.color2 = None

    def combine(a, b):
        # hmmm, find a way to rip this shit
        res = MAPGEOVertex()
        res.position = b.position if (
            b.position and not a.position) else a.position
        res.normal = b.normal if (b.normal and not a.normal) else a.normal
        res.diffuse_uv = b.diffuse_uv if (
            b.diffuse_uv and not a.diffuse_uv) else a.diffuse_uv
        res.lightmap_uv = b.lightmap_uv if (
            b.lightmap_uv and not a.lightmap_uv) else a.lightmap_uv
        # res.color2 = b.color2 if b.color2 and not a.color2 else a.color2
        return res


class MAPGEOSubmesh:
    def __init__(self):
        # self.parent = None
        # self.hash = None
        self.name = None
        self.index_start = None
        self.index_count = None
        self.vertex_start = None
        self.vertex_count = None


class MAPGeoModel:
    def __init__(self):
        self.name = None
        self.submeshes = []
        self.vertices = []
        self.indices = []

        # transform
        self.translation = None
        self.scale = None
        self.rotation = None

        # depend which layer that model appear on
        self.layer = None

        # empty = no light map
        self.lightmap = None


class MAPGEO:
    def __init__(self):
        self.models = []
        # self.bucket_grid = None

    def flip(self):
        for model in self.models:
            for vertex in model.vertices:
                vertex.position.x *= -1.0
                vertex.normal.y *= -1.0
                vertex.normal.z *= -1.0
            model.translation.x *= -1.0
            model.rotation.y *= -1.0
            model.rotation.z *= -1.0

    def read(self, path):
        with open(path, 'rb') as f:
            bs = BinaryStream(f)

            magic = bs.read_bytes(4).decode('ascii')
            if magic != 'OEGM':
                print('wrog magic')
                return

            version = bs.read_uint32()
            if version not in [5, 6, 7, 9, 11]:
                print('wrong version')
                return

            use_seperate_point_lights = False
            if version < 7:
                use_seperate_point_lights = bs.read_bool()

            if version >= 9:
                # unknown str 1
                bs.read_bytes(bs.read_int32()).decode('ascii')
                if version >= 11:
                    # unknown str 2
                    bs.read_bytes(bs.read_int32()).decode('ascii')

            # vertex elements
            vegs = []  # list of ve groups
            veg_count = bs.read_uint32()
            for i in range(0, veg_count):
                veg = MAPGEOVertexElementGroup()  # group of ves
                bs.read_uint32()  # usage

                ve_count = bs.read_uint32()
                for j in range(0, ve_count):
                    ve = MAPGEOVertexElement()  # ve
                    ve.name = bs.read_uint32()
                    bs.read_uint32()  # format
                    veg.vertex_elements.append(ve)

                vegs.append(veg)
                bs.stream.seek(8 * (15 - ve_count), 1)

            # vertex buffers
            vbs = []
            vb_count = bs.read_uint32()
            for i in range(0, vb_count):
                buffer = bs.read_uint32()
                vbs.append(bs.stream.tell())
                bs.stream.seek(buffer, 1)

            # index buffers
            ibs = []  # list of list
            ib_count = bs.read_uint32()
            for i in range(0, ib_count):
                buffer = bs.read_uint32()

                ib = []
                for j in range(0, buffer // 2):
                    ib.append(bs.read_uint16())

                ibs.append(ib)  # list of list

            model_count = bs.read_uint32()
            for m in range(0, model_count):
                model = MAPGeoModel()
                model.name = bs.read_bytes(bs.read_int32()).decode('ascii')

                vertex_count = bs.read_uint32()
                vb_count = bs.read_uint32()
                veg = bs.read_int32()

                # init vertices
                for i in range(0, vertex_count):
                    model.vertices.append(MAPGEOVertex())

                for i in range(0, vb_count):
                    vb_id = bs.read_int32()
                    return_offset = bs.stream.tell()
                    bs.stream.seek(vbs[vb_id])

                    for j in range(0, vertex_count):
                        vertex = MAPGEOVertex()
                        for element in vegs[veg+i].vertex_elements:
                            if element.name == 0:
                                vertex.position = bs.read_vec3()
                            elif element.name == 2:
                                vertex.normal = bs.read_vec3()
                            elif element.name == 7:
                                vertex.diffuse_uv = bs.read_vec2()
                            elif element.name == 14:
                                vertex.lightmap_uv = bs.read_vec2()
                            elif element.name == 4:
                                bs.read_byte()
                                bs.read_byte()
                                bs.read_byte()
                                bs.read_byte()  # pad color idk
                            else:
                                print('error unknown element')
                                return
                        model.vertices[j] = MAPGEOVertex.combine(
                            model.vertices[j], vertex)

                    bs.stream.seek(return_offset)

                # indices
                bs.read_uint32()  # index_count
                ib = bs.read_int32()
                model.indices += ibs[ib]

                # submeshes
                submesh_count = bs.read_uint32()
                for i in range(0, submesh_count):
                    submesh = MAPGEOSubmesh()
                    # submesh.parent = model
                    bs.read_uint32()  # hash
                    submesh.name = bs.read_bytes(
                        bs.read_int32()).decode('ascii')

                    # maya doesnt allow '/' in name, so use __ instead, bruh
                    submesh.name = submesh.name.replace('/', '__')

                    submesh.index_start = bs.read_uint32()
                    submesh.index_count = bs.read_uint32()
                    submesh.vertex_start = bs.read_uint32()
                    submesh.vertex_count = bs.read_uint32()
                    if submesh.vertex_start > 0:
                        submesh.vertex_start -= 1
                    model.submeshes.append(submesh)

                if version != 5:
                    bs.read_bool()  # flip normals

                # bounding box
                bs.read_vec3()
                bs.read_vec3()

                # transform matrix
                py_list = []
                for i in range(0, 4):
                    for j in range(0, 4):
                        py_list.append(bs.read_float())

                matrix = MMatrix()
                MScriptUtil.createMatrixFromList(py_list, matrix)
                model.translation, model.scale, model.rotation = MTransform.decompose(
                    MTransformationMatrix(matrix),
                    MSpace.kWorld
                )

                bs.read_byte()  # flags

                # layer data - 8 char binary string of an byte, example: 10101010
                # need to be reversed
                # if the char at index 3 is '1' -> the object appear on layer index 3
                # default layer data: 11111111 - appear on all layers
                model.layer = f'{255:08b}'[::-1]
                if version >= 7:
                    model.layer = f'{bs.read_byte()[0]:08b}'[::-1]
                    if version >= 11:
                        # unknown byte
                        bs.read_byte()

                if use_seperate_point_lights and version < 7:
                    # pad seperated point light
                    t = bs.read_vec3()
                    print('point light: ', t.x, t.y, t.z)
                    # need to create pointlight?

                if version < 9:
                    for i in range(0, 9):
                        bs.read_vec3()
                        # pad unknow vec3

                model.lightmap = bs.read_bytes(
                    bs.read_int32()).decode('ascii')
                model.lm_scale_u = bs.read_float()  # this is not color
                model.lm_scale_v = bs.read_float()  # we have been tricked
                model.lm_offset_u = bs.read_float()  # backstabbed
                model.lm_offset_v = bs.read_float()  # and possibly, bamboozled

                if version >= 9:
                    # pretty pad all this since idk
                    bs.read_bytes(bs.read_int32()).decode(
                        'ascii')  # baked paintexture
                    bs.read_float()  # probably some UVs things
                    bs.read_float()
                    bs.read_float()
                    bs.read_float()

                self.models.append(model)

            # bro its actually still bucked grid here, forgot

    def load(self, mgbin=None):
        def load_both():
            # ensure far clip plane, allow to see big objects like whole map
            MGlobal.executeCommand('setAttr "perspShape.farClipPlane" 300000')

            # render with alpha cut
            MGlobal.executeCommand(
                'setAttr "hardwareRenderingGlobals.transparencyAlgorithm" 5')

            meshes = []
            for model in self.models:
                mesh = MFnMesh()
                vertices_count = len(model.vertices)
                indices_count = len(model.indices)

                # create mesh with vertices, indices
                vertices = MFloatPointArray()
                for i in range(0, vertices_count):
                    vertex = model.vertices[i]
                    vertices.append(MFloatPoint(
                        vertex.position.x, vertex.position.y, vertex.position.z))
                poly_index_count = MIntArray(indices_count // 3, 3)
                poly_indices = MIntArray()
                MScriptUtil.createIntArrayFromList(model.indices, poly_indices)
                mesh.create(
                    vertices_count,
                    indices_count // 3,
                    vertices,
                    poly_index_count,
                    poly_indices,
                )

                # assign uv
                # lightmap not always available, skip if we dont have
                u_values = MFloatArray(vertices_count)
                v_values = MFloatArray(vertices_count)
                if len(model.lightmap) > 0:
                    u_values_lm = MFloatArray(vertices_count)
                    v_values_lm = MFloatArray(vertices_count)
                for i in range(0, vertices_count):
                    vertex = model.vertices[i]
                    u_values[i] = vertex.diffuse_uv.x
                    v_values[i] = 1.0 - vertex.diffuse_uv.y
                    if len(model.lightmap) > 0:
                        u_values_lm[i] = vertex.lightmap_uv.x * \
                            model.lm_scale_u + model.lm_offset_u
                        v_values_lm[i] = 1.0-(vertex.lightmap_uv.y *
                                              model.lm_scale_v + model.lm_offset_v)

                mesh.setUVs(
                    u_values, v_values
                )
                mesh.assignUVs(
                    poly_index_count, poly_indices
                )
                if len(model.lightmap) > 0:
                    mesh.createUVSetWithName('lightmap')
                    mesh.setUVs(
                        u_values_lm, v_values_lm, 'lightmap'
                    )

                    mesh.assignUVs(
                        poly_index_count, poly_indices, 'lightmap'
                    )

                # normal
                normals = MVectorArray()
                normal_indices = MIntArray(vertices_count)
                for i in range(0, vertices_count):
                    vertex = model.vertices[i]
                    normals.append(
                        MVector(vertex.normal.x, vertex.normal.y, vertex.normal.z))
                    normal_indices[i] = i
                mesh.setVertexNormals(
                    normals,
                    normal_indices
                )

                # dagpath and name
                mesh_dagpath = MDagPath()
                mesh.getPath(mesh_dagpath)
                mesh.setName(model.name)
                transform_node = MFnTransform(mesh.parent(0))
                transform_node.setName(model.name)

                transform_node.set(MTransform.compose(
                    model.translation, model.scale, model.rotation,
                    MSpace.kWorld
                ))

                model.mesh_dagpath = MDagPath(mesh_dagpath)
                meshes.append(mesh)

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
            shader_models = {}
            for model in self.models:
                for submesh in model.submeshes:
                    if submesh.name not in shader_models:
                        shader_models[submesh.name] = []
                    if model not in shader_models[submesh.name]:
                        shader_models[submesh.name].append(model)

            # lightmaps
            lightmap_models = {}
            for model in self.models:
                if model.lightmap:
                    if model.lightmap not in lightmap_models:
                        lightmap_models[model.lightmap] = []
                    if model not in lightmap_models[model.lightmap]:
                        lightmap_models[model.lightmap].append(model)

            modifier = MDGModifier()
            set = MFnSet()
            for model in self.models:
                for submesh in model.submeshes:
                    submesh_name = submesh.name
                    lambert = MFnLambertShader()
                    lambert.create(True)
                    lambert.setName(f'{model.name}__{submesh_name}')

                    # some shader stuffs
                    dependency_node = MFnDependencyNode()
                    shading_engine = dependency_node.create(
                        'shadingEngine', f'{model.name}__{submesh_name}_SG')
                    material_info = dependency_node.create(
                        'materialInfo', f'{model.name}__{submesh_name}_MaterialInfo')

                    if found_rp:
                        partition = MFnDependencyNode(
                            shading_engine).findPlug('partition')

                        sets = render_partition.findPlug('sets')
                        the_plug_we_need = None
                        count = 0
                        while True:
                            the_plug_we_need = sets.elementByLogicalIndex(
                                count)
                            if not the_plug_we_need.isConnected():  # find the one that not connected
                                break
                            count += 1

                    modifier.connect(partition, the_plug_we_need)

                    # connect node
                    out_color = lambert.findPlug('outColor')
                    surface_shader = MFnDependencyNode(
                        shading_engine).findPlug('surfaceShader')
                    modifier.connect(out_color, surface_shader)

                    message = MFnDependencyNode(
                        shading_engine).findPlug('message')
                    shading_group = MFnDependencyNode(
                        material_info).findPlug('shadingGroup')
                    modifier.connect(message, shading_group)

                    modifier.doIt()

                    set.setObject(shading_engine)
                    # assign face to material
                    component = MFnSingleIndexedComponent()
                    face_component = component.create(
                        MFn.kMeshPolygonComponent)
                    group_poly_indices = MIntArray()
                    for index in range(submesh.index_start // 3, (submesh.index_start + submesh.index_count) // 3):
                        group_poly_indices.append(index)
                    component.addElements(group_poly_indices)

                    set.addMember(model.mesh_dagpath, face_component)

                    MGlobal.executeCommand(
                        f'shadingNode -asUtility blendColors -name "{model.name}__{submesh_name}_BC"')
                    MGlobal.executeCommand(
                        f'connectAttr -f {model.name}__{submesh_name}_BC.output {model.name}__{submesh_name}.color')

            # parsing the py and assign material attributes
            # little bit complicated
            if mgbin:
                for submesh_name in shader_models:
                    material = mgbin.materials[submesh_name]
                    if material.texture:
                        # create file node
                        MGlobal.executeCommand(
                            f'shadingNode -asTexture -isColorManaged file -name "{submesh_name}_file"')
                        MGlobal.executeCommand(
                            f'setAttr {submesh_name}_file.fileTextureName -type "string" "{mgbin.full_path(material.texture)}"')

                        # create place2dTexture node (p2d)
                        MGlobal.executeCommand(
                            f'shadingNode -asUtility place2dTexture -name "{submesh_name}_p2d"')

                        # connect p2d - file
                        attributes = [
                            'coverage',
                            'translateFrame',
                            'rotateFrame',
                            'mirrorU',
                            'mirrorV',
                            'stagger',
                            'wrapU',
                            'wrapV',
                            'repeatUV',
                            'offset',
                            'rotateUV',
                            'noiseUV',
                            'vertexUvOne',
                            'vertexUvTwo',
                            'vertexUvThree',
                            'vertexCameraOne'
                        ]
                        for attribute in attributes:
                            MGlobal.executeCommand(
                                f'connectAttr -f {submesh_name}_p2d.{attribute} {submesh_name}_file.{attribute}')
                        MGlobal.executeCommand(
                            f'connectAttr -f {submesh_name}_p2d.outUV {submesh_name}_file.uv')
                        MGlobal.executeCommand(
                            f'connectAttr -f {submesh_name}_p2d.outUvFilterSize {submesh_name}_file.uvFilterSize')

                        for model in shader_models[submesh_name]:
                            MGlobal.executeCommand(
                                f'connectAttr -f {submesh_name}_file.outColor {model.name}__{submesh_name}_BC.color1')
                            MGlobal.executeCommand(
                                f'connectAttr -f {submesh_name}_file.outColor {model.name}__{submesh_name}_BC.color2')

                            MGlobal.executeCommand(
                                f'connectAttr -f {submesh_name}_file.outTransparency {model.name}__{submesh_name}.transparency')
                    else:
                        if material.color:
                            for model in shader_models[submesh_name]:
                                MGlobal.executeCommand(
                                    f'setAttr "{model.name}__{submesh_name}_BC.color1" -type double3 {material.color[0]} {material.color[1]} {material.color[2]}')
                                MGlobal.executeCommand(
                                    f'setAttr "{model.name}__{submesh_name}_BC.color2" -type double3 {material.color[0]} {material.color[1]} {material.color[2]}')
                    if material.ambient:
                        for model in shader_models[submesh_name]:
                            MGlobal.executeCommand(
                                f'setAttr "{model.name}__{submesh_name}.ambientColor" -type double3 {material.ambient[0]} {material.ambient[1]} {material.ambient[2]}')
                    if material.incandescence:
                        for model in shader_models[submesh_name]:
                            MGlobal.executeCommand(
                                f'setAttr "{model.name}__{submesh_name}.incandescence" -type double3 {material.incandescence[0]} {material.incandescence[1]} {material.incandescence[2]}')

                for lightmap in lightmap_models:
                    lightmap_name = 'LM_' + \
                        lightmap.split('/')[-1].split('.dds')[0]
                    # create file node
                    MGlobal.executeCommand(
                        f'shadingNode -asTexture -isColorManaged file -name "{lightmap_name}_fileLM"')
                    MGlobal.executeCommand(
                        f'setAttr {lightmap_name}_fileLM.fileTextureName -type "string" "{mgbin.full_path(lightmap)}"')

                    # create place2dTexture node (p2d)
                    MGlobal.executeCommand(
                        f'shadingNode -asUtility place2dTexture -name "{lightmap_name}_p2dLM"')

                    # connect p2d - file
                    attributes = [
                        'coverage',
                        'translateFrame',
                        'rotateFrame',
                        'mirrorU',
                        'mirrorV',
                        'stagger',
                        'wrapU',
                        'wrapV',
                        'repeatUV',
                        'offset',
                        'rotateUV',
                        'noiseUV',
                        'vertexUvOne',
                        'vertexUvTwo',
                        'vertexUvThree',
                        'vertexCameraOne'
                    ]
                    for attribute in attributes:
                        MGlobal.executeCommand(
                            f'connectAttr -f {lightmap_name}_p2dLM.{attribute} {lightmap_name}_fileLM.{attribute}')
                    MGlobal.executeCommand(
                        f'connectAttr -f {lightmap_name}_p2dLM.outUV {lightmap_name}_fileLM.uv')
                    MGlobal.executeCommand(
                        f'connectAttr -f {lightmap_name}_p2dLM.outUvFilterSize {lightmap_name}_fileLM.uvFilterSize')

                    for model in lightmap_models[lightmap]:
                        for submesh in model.submeshes:
                            submesh_name = submesh.name
                            MGlobal.executeCommand(
                                f'connectAttr -f {lightmap_name}_fileLM.outColor {model.name}__{submesh_name}_BC.color2')

                        MGlobal.executeCommand(
                            f'uvLink -uvSet "|{model.name}|{model.name}.uvSet[1].uvSetName" -texture "{lightmap_name}_fileLM"')

            for mesh in meshes:
                mesh.updateSurface()

        def load_diffuse():
            # ensure far clip plane, allow to see big objects like whole map
            MGlobal.executeCommand('setAttr "perspShape.farClipPlane" 300000')

            # render with alpha cut
            MGlobal.executeCommand(
                'setAttr "hardwareRenderingGlobals.transparencyAlgorithm" 5')

            # create 8 layers, because max = 8, bruh
            layer_models = {}
            for i in range(0, 8):
                MGlobal.executeCommand(
                    f'createDisplayLayer -name "layer{i}" -number {i} -empty')
                layer_models[i] = []

            meshes = []
            for model in self.models:
                mesh = MFnMesh()
                vertices_count = len(model.vertices)
                indices_count = len(model.indices)

                # create mesh with vertices, indices
                vertices = MFloatPointArray()
                for i in range(0, vertices_count):
                    vertex = model.vertices[i]
                    vertices.append(MFloatPoint(
                        vertex.position.x, vertex.position.y, vertex.position.z))
                poly_index_count = MIntArray(indices_count // 3, 3)
                poly_indices = MIntArray()
                MScriptUtil.createIntArrayFromList(model.indices, poly_indices)
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
                    vertex = model.vertices[i]
                    u_values[i] = vertex.diffuse_uv.x
                    v_values[i] = 1.0 - vertex.diffuse_uv.y

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
                    vertex = model.vertices[i]
                    normals.append(
                        MVector(vertex.normal.x, vertex.normal.y, vertex.normal.z))
                    normal_indices[i] = i
                mesh.setVertexNormals(
                    normals,
                    normal_indices
                )

                # dagpath and name
                mesh_dagpath = MDagPath()
                mesh.getPath(mesh_dagpath)
                mesh.setName(model.name)
                transform_node = MFnTransform(mesh.parent(0))
                transform_node.setName(model.name)

                # transform
                transform_node.set(MTransform.compose(
                    model.translation, model.scale, model.rotation,
                    MSpace.kWorld
                ))

                # add the model to the layer, where it belong too
                for i in range(0, 8):
                    if model.layer[i] == '1':
                        layer_models[i].append(f'|{model.name}')

                model.mesh_dagpath = MDagPath(mesh_dagpath)
                meshes.append(mesh)

            # set model to layers
            for i in range(0, 8):
                model_names = ' '.join(layer_models[i])
                MGlobal.executeCommand(
                    f'editDisplayLayerMembers -noRecurse layer{i} {model_names}')

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
            shader_models = {}
            for model in self.models:
                for submesh in model.submeshes:
                    if submesh.name not in shader_models:
                        shader_models[submesh.name] = []
                    if model not in shader_models[submesh.name]:
                        shader_models[submesh.name].append(model)

            modifier = MDGModifier()
            set = MFnSet()
            for submesh_name in shader_models:
                lambert = MFnLambertShader()
                lambert.create(True)
                lambert.setName(f'{submesh_name}')

                # some shader stuffs
                dependency_node = MFnDependencyNode()
                shading_engine = dependency_node.create(
                    'shadingEngine', f'{submesh_name}_SG')
                material_info = dependency_node.create(
                    'materialInfo', f'{submesh_name}_MaterialInfo')

                if found_rp:
                    partition = MFnDependencyNode(
                        shading_engine).findPlug('partition')

                    sets = render_partition.findPlug('sets')
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

                set.setObject(shading_engine)
                # assign face to material
                for model in shader_models[submesh_name]:
                    component = MFnSingleIndexedComponent()
                    face_component = component.create(
                        MFn.kMeshPolygonComponent)
                    for submesh in model.submeshes:
                        if submesh.name == submesh_name:
                            break
                    group_poly_indices = MIntArray()
                    for index in range(submesh.index_start // 3, (submesh.index_start + submesh.index_count) // 3):
                        group_poly_indices.append(index)
                    component.addElements(group_poly_indices)

                    set.addMember(model.mesh_dagpath, face_component)

            # parsing the py and assign material attributes
            # little bit complicated
            if mgbin:
                for submesh_name in shader_models:
                    material = mgbin.materials[submesh_name]
                    if material.texture:
                        # create file node
                        MGlobal.executeCommand(
                            f'shadingNode -asTexture -isColorManaged file -name "{submesh_name}_file"')
                        MGlobal.executeCommand(
                            f'setAttr {submesh_name}_file.fileTextureName -type "string" "{mgbin.full_path(material.texture)}"')

                        # create place2dTexture node (p2d)
                        MGlobal.executeCommand(
                            f'shadingNode -asUtility place2dTexture -name "{submesh_name}_p2d"')

                        # connect p2d - file
                        attributes = [
                            'coverage',
                            'translateFrame',
                            'rotateFrame',
                            'mirrorU',
                            'mirrorV',
                            'stagger',
                            'wrapU',
                            'wrapV',
                            'repeatUV',
                            'offset',
                            'rotateUV',
                            'noiseUV',
                            'vertexUvOne',
                            'vertexUvTwo',
                            'vertexUvThree',
                            'vertexCameraOne'
                        ]
                        for attribute in attributes:
                            MGlobal.executeCommand(
                                f'connectAttr -f {submesh_name}_p2d.{attribute} {submesh_name}_file.{attribute}')
                        MGlobal.executeCommand(
                            f'connectAttr -f {submesh_name}_p2d.outUV {submesh_name}_file.uv')
                        MGlobal.executeCommand(
                            f'connectAttr -f {submesh_name}_p2d.outUvFilterSize {submesh_name}_file.uvFilterSize')

                        MGlobal.executeCommand(
                            f'connectAttr -f {submesh_name}_file.outColor {submesh_name}.color')
                        MGlobal.executeCommand(
                            f'connectAttr -f {submesh_name}_file.outTransparency {submesh_name}.transparency')

                    else:
                        if material.color:
                            MGlobal.executeCommand(
                                f'setAttr "{submesh_name}.color" -type double3 {material.color[0]} {material.color[1]} {material.color[2]}')
                    if material.ambient:
                        MGlobal.executeCommand(
                            f'setAttr "{submesh_name}.ambientColor" -type double3 {material.ambient[0]} {material.ambient[1]} {material.ambient[2]}')
                    if material.incandescence:
                        MGlobal.executeCommand(
                            f'setAttr "{submesh_name}.incandescence" -type double3 {material.incandescence[0]} {material.incandescence[1]} {material.incandescence[2]}')

            for mesh in meshes:
                mesh.updateSurface()

        def load_lightmap():
            # ensure far clip plane, allow to see big objects like whole map
            MGlobal.executeCommand('setAttr "perspShape.farClipPlane" 300000')

            # render with alpha cut
            MGlobal.executeCommand(
                'setAttr "hardwareRenderingGlobals.transparencyAlgorithm" 5')

            meshes = []
            for model in self.models:
                if not model.lightmap:
                    raise FunnyError(
                        '[MAPGEO.load().load_lightmap()]: No lightmap data found.')

                mesh = MFnMesh()
                vertices_count = len(model.vertices)
                indices_count = len(model.indices)

                # create mesh with vertices, indices
                vertices = MFloatPointArray()
                for i in range(0, vertices_count):
                    vertex = model.vertices[i]
                    vertices.append(MFloatPoint(
                        vertex.position.x, vertex.position.y, vertex.position.z))
                poly_index_count = MIntArray(indices_count // 3, 3)
                poly_indices = MIntArray()
                MScriptUtil.createIntArrayFromList(model.indices, poly_indices)
                mesh.create(
                    vertices_count,
                    indices_count // 3,
                    vertices,
                    poly_index_count,
                    poly_indices,
                )

                # assign uv
                # lightmap not always available, skip if we dont have
                u_values_lm = MFloatArray(vertices_count)
                v_values_lm = MFloatArray(vertices_count)
                for i in range(0, vertices_count):
                    vertex = model.vertices[i]
                    u_values_lm[i] = vertex.lightmap_uv.x * \
                        model.lm_scale_u + model.lm_offset_u
                    v_values_lm[i] = 1.0-(vertex.lightmap_uv.y *
                                          model.lm_scale_v + model.lm_offset_v)
                mesh.setUVs(
                    u_values_lm, v_values_lm
                )
                mesh.assignUVs(
                    poly_index_count, poly_indices
                )

                # normal
                normals = MVectorArray()
                normal_indices = MIntArray(vertices_count)
                for i in range(0, vertices_count):
                    vertex = model.vertices[i]
                    normals.append(
                        MVector(vertex.normal.x, vertex.normal.y, vertex.normal.z))
                    normal_indices[i] = i
                mesh.setVertexNormals(
                    normals,
                    normal_indices
                )

                # dagpath and name
                mesh_dagpath = MDagPath()
                mesh.getPath(mesh_dagpath)
                mesh.setName(model.name)
                transform_node = MFnTransform(mesh.parent(0))
                transform_node.setName(model.name)

                # transform
                transform_node.set(MTransform.compose(
                    model.translation, model.scale, model.rotation,
                    MSpace.kWorld
                ))

                model.mesh_dagpath = MDagPath(mesh_dagpath)
                meshes.append(mesh)

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

            # lightmaps
            lightmap_models = {}
            for model in self.models:
                if model.lightmap not in lightmap_models:
                    lightmap_models[model.lightmap] = []
                if model not in lightmap_models[model.lightmap]:
                    lightmap_models[model.lightmap].append(model)

            modifier = MDGModifier()
            set = MFnSet()
            for lightmap in lightmap_models:
                lightmap_name = lightmap.replace('/', '__').split('.dds')[0]
                lambert = MFnLambertShader()
                lambert.create(True)
                lambert.setName(f'{lightmap_name}')

                # some shader stuffs
                dependency_node = MFnDependencyNode()
                shading_engine = dependency_node.create(
                    'shadingEngine', f'{lightmap_name}_SG')
                material_info = dependency_node.create(
                    'materialInfo', f'{lightmap_name}_MaterialInfo')

                if found_rp:
                    partition = MFnDependencyNode(
                        shading_engine).findPlug('partition')

                    sets = render_partition.findPlug('sets')
                    the_plug_we_need = None
                    count = 0
                    while True:
                        the_plug_we_need = sets.elementByLogicalIndex(
                            count)
                        if not the_plug_we_need.isConnected():  # find the one that not connected
                            break
                        count += 1

                modifier.connect(partition, the_plug_we_need)

                # connect node
                out_color = lambert.findPlug('outColor')
                surface_shader = MFnDependencyNode(
                    shading_engine).findPlug('surfaceShader')
                modifier.connect(out_color, surface_shader)

                message = MFnDependencyNode(
                    shading_engine).findPlug('message')
                shading_group = MFnDependencyNode(
                    material_info).findPlug('shadingGroup')
                modifier.connect(message, shading_group)

                modifier.doIt()

                set.setObject(shading_engine)
                for model in lightmap_models[lightmap]:
                    # assign face to material
                    component = MFnSingleIndexedComponent()
                    face_component = component.create(
                        MFn.kMeshPolygonComponent)
                    group_poly_indices = MIntArray()
                    for index in range(0, len(model.indices) // 3):
                        group_poly_indices.append(index)
                    component.addElements(group_poly_indices)

                    set.addMember(model.mesh_dagpath, face_component)

            # parsing the py and assign material attributes
            # little bit complicated
            if mgbin:
                for lightmap in lightmap_models:
                    lightmap_name = lightmap.replace(
                        '/', '__').split('.dds')[0]
                    # create file node
                    MGlobal.executeCommand(
                        f'shadingNode -asTexture -isColorManaged file -name "{lightmap_name}_file"')
                    MGlobal.executeCommand(
                        f'setAttr {lightmap_name}_file.fileTextureName -type "string" "{mgbin.full_path(lightmap)}"')

                    # create place2dTexture node (p2d)
                    MGlobal.executeCommand(
                        f'shadingNode -asUtility place2dTexture -name "{lightmap_name}_p2d"')

                    # connect p2d - file
                    attributes = [
                        'coverage',
                        'translateFrame',
                        'rotateFrame',
                        'mirrorU',
                        'mirrorV',
                        'stagger',
                        'wrapU',
                        'wrapV',
                        'repeatUV',
                        'offset',
                        'rotateUV',
                        'noiseUV',
                        'vertexUvOne',
                        'vertexUvTwo',
                        'vertexUvThree',
                        'vertexCameraOne'
                    ]
                    for attribute in attributes:
                        MGlobal.executeCommand(
                            f'connectAttr -f {lightmap_name}_p2d.{attribute} {lightmap_name}_file.{attribute}')
                    MGlobal.executeCommand(
                        f'connectAttr -f {lightmap_name}_p2d.outUV {lightmap_name}_file.uv')
                    MGlobal.executeCommand(
                        f'connectAttr -f {lightmap_name}_p2d.outUvFilterSize {lightmap_name}_file.uvFilterSize')

                    MGlobal.executeCommand(
                        f'connectAttr -f {lightmap_name}_file.outColor {lightmap_name}.color')

            for mesh in meshes:
                mesh.updateSurface()

        """ hmmmmmmmmmmm
        mode = MGlobal.executeCommandStringResult(
            'confirmDialog -title "[MAPGEO.load()]:" -message "Choose load mode:" -button "Diffuse" -button "Lightmap" -button "Both (unable to export)" -icon "question" -defaultButton "Diffuse"')
        if mode == 'Diffuse':
            load_diffuse()
        elif mode == 'Lightmap':
            load_lightmap()
        elif mode == 'Both (unable to export)':
            load_both()
        else:
            MGlobal.displayInfo(f'[MAPGEO.load()]: Invalid mode: {mode}')
        """
        load_diffuse()

# sorry its a py instead
# probaly need a sub class


class MAPGEOMaterial():
    def __init__(self):
        self.texture = None
        self.color = None
        self.ambient = None
        self.incandescence = None


class MAPGEOBin():
    def __init__(self):
        self.path = None  # this is the dynamic parent path of assets
        self.materials = {}

    def full_path(self, path):
        return self.path + '/' + path

    def read(self, path):
        with open(path, 'r') as f:
            self.path = '/'.join(path.split('/')[:-1])
            lines = f.readlines()

            i = 0
            len12345 = len(lines)
            mat_lines = []
            while i < len12345:
                if 'StaticMaterialDef' in lines[i]:
                    a = i
                    for j in range(a, len12345):
                        if lines[j] == '    }\n':
                            b = j
                            break
                    mat_lines.append((a, b))
                    i = b
                i += 1

            # redo this
            for a, b in mat_lines:
                path = None
                material = MAPGEOMaterial()
                for i in range(a, b):
                    if 'StaticMaterialDef' in lines[i]:
                        material.name = lines[i+1].split('=')[1][1:-
                                                                 1].replace('"', '').replace('/', '__')
                    if 'DiffuseTexture' in lines[i]:
                        material.texture = lines[i +
                                                 1].split('=')[1][1:].replace('"', '')[:-1]
                    if 'Diffuse_Texture' in lines[i]:
                        material.texture = lines[i +
                                                 1].split('=')[1][1:].replace('"', '')[:-1]
                    if 'Mask_Textures' in lines[i]:
                        material.texture = lines[i +
                                                 1].split('=')[1][1:].replace('"', '')[:-1]
                        for j in range(i+2, b):
                            if 'BaseColor' in lines[j]:
                                colors = lines[j+1].split('=')[1][2:-
                                                                  1].replace(' ', '').split(',')
                                material.ambient = (
                                    float(colors[0]), float(colors[1]), float(colors[2]))
                        break
                    if 'GlowTexture' in lines[i]:
                        material.texture = lines[i +
                                                 1].split('=')[1][1:].replace('"', '')[:-1]
                        for j in range(i+2, b):
                            if 'Color_Mult' in lines[j]:
                                colors = lines[j+1].split('=')[1][2:-
                                                                  1].replace(' ', '').split(',')
                                material.incandescence = (
                                    float(colors[0]), float(colors[1]), float(colors[2]))
                        break
                    if 'Emissive_Color' in lines[i]:
                        colors = lines[i+1].split('=')[1][2:-
                                                          1].replace(' ', '').split(',')
                        material.color = (float(colors[0]), float(
                            colors[1]), float(colors[2]))
                        material.ambient = material.color
                self.materials[material.name] = material


def db():
    mgbin = MAPGEOBin()
    mgbin.read('D:/base_srx.materials.py')
    # for path in mgbin.textures:
    #    print(path, mgbin.textures[path])
    mg = MAPGEO()
    mg.read('D:/base_srx.mapgeo')
    mg.flip()
    mg.load(mgbin=mgbin)
