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

        # check options to load separated mesh by materials
        # for combined mesh on 1 mat
        sepmat = False
        if 'sepmat=1' in options:
            sepmat = True
        if len(skn.submeshes) == 1:
            sepmat = False

        skn.load(skl, sepmat)
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

        # read riot.skl, path1 = riot_{same name}.skl > path2 = riot.skl
        riot_skl = None
        dirname, basename = MFPath.split(path)
        path1 = dirname + '/' + 'riot_' + basename.split('.skn')[0] + '.skl'
        if MFPath.exsist(path1):
            riot_skl = SKL()
            riot_skl.read(path1)
        else:
            path2 = dirname + '/' + 'riot.skl'
            if MFPath.exsist(path2):
                riot_skl = SKL()
                riot_skl.read(path2)

        skl = SKL()
        skl.dump(riot_skl)
        skl.flip()
        skl.write(dirname + '/' + basename.split('.skn')[0] + '.skl')

        # read riot.skn, path1 = riot_{same name}.skn > path2 = riot.skn
        riot_skn = None
        dirname, basename = MFPath.split(path)
        path1 = dirname + '/' + 'riot_' + basename
        if MFPath.exsist(path1):
            riot_skn = SKN()
            riot_skn.read(path1)
        else:
            path2 = dirname + '/' + 'riot.skn'
            if MFPath.exsist(path2):
                riot_skn = SKN()
                riot_skn.read(path2)

        skn = SKN()
        skn.dump(skl, riot_skn)
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

        # auto delete channel data before load
        delchannel = True
        if 'delchannel=1' not in options:
            delchannel = False

        anm = ANM()
        anm.read(path)
        anm.flip()
        anm.load(delchannel)
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

        # read riot.sco, path1 = riot_{same name}.sco > path2 = riot.sco
        riot_sco = None
        dirname, basename = MFPath.split(path)
        path1 = dirname + '/' + 'riot_' + basename
        if MFPath.exsist(path1):
            riot_sco = SO()
            riot_sco.read_sco(path1)
        else:
            path2 = dirname + '/' + 'riot.sco'
            if MFPath.exsist(path2):
                riot_sco = SO()
                riot_sco.read_sco(path2)

        so = SO()
        so.dump(riot=riot_sco)
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

        # read riot.scb, path1 = riot_{same name}.scb > path2 = riot.scb
        riot_scb = None
        dirname, basename = MFPath.split(path)
        path1 = dirname + '/' + 'riot_' + basename
        if MFPath.exsist(path1):
            riot_scb = SO()
            riot_scb.read_scb(path1)
        else:
            path2 = dirname + '/' + 'riot.scb'
            if MFPath.exsist(path2):
                riot_scb = SO()
                riot_scb.read_scb(path2)

        so = SO()
        so.dump(riot=riot_scb)
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

        # use standard surface material instead of lambert
        ssmat = True
        if 'ssmat=1' not in options:
            ssmat = False

        mg = MAPGEO()
        mg.read(path)
        mg.flip()
        mg.load(ssmat)
        return True

    def writer(self, file, options, access):
        if access != MPxFileTranslator.kExportActiveAccessMode:
            raise FunnyError(
                f'[MAPGEOTranslator.writer()]: Stop! u violated the law, use Export Selection or i violate u UwU.')

        path = file.expandedFullName()
        if not path.endswith('.mapgeo'):
            path += '.mapgeo'

        # read riot.mapgeo, path1 = riot_{same name}.mapgeo > path2 = riot.mapgeo
        riot_mapgeo = None
        dirname, basename = MFPath.split(path)
        path1 = dirname + '/' + 'riot_' + basename
        if MFPath.exsist(path1):
            riot_mapgeo = MAPGEO()
            riot_mapgeo.read(path1)
        else:
            path2 = dirname + '/' + 'riot.mapgeo'
            if MFPath.exsist(path2):
                riot_mapgeo = MAPGEO()
                riot_mapgeo.read(path2)

        mg = MAPGEO()
        mg.dump(riot=riot_mapgeo)
        mg.flip()
        mg.write(path)
        return True


# plugin register
def initializePlugin(obj):
    # totally not copied code
    plugin = MFnPlugin(obj, 'tarngaina', '3.6.0')
    try:
        plugin.registerFileTranslator(
            SKNTranslator.name,
            None,
            SKNTranslator.creator,
            'SKNTranslatorOpts',
            'skl=1;sepmat=0',
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
            'ANMTranslatorOpts',
            'delchannel=0',
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
            'MAPGEOTranslatorOpts',
            'ssmat=0',
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
# maya funny path functions
class MFPath():
    def exsist(path):
        mfo = MFileObject()
        mfo.setRawFullName(path)
        return mfo.exists()

    def split(path):
        temp = path.split('/')
        dirname = '/'.join(temp[:-1])
        basename = temp[-1]
        return dirname, basename


# funny error to catch
class FunnyError(Exception):
    def __init__(self, message):
        self.show(message)

    def show(self, message):
        title = 'Error:'
        if ']: ' in message:
            temp = message.split(']: ')
            title = temp[0][1:] + ':'
            message = temp[1]
        message = repr(message)[1:-1]
        button = choice(
            [
                'UwU', '<(\")', 'ok boomer', 'funny man', 'jesus', 'bruh', 'bro', 'please', 'man',
                'stop', 'get some help', 'haha', 'lmao', 'ay yo', 'SUS', 'sOcIEtY.', 'yeah', 'whatever',
                'gurl', 'fck', 'im ded', '(~`u`)~', 't(^u^t)', '(>w<)'
            ]
        )
        MGlobal.executeCommandStringResult(
            f'confirmDialog -title "{title}" -message "{message}" -button "{button}" -icon "critical"  -defaultButton "{button}"')


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

    def end(self):
        cur = self.stream.tell()
        self.stream.seek(0, 2)
        res = self.stream.tell()
        self.stream.seek(cur)
        return res

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
        res = []
        while True:
            c = self.read_char()
            if c == 0:
                break
            res.append(chr(c))
        return ''.join(res)

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

    def write_bool(self, value):
        self.write_bytes(pack('?', value))

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
            rsize = bs.end()
            bs.seek(0)
            bs.write_uint32(rsize)

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
        riot_id = 0
        execmd = ''
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

            # add custom attribute: Riot ID
            util = MScriptUtil()
            ptr = util.asIntPtr()
            MGlobal.executeCommand(
                f'attributeQuery -ex -n "{joint.name}" "riotid"', ptr)
            if util.getInt(ptr) == 0:
                execmd += f'addAttr -ln "riotid" -nn "Riot ID" -at byte -min 0 -max 255 -dv {riot_id} "{joint.name}";'
            execmd += f'setAttr {joint.name}.riotid {riot_id};'
            riot_id += 1

            ik_joint.set(MTransform.compose(
                joint.local_translation, joint.local_scale, joint.local_rotation, MSpace.kWorld
            ))
        MGlobal.executeCommand(execmd)

        # link parent
        for joint in self.joints:
            if joint.parent > -1:
                parent_node = MFnIkJoint(self.joints[joint.parent].dagpath)
                child_node = MFnIkJoint(joint.dagpath)
                if not parent_node.isParentOf(child_node.object()):
                    parent_node.addChild(child_node.object())

    def dump(self, riot=None):
        # iterator on all joints
        # to dump dagpath, name and transform
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
                # fill empty joint
                if not found:
                    MGlobal.displayWarning(
                        f'[SKL.dump(riot.skl)]: Missing riot joint: {riot_joint.name}')
                    joint = SKLJoint()
                    joint.dagpath = None
                    joint.name = riot_joint.name
                    joint.parent = -1
                    joint.local_translation = MVector(0.0, 0.0, 0.0)
                    joint.local_rotation = MQuaternion(0.0, 0.0, 0.0, 0.0)
                    joint.local_scale = MVector(0.0, 0.0, 0.0)
                    joint.iglobal_translation = MVector(0.0, 0.0, 0.0)
                    joint.iglobal_rotation = MQuaternion(0.0, 0.0, 0.0, 0.0)
                    joint.iglobal_scale = MVector(0.0, 0.0, 0.0)
                    new_joints.append(joint)

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
        else:
            new_joints = []

            # init things
            joint_count = len(self.joints)
            riot_joints = list([None]*joint_count)
            other_joints = []
            flags = list([True] * joint_count)

            # get riot joints through ID attribute
            for joint in self.joints:
                util = MScriptUtil()
                ptr = util.asIntPtr()
                MGlobal.executeCommand(
                    f'attributeQuery -ex -n "{joint.name}" "riotid"', ptr)

                # if joint doesnt have attribute -> new joint
                if util.getInt(ptr) == 0:
                    other_joints.append(joint)
                    continue

                util = MScriptUtil()
                ptr = util.asIntPtr()
                MGlobal.executeCommand(
                    f'getAttr {joint.name}.riotid', ptr)
                id = util.getInt(ptr)

                # if id out of range -> bad joint
                if id < 0 or id >= joint_count:
                    other_joints.append(joint)
                    continue

                # if duplicated ID -> bad joint
                if riot_joints[id]:
                    other_joints.append(joint)
                    continue

                riot_joints[id] = joint

            # add good joints first
            for joint in riot_joints:
                if joint != None:
                    new_joints.append(joint)

            # add new/bad joints at the end
            for joint in other_joints:
                new_joints.append(joint)

            self.joints = new_joints

        # link parent
        joint_count = len(self.joints)
        for joint in self.joints:
            if not joint.dagpath:
                continue
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
        self.new_index = None


class SKNSubmesh:
    def __init__(self):
        self.name = None
        self.vertex_start = None
        self.vertex_count = None
        self.index_start = None
        self.index_count = None

        # for dumping
        self.indices = []
        self.vertices = []


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

    def load(self, skl=None, sepmat=False):
        def load_combined():
            vertex_count = len(self.vertices)
            index_count = len(self.indices)
            face_count = index_count // 3

            # create mesh
            vertices = MFloatPointArray()
            u_values = MFloatArray()
            v_values = MFloatArray()
            for vertex in self.vertices:
                vertices.append(MFloatPoint(
                    vertex.position.x, vertex.position.y, vertex.position.z))
                u_values.append(vertex.uv.x)
                v_values.append(1.0 - vertex.uv.y)

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

            # name
            mesh.setName(f'{self.name}Shape')
            mesh_name = mesh.name()
            MFnTransform(mesh.parent(0)).setName(f'mesh_{self.name}')

            # materials
            execmd = ''
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
                # create renderable, independent shading group
                execmd += f'sets -renderable true -noSurfaceShader true -empty -name "{lambert_name}_SG";'
                # add submesh faces to shading group
                execmd += f'sets -e -forceElement "{lambert_name}_SG" {mesh_name}.f[{face_start}:{face_end}];'
                # connect lambert to shading group
                execmd += f'connectAttr -f {lambert_name}.outColor {lambert_name}_SG.surfaceShader;'
            MGlobal.executeCommand(execmd)

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
                plugs = MPlugArray()
                in_mesh.connectedTo(plugs, True, False)
                skin_cluster = MFnSkinCluster(plugs[0].node())
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
                # empty vertex_component = all vertices
                vertex_component = component.create(MFn.kMeshVertComponent)
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
                MGlobal.executeCommand((
                    f'setAttr {skin_cluster_name}.normalizeWeights 1;'
                    f'skinPercent -normalize true {skin_cluster_name} {mesh_name};'
                ))

            MGlobal.executeCommand('select -cl')
            # shud be final line
            mesh.updateSurface()

        def load_separated():
            # group of meshes
            group_transform = MFnTransform()
            group_transform.create()
            group_transform.setName(f'group_{self.name}')

            # init seperated meshes data
            shader_count = len(self.submeshes)
            shader_vertices = {}
            shader_indices = {}
            shader_meshes = []
            for shader_index in range(0, shader_count):
                submesh = self.submeshes[shader_index]
                shader_vertices[shader_index] = self.vertices[submesh.vertex_start:
                                                              submesh.vertex_start+submesh.vertex_count]
                shader_indices[shader_index] = self.indices[submesh.index_start:
                                                            submesh.index_start+submesh.index_count]
                min_vertex = min(shader_indices[shader_index])
                shader_index_count = len(shader_indices[shader_index])
                for i in range(0, shader_index_count):
                    shader_indices[shader_index][i] -= min_vertex

            execmd = ''
            for shader_index in range(0, shader_count):
                vertex_count = len(shader_vertices[shader_index])
                index_count = len(shader_indices[shader_index])
                face_count = index_count // 3

                # create mesh
                vertices = MFloatPointArray()
                u_values = MFloatArray()
                v_values = MFloatArray()
                for vertex in shader_vertices[shader_index]:
                    vertices.append(MFloatPoint(
                        vertex.position.x, vertex.position.y, vertex.position.z))
                    u_values.append(vertex.uv.x)
                    v_values.append(1.0 - vertex.uv.y)

                poly_count = MIntArray(face_count, 3)
                poly_indices = MIntArray()
                MScriptUtil.createIntArrayFromList(
                    shader_indices[shader_index], poly_indices)
                normal_indices = MIntArray()
                MScriptUtil.createIntArrayFromList(
                    list(range(vertex_count)), normal_indices)

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

                # save the MFnMesh to bind later
                shader_meshes.append(mesh)

                # name
                submesh = self.submeshes[shader_index]
                mesh.setName(f'{self.name}_{submesh.name}Shape')
                mesh_name = mesh.name()
                mesh_transform = MFnTransform(mesh.parent(0))
                mesh_transform.setName(
                    f'mesh_{submesh.name}')

                # add mesh to group
                group_transform.addChild(mesh_transform.object())

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
                # create renderable, independent shading group
                execmd += f'sets -renderable true -noSurfaceShader true -empty -name "{lambert_name}_SG";'
                # add submesh faces to shading group
                execmd += f'sets -e -forceElement "{lambert_name}_SG" {mesh_name}.f[0:{face_count}];'
                # connect lambert to shading group
                execmd += f'connectAttr -f {lambert_name}.outColor {lambert_name}_SG.surfaceShader;'
            MGlobal.executeCommand(execmd)

            if skl:
                for shader_index in range(0, shader_count):
                    # get mesh base on shader
                    mesh = shader_meshes[shader_index]
                    mesh_name = mesh.name()
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
                        f'skinCluster -mi 4 -tsb -n {mesh_name}_skinCluster')

                    # get skin cluster
                    in_mesh = mesh.findPlug('inMesh')
                    plugs = MPlugArray()
                    in_mesh.connectedTo(plugs, True, False)
                    skin_cluster = MFnSkinCluster(
                        plugs[0].node())
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
                    vertex_count = len(shader_vertices[shader_index])
                    weights = MDoubleArray(vertex_count * influence_count)
                    for i in range(0, vertex_count):
                        vertex = shader_vertices[shader_index][i]
                        for j in range(0, 4):
                            weight = vertex.weights[j]
                            influence = vertex.influences[j][0]
                            if weight > 0:
                                weights[i * influence_count +
                                        influence] = weight
                    skin_cluster.setWeights(
                        mesh_dagpath, vertex_component, mask_influence, weights, False)
                    MGlobal.executeCommand((
                        f'setAttr {skin_cluster_name}.normalizeWeights 1;'
                        f'skinPercent -normalize true {skin_cluster_name} {mesh_name}'
                    ))

            MGlobal.executeCommand('select -cl')
            # shud be final line
            for mesh in shader_meshes:
                mesh.updateSurface()

        if sepmat:
            load_separated()
        else:
            load_combined()

    def dump(self, skl, riot=None):
        def dump_mesh(mesh):
            # get mesh DAG path
            mesh_dagpath = MDagPath()
            mesh.getPath(mesh_dagpath)
            mesh_vertex_count = mesh.numVertices()

            iterator = MItDependencyGraph(
                mesh.object(), MFn.kSkinClusterFilter, MItDependencyGraph.kUpstream)
            if iterator.isDone():
                raise FunnyError(
                    f'[SKN.dump({mesh.name()})]: No skin cluster found, make sure you bound the skin.')
            skin_cluster = MFnSkinCluster(iterator.currentItem())

            # check holes
            hole_info = MIntArray()
            hole_vertices = MIntArray()
            mesh.getHoles(hole_info, hole_vertices)
            if hole_info.length() != 0:
                raise FunnyError(
                    f'[SKN.dump({mesh.name()})]: Mesh contains {hole_info.length()} holes.')

            # get shader/materials
            shaders = MObjectArray()
            face_shader = MIntArray()
            instance = mesh_dagpath.instanceNumber() if mesh_dagpath.isInstanced() else 0
            mesh.getConnectedShaders(instance, shaders, face_shader)
            shader_count = shaders.length()
            # check no material assigned
            if shader_count < 1:
                raise FunnyError(
                    f'[SKN.dump({mesh.name()})]: No material assigned to this mesh, please assign one.')
            # init shaders data to work on multiple shader
            shader_vertices = []
            shader_indices = []
            shader_names = []
            for i in range(0, shader_count):
                shader_vertices.append([])
                shader_indices.append([])
                # get shader name
                ss = MFnDependencyNode(
                    shaders[i]).findPlug('surfaceShader')
                plugs = MPlugArray()
                ss.connectedTo(plugs, True, False)
                shader_node = MFnDependencyNode(plugs[0].node())
                shader_names.append(shader_node.name())

            # iterator on faces - 1st
            # to get vertex_shader first base on face_shader
            # extra checking stuffs
            bad_faces = MIntArray()  # invalid triangulation polygon
            bad_faces2 = MIntArray()  # no material assigned
            bad_faces3 = MIntArray()  # no uv assigned
            bad_vertices = MIntArray()  # shared vertices
            vertex_shader = MIntArray(mesh_vertex_count, -1)
            iterator = MItMeshPolygon(mesh_dagpath)
            iterator.reset()
            while not iterator.isDone():
                # get shader of this face
                face_index = iterator.index()
                shader_index = face_shader[face_index]

                # check valid triangulation
                if not iterator.hasValidTriangulation():
                    if face_index not in bad_faces:
                        bad_faces.append(face_index)
                # check face with no material assigned
                if shader_index == -1:
                    if face_index not in bad_faces2:
                        bad_faces2.append(face_index)
                # check if face has no UVs
                if not iterator.hasUVs():
                    if face_index not in bad_faces3:
                        bad_faces3.append(face_index)
                # get face vertices
                vertices = MIntArray()
                iterator.getVertices(vertices)
                face_vertex_count = vertices.length()
                # check if each vertex is shared by mutiple materials
                for i in range(0, face_vertex_count):
                    vertex = vertices[i]
                    if vertex_shader[vertex] not in [-1, shader_index]:
                        if vertex not in bad_vertices:
                            bad_vertices.append(vertex)
                        continue
                    vertex_shader[vertex] = shader_index
                iterator.next()
            if bad_faces.length() > 0:
                component = MFnSingleIndexedComponent()
                face_component = component.create(
                    MFn.kMeshPolygonComponent)
                component.addElements(bad_faces)
                selections = MSelectionList()
                selections.add(mesh_dagpath, face_component)
                MGlobal.selectCommand(selections)
                raise FunnyError(
                    f'[SKN.dump({mesh.name()})]: Mesh contains {bad_faces.length()} invalid triangulation faces, those faces will be selected in scene.\nBonus: If there is nothing selected (or they are invisible) after this error message, consider to delete history and rebind the skin, that might fix the problem.')
            if bad_faces2.length() > 0:
                component = MFnSingleIndexedComponent()
                face_component = component.create(
                    MFn.kMeshPolygonComponent)
                component.addElements(bad_faces2)
                selections = MSelectionList()
                selections.add(mesh_dagpath, face_component)
                MGlobal.selectCommand(selections)
                raise FunnyError(
                    f'[SKN.dump({mesh.name()})]: Mesh contains {bad_faces2.length()} faces have no material assigned, those faces will be selected in scene.\nBonus: If there is nothing selected (or they are invisible) after this error message, consider to delete history and rebind the skin, that might fix the problem.')
            if bad_faces3.length() > 0:
                component = MFnSingleIndexedComponent()
                face_component = component.create(
                    MFn.kMeshPolygonComponent)
                component.addElements(bad_faces3)
                selections = MSelectionList()
                selections.add(mesh_dagpath, face_component)
                MGlobal.selectCommand(selections)
                raise FunnyError(
                    f'[SKN.dump({mesh.name()})]: Mesh contains {bad_faces3.length()} faces have no UVs assigned, or, those faces UVs are not in current UV set, those faces will be selected in scene.\nBonus: If there is nothing selected (or they are invisible) after this error message, consider to delete history and rebind the skin, that might fix the problem.')
            if bad_vertices.length() > 0:
                component = MFnSingleIndexedComponent()
                vertex_component = component.create(
                    MFn.kMeshVertComponent)
                component.addElements(bad_vertices)
                selections = MSelectionList()
                selections.add(mesh_dagpath, vertex_component)
                MGlobal.selectCommand(selections)
                raise FunnyError((
                    f'[SKN.dump({mesh.name()})]: Mesh contains {bad_vertices.length()} vertices are shared by mutiple materials, those vertices will be selected in scene.\n'
                    'Save/backup scene first, try one of following methods to fix:\n'
                    '1. Seperate all connected faces that shared those vertices.\n'
                    '2. Check and reassign correct material.\n'
                    '3. [recommended] Try auto fix shared vertices button on shelf.'
                    '\nBonus: If there is nothing selected (or they are invisible) after this error message, consider to delete history and rebind the skin, that might fix the problem.'
                ))

            # get weights of all vertices
            # empty vertex component = all vertices
            component = MFnSingleIndexedComponent()
            vertex_component = component.create(MFn.kMeshVertComponent)
            weights = MDoubleArray()
            util = MScriptUtil()
            ptr = util.asUintPtr()
            skin_cluster.getWeights(
                mesh_dagpath, vertex_component, weights, ptr)
            weight_influence_count = util.getUint(ptr)
            # map influence indices by skl joints
            influence_dagpaths = MDagPathArray()
            influence_count = skin_cluster.influenceObjects(influence_dagpaths)
            map_influence_indices = MIntArray(influence_count)
            joint_count = len(skl.joints)
            for i in range(0, influence_count):
                dagpath = influence_dagpaths[i]
                for j in range(0, joint_count):
                    if dagpath == skl.joints[j].dagpath:
                        map_influence_indices[i] = j
                        break
            # get all uvs
            u_values = MFloatArray()
            v_values = MFloatArray()
            mesh.getUVs(u_values, v_values)
            # iterator on vertices
            # to dump all new vertices base on unique uv
            bad_vertices = MIntArray()  # vertex has 4+ influences
            bad_vertices2 = MIntArray()  # vertex has no UVs
            iterator = MItMeshVertex(mesh_dagpath)
            iterator.reset()
            while not iterator.isDone():
                # get shader of this vertex
                vertex_index = iterator.index()
                shader_index = vertex_shader[vertex_index]
                if shader_index == -1:
                    # a strange vertex with no shader ?
                    # let say this vertex is alone and not in any face
                    # just ignore it?
                    iterator.next()
                    continue

                # influence and weight
                influences = [bytes([0]), bytes(
                    [0]), bytes([0]), bytes([0])]
                vertex_weights = [0.0, 0.0, 0.0, 0.0]
                inf_index = 0
                for influence in range(0, weight_influence_count):
                    weight = weights[vertex_index *
                                     weight_influence_count + influence]
                    if weight > 0.001:  # prune weight 0.001
                        # check 4+ influneces
                        if inf_index > 3:
                            bad_vertices.append(vertex_index)
                            break
                        influences[inf_index] = bytes(
                            [map_influence_indices[influence]])
                        vertex_weights[inf_index] = weight
                        inf_index += 1
                # normalize weight
                weight_sum = sum(vertex_weights)
                if weight_sum > 0:
                    for i in range(0, 4):
                        if vertex_weights[i] > 0:
                            vertex_weights[i] /= weight_sum
                # position
                position = iterator.position(MSpace.kWorld)
                # average of normals of all faces connect to this vertex
                normals = MVectorArray()
                iterator.getNormals(normals)
                normal_count = normals.length()
                normal = MVector(0.0, 0.0, 0.0)
                for i in range(0, normal_count):
                    normal += normals[i]
                normal /= normal_count
                # unique uv
                uv_indices = MIntArray()
                iterator.getUVIndices(uv_indices)
                uv_count = uv_indices.length()
                if uv_count > 0:
                    seen = []
                    for i in range(0, uv_count):
                        uv_index = uv_indices[i]
                        if uv_index == -1:
                            continue
                            # raise FunnyError(
                            #    f'[SKN.dump({mesh.name()})]: No uv_index found on a vertex, this error should not happen. Possibly caused by bad mesh history and delete/bake history might fix this problem.')
                        if uv_index not in seen:
                            seen.append(uv_index)
                            uv = MVector(
                                u_values[uv_index],
                                1.0 - v_values[uv_index]
                            )
                            # dump vertices
                            vertex = SKNVertex()
                            vertex.position = MVector(position)
                            vertex.normal = MVector(normal)
                            vertex.influences = influences
                            vertex.weights = vertex_weights
                            vertex.uv = uv
                            vertex.uv_index = uv_index
                            vertex.new_index = len(
                                shader_vertices[shader_index])
                            shader_vertices[shader_index].append(vertex)
                else:
                    if vertex_index not in bad_vertices:
                        bad_vertices.append(vertex_index)
                iterator.next()
            if bad_vertices.length() > 0:
                component = MFnSingleIndexedComponent()
                vertex_component = component.create(
                    MFn.kMeshVertComponent)
                component.addElements(bad_vertices)
                selections = MSelectionList()
                selections.add(mesh_dagpath, vertex_component)
                MGlobal.selectCommand(selections)
                raise FunnyError((
                    f'[SKN.dump({mesh.name()})]: Mesh contains {bad_vertices.length()} vertices that have weight on 4+ influences, those vertices will be selected in scene.\n'
                    'Save/backup scene first, try one of following methods to fix:\n'
                    '1. Repaint weight on those vertices.\n'
                    '2. Prune small weights.\n'
                    '3. [recommended] Try auto fix 4 influences button on shelf.'
                    '\nBonus: If there is nothing selected (or they are invisible) after this error message, consider to delete history and rebind the skin, that might fix the problem.'
                ))
            if bad_vertices2.length() > 0:
                raise FunnyError(
                    f'[SKN.dump({mesh.name()})]: Mesh contains {bad_vertices2.length()} vertices have no UVs assigned, those vertices will be selected in scene.\nBonus: If there is nothing selected (or they are invisible) after this error message, consider to delete history and rebind the skin, that might fix the problem.')

            # map new vertices base on uv_index
            map_vertices = {}
            for shader_index in range(0, shader_count):
                map_vertices[shader_index] = {}
                for vertex in shader_vertices[shader_index]:
                    map_vertices[shader_index][vertex.uv_index] = vertex.new_index
            # iterator on faces - 2nd
            # to dump indices:
            # - triangulated indices
            # - new indices base on new unique uv vertices
            iterator = MItMeshPolygon(mesh_dagpath)
            iterator.reset()
            while not iterator.isDone():
                # get shader of this face
                face_index = iterator.index()
                shader_index = face_shader[face_index]

                # get triangulated face indices
                points = MPointArray()
                indices = MIntArray()
                iterator.getTriangles(points, indices)
                face_index_count = indices.length()
                # get face vertices
                map_indices = {}
                vertices = MIntArray()
                iterator.getVertices(vertices)
                face_vertex_count = vertices.length()
                # map face indices by uv_index
                for i in range(0, face_vertex_count):
                    util = MScriptUtil()
                    ptr = util.asIntPtr()
                    iterator.getUVIndex(i, ptr)
                    uv_index = util.getInt(ptr)
                    map_indices[vertices[i]] = uv_index
                # map new indices by new vertices through uv_index
                # and add new indices
                for i in range(0, face_index_count):
                    new_indices = map_vertices[shader_index][map_indices[indices[i]]]
                    shader_indices[shader_index].append(new_indices)
                iterator.next()

            # return list of submeshes dumped out of this mesh
            submeshes = []
            for i in range(0, shader_count):
                submesh = SKNSubmesh()
                submesh.name = shader_names[i]
                submesh.indices = shader_indices[i]
                submesh.vertices = shader_vertices[i]
                submeshes.append(submesh)
            return submeshes

        # find mesh in selections
        selections = MSelectionList()
        MGlobal.getActiveSelectionList(selections)
        iterator = MItSelectionList(selections, MFn.kTransform)
        if iterator.isDone():
            raise FunnyError(
                f'[SKN.dump()]: Please select a mesh or group of meshes.')

        # get first transform dagpath in selections
        iterator_dagpath = MDagPath()
        selected_dagpath = None
        while not iterator.isDone():
            iterator.getDagPath(iterator_dagpath)
            if iterator_dagpath.apiType() == MFn.kTransform:
                if not selected_dagpath:
                    selected_dagpath = MDagPath(iterator_dagpath)
                else:
                    raise FunnyError(
                        '[SKN.dump()]: Too many selected objects. Please select only a mesh or group of meshes.')
            iterator.next()

        # dump all submeshes data out of selected transform
        submeshes = []
        selected_transform = MFnTransform(selected_dagpath)
        selected_child_count = selected_transform.childCount()
        if selected_child_count == 0:
            raise FunnyError(
                f'[SKN.dump({selected_transform.name()})]: Selected object is not a mesh or group of meshes?')
        first_child = selected_transform.child(0)
        if first_child.apiType() == MFn.kMesh:
            submeshes += dump_mesh(MFnMesh(first_child))
        else:
            for i in range(0, selected_child_count):
                transform_child = MFnTransform(
                    selected_transform.child(i))
                if transform_child.childCount() > 0:
                    first_grand_child = transform_child.child(0)
                    if first_grand_child.apiType() == MFn.kMesh:
                        submeshes += dump_mesh(MFnMesh(first_grand_child))

        # map submeshes by name
        map_submeshes = {}
        for submesh in submeshes:
            if submesh.name not in map_submeshes:
                map_submeshes[submesh.name] = []
            map_submeshes[submesh.name].append(submesh)
        # combine all submesh that has same name
        # save to SKN submeshes
        for submesh_name in map_submeshes:
            # check submesh name's length
            if len(submesh_name) > 64:
                raise FunnyError(
                    f'[SKN.dump()]: Material name is too long: {submesh_name} with {len(submesh_name)} chars, max allowed: 64 chars.')
            combined_submesh = SKNSubmesh()
            combined_submesh.name = submesh_name
            previous_max_index = 0
            for submesh in map_submeshes[submesh_name]:
                combined_submesh.vertices += submesh.vertices
                if previous_max_index > 0:
                    previous_max_index += 1
                combined_submesh.indices += [index +
                                             previous_max_index for index in submesh.indices]
                previous_max_index = max(combined_submesh.indices)
            self.submeshes.append(combined_submesh)

        # calculate SKN indices, vertices and update SKN submeshes data
        # for first submesh
        self.submeshes[0].index_start = 0
        self.submeshes[0].index_count = len(self.submeshes[0].indices)
        self.indices += self.submeshes[0].indices
        self.submeshes[0].vertex_start = 0
        self.submeshes[0].vertex_count = len(self.submeshes[0].vertices)
        self.vertices += self.submeshes[0].vertices
        # for the rest if more than 1 submeshes
        submesh_count = len(self.submeshes)
        if submesh_count > 1:
            index_start = self.submeshes[0].index_count
            vertex_start = self.submeshes[0].vertex_count
            for i in range(1, submesh_count):
                self.submeshes[i].index_start = index_start
                self.submeshes[i].index_count = len(self.submeshes[i].indices)
                max_index = max(self.submeshes[i-1].indices)
                self.submeshes[i].indices = [
                    index + max_index+1 for index in self.submeshes[i].indices]
                self.indices += self.submeshes[i].indices

                self.submeshes[i].vertex_start = vertex_start
                self.submeshes[i].vertex_count = len(
                    self.submeshes[i].vertices)
                self.vertices += self.submeshes[i].vertices

                index_start += self.submeshes[i].index_count
                vertex_start += self.submeshes[i].vertex_count

        # sort submesh to match riot.skn submeshes order
        if riot:
            MGlobal.displayInfo(
                '[SKL.dump(riot.skn)]: Found riot.skn, sorting materials...')

            new_submeshes = []
            submesh_count = len(self.submeshes)
            # for adding extra material at the end of list
            flags = list([True] * submesh_count)

            # find riot submesh in scene
            for riot_submesh in riot.submeshes:
                riot_submesh_name = riot_submesh.name.lower()
                found = False
                for i in range(0, submesh_count):
                    if flags[i] and self.submeshes[i].name.lower() == riot_submesh_name:
                        new_submeshes.append(self.submeshes[i])
                        MGlobal.displayInfo(
                            f'[SKL.dump(riot.skn)]: Found material: {self.submeshes[i].name}')
                        flags[i] = False
                        found = True
                        break

                # submesh that not found
                if not found:
                    MGlobal.displayWarning(
                        f'[SKL.dump(riot.skn)]: Missing riot material: {riot_submesh.name}')

            # add extra/addtional materials to the end of list
            for i in range(0, submesh_count):
                if flags[i]:
                    new_submeshes.append(self.submeshes[i])
                    flags[i] = False
                    MGlobal.displayInfo(
                        f'[SKL.dump(riot.skn)]: New material: {self.submeshes[i].name}')

            # assign new list
            self.submeshes = new_submeshes

        # check limit vertices
        vertices_count = len(self.vertices)
        if vertices_count > 65535:
            raise FunnyError(
                f'[SKN.dump()]: Too many vertices found: {vertices_count}, max allowed: 65535 vertices. (base on UVs)')

        # check limit submeshes
        submesh_count = len(self.submeshes)
        if submesh_count > 32:
            raise FunnyError(
                f'[SKN.dump()]: Too many materials assigned: {submesh_count}, max allowed: 32 materials.')


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

    def write(self, path):
        # build unique vecs + quats
        uni_vecs = {}
        uni_quats = {}

        vec_index = 0
        quat_index = 0
        for time in range(1, self.duration+1):
            for track in self.tracks:
                pose = track.poses[time]
                translation_key = f'{pose.translation.x:.6f} {pose.translation.y:.6f} {pose.translation.z:.6f}'
                scale_key = f'{pose.scale.x:.6f} {pose.scale.y:.6f} {pose.scale.z:.6f}'
                rotation_key = f'{pose.rotation.x:.6f} {pose.rotation.y:.6f} {pose.rotation.z:.6f} {pose.rotation.w:.6f}'
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
            rsize = bs.end()
            bs.seek(12)
            bs.write_uint32(rsize)

    def load(self, delchannel=False):
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

        execmd = ''
        # delete all channel data
        if delchannel:
            execmd += 'delete -all -c;'

        # ensure scene fps
        # this only ensure the "import scene", not the "opening/existing scene" in maya, to make this work:
        # select "Override to Math Source" for both Framerate % Animation Range in Maya's import options panel
        if self.fps > 59:
            execmd += 'currentUnit -time ntscf;'
        else:
            execmd += 'currentUnit -time ntsc;'

        # get current time
        util = MScriptUtil()
        ptr = util.asDoublePtr()
        MGlobal.executeCommand('currentTime -q', ptr)
        current = util.getDouble(ptr)

        # adjust animation range
        end = self.duration * self.fps
        execmd += f'playbackOptions -e -min 0 -max {current+end} -animationStartTime 0 -animationEndTime {current+end} -playbackSpeed 1;'

        # bind current pose to frame 0 - very helpful if its bind pose
        joint_names = ' '.join([track.joint_name for track in scene_tracks])
        execmd += f'currentTime 0;setKeyframe -breakdown 0 -hierarchy none -controlPoints 0 -shape 0 -at translateX -at translateY -at translateZ -at scaleX -at scaleY -at scaleZ -at rotateX -at rotateY -at rotateZ {joint_names};'

        MGlobal.executeCommand(execmd)

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
            MGlobal.executeCommand(
                f'currentTime {current + time * self.fps + 1};')

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
                        ekf = False
                    # scale
                    if pose.scale:
                        scale = pose.scale
                        util = MScriptUtil()
                        util.createFromDouble(scale.x, scale.y, scale.z)
                        ptr = util.asDoublePtr()
                        ik_joint.setScale(ptr)
                        setKeyFrame += f' {track.joint_name}.scaleX {track.joint_name}.scaleY {track.joint_name}.scaleZ'
                        ekf = False
                    # rotation
                    if pose.rotation:
                        orient = MQuaternion()
                        ik_joint.getOrientation(orient)
                        axe = ik_joint.rotateOrientation(MSpace.kTransform)
                        ik_joint.setRotation(
                            axe.inverse() * pose.rotation * orient.inverse(), MSpace.kTransform)
                        setKeyFrame += f' {track.joint_name}.rotateX {track.joint_name}.rotateY {track.joint_name}.rotateZ'
                        ekf = False

            if not ekf:
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


# static object - sco/scb
class SO:
    def __init__(self):
        self.name = None
        self.central = None

        # for sco only
        self.pivot = None

        # assume sco/scb only have 1 material
        self.material = None
        self.indices = []
        # important: uv can be different at each index, can not map this uv data by vertex
        self.uvs = []
        # not actual vertex, its a position of vertex, no reason to create a class
        self.vertices = []

        # for scb only
        # 1 - vertex color
        # 2 - local origin locator and pivot
        self.scb_flag = 2

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

            self.scb_flag = bs.read_uint32()
            bs.pad(24)  # bouding box

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
                    f'{self.uvs[index+1].x:.12f} {self.uvs[index+1].y:.12f} ')
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
            bs.write_uint32(self.scb_flag)

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
            # connect lambert to shading group
            f'connectAttr -f {lambert_name}.outColor {lambert_name}_SG.surfaceShader;'
        ))

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

        MGlobal.executeCommand('select -cl')
        mesh.updateSurface()

    def dump(self, riot=None):
        # find mesh in selections
        selections = MSelectionList()
        MGlobal.getActiveSelectionList(selections)
        iterator = MItSelectionList(selections, MFn.kMesh)
        if iterator.isDone():
            raise FunnyError(
                f'[SO.dump()]: Please select a mesh.')
        # get mesh's DAG path
        mesh_dagpath = MDagPath()
        iterator.getDagPath(mesh_dagpath)
        iterator.next()
        if not iterator.isDone():
            raise FunnyError(
                f'[SO.dump()]: More than 1 mesh selected.')
        # get MFnMesh
        mesh = MFnMesh(mesh_dagpath)

        # get name
        self.name = mesh.name()

        # central point: translation of mesh
        transform = MFnTransform(mesh.parent(0))
        self.central = transform.getTranslation(MSpace.kTransform)

        # check hole
        hole_info = MIntArray()
        hole_vertex = MIntArray()
        mesh.getHoles(hole_info, hole_vertex)
        if hole_info.length() > 0:
            raise FunnyError(f'[SO.dump({mesh.name()})]: Mesh contains holes.')

        # SCO only: find pivot joint through skin cluster
        iterator = MItDependencyGraph(
            mesh.object(), MFn.kSkinClusterFilter, MItDependencyGraph.kUpstream)
        if not iterator.isDone():
            skin_cluster = MFnSkinCluster(iterator.currentItem())
            influences_dagpath = MDagPathArray()
            influence_count = skin_cluster.influenceObjects(
                influences_dagpath)
            if influence_count > 1:
                raise FunnyError(
                    f'[SO.dump({mesh.name()})]: There is more than 1 joint bound with this mesh.')
            ik_joint = MFnTransform(influences_dagpath[0])
            translation = ik_joint.getTranslation(MSpace.kTransform)
            self.pivot = self.central - translation

        # dumb vertices
        vertex_count = mesh.numVertices()
        points = MFloatPointArray()
        mesh.getPoints(points, MSpace.kWorld)
        for i in range(0, vertex_count):
            self.vertices.append(
                MVector(points[i].x, points[i].y, points[i].z))

        # dump uvs outside loop
        u_values = MFloatArray()
        v_values = MFloatArray()
        mesh.getUVs(u_values, v_values)
        # iterator on faces
        # to dump face indices and UVs
        # extra checking
        bad_faces = MIntArray()  # invalid triangulation face
        bad_faces2 = MIntArray()  # no UV face
        iterator = MItMeshPolygon(mesh_dagpath)
        iterator.reset()
        while not iterator.isDone():
            face_index = iterator.index()

            # check valid triangulation
            if not iterator.hasValidTriangulation():
                if face_index not in bad_faces:
                    bad_faces.append(face_index)
            # check if face has no UVs
            if not iterator.hasUVs():
                if face_index not in bad_faces2:
                    bad_faces2.append(face_index)

            # get triangulated face indices
            points = MPointArray()
            indices = MIntArray()
            iterator.getTriangles(points, indices)
            face_index_count = indices.length()
            # get face vertices
            map_indices = {}
            vertices = MIntArray()
            iterator.getVertices(vertices)
            face_vertex_count = vertices.length()
            # map face indices by uv_index
            for i in range(0, face_vertex_count):
                util = MScriptUtil()
                ptr = util.asIntPtr()
                iterator.getUVIndex(i, ptr)
                uv_index = util.getInt(ptr)
                map_indices[vertices[i]] = uv_index
            # dump indices and uvs
            for i in range(0, face_index_count):
                index = indices[i]
                self.indices.append(index)
                uv_index = map_indices[index]
                self.uvs.append(MVector(
                    u_values[uv_index],
                    1.0 - v_values[uv_index]
                ))
            iterator.next()
        if bad_faces.length() > 0:
            component = MFnSingleIndexedComponent()
            face_component = component.create(
                MFn.kMeshPolygonComponent)
            component.addElements(bad_faces)
            selections = MSelectionList()
            selections.add(mesh_dagpath, face_component)
            MGlobal.selectCommand(selections)
            raise FunnyError(
                f'[SO.dump({mesh.name()})]: Mesh contains {bad_faces.length()} invalid triangulation faces, those faces will be selected in scene.\nBonus: If there is nothing selected (or they are invisible) after this error message, consider to delete history, that might fix the problem.')
        if bad_faces2.length() > 0:
            component = MFnSingleIndexedComponent()
            face_component = component.create(
                MFn.kMeshPolygonComponent)
            component.addElements(bad_faces2)
            selections = MSelectionList()
            selections.add(mesh_dagpath, face_component)
            MGlobal.selectCommand(selections)
            raise FunnyError(
                f'[SO.dump({mesh.name()})]: Mesh contains {bad_faces2.length()} faces have no UVs assigned, or, those faces UVs are not in current UV set, those faces will be selected in scene.\nBonus: If there is nothing selected (or they are invisible) after this error message, consider to delete history, that might fix the problem.')

        # get shader
        instance = mesh_dagpath.instanceNumber() if mesh_dagpath.isInstanced() else 0
        shaders = MObjectArray()
        face_shader = MIntArray()
        mesh.getConnectedShaders(instance, shaders, face_shader)
        if shaders.length() > 1:
            raise FunnyError(
                f'[SO.dump({mesh.name()})]: There are more than 1 material assigned to this mesh.')
        # material name
        if shaders.length() > 0:
            ss = MFnDependencyNode(
                shaders[0]).findPlug('surfaceShader')
            plugs = MPlugArray()
            ss.connectedTo(plugs, True, False)
            material = MFnDependencyNode(plugs[0].node())
            self.material = material.name()
            if len(self.material) > 64:
                raise FunnyError(
                    f'[SO.dump()]: Material name is too long: {self.material} with {len(self.material)} chars, max allowed: 64 chars.')
        else:
            # its only allow 1 material anyway
            self.material = 'lambert'

        # export base on riot file
        if riot:
            self.central = riot.central
            self.pivot = riot.pivot
            self.scb_flag = riot.scb_flag
            MGlobal.displayInfo(
                '[SO.dump(riot.so)]: Found riot.so (scb/sco), updated value.')


class MAPGEOVertex:
    def __init__(self):
        self.position = None
        self.normal = None
        self.diffuse_uv = None
        self.lightmap_uv = None


class MAPGEOSubmesh:
    def __init__(self):
        self.name = None
        self.index_start = None
        self.index_count = None
        self.vertex_start = None
        self.vertex_count = None


class MAPGEOModel:
    def __init__(self):
        self.name = None
        self.submeshes = []
        self.vertices = []
        self.indices = []

        # bounding box
        self.bb = None

        self.layer = None
        self.lightmap = None


class MAPGEOBucketGrid:
    def __init__(self):
        self.header = None
        self.vertices = []
        self.indices = []
        self.buckets = []
        self.face_flags = []


class MAPGEOPlanarReflector:  # hope its the last name
    def __init__(self):
        self.prs = []


class MAPGEO:
    def __init__(self):
        self.models = []
        self.bucket_grid = None
        self.planar_reflector = None

    def flip(self):
        for model in self.models:
            for vertex in model.vertices:
                vertex.position.x *= -1.0
                if vertex.normal:
                    vertex.normal.y *= -1.0
                    vertex.normal.z *= -1.0

    def read(self, path):
        with open(path, 'rb') as f:
            bs = BinaryStream(f)

            magic = bs.read_bytes(4).decode('ascii')
            if magic != 'OEGM':
                raise FunnyError(
                    f'[MAPGEO.read()]: Wrong file signature: {magic}')

            version = bs.read_uint32()
            if version not in [5, 6, 7, 9, 11, 12, 13]:
                raise FunnyError(
                    f'[MAPGEO.read()]: Unsupported file version: {version}')

            use_seperate_point_lights = False
            if version < 7:
                use_seperate_point_lights = bs.read_bool()

            if version >= 9:
                # shader sampler 1
                bs.pad(bs.read_int32())
                if version >= 11:
                    # shader sampler 2
                    bs.pad(bs.read_int32())

            # vertex elements
            vegs = []
            veg_count = bs.read_uint32()
            for i in range(0, veg_count):
                bs.pad(4)  # usage
                veg = []
                ve_count = bs.read_uint32()
                for j in range(0, ve_count):
                    ve = (bs.read_uint32(), 0)
                    veg.append(ve)
                    bs.pad(4)  # format
                vegs.append(veg)
                bs.pad(8 * (15 - ve_count))  # pad empty vertex elements

            # vertex buffers offsets
            # -> to read vertex later using vertex elements
            vbos = []
            vb_count = bs.read_uint32()
            for i in range(0, vb_count):
                if version >= 13:
                    bs.pad(1)  # layer
                size = bs.read_uint32()
                vbos.append(bs.tell())
                bs.pad(size)

            # index buffers
            ibs = []
            ib_count = bs.read_uint32()
            for i in range(0, ib_count):
                if version >= 13:
                    bs.pad(1)  # layer
                size = bs.read_uint32()
                ibs.append([bs.read_uint16() for j in range(0, size // 2)])

            model_count = bs.read_uint32()
            for m in range(0, model_count):
                model = MAPGEOModel()
                if version < 12:
                    model.name = bs.read_bytes(bs.read_int32()).decode('ascii')
                else:
                    model.name = f'MapGeo_Instance_{m}'
                vertex_count = bs.read_uint32()
                vb_count = bs.read_uint32()
                veg_id = bs.read_int32()

                # init vertices
                for i in range(0, vertex_count):
                    model.vertices.append(MAPGEOVertex())
                # read vertex using vertex buffers offsets and vertex elements
                for i in range(0, vb_count):
                    vb_id = bs.read_int32()
                    return_offset = bs.tell()
                    bs.seek(vbos[vb_id])
                    for j in range(0, vertex_count):
                        for ve in vegs[veg_id+i]:
                            if ve[0] == 0:
                                model.vertices[j].position = bs.read_vec3()
                            elif ve[0] == 2:
                                model.vertices[j].normal = bs.read_vec3()
                            elif ve[0] == 7:
                                model.vertices[j].diffuse_uv = bs.read_vec2()
                            elif ve[0] == 14:
                                model.vertices[j].lightmap_uv = bs.read_vec2()
                            elif ve[0] == 4:
                                bs.pad(4)  # pad color idk
                            else:
                                raise FunnyError(
                                    f'[MAPGEO.read()]: Unknown element: {ve[0]}')
                    bs.seek(return_offset)

                # indices
                bs.pad(4)  # index_count
                ib_id = bs.read_int32()
                model.indices += ibs[ib_id]

                # layer
                model.layer = bytes([255])
                if version >= 13:
                    model.layer = bs.read_byte()

                # submeshes
                submesh_count = bs.read_uint32()
                for i in range(0, submesh_count):
                    submesh = MAPGEOSubmesh()
                    bs.pad(4)  # hash
                    submesh.name = bs.read_bytes(
                        bs.read_int32()).decode('ascii')
                    # maya doesnt allow '/' in name, so use __ instead, bruh
                    submesh.name = submesh.name.replace('/', '__')
                    submesh.index_start = bs.read_uint32()
                    submesh.index_count = bs.read_uint32()
                    submesh.vertex_start = bs.read_uint32()
                    submesh.vertex_count = bs.read_uint32()
                    model.submeshes.append(submesh)

                if version != 5:
                    # flip normals
                    bs.pad(1)

                # bounding box
                bs.pad(24)

                # transform matrix
                # its all identity matrix plus we freeze when export
                bs.pad(64)

                # quality: 1, 2, 4, 8, 16
                # all quality = 1|2|4|8|16 = 31
                bs.pad(1)

                # layer - below version 13
                if version >= 7 and version <= 12:
                    model.layer = bs.read_byte()

                if version >= 11:
                    # render flag
                    bs.pad(1)

                if use_seperate_point_lights and version < 7:
                    # pad seperated point light
                    bs.pad(12)

                if version < 9:
                    # pad 9 unknown vector3
                    bs.pad(108)

                # lightmap
                model.lightmap = bs.read_bytes(
                    bs.read_int32()).decode('ascii')
                # this is not color, we have been tricked, backstabbed and possibly, bamboozled
                # real lightmap uv = lightmap uv * lightmap scale + lightmap pos
                model.lightmap_scale = bs.read_vec2()
                model.lightmap_pos = bs.read_vec2()

                if version >= 9:
                    # baked light texture
                    bs.pad(bs.read_int32())
                    # color, OR UV
                    bs.pad(16)

                    if version >= 12:
                        # baked paint texture
                        bs.pad(bs.read_int32())
                        # color, OR UV
                        bs.pad(16)

                self.models.append(model)

            # for modded file with no bucket grid, planar reflector: stop reading
            current = bs.tell()
            end = bs.end()
            if current == end:
                return

            # there is no reason to read bucket grid below v13
            if version >= 13:
                # bucket grid
                self.bucket_grid = MAPGEOBucketGrid()
                # min/max x/z(16), max out stick x/z(8), bucket size x/z(8)
                self.bucket_grid.header = bs.read_bytes(32)
                bucket_size = bs.read_uint16()
                no_bucket = bs.read_bool()
                bucket_flag = bs.read_byte()
                vertex_count = bs.read_uint32()
                index_count = bs.read_uint32()
                if not no_bucket:
                    self.bucket_grid.vertices = bs.read_bytes(12*vertex_count)
                    self.bucket_grid.indices = bs.read_bytes(2*index_count)
                    # max stick out x/z(8)
                    # start index + base vertex(8)
                    # inside face count + sticking out face count(4)
                    self.bucket_grid.buckets = bs.read_bytes(
                        20*bucket_size*bucket_size)
                    if bucket_flag[0] & 1 == 1:
                        # if first bit = 1, read face flags
                        self.bucket_grid.face_flags = bs.read_bytes(
                            index_count//3)

                self.planar_reflector = MAPGEOPlanarReflector()
                pr_count = bs.read_uint32()
                for i in range(0, pr_count):
                    # matrix4 transform of viewpoint?
                    m = bs.read_bytes(64) 
                    # 2 vec3 position to indicate the plane
                    b = bs.read_bytes(24) 
                    # vec3 normal, direction of plane
                    v = bs.read_bytes(12) 
                    self.planar_reflector.prs.append((m, b, v))

    def write(self, path):
        def prepare():
            vegs = []
            vbs = []
            ibs = []
            for model in self.models:
                # vertex elements
                veg = []
                vertex = model.vertices[0]
                if vertex.position:
                    veg.append((0, 2))
                if vertex.normal:
                    veg.append((2, 2))
                if vertex.diffuse_uv:
                    veg.append((7, 1))
                if vertex.lightmap_uv:
                    veg.append((14, 1))
                vegs.append(veg)

                # vertex buffers + find bounding box in same loop
                vb = []
                min = MVector(float('inf'), float('inf'), float('inf'))
                max = MVector(float('-inf'), float('-inf'), float('-inf'))
                for vertex in model.vertices:
                    # model vertex to bytes
                    if vertex.position:
                        vb.append(pack('f', vertex.position.x))
                        vb.append(pack('f', vertex.position.y))
                        vb.append(pack('f', vertex.position.z))
                    if vertex.normal:
                        vb.append(pack('f', vertex.normal.x))
                        vb.append(pack('f', vertex.normal.y))
                        vb.append(pack('f', vertex.normal.z))
                    if vertex.diffuse_uv:
                        vb.append(pack('f', vertex.diffuse_uv.x))
                        vb.append(pack('f', vertex.diffuse_uv.y))
                    if vertex.lightmap_uv:
                        vb.append(pack('f', vertex.lightmap_uv.x))
                        vb.append(pack('f', vertex.lightmap_uv.y))
                    # bounding box
                    if min.x > vertex.position.x:
                        min.x = vertex.position.x
                    if min.y > vertex.position.y:
                        min.y = vertex.position.y
                    if min.z > vertex.position.z:
                        min.z = vertex.position.z
                    if max.x < vertex.position.x:
                        max.x = vertex.position.x
                    if max.y < vertex.position.y:
                        max.y = vertex.position.y
                    if max.z < vertex.position.z:
                        max.z = vertex.position.z
                model.bb = (min, max)
                vbs.append((model.layer, b''.join(vb)))

                # index buffers
                ib = [pack('H', index) for index in model.indices]
                ibs.append((model.layer, b''.join(ib)))

            # use identity matrix for all models
            im = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            imb = b''.join([pack('f', f) for f in im])

            return vegs, vbs, ibs, imb

        with open(path, 'wb+') as f:
            bs = BinaryStream(f)

            bs.write_bytes('OEGM'.encode('ascii'))
            bs.write_uint32(13)

            # shader sampler 1
            bs.write_int32(0)
            bs.write_bytes(bytes())
            # shader sampler 2
            bs.write_int32(0)
            bs.write_bytes(bytes())

            vegs, vbs, ibs, imb = prepare()

            # vertex elements
            bs.write_uint32(len(vegs))
            for veg in vegs:
                bs.write_uint32(0)  # usage - static
                ve_count = len(veg)
                bs.write_uint32(ve_count)
                for ve in veg:
                    bs.write_uint32(ve[0])
                    bs.write_uint32(ve[1])
                # fill empty vertex elements
                for i in range(0, 15-ve_count):
                    bs.write_uint32(0)
                    bs.write_uint32(2)

            # vertex buffers
            bs.write_uint32(len(vbs))
            for layer, vb in vbs:
                bs.write_bytes(layer)
                bs.write_uint32(len(vb))
                bs.write_bytes(vb)

            # index buffers
            bs.write_uint32(len(ibs))
            for layer, ib in ibs:
                bs.write_bytes(layer)
                bs.write_uint32(len(ib))
                bs.write_bytes(ib)

            # model
            bs.write_uint32(len(self.models))
            model_id = 0
            for model in self.models:
                # count and buffer id
                bs.write_uint32(len(model.vertices))
                bs.write_uint32(1)  # vertex buffers count
                bs.write_int32(model_id)  # veg id
                bs.write_int32(model_id)  # vb id
                bs.write_uint32(len(model.indices))
                bs.write_int32(model_id)  # ib id
                model_id += 1

                # layer
                bs.write_bytes(model.layer)

                # submeshes
                bs.write_uint32(len(model.submeshes))
                for submesh in model.submeshes:
                    submesh.name = submesh.name.replace('__', '/')

                    bs.write_uint32(0)  # hash, always 0?
                    bs.write_int32(len(submesh.name))
                    bs.write_bytes(submesh.name.encode('ascii'))

                    bs.write_uint32(submesh.index_start)
                    bs.write_uint32(submesh.index_count)
                    bs.write_uint32(submesh.vertex_start)
                    bs.write_uint32(submesh.vertex_count)

                # flip normals
                bs.write_bool(False)

                # bounding box
                bs.write_vec3(model.bb[0])
                bs.write_vec3(model.bb[1])

                # transform
                bs.write_bytes(imb)

                # quality, 31 = all quality
                bs.write_bytes(bytes([31]))

                # render flag
                bs.write_bytes(bytes([0]))

                # lightmap
                if model.lightmap:
                    bs.write_int32(len(model.lightmap))
                    bs.write_bytes(model.lightmap.encode('ascii'))
                    bs.write_float(1)  # scale
                    bs.write_float(1)
                    bs.write_float(0)  # pos
                    bs.write_float(0)
                else:
                    bs.write_bytes(bytes([0])*20)

                # baked light texture
                bs.write_bytes(bytes([0])*20)

                # baked paint texture
                bs.write_bytes(bytes([0])*20)

            # bucket grid
            if self.bucket_grid:
                bs.write_bytes(self.bucket_grid.header)
                # bucket size
                bs.write_uint16(int(sqrt(len(self.bucket_grid.buckets)//20)))
                bs.write_bool(False)  # no bucket grid
                bs.write_bytes(bytes([1]))  # bucket flag
                bs.write_uint32(len(self.bucket_grid.vertices)//12)
                bs.write_uint32(len(self.bucket_grid.indices)//2)
                bs.write_bytes(self.bucket_grid.vertices)
                bs.write_bytes(self.bucket_grid.indices)
                bs.write_bytes(self.bucket_grid.buckets)
                bs.write_bytes(self.bucket_grid.face_flags)

            if self.planar_reflector:
                bs.write_uint32(len(self.planar_reflector.prs))
                for m, b, v in self.planar_reflector.prs:
                    bs.write_bytes(m)
                    bs.write_bytes(b)
                    bs.write_bytes(v)

    def load(self, ssmat=False):
        # to call only 1 cmd
        execmd = ''

        # ensure far clip plane, to see whole map
        execmd += 'setAttr "perspShape.farClipPlane" 300000;'
        # render with alpha cut
        execmd += 'setAttr "hardwareRenderingGlobals.transparencyAlgorithm" 5;'

        # init 8 layers
        layer_models = {}
        for i in range(0, 8):
            layer_models[i] = []

        # map submeshes by name
        submesh_names = []
        for model in self.models:
            for submesh in model.submeshes:
                if submesh.name not in submesh_names:
                    submesh_names.append(submesh.name)

        # create shared material
        material_type = 'standardSurface' if ssmat else 'lambert'
        for submesh_name in submesh_names:
            # material
            execmd += f'shadingNode -asShader {material_type} -name "{submesh_name}";'
            # create renderable, independent shading group
            execmd += f'sets -renderable true -noSurfaceShader true -empty -name "{submesh_name}_SG";'
            # connect material to shading group
            execmd += f'connectAttr -f {submesh_name}.outColor {submesh_name}_SG.surfaceShader;'

        # the group of all meshes, the name of this group = map ID, for example: Map11, Map12
        group_transform = MFnTransform()
        group_transform.create()
        group_transform.setName('MapID')

        for model in self.models:
            MGlobal.displayInfo(f'[MAPGEO.load()]: Loading {model.name}')

            vertex_count = len(model.vertices)
            index_count = len(model.indices)
            face_count = index_count // 3

            # create mesh
            vertices = MFloatPointArray()
            u_values = MFloatArray()
            v_values = MFloatArray()
            for vertex in model.vertices:
                vertices.append(MFloatPoint(
                    vertex.position.x, vertex.position.y, vertex.position.z))
                u_values.append(vertex.diffuse_uv.x)
                v_values.append(1.0 - vertex.diffuse_uv.y)

            poly_count = MIntArray(face_count, 3)
            poly_indices = MIntArray()
            MScriptUtil.createIntArrayFromList(model.indices, poly_indices)
            normal_indices = MIntArray()
            MScriptUtil.createIntArrayFromList(
                list(range(len(model.vertices))), normal_indices)

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

            # name and transform
            mesh.setName(f'{model.name}Shape')
            mesh_name = mesh.name()
            transform = MFnTransform(mesh.parent(0))
            transform.setName(model.name)
            transform_name = transform.name()

            # lightmap uv
            if model.lightmap:
                short_lightmap = model.lightmap.split('/')[-1].split('.')[0]
                mesh.createUVSetWithName(short_lightmap)
                lightmap_u_values = MFloatArray()
                lightmap_v_values = MFloatArray()
                for vertex in model.vertices:
                    lightmap_u_values.append(vertex.lightmap_uv.x *
                                             model.lightmap_scale.x + model.lightmap_pos.x)
                    lightmap_v_values.append(1.0-(vertex.lightmap_uv.y *
                                                  model.lightmap_scale.y + model.lightmap_pos.y))

                mesh.setUVs(
                    lightmap_u_values, lightmap_v_values, short_lightmap
                )
                mesh.assignUVs(
                    poly_count, poly_indices, short_lightmap
                )

            for submesh in model.submeshes:
                submesh_name = submesh.name
                # shading group
                face_start = submesh.index_start // 3
                face_end = (submesh.index_start + submesh.index_count) // 3
                # add submesh faces to shading group
                execmd += f'sets -e -forceElement "{submesh_name}_SG" {mesh_name}.f[{face_start}:{face_end}];'

            mesh.updateSurface()

            # convert layer in byte to 8 char binary string, example: 10101010
            # from RIGHT to LEFT, if the char at index 3 is '1' -> the object appear on layer index 3
            # default for no layer data: 11111111
            model.layer = f'{model.layer[0]:08b}'[::-1]

            # add the model to the layer data, where it belong to
            for i in range(0, 8):
                if model.layer[i] == '1':
                    layer_models[i].append(transform_name)

            group_transform.addChild(transform.object())

        # create set and assign mesh to set
        for i in range(0, 8):
            model_names = ' '.join(layer_models[i])
            execmd += f'sets -name "set{i+1}";sets -addElement set{i+1} {model_names};'
        execmd += 'select -cl;'
        MGlobal.executeCommand(execmd)

    def dump(self, riot=None):
        # get transform in selections
        selections = MSelectionList()
        MGlobal.getActiveSelectionList(selections)
        iterator = MItSelectionList(selections, MFn.kTransform)
        if iterator.isDone():
            raise FunnyError(
                f'[MAPGEO.dump()]: Please select the group of meshes.')
        selected_dagpath = MDagPath()
        iterator.getDagPath(selected_dagpath)
        iterator.next()
        if not iterator.isDone():
            raise FunnyError(
                f'[MAPGEO.dump()]: More than 1 group selected.')
        group_transform = MFnTransform(selected_dagpath)
        group_name = group_transform.name()

        # auto freeze selected group transform
        MGlobal.executeCommand(
            f'makeIdentity -apply true -t 1 -r 1 -s 1 -n 0 -pn 1 -jointOrient;')

        layer_models = {}
        for i in range(0, 8):
            util = MScriptUtil()
            ptr = util.asIntPtr()
            MGlobal.executeCommand(f'objExists set{i+1}', ptr)
            if util.getInt(ptr) == 0:
                raise FunnyError(
                    f'[MAPGEO.dump()]: There is no set{i+1} found in scene. Please create 8 sets as 8 layers of mapgeo and set up objects with them.\n'
                    'How to create a set: Maya toolbar -> Create -> Sets -> Set.'
                )
            layer_models[i] = []
            MGlobal.executeCommand(
                f'sets -q set{i+1}', layer_models[i])

        # iterator all meshes in group transform
        iteratorMesh = MItDag(MItDag.kDepthFirst, MFn.kMesh)
        iteratorMesh.reset(group_transform.object())
        while not iteratorMesh.isDone():
            mesh_dagpath = MDagPath()
            iteratorMesh.getPath(mesh_dagpath)
            if mesh_dagpath == selected_dagpath:
                iteratorMesh.next()
                continue
            if mesh_dagpath.apiType() != MFn.kMesh:
                iteratorMesh.next()
                continue
            mesh = MFnMesh(mesh_dagpath)
            mesh_vertex_count = mesh.numVertices()
            model = MAPGEOModel()

            # name and transform
            transform = MFnTransform(mesh.parent(0))
            model.name = transform.name()
            MGlobal.displayInfo(f'[MAPGEO.dump()]: Dumping {model.name}')

            # layer
            model.layer = ''
            for i in range(0, 8):
                if model.name in layer_models[i]:
                    model.layer = '1' + model.layer
                else:
                    model.layer = '0' + model.layer

            # convert layer from 8 char binary string to byte
            model.layer = bytes([int(model.layer, 2)])

            # get shader/materials
            shaders = MObjectArray()
            face_shader = MIntArray()
            instance = mesh_dagpath.instanceNumber() if mesh_dagpath.isInstanced() else 0
            mesh.getConnectedShaders(instance, shaders, face_shader)
            shader_count = shaders.length()
            if shader_count < 1:
                raise FunnyError(
                    f'[MAPGEO.dump({mesh.name()})]: No material assigned to this mesh, please assign one.')

            # init shaders data to work on multiple shader
            shader_vertices = []
            shader_indices = []
            for i in range(0, shader_count):
                shader_vertices.append([])
                shader_indices.append([])

            # iterator on faces - 1st
            # to get vertex_shader first base on face_shader
            # extra checking stuffs
            bad_faces = MIntArray()  # invalid triangulation polygon
            bad_faces2 = MIntArray()  # no material assigned
            bad_faces3 = MIntArray()  # no uv assigned
            bad_vertices = MIntArray()  # shared vertices
            vertex_shader = MIntArray(mesh_vertex_count, -1)
            iterator = MItMeshPolygon(mesh_dagpath)
            iterator.reset()
            while not iterator.isDone():
                # get shader of this face
                face_index = iterator.index()
                shader_index = face_shader[face_index]

                # check valid triangulation
                if not iterator.hasValidTriangulation():
                    if face_index not in bad_faces:
                        bad_faces.append(face_index)
                # check face with no material assigned
                if shader_index == -1:
                    if face_index not in bad_faces2:
                        bad_faces2.append(face_index)
                # check if face has no UVs
                if not iterator.hasUVs():
                    if face_index not in bad_faces3:
                        bad_faces3.append(face_index)
                # get face vertices
                vertices = MIntArray()
                iterator.getVertices(vertices)
                face_vertex_count = vertices.length()
                # check if each vertex is shared by mutiple materials
                for i in range(0, face_vertex_count):
                    vertex = vertices[i]
                    if vertex_shader[vertex] not in [-1, shader_index]:
                        if vertex not in bad_vertices:
                            bad_vertices.append(vertex)
                        continue
                    vertex_shader[vertex] = shader_index
                iterator.next()
            if bad_faces.length() > 0:
                component = MFnSingleIndexedComponent()
                face_component = component.create(
                    MFn.kMeshPolygonComponent)
                component.addElements(bad_faces)
                selections = MSelectionList()
                selections.add(mesh_dagpath, face_component)
                MGlobal.selectCommand(selections)
                raise FunnyError(
                    f'[MAPGEO.dump({mesh.name()})]: Mesh contains {bad_faces.length()} invalid triangulation faces, those faces will be selected in scene.\nBonus: If there is nothing selected (or they are invisible) after this error message, consider to delete history, that might fix the problem.')
            if bad_faces2.length() > 0:
                component = MFnSingleIndexedComponent()
                face_component = component.create(
                    MFn.kMeshPolygonComponent)
                component.addElements(bad_faces2)
                selections = MSelectionList()
                selections.add(mesh_dagpath, face_component)
                MGlobal.selectCommand(selections)
                raise FunnyError(
                    f'[MAPGEO.dump({mesh.name()})]: Mesh contains {bad_faces2.length()} faces have no material assigned, those faces will be selected in scene.\nBonus: If there is nothing selected (or they are invisible) after this error message, consider to delete history, that might fix the problem.')
            if bad_faces3.length() > 0:
                component = MFnSingleIndexedComponent()
                face_component = component.create(
                    MFn.kMeshPolygonComponent)
                component.addElements(bad_faces3)
                selections = MSelectionList()
                selections.add(mesh_dagpath, face_component)
                MGlobal.selectCommand(selections)
                raise FunnyError(
                    f'[MAPGEO.dump({mesh.name()})]: Mesh contains {bad_faces3.length()} faces have no UVs assigned, or, those faces UVs are not in first UV set, those faces will be selected in scene.\nBonus: If there is nothing selected (or they are invisible) after this error message, consider to delete history, that might fix the problem.')
            if bad_vertices.length() > 0:
                component = MFnSingleIndexedComponent()
                vertex_component = component.create(
                    MFn.kMeshVertComponent)
                component.addElements(bad_vertices)
                selections = MSelectionList()
                selections.add(mesh_dagpath, vertex_component)
                MGlobal.selectCommand(selections)
                raise FunnyError((
                    f'[MAPGEO.dump({mesh.name()})]: Mesh contains {bad_vertices.length()} vertices are shared by mutiple materials, those vertices will be selected in scene.\n'
                    'Save/backup scene first, try one of following methods to fix:\n'
                    '1. Seperate all connected faces that shared those vertices.\n'
                    '2. Check and reassign correct material.\n'
                    '3. [recommended] Try auto fix shared vertices button on shelf.'
                    '\nBonus: If there is nothing selected (or they are invisible) after this error message, consider to delete history, that might fix the problem.'
                ))

            # get all UV sets
            # first uv set = diffuse
            # second uv set = lightmap
            # ignore other sets
            uv_names = []
            mesh.getUVSetNames(uv_names)
            if len(uv_names) > 1:
                model.lightmap = f'ASSETS/Maps/Lightmaps/Maps/MapGeometry/{group_name}/Base/{uv_names[1]}.dds'
            u_values = MFloatArray()
            v_values = MFloatArray()
            mesh.getUVs(u_values, v_values, uv_names[0])
            if model.lightmap:
                lightmap_u_values = MFloatArray()
                lightmap_v_values = MFloatArray()
                mesh.getUVs(lightmap_u_values,
                            lightmap_v_values, uv_names[1])
            # iterator on vertices
            # to dump all new vertices base on unique uv
            bad_vertices2 = MIntArray()  # vertex has no UVs
            iterator = MItMeshVertex(mesh_dagpath)
            iterator.reset()
            while not iterator.isDone():
                # get shader of this vertex
                vertex_index = iterator.index()
                shader_index = vertex_shader[vertex_index]
                if shader_index == -1:
                    # a strange vertex with no shader ?
                    # let say this vertex is alone and not in any face
                    # just ignore it?
                    iterator.next()
                    continue

                # position
                position = iterator.position(MSpace.kWorld)
                # average of normals of all faces connect to this vertex
                normals = MVectorArray()
                iterator.getNormals(normals)
                normal_count = normals.length()
                normal = MVector(0.0, 0.0, 0.0)
                for i in range(0, normal_count):
                    normal += normals[i]
                normal /= normal_count
                # unique uv
                uv_indices = MIntArray()
                iterator.getUVIndices(uv_indices)
                uv_count = uv_indices.length()
                if uv_count > 0:
                    seen = []
                    for i in range(0, uv_count):
                        uv_index = uv_indices[i]
                        if uv_index == -1:
                            continue
                        if uv_index not in seen:
                            seen.append(uv_index)
                            # dump vertices
                            vertex = MAPGEOVertex()
                            vertex.position = MVector(position)
                            vertex.normal = MVector(normal)
                            vertex.diffuse_uv = MVector(
                                u_values[uv_index],
                                1.0 - v_values[uv_index]
                            )
                            if model.lightmap:
                                vertex.lightmap_uv = MVector(
                                    lightmap_u_values[uv_index],
                                    1.0 - lightmap_v_values[uv_index]
                                )
                            vertex.uv_index = uv_index
                            vertex.new_index = len(
                                shader_vertices[shader_index])
                            shader_vertices[shader_index].append(vertex)
                else:
                    if vertex_index not in bad_vertices2:
                        bad_vertices2.append(vertex_index)
                iterator.next()
            if bad_vertices2.length() > 0:
                raise FunnyError(
                    f'[MAPGEO.dump({mesh.name()})]: Mesh contains {bad_vertices2.length()} vertices have no UVs assigned, those vertices will be selected in scene.\nBonus: If there is nothing selected (or they are invisible) after this error message, consider to delete history, that might fix the problem.')

            # map new vertices base on uv_index
            map_vertices = {}
            for shader_index in range(0, shader_count):
                map_vertices[shader_index] = {}
                for vertex in shader_vertices[shader_index]:
                    map_vertices[shader_index][vertex.uv_index] = vertex.new_index

            # iterator on faces - 2nd
            # to dump indices:
            # - triangulated indices
            # - new indices base on new unique uv vertices
            iterator = MItMeshPolygon(mesh_dagpath)
            iterator.reset()
            while not iterator.isDone():
                # get shader of this face
                face_index = iterator.index()
                shader_index = face_shader[face_index]

                # get triangulated face indices
                points = MPointArray()
                indices = MIntArray()
                iterator.getTriangles(points, indices)
                face_index_count = indices.length()
                # get face vertices
                map_indices = {}
                vertices = MIntArray()
                iterator.getVertices(vertices)
                face_vertex_count = vertices.length()
                # map face indices by uv_index
                for i in range(0, face_vertex_count):
                    util = MScriptUtil()
                    ptr = util.asIntPtr()
                    iterator.getUVIndex(i, ptr)
                    uv_index = util.getInt(ptr)
                    map_indices[vertices[i]] = uv_index
                # map new indices by new vertices through uv_index
                # and add new indices
                for i in range(0, face_index_count):
                    new_indices = map_vertices[shader_index][map_indices[indices[i]]]
                    shader_indices[shader_index].append(new_indices)
                iterator.next()

            # combine material's indices
            for i in range(0, shader_count):
                if i > 0:
                    max_vertex = max(shader_indices[i-1])
                    shader_index_count = len(shader_indices[i])
                    for j in range(0, shader_index_count):
                        shader_indices[i][j] += max_vertex+1

            # create MAPGEOModel data base on shader_indices and shader_vertices
            index_start = 0
            vertex_start = 0
            for i in range(0, shader_count):
                index_count = len(shader_indices[i])
                vertex_count = len(shader_vertices[i])

                # get shader node -> get shader name
                ss = MFnDependencyNode(
                    shaders[i]).findPlug('surfaceShader')
                plugs = MPlugArray()
                ss.connectedTo(plugs, True, False)
                material = MFnDependencyNode(plugs[0].node())
                # dump SKN data: submeshes, indices and vertices
                submesh = MAPGEOSubmesh()
                submesh.name = material.name()
                submesh.index_start = index_start
                submesh.vertex_start = vertex_start
                submesh.index_count = index_count
                submesh.vertex_count = vertex_count
                model.submeshes.append(submesh)
                model.indices += shader_indices[i]
                model.vertices += shader_vertices[i]

                index_start += index_count
                vertex_start += vertex_count

            # check limit vertices
            vertices_count = max(model.indices)
            if vertices_count > 65535:
                raise FunnyError(
                    f'[MAPGEO.dump({mesh.name()})]: Too many vertices found: {vertices_count}, max allowed: 65535 vertices.')

            # check limit submeshes
            submesh_count = len(model.submeshes)
            if submesh_count > 64:
                raise FunnyError(
                    f'[MAPGEO.dump({mesh.name()})]: Too many materials assigned on this mesh: {submesh_count}, max allowed: 64 materials on each mesh.')
            self.models.append(model)
            iteratorMesh.next()

        if riot:
            MGlobal.displayInfo(
                '[MAPGEO.dump(riot.mapgeo)]: Found riot.mapgeo, copying bucket grids...')
            self.bucket_grid = riot.bucket_grid
            self.planar_reflector = riot.planar_reflector
        else:
            MGlobal.displayWarning(
                '[MAPGEO.dump()]: No riot.mapgeo found, map can be crashed due to missing bucket grids...')
