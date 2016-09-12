#this is python-stl v1.3.3, slightly modified

import os
import numpy
import struct
import datetime
import collections

#__package_name__ = 'numpy-stl'
#__import_name__ = 'stl'
#__version__ = '1.3.3'
#__author__ = 'Rick van Hattem'
#__author_email__ = 'Wolph@Wol.ph'
#__description__ = '''
#Library to make reading, writing and modifying both binary and ascii STL files
#easy.
#'''
#__url__ = 'https://github.com/WoLpH/numpy-stl/'


AREA_SIZE_THRESHOLD = 1e-6
VECTORS = 3
DIMENSIONS = 3
X = 0
Y = 1
Z = 2


#class Mesh(logger.Logged, collections.Mapping):
class Mesh(collections.Mapping):
    '''
    Mesh object with easy access to the vectors through v0, v1 and v2. An

    :param numpy.array data: The data for this mesh
    :param bool calculate_normals: Whehter to calculate the normals
    :param bool remove_empty_areas: Whether to remove triangles with 0 area
            (due to rounding errors for example)

    >>> data = numpy.zeros(10, dtype=Mesh.dtype)
    >>> mesh = Mesh(data, remove_empty_areas=False)
    >>> # Increment vector 0 item 0
    >>> mesh.v0[0] += 1
    >>> mesh.v1[0] += 2

    # Check item 0 (contains v0, v1 and v2)
    >>> mesh[0]
    array([ 1.,  1.,  1.,  2.,  2.,  2.,  0.,  0.,  0.], dtype=float32)
    >>> mesh.vectors[0]
    array([[ 1.,  1.,  1.],
           [ 2.,  2.,  2.],
           [ 0.,  0.,  0.]], dtype=float32)
    >>> mesh.v0[0]
    array([ 1.,  1.,  1.], dtype=float32)
    >>> mesh.points[0]
    array([ 1.,  1.,  1.,  2.,  2.,  2.,  0.,  0.,  0.], dtype=float32)
    >>> mesh.data[0]
    ([0.0, 0.0, 0.0], [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.0, 0.0, 0.0]], [0])
    >>> mesh.x[0]
    array([ 1.,  2.,  0.], dtype=float32)

    >>> mesh[0] = 3
    >>> mesh[0]
    array([ 3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.], dtype=float32)

    >>> len(mesh) == len(list(mesh))
    True
    >>> (mesh.min_ < mesh.max_).all()
    True
    >>> mesh.update_normals()
    >>> mesh.units.sum()
    0.0
    >>> mesh.v0[:] = mesh.v1[:] = mesh.v2[:] = 0
    >>> mesh.points.sum()
    0.0
    '''
    dtype = numpy.dtype([
        ('normals', numpy.float32, (3, )),
        ('vectors', numpy.float32, (3, 3)),
        ('attr', 'u2', (1, )),
    ])

    def __init__(self, data, calculate_normals=True, remove_empty_areas=True):
        super(Mesh, self).__init__()
        if remove_empty_areas:
            data = self.remove_empty_areas(data)

        self.data = data

        points = self.points = data['vectors']
        self.points.shape = data.size, 9
        self.x = points[:, X::3]
        self.y = points[:, Y::3]
        self.z = points[:, Z::3]
        self.v0 = data['vectors'][:, 0]
        self.v1 = data['vectors'][:, 1]
        self.v2 = data['vectors'][:, 2]
        self.normals = data['normals']
        self.vectors = data['vectors']
        self.attr = data['attr']

        if calculate_normals:
            self.update_normals()

    @classmethod
    def remove_empty_areas(cls, data):
        vectors = data['vectors']
        v0 = vectors[:, 0]
        v1 = vectors[:, 1]
        v2 = vectors[:, 2]
        normals = numpy.cross(v1 - v0, v2 - v0)
        areas = numpy.sqrt((normals ** 2).sum(axis=1))
        return data[areas > AREA_SIZE_THRESHOLD]

    def update_normals(self):
        '''Update the normals for all points'''
        self.normals[:] = numpy.cross(self.v1 - self.v0, self.v2 - self.v0)

    def update_min(self):
        self._min = self.vectors.min(axis=(0, 1))

    def update_max(self):
        self._max = self.vectors.max(axis=(0, 1))

    def update_areas(self):
        areas = .5 * numpy.sqrt((self.normals ** 2).sum(axis=1))
        self.areas = areas.reshape((areas.size, 1))

    def update_units(self):
        units = self.normals.copy()
        non_zero_areas = self.areas > 0
        areas = self.areas

        if non_zero_areas.any():
            non_zero_areas.shape = non_zero_areas.shape[0]
            areas = numpy.hstack((2 * areas[non_zero_areas],) * DIMENSIONS)
            units[non_zero_areas] /= areas

        self.units = units

    def _get_or_update(key):
        def _get(self):
            if not hasattr(self, '_%s' % key):
                getattr(self, 'update_%s' % key)()
            return getattr(self, '_%s' % key)

        return _get

    def _set(key):
        def _set(self, value):
            setattr(self, '_%s' % key, value)

        return _set

    min_ = property(_get_or_update('min'), _set('min'),
                    doc='Mesh minimum value')
    max_ = property(_get_or_update('max'), _set('max'),
                    doc='Mesh maximum value')
    areas = property(_get_or_update('areas'), _set('areas'),
                     doc='Mesh areas')
    units = property(_get_or_update('units'), _set('units'),
                     doc='Mesh unit vectors')

    def __getitem__(self, k):
        return self.points[k]

    def __setitem__(self, k, v):
        self.points[k] = v

    def __len__(self):
        return self.points.shape[0]

    def __iter__(self):
        for point in self.points:
            yield point



#: Automatically detect whether the output is a TTY, if so, write ASCII
#: otherwise write BINARY
AUTOMATIC = 0
#: Force writing ASCII
ASCII = 1
#: Force writing BINARY
BINARY = 2

#: Amount of bytes to read while using buffered reading
BUFFER_SIZE = 4096
#: The amount of bytes in the header field
HEADER_SIZE = 80
#: The amount of bytes in the count field
COUNT_SIZE = 4
#: The maximum amount of triangles we can read from binary files
MAX_COUNT = 1e6


class StlMesh(Mesh):
    '''Load a mesh from a STL file

    :param str filename: The file to load
    :param bool update_normals: Whether to update the normals
    :param file fh: The file handle to open
    '''
    def __init__(self, filename, update_normals=True, fh=None, **kwargs):
        self.filename = filename
        if fh:
            data = self.load(fh)
        else:
            with open(filename, 'rb') as fh:
                data = self.load(fh)

        Mesh.__init__(self, data, update_normals, **kwargs)

    def load(self, fh, mode=AUTOMATIC):
        '''Load Mesh from STL file

        Automatically detects binary versus ascii STL files.

        :param file fh: The file handle to open
        :param int mode: Automatically detect the filetype or force binary
        '''
        header = fh.read(HEADER_SIZE).lower()
        if mode in (AUTOMATIC, ASCII) and header.startswith('solid'):
            try:
                data = self._load_ascii(fh, header)
            except RuntimeError, (recoverable, e):
                if recoverable:  # Recoverable?
                    data = self._load_binary(fh, header, check_size=False)
                else:
                    # Apparently we've read beyond the header. Let's try
                    # seeking :)
                    # Note that this fails when reading from stdin, we can't
                    # recover from that.
                    fh.seek(HEADER_SIZE)

                    # Since we know this is a seekable file now and we're not
                    # 100% certain it's binary, check the size while reading
                    data = self._load_binary(fh, header, check_size=True)
        else:
            data = self._load_binary(fh, header)

        return data

    def _load_binary(self, fh, header, check_size=False):
        # Read the triangle count
        count, = struct.unpack('@i', fh.read(COUNT_SIZE))
        assert count < MAX_COUNT, ('File too large, got %d triangles which '
                                   'exceeds the maximum of %d') % (
                                       count, MAX_COUNT)

        if check_size:
            try:
                # Check the size of the file
                fh.seek(0, os.SEEK_END)
                raw_size = fh.tell() - HEADER_SIZE - COUNT_SIZE
                expected_count = raw_size / self.dtype.itemsize
                assert expected_count == count, ('Expected %d vectors but '
                                                 'header indicates %d') % (
                                                     expected_count, count)
                fh.seek(HEADER_SIZE + COUNT_SIZE)
            except IOError:  # pragma: no cover
                pass

        # Read the rest of the binary data
        return numpy.fromfile(fh, dtype=self.dtype, count=count)

    def _ascii_reader(self, fh, header):
        lines = header.split('\n')
        recoverable = [True]

        def get(prefix=''):
            if lines:
                line = lines.pop(0)
            else:
                raise RuntimeError(recoverable[0], 'Unable to find more lines')
            if not lines:
                recoverable[0] = False

                # Read more lines and make sure we prepend any old data
                lines[:] = fh.read(BUFFER_SIZE).split('\n')
                line += lines.pop(0)

            line = line.lower().strip()
            if prefix:
                if line.startswith(prefix):
                    values = line.replace(prefix, '', 1).strip().split()
                elif line.startswith('endsolid'):
                    raise StopIteration
                else:
                    raise RuntimeError(recoverable[0],
                                       '%r should start with %r' % (line,
                                                                    prefix))

                if len(values) == 3:
                    return [float(v) for v in values]
                else:  # pragma: no cover
                    raise RuntimeError(recoverable[0],
                                       'Incorrect value %r' % line)
            else:
                return line

        assert get().startswith('solid ')

        if not lines:
            raise RuntimeError(recoverable[0],
                               'No lines found, impossible to read')

        while True:
            # Read from the header lines first, until that point we can recover
            # and go to the binary option. After that we cannot due to
            # unseekable files such as sys.stdin
            #
            # Numpy doesn't support any non-file types so wrapping with a
            # buffer and/or StringIO does not work.
            try:
                normals = get('facet normal')
                assert get() == 'outer loop'
                v0 = get('vertex')
                v1 = get('vertex')
                v2 = get('vertex')
                assert get() == 'endloop'
                assert get() == 'endfacet'
                attrs = 0
                yield (normals, (v0, v1, v2), attrs)
            except AssertionError, e:
                raise RuntimeError(recoverable[0], e)

    def _load_ascii(self, fh, header):
        return numpy.fromiter(self._ascii_reader(fh, header), dtype=self.dtype)

    def save(self, filename, fh=None, mode=AUTOMATIC, update_normals=True):
        '''Save the STL to a (binary) file

        If mode is :py:data:`AUTOMATIC` an :py:data:`ASCII` file will be
        written if the output is a TTY and a :py:data:`BINARY` file otherwise.

        :param str filename: The file to load
        :param file fh: The file handle to open
        :param int mode: The mode to write, default is :py:data:`AUTOMATIC`.
        :param bool update_normals: Whether to update the normals
        '''
        assert filename, 'Filename is required for the STL headers'
        if update_normals:
            self.update_normals()

        if mode is AUTOMATIC:
            if fh and os.isatty(fh.fileno()):  # pragma: no cover
                write = self._write_ascii
            else:
                write = self._write_binary
        elif mode is BINARY:
            write = self._write_binary
        elif mode is ASCII:
            write = self._write_ascii
        else:
            raise ValueError('Mode %r is invalid' % mode)

        name = os.path.split(filename)[-1]
        try:
            if fh:
                write(fh, name)
            else:
                with open(name, 'wb') as fh:
                    write(fh, filename)
        except IOError:  # pragma: no cover
            pass

    def _write_ascii(self, fh, name):
        print >>fh, 'solid %s' % name

        for row in self.data:
            vectors = row['vectors']
            print >>fh, 'facet normal %f %f %f' % tuple(row['normals'])
            print >>fh, '  outer loop'
            print >>fh, '    vertex %f %f %f' % tuple(vectors[0])
            print >>fh, '    vertex %f %f %f' % tuple(vectors[1])
            print >>fh, '    vertex %f %f %f' % tuple(vectors[2])
            print >>fh, '  endloop'
            print >>fh, 'endfacet'

        print >>fh, 'endsolid %s' % name

    def _write_binary(self, fh, name):
        fh.write(('%s (%s) %s %s' % (
            'numpy-stl',
            '1.3.3',
            datetime.datetime.now(),
            name,
        ))[:80].ljust(80, ' '))
        fh.write(struct.pack('@i', self.data.size))
        self.data.tofile(fh)


#import sys
#import random
#import argparse
#
#
#def _get_parser(description):
#    parser = argparse.ArgumentParser(description=description)
#    parser.add_argument('infile', nargs='?', type=argparse.FileType('r'),
#                        default=sys.stdin, help='STL file to read')
#    parser.add_argument('outfile', nargs='?', type=argparse.FileType('w'),
#                        default=sys.stdout, help='STL file to write')
#    parser.add_argument('--name', nargs='?', help='Name of the mesh')
#    parser.add_argument(
#        '-n', '--use-file-normals', action='store_true',
#        help='Read the normals from the file instead of recalculating them')
#    return parser
#
#
#def _get_name(args):
#    if args.name:
#        name = args.name
#    elif not getattr(args.outfile, 'name', '<').startswith('<'):
#        name = args.outfile.name
#    elif not getattr(args.infile, 'name', '<').startswith('<'):
#        name = args.infile.name
#    else:
#        name = 'numpy-stl-%06d' % random.randint(0, 1e6)
#    return name
#
#
#def main():
#    parser = _get_parser('Convert STL files from ascii to binary and back')
#    parser.add_argument('-a', '--ascii', action='store_true',
#                        help='Write ASCII file (default is binary)')
#    parser.add_argument('-b', '--binary', action='store_true',
#                        help='Force binary file (for TTYs)')
#
#    args = parser.parse_args()
#    name = _get_name(args)
#    stl_file = stl.StlMesh(filename=name, fh=args.infile,
#                           update_normals=False)
#
#    if args.binary:
#        mode = stl.BINARY
#    elif args.ascii:
#        mode = stl.ASCII
#    else:
#        mode = stl.AUTOMATIC
#
#    stl_file.save(name, args.outfile, mode=mode,
#                  update_normals=not args.use_file_normals)
#
#
#def to_ascii():
#    parser = _get_parser('Convert STL files to ASCII (text) format')
#    args = parser.parse_args()
#    name = _get_name(args)
#    stl_file = stl.StlMesh(filename=name, fh=args.infile,
#                           update_normals=False)
#    stl_file.save(name, args.outfile, mode=stl.ASCII,
#                  update_normals=not args.use_file_normals)
#
#
#def to_binary():
#    parser = _get_parser('Convert STL files to ASCII (text) format')
#    args = parser.parse_args()
#    name = _get_name(args)
#    stl_file = stl.StlMesh(filename=name, fh=args.infile,
#                           update_normals=False)
#    stl_file.save(name, args.outfile, mode=stl.BINARY,
#                  update_normals=not args.use_file_normals)
#
