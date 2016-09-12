import os
import struct
import numpy as n

#adapted from gwyddion's PLU file format reader in
#http://www.sourcecodebrowser.com/gwyddion/2.19/sensofar_8c_source.html

#enum {
DATE_SIZE = 128
COMMENT_SIZE = 256
HEADER_SIZE = 500
#};
#typedef enum {
MES_IMATGE      = 0
MES_PERFIL      = 1
MES_MULTIPERFIL = 2
MES_TOPO        = 3
MES_COORDENADES = 4
MES_GRUIX       = 5
MES_CUSTOM      = 6
#} MeasurementType;
#typedef enum {
ALGOR_CONFOCAL_INTENSITY        = 0
ALGOR_CONFOCAL_GRADIENT         = 1
ALGOR_INTERFEROMETRIC_PSI       = 2
ALGOR_INTERFEROMETRIC_VSI       = 3
ALGOR_INTERFEROMETRIC_ePSI      = 3
ALGOR_CONFOCAL_THICKNESS        = 4
ALGOR_INTERFEROMETRIC_THICKNESS = 5
#} AcquisitionAlgorithm;
#/* This seems to be context-dependent */
#typedef enum {
METHOD_CONVENTIONAL              = 0
METHOD_CONFOCAL                  = 1
METHOD_SINGLE_PROFILE            = 0
METHOD_EXTENDED_PROFILE          = 1
METHOD_TOPOGRAPHY                = 0
METHOD_EXTENDED_TOPOGRAPHY       = 1
METHOD_MULTIPLE_PROFILE          = 0
METHOD_EXTENDED_MULTIPLE_PROFILE = 1
#} MethodType;
#typedef enum {
OBJ_SLWD_10x  = 0
OBJ_SLWD_20x  = 1
OBJ_SLWD_50x  = 2
OBJ_SLWD_100x = 3
OBJ_EPI_20x   = 4
OBJ_EPI_50x   = 5
OBJ_EPI_10x   = 6
OBJ_EPI_100x  = 7
OBJ_ELWD_10x  = 8
OBJ_ELWD_20x  = 9
OBJ_ELWD_50x  = 10
OBJ_ELWD_100x = 11
OBJ_TI_2_5x   = 12
OBJ_TI_5x     = 13
OBJ_DI_10x    = 14
OBJ_DI_20x    = 15
OBJ_DI_50x    = 16
OBJ_EPI_5x    = 17
OBJ_EPI_150x  = 18
#} ObjectiveType;
#typedef enum {
AREA_128  = 0
AREA_256  = 1
AREA_512  = 2
AREA_MAX  = 3
AREA_L256 = 4
AREA_L128 = 5
#} AreaType;
#typedef enum {
PLU             = 0
PLU_2300_XGA    = 1
PLU_2300_XGA_T5 = 2
PLU_2300_SXGA   = 3
PLU_3300        = 4

FORMAT_VERSION_2000 = 0
FORMAT_VERSION_2006 = 255

def readPLU(filename):
  contents = n.fromfile(filename, dtype=n.uint8)
  #with open(filename, 'r') as f:
  #  contents = f.read()
  header = contents[0:512].view('S512')[0]
  date   = header[0:DATE_SIZE]
  date_t = struct.unpack_from('<I', header, DATE_SIZE)
  user_comment = header[DATE_SIZE+4:DATE_SIZE+4+COMMENT_SIZE]
  
  axes_config_fields    = ['yres', 'xres', 'N_tall', 'dy_multip', 'mppx', 'mppy', 'x_0', 'y_0', 'mpp_tall', 'z_0']
  axes_config_fmt       = 'I'*3+'f'*7
  measure_config_fields = ['type', 'algorithm', 'method', 'objective', 'area', 'xres_area', 'yres_area', 'xres', 'yres', 'na', 'incr_z', 'range', 'n_planes', 'tpc_umbral_F', 'restore', 'num_layers', 'version', 'config_hardware', 'stack_im_num', 'reserved', 'factor_delmacio', 'yres', 'xres']
  measure_config_fmt    = 'I'*10+'dfII?BBBBBxxIII'
  
  fmts               = axes_config_fmt+measure_config_fmt
  tupla              = struct.unpack_from('<'+fmts, header, DATE_SIZE+4+COMMENT_SIZE)
  sz                 = struct.calcsize(fmts)
  axes_config_raw    = zip(axes_config_fields, tupla[0:len(axes_config_fields)])
  axes_config        = dict(axes_config_raw)
  measure_config_raw = zip(measure_config_fields, tupla[len(axes_config_fields):])
  measure_config     = dict(measure_config_raw)
  if not measure_config['type'] in [MES_TOPO, MES_IMATGE]:
    raise Exception('The PLU file format has several data modes. The data mode of this file is not supported: '+filename)
  shape           = (axes_config['yres'], axes_config['xres'])
  for x in ['mppx', 'mppy']:
    if axes_config[x]==0.0:
      axes_config[x] = 1.0
  #print axes_config_raw
  #print measure_config_raw
  data            = contents[DATE_SIZE+4+COMMENT_SIZE+sz:].view(n.float32)[0:(shape[0]*shape[1])].reshape(shape)
  data[data==1000001.0] = n.nan
  return (axes_config, measure_config, data, date,date_t,user_comment)

def PLU2XYZ(axes_config, data):
#  #rebase heightmap to cancel z_0 away
#  data -= axes_config['z_0']
  #create XY grid
  xs = n.linspace(0, axes_config['xres']*axes_config['mppx'], axes_config['xres'])
  ys = n.linspace(0, axes_config['yres']*axes_config['mppy'], axes_config['yres'])
  
  grid = n.meshgrid(xs, ys)
  
  xyz = n.column_stack((grid[0].ravel(), grid[1].ravel(), data.ravel()))
  return xyz

