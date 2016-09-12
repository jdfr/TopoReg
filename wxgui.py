#!/usr/bin/python


import os
import os.path as op
import traceback
import sys
import threading as thr
import itertools as it

FROZEN = getattr(sys, 'frozen', False)

if FROZEN:
    thisdir = op.dirname(sys.executable) # frozen
else:
    thisdir = op.dirname(op.realpath(__file__)) # unfrozen

def terminate():
  if not FROZEN:
    try:
      import winpython
      inWinPython = True
    except:
      inWinPython = False
    if inWinPython: #the command line window may close too fast for the error to see. put a raw_input here
      print "Please press enter to close this command window"
      raw_input()
  sys.exit(-1)

try:
  import wx
  import wx.lib.dialogs as wxd
except:
  print 'The GUI could not be loaded. Is wxPython installed? The error is shown below: '
  traceback.print_exc()
  terminate()


IS_GTK = 'wxGTK' in wx.PlatformInfo
IS_WIN = 'wxMSW' in wx.PlatformInfo
IS_MAC = 'wxMac' in wx.PlatformInfo

try:
  import numpy as np
except:
  print 'The numpy library could not be loaded. Is the scipy stack installed? The error is shown below: '
  traceback.print_exc()
  terminate()

np.seterr(all='ignore') #avoid annoying warning messages

try:
  import matplotlib
  matplotlib.use('WXAgg') #this is in case we render plots besides the embedded one
  from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
#  from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
#  from matplotlib.backends.backend_wx import NavigationToolbar2Wx
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  from mpl_toolkits.mplot3d.art3d import Poly3DCollection
  #from pylab import cm 
except:
  print 'The matplotlib library could not be loaded. Is the scipy stack installed? The error is shown below: '
  traceback.print_exc()
  terminate()



try:
  import heightmaps as h
except:
  print 'The module heightmaps.py could not be loaded. It should be in the same directory as wxgui.py: '+thisdir
  traceback.print_exc()
  terminate()
try:
  from wxsimpleapp import AllFrame, gettext, wxstc
except:
  print 'The module wxsimpleapp.py could not be loaded. It should be in the same directory as wxgui.py: '+thisdir
  traceback.print_exc()
  terminate()


########################################################################
### HELPER FUNCTIONS AND CLASSES ###
########################################################################

# Define notification event for thread completion
EVT_RESULT_ID = wx.NewId()
def EVT_RESULT(win, func):
    """Define Result Event."""
    win.Connect(-1, -1, EVT_RESULT_ID, func)
class ResultEvent(wx.PyEvent):
    """Simple event to carry arbitrary result data."""
    def __init__(self, data=None):
        """Init Result Event."""
        wx.PyEvent.__init__(self)
        self.SetEventType(EVT_RESULT_ID)
        self.data = data

class RedirectText(object):
    """Simple class to redirect standard output"""
    def __init__(self,aWxTextCtrl):
        self.out=aWxTextCtrl
 
    def deferredWrite(self, string):
      self.out.SetInsertionPointEnd()
      self.out.WriteText(string)
 
    def write(self,string):
        #self.out.WriteText(string)
        wx.CallAfter(self.deferredWrite, string) #thread-safe

# Thread class that executes non-blocking processing. This has been adapted from
#http://wiki.wxpython.org/LongRunningTasks
class WorkerThread(thr.Thread):
    """Worker Thread Class."""
    def __init__(self, notify_window, closure, eventendid):
        """Init Worker Thread Class."""
        thr.Thread.__init__(self)
        self.daemon = True #make sure that the app will exit abruptly instead of waiting for the thread to end (bad, and hacky, I know...)
        self._notify_window = notify_window
        self.closure = closure
        self._want_abort = False
        self.eventendid = eventendid
        # This starts the thread running on creation, but you could
        # also make the GUI thread responsible for calling this
        self.start()

    def run(self):
        """Run Worker Thread."""
        fun      = self.closure[0]
        args     = self.closure[1:-1]
        dictargs = self.closure[-1]
        self._want_abort = False
        ok = False
        try:
          fun(self.notifyFun, self.queryAbort, *args, **dictargs)
          ok = True
#        except:
#          print 'XKCD'          
#          traceback.print_exc()
        finally:
          if self.eventendid is not None:
            self.notifyFun((self.eventendid, ok, self._want_abort))
          #wx.PostEvent(self._notify_window, ResultEvent(('end', ok)))
#            if self._want_abort:
#                # Use a result of None to acknowledge the abort (of
#                # course you can use whatever you'd like or even
#                # a separate event type)
#                wx.PostEvent(self._notify_window, ResultEvent(None))
#                return

    def notifyFun(self, data):
      wx.PostEvent(self._notify_window, ResultEvent(data))
      
    def queryAbort(self):
      return self._want_abort
      
    def abort(self):
        """abort worker thread."""
        # Method for use by main thread to signal an abort
        self._want_abort = True

def doComputeFun(notifyFun, queryAbortFun, regtools, heightmapFiles, processSpecification, conf, path):
  """Helper function to execute closure for a registration process in a separate thread"""
  def callbackFun():
    #if we have just entered a state where the last state recomputed a rectangle:
    if ( (regtools.state[0] in [h.C.STATE.COMPUTEPOINTCLOUD, h.C.STATE.FIRSTPHASE, h.C.STATE.END]) or 
         (regtools.state[0]==h.C.STATE.LOADIMAGES) and (regtools.state[1]==1) ):
      processed = regtools.processed.copy()
      if regtools.state[0]==h.C.STATE.COMPUTEPOINTCLOUD:
        i   = regtools.state[1]
        idx = regtools.processOrder[i]
        processed[idx] = True
      if regtools.state[0]==h.C.STATE.LOADIMAGES:
        processed[regtools.processOrder[0]] = True
      notifyFun(('draw', processed, regtools.rectangles))
    return queryAbortFun()
    
  if path[-1]!=os.sep:
    path = path+os.sep
  conf.debugSavePath = path
  conf.debugXYZ = True
  #result = regtools.computeRegistration(
  regtools.computeRegistration(
     conf=conf, 
     num=len(heightmapFiles), 
     heightmaps=heightmapFiles,
     loader=h.SimpleUniversalImageLoader().loader,
     processSpecification=processSpecification,
     forceNew=True,
     callbackFun=callbackFun)

def msgbox(message):
  """use a dialog box to notify the user of something. If the message has several
  lines (possibly because of containing a exception), the dialog allows to select
  the text"""
  if '\n' in message:
    wxd.scrolledMessageDialog(None, message, '')#, pos, size)
  else:
    dlg = wx.MessageDialog(None, message, '', wx.OK)
    #dlg = ScrolledMessageDialog(None, message, '', (20,20), None, wx.OK)
    dlg.ShowModal()
    dlg.Destroy()


def openDirDialog(text, msge, flag):
  """event handler for the SELECT FILE buttons. It has to open a FileDialog
  with the appropriate parameters and (if successful, fill the corresponding textcontrol)"""
  dr = text.GetValue()
  if not op.isdir(dr):
    dr = os.getcwd()
  dlg = wx.DirDialog(None, msge, dr, flag)
  if dlg.ShowModal() == wx.ID_OK:
    result = dlg.GetPath()
    if result[-1]!=os.sep:
      result += os.sep
    text.SetValue(result)
  else:
    result = None
  dlg.Destroy()
  return result


def genericFileDialog(msge, dr, nm, typ, flag):
  """event handler for the SELECT FILE buttons. It has to open a FileDialog
  with the appropriate parameters"""
  if not op.isdir(dr):
    dr = os.getcwd()
  dlg = wx.FileDialog(None, msge, dr, nm, typ, flag)
  if dlg.ShowModal() == wx.ID_OK:
    if flag & wx.FD_MULTIPLE:
      result = dlg.GetPaths()
    else:
      result = dlg.GetPath()
  else:
    result = None
  dlg.Destroy()
  return result


def helperValidateBool(page, chckctrl, confname):
  """Validate checkboxes to boolean values"""
  return [True, chckctrl.GetValue(), confname]

def helperValidateBoolToValues(page, chckctrl, values, confname):
  """Validate checkboxes to arbitrary values"""
  return [True, values[chckctrl.GetValue()], confname]

def helperValidateNumeric(page, txtctrl, typ, msg, confname, numrange=None):
  """Validate numeric text input"""
  if isinstance(txtctrl, basestring):
    val = txtctrl
  else:
    val = txtctrl.GetValue()
  try:
    val = typ(val)
  except:
    traceback.print_exc()
    return (False, msg+val, (page, txtctrl))
  if (numrange is not None) and ( (val<numrange[0]) or (val>numrange[1]) ):
    return (False, msg+str(val), (page, txtctrl))
  return [True, val, confname]

def helperValidateLambdaWithNumber(page, txtctrl, typ, lambdastr, msg, confname, numrange=None):
  """Validate numeric text input to be embedded in a string specifying a lambda function"""
  ret = helperValidateNumeric(page, txtctrl, typ, msg, confname, numrange)
  if ret[0]:
    ret[1] = lambdastr % txtctrl.GetValue()
  return ret

def helperRadioBox(page, radiobox, msgNotFound, msgNoValue, values, confname):
  """validate multioption radioboxes"""
  val = radiobox.GetSelection()
  if val==wx.NOT_FOUND:
    return (False, msgNotFound, (page, radiobox))
  if val<len(values):
    return [True, values[val], confname]
  else:
    return (False, msgNoValue, (page, radiobox))

def helperValidateIncreasingNumericSequence(page, txtctrl, typ, msgnook, msgnoincr, confname, numrange=None):
  """validate text inputs for increasing numeric sequences"""
  vals = txtctrl.GetValue()
  nums = vals.split(',')
  for i, n in enumerate(nums):
    ret = helperValidateNumeric(page, n.strip(), typ, '', confname, numrange)
    if not ret[0]:
      return (False, msgnook % (i, vals), (page, txtctrl))
    nums[i] = ret[1]
    if (i>0) and (nums[i-1]>=nums[i]):
      return (False, msgnoincr % (i, vals), (page, txtctrl))
  return [True, nums, confname]
    

def processMultiline(string, removeEmpty=True, removeComments=False, doStrip=True):
  """split a string into lines, with some default post-processing. Caution if using removeEmpty=False and removeComments==True, it will fail if empty lines are present"""
  lines = string.split('\n')
  if doStrip:
    lines = (f.strip() for f in lines)
  if removeEmpty:
    lines = [f for f in lines if len(f)>0]
  if removeComments:
    lines = [f for f in lines if f[0]!='#']
    #lines = [f for f in lines if (len(f)>0) and f[0]!='#']
  return lines


def fake3dEqualAxes(ax, X, Y, Z):
  """ugly hack to set 3D matplotlib axes to have eqaul aspect ratios in all three directions"""
  xmax, xmin, ymax, ymin, zmax, zmin = X.max(), X.min(), Y.max(), Y.min(), Z.max(), Z.min()
  # Create cubic bounding box to simulate equal aspect ratio
  max_range = np.array([xmax-xmin, ymax-ymin, zmax-zmin]).max()
  q = np.mgrid[-1:2:2,-1:2:2,-1:2:2]
  Xb = 0.5*max_range*q[0].flatten() + 0.5*(xmax+xmin)
  Yb = 0.5*max_range*q[1].flatten() + 0.5*(ymax+ymin)
  Zb = 0.5*max_range*q[2].flatten() + 0.5*(zmax+zmin)
  # Comment or uncomment following both lines to test the fake bounding box:
  for xb, yb, zb in zip(Xb, Yb, Zb):
     ax.plot([xb], [yb], [zb], 'w')
  return Xb, Yb, Zb

#adapted from  https://stuff.mit.edu/afs/sipb/project/python-lib/src/wxPython-src-2.5.3.1/wxPython/demo/StyledTextCtrl_2.py
def setPythonSyntax(ctrl):
  """Set a wx.stc.StyledTextCtrl for line numbers and python syntax highlighting"""
  #ctrl must be a wx.stc.StyledTextCtrl
  if wx.Platform == '__WXMSW__':
      faces = { 'times': 'Times New Roman',
                'mono' : 'Courier New',
                'helv' : 'Arial',
                'other': 'Comic Sans MS',
                'size' : 10,
                'size2': 8,
               }
  else:
      faces = { 'times': 'Times',
                'mono' : 'Courier',
                'helv' : 'Helvetica',
                'other': 'new century schoolbook',
                'size' : 12,
                'size2': 10,
               }
  
  ctrl.SetLexer(wxstc.STC_LEX_PYTHON)
  # Global default styles for all languages
  ctrl.StyleSetSpec(wxstc.STC_STYLE_DEFAULT,     "face:%(helv)s,size:%(size)d" % faces)
  ctrl.StyleSetSpec(wxstc.STC_STYLE_LINENUMBER,  "back:#C0C0C0,face:%(helv)s,size:%(size2)d" % faces)
  ctrl.StyleSetSpec(wxstc.STC_STYLE_CONTROLCHAR, "face:%(other)s" % faces)
  ctrl.StyleSetSpec(wxstc.STC_STYLE_BRACELIGHT,  "fore:#FFFFFF,back:#0000FF,bold")
  ctrl.StyleSetSpec(wxstc.STC_STYLE_BRACEBAD,    "fore:#000000,back:#FF0000,bold")
  
  # Python styles
  # White space
  ctrl.StyleSetSpec(wxstc.STC_P_DEFAULT, "fore:#808080,face:%(helv)s,size:%(size)d" % faces)
  # Comment
  ctrl.StyleSetSpec(wxstc.STC_P_COMMENTLINE, "fore:#007F00,face:%(other)s,size:%(size)d" % faces)
  # Number
  ctrl.StyleSetSpec(wxstc.STC_P_NUMBER, "fore:#007F7F,size:%(size)d" % faces)
  # String
  ctrl.StyleSetSpec(wxstc.STC_P_STRING, "fore:#7F007F,italic,face:%(times)s,size:%(size)d" % faces)
  # Single quoted string
  ctrl.StyleSetSpec(wxstc.STC_P_CHARACTER, "fore:#7F007F,italic,face:%(times)s,size:%(size)d" % faces)
  # Keyword
  ctrl.StyleSetSpec(wxstc.STC_P_WORD, "fore:#00007F,bold,size:%(size)d" % faces)
  # Triple quotes
  ctrl.StyleSetSpec(wxstc.STC_P_TRIPLE, "fore:#7F0000,size:%(size)d" % faces)
  # Triple double quotes
  ctrl.StyleSetSpec(wxstc.STC_P_TRIPLEDOUBLE, "fore:#7F0000,size:%(size)d" % faces)
  # Class name definition
  ctrl.StyleSetSpec(wxstc.STC_P_CLASSNAME, "fore:#0000FF,bold,underline,size:%(size)d" % faces)
  # Function or method name definition
  ctrl.StyleSetSpec(wxstc.STC_P_DEFNAME, "fore:#007F7F,bold,size:%(size)d" % faces)
  # Operators
  ctrl.StyleSetSpec(wxstc.STC_P_OPERATOR, "bold,size:%(size)d" % faces)
  # Identifiers
  ctrl.StyleSetSpec(wxstc.STC_P_IDENTIFIER, "fore:#808080,face:%(helv)s,size:%(size)d" % faces)
  # Comment-blocks
  ctrl.StyleSetSpec(wxstc.STC_P_COMMENTBLOCK, "fore:#7F7F7F,size:%(size)d" % faces)
  # End of line where string is not closed
  ctrl.StyleSetSpec(wxstc.STC_P_STRINGEOL, "fore:#000000,face:%(mono)s,back:#E0C0E0,eol,size:%(size)d" % faces)
  ctrl.SetMarginType(1, wxstc.STC_MARGIN_NUMBER)
  ctrl.SetMarginWidth(1, 24)#16

#Default text for the output script window
defaultOutputScript = (
"""#Introduce here arbitrary python one-line statements to compute
#images and point clouds from the registered data.

#A default script is provided. Sections may be added, deleted
#or modified to compute, render and show different image files.

#save all registered images as a big point cloud in PLY format,
#color-coded by image
rt.saveRegisteredHeightmapstoPLY(rt.savePath+'final.registered.ply', removeNANs=True, smoothByOldest=False)

rt.log('final.registered.ply SAVED\\n') #show message in the log tab

#this statement computes three images from the registered data:
#  In the image 'first', pixels are taken from the oldest registered
#  image (oldest in the registration order).
#  In the image 'diff', pixels are the difference between the minimum
#  and maximum value for all images over that pixel
#  In the image 'avg', pixels are the average for all images over that pixel
#All three images are computed with NAN values where no data or invalid
#data were present. However, if the parameter removeNANs=False is used,
#invalid data is interpolated from the neighbouring valid data (non-existent
#data is still rendered as NAN values)
imgs, pixelstep = rt.getImages(removeNANs=True, outputs=['first', 'diff', 'avg'])
#imgs, pixelstep = rt.getImages(removeNANs=False, outputs=['first', 'diff', 'avg']) #use this to get interpolated data

rt.log('Images COMPUTED\\n')

#These lines are intended to rotate the 'first' and 'avg' images,
#since residual bias may be present and amplified by the large
#registered surface
R = rt.getImageRotation(imgs['avg'], pixelstep, method='SVD') # VERY FAST
#R = rt.getImageRotation(imgs['avg'], pixelstep, method='RANSAC') #VERY SLOW

rt.rotateImage(imgs['avg'], pixelstep, R)
rt.rotateImage(imgs['first'], pixelstep, R)

rt.log('Images ROTATED\\n')

#save the images in several formats

#the TIFF format allows to save the images as matrices of unbounded double
#floating point values. They are good to save the raw images, but some image
#viewers may be unable to display this kind of image.

rt.saveImage(imgs['avg'],  rt.savePath+'final.avg.tif')
rt.saveImage(imgs['first'],  rt.savePath+'final.first.tif')
rt.saveImage(imgs['diff'],  rt.savePath+'final.diff.tif')

#The PNG format is popular. A color map is used to render the Z values as colors
#In the case of the 'diff' image, a bounded log scale is used to display more
#effectively what areas have small or large values
rt.saveImage(imgs['avg'],  rt.savePath+'final.avg.png')
rt.saveImage(imgs['first'],  rt.savePath+'final.first.png')
rt.saveImage(imgs['diff'],  rt.savePath+'final.diff.png', uselog=True, vmin=0.1, vmax=10)

#save the 'first' and 'avg' images as point clouds, color-coded by Z values.
rt.saveImage(imgs['avg'],  rt.savePath+'final.avg.ply')
rt.saveImage(imgs['first'],  rt.savePath+'final.first.ply')

rt.log('images SAVED\\n')

#show the images in separate windows with colorbars,
#with the ability to zoom, pan, and save snapshots of the images.
#In the case of the 'diff' image, a bounded log scale is used to display more
#effectively what areas have small or large values
->rt.showImage(imgs['avg'])
->rt.showImage(imgs['first'])
->rt.showImage(imgs['diff'], uselog=True, vmin=0.1, vmax=10)

rt.log('output script FINISHED\\n')""")

#Default text for the freeform params window
defaultFreeFormParams = (
"""#Introduce here arbitrary python one-line statements to modify the 
#configuration object. Modify only if you know what you are doing. 
#The parameters can be found in the class ConfigurationRegister
#in heightmaps.py, or simpleRegister.py
#
#Examples:
#conf.RANSAC_fitplane_enable = False
#conf.PhaseCorrWhitening = False
#conf.PhaseCorrMinratio    = 0.01""")

########################################################################
### APP CODE, EVENT HANDLERS ###
########################################################################


class AllFrameConcrete(AllFrame):
  """Subclassing wxGlade's object"""
  def __init__(self, *args, **kwds):
    """giant, dumb constructor. It should really be pieced into meaningful, self-contained  snippets"""
    AllFrame.__init__(self, *args, **kwds)

    #setup python scripting windows    
    setPythonSyntax(self.outputscript)
    self.outputscript.SetText(defaultOutputScript)
    setPythonSyntax(self.freeformparams)
    self.freeformparams.SetText(defaultFreeFormParams)
     
    #setup input filetypes
    self.resetFileList()
    template = lambda txt: '*.%s;*.%s' % (txt.upper(), txt.lower())
    self.filetypes = ';'.join((template('plu'), template('png'), template('tif'), template('tiff')))
    
    #self.openfoldertxt.SetValue('/home/josedavid/3dprint/software/pypcl/a')
    
    #setup logging-log window bridge
    self.mystdout = RedirectText(self.logctrl)
    
    # Set up event handler for any worker thread results
    EVT_RESULT(self,self.OnResult)

    #matplotlib integration taken from these two webs:
    #https://sukhbinder.wordpress.com/2013/12/19/matplotlib-with-wxpython-example-with-panzoom-functionality/   
    #http://wiki.scipy.org/Matplotlib_figure_in_a_wx_panel
    self.figure = plt.figure()
    self.canvas = FigureCanvas(self.visupanel,-1, self.figure)
    #self.toolbar = NavigationToolbar(self.canvas)
    #self.toolbar.Hide()
    self.visupanel.Bind(wx.EVT_IDLE, self._onIdle)
    self.visupanel.Bind(wx.EVT_SIZE, self._onSize)
    self._resizeflag = True
    self._SetSize()
    #d1 = np.random.random((25,))
    #d2 = np.random.random((25,))
    #d3 = np.random.random((25,))
    ax = Axes3D(self.figure) #self.figure.add_subplot(111, projection='3d')
    ax.hold(True)
    #ax.plot(d1, '*')
    #ax.plot(d1, d2, d3, '*')
    self.canvas.draw()    
    self.ax = ax
    
    
    #initialize variables to communicate with working threads
    self.worker = None
    self.rt = None
    self.rectangles = None
    
    self.htmlw.LoadPage(op.join(thisdir, "help.html")) #for wx.html.HtmlWindow (static HTML)
    #self.htmlw.LoadURL(op.join(thisdir, "help.html")) #for wx.html2.WebView (accepts javascript)
    
    #self.alltabs.SetSelection(1)
    #switch specs are 3-tuples with the widget causing the switch, the boolean function, and the list of dependent widgets to enable/disable.
    #The list (last element of the tuple) may containt nested switch specs, to allow for recursive switching
    switchGrid  = (self.usegridmode,  self.usegridmode.GetValue,                  [self.gridcomplete, self.nrows, self.ncols, self.fillmode, self.gridmode, self.registerorder, self.labelnrows, self.labelncols])
    switchRotR  = (self.rotationalgo, lambda: self.rotationalgo.GetSelection()>0, [self.ransacfpk, self.ransacfpt, self.ransacfpr, self.labelransac1, self.labelransac2, self.labelransac3, self.sizerRotRANSACParamsLabel_staticbox])
    switchRot   = (self.rotationmode, lambda: self.rotationmode.GetSelection()>0, [self.sizerRotParamLabel_staticbox, self.rotationalgo, switchRotR])
    switchPCS   = (self.subpixelpc,   self.subpixelpc.GetValue,                   [self.labelpc1, self.subpixelfacpc])
    switchPC    = (self.enablepc,     self.enablepc.GetValue,                     [self.whiteningpc, self.subpixelpc, self.corrcoefpc, self.numpeakspc, self.labelpc2, self.labelpc3, self.labelpc4, self.minratiopc, switchPCS])
    self.switchspecs = (switchGrid, switchRotR, switchRot, switchPCS,switchPC)
    #initialize top-level specs    
    for spec in (switchGrid, switchRot, switchPC):
      self.switchOptions(spec)
      
    #validation specs are pairs with a boolean function and a list of closures to validate the widgets
    #The closures contain a dummy string (for visual identification purposes), a function, and a list of arguments for the function)
    #If the boolean function (first element of the pair) returns false, the list is not validated
    validationScaling = (
      lambda: True, [
      ('Z',          helperValidateNumeric, (1, self.zctrl, float, 'This is not a valid value for the Z scale: ', 'defaultPixelStep')),
      ('PixelStep',  helperValidateNumeric, (1, self.stepctrl, float, 'This is not a valid value for the Pixel scale: ', 'zfac')),
      ('SubsMedian', helperValidateBool,    (1, self.rebasebymedian, 'substractMedian')),
      #the rotation mode radiobox has to be validated inconditionally, so we include it here
      ('ROTMODE',    helperRadioBox,
          (1, self.rotationmode, 
          'Please select a plane rotation mode',
          'Please select a valid plane rotation mode', 
          [h.C.PLANEROT.NOROTATION, h.C.PLANEROT.JUSTFIRST, h.C.PLANEROT.ALLBYFIRST, h.C.PLANEROT.ALLINDEPENDENT], 'rotateMode'))
      ])
    validationRotParams = (
      lambda: self.rotationmode.GetSelection()>0, [
      ('ROTALGO',    helperRadioBox,
          (1, self.rotationalgo, 
          'Please select an algorithm for the plane rotation',
          'Please select a valid algorithm for the plane rotation', 
          [False, True], 'RANSAC_fitplane_enable')),
       ])
    validationRotRANSACParams = (
      lambda: (self.rotationmode.GetSelection()>0) and (self.rotationalgo.GetSelection()>0), [
      ('RANSACFPK',  helperValidateNumeric,
          (1, self.ransacfpk, int, 'This is not a valid value for the number of iterations (1-1000): ', 'RANSAC_fitplane_k', (1, 1000))),
      ('RANSACFPT',  helperValidateLambdaWithNumber, 
          (1, self.ransacfpt, float, 'lambda maxstep: (maxstep*%s)', 'This is not a valid value for the distance to the plane: ', 'RANSAC_fitplane_tfun', (0, np.inf))),
      ('RANSACFPR', helperValidateNumeric, 
          (1, self.ransacfpr, float, 'This is not a valid value for the ratio of points (0..1): ', 'RANSAC_fitplane_planeratio', (0, 1))),
      ])
    validationPhaseCorrelation = (
      self.enablepc.GetValue, [
      ('PCPREWHT', helperValidateBool,         (2, self.whiteningpc, 'PhaseCorrWhitening')),
      ('PCSUBPIX', helperValidateBool,         (2, self.subpixelpc,  'PhaseCorrSubpixel')),
      ('PCCORRCO', helperValidateNumeric,      (2, self.corrcoefpc,    float, 'This is not a valid value for the correlation coefficient threshold (0..1): ', 'PhaseCorrRecommendableCorrCoef', (0, 1))),
      ('PCCORRCO', helperValidateNumeric,      (2, self.minratiopc,    float, 'This is not a valid value for the correlation coefficient threshold (0..1): ', 'PhaseCorrMinratio', (0, 1))),
      ('PCPKSEQ',  helperValidateIncreasingNumericSequence,
             (2, self.numpeakspc, int,
             'This sequence contains a non-valid value (1-1000000) in position %d: %s',
             'This sequence contains a non-increasing value in position %d: %s',
             'PhaseCorrNumPeaks', (1, 1000000)))
      ])
    validationPhaseCorrelationSP = (
      lambda: self.enablepc.GetValue() and self.subpixelpc.GetValue(), [
      ('PCSUBFAC', helperValidateNumeric,      (2, self.subpixelfacpc, float, 'This is not a valid value for the subpixel scale factor scale (1..1000): ',       'PhaseCorrScale', (1, 1000))),
      ])

    validationOtherParams = (
      lambda: True, [
      #the phase correlation checkbox has to be validated inconditionally, so we include it here (not a problem for the order, since the validation always succeeds...)
      ('ENABLEPC', helperValidateBoolToValues, (2, self.enablepc, [h.C.FIRSTPHASE.RANSACROTATION, h.C.FIRSTPHASE.PHASECORRELATION], 'firstPhase')),
      ('KEYTHR',   helperValidateNumeric,      (2, self.matchthr, float, 'This is not a valid value for the keypoint matching threshold (0..1): ',       'matcherThreshold', (0, 1))),
      ('RANSACRTK',  helperValidateNumeric,
          (2, self.ransacrtk, int, 'This is not a valid value for the number of iterations (1-1000): ', 'RANSAC_k', (1, 1000))),
      ('RANSACRTT',  helperValidateLambdaWithNumber, 
          (2, self.ransacrtt, float, 'lambda maxstep: (maxstep*%s)**2', 'This is not a valid value for the distance between matching points after the rotation: ', 'RANSAC_tfun', (0, np.inf))),
      ('RANSACRTR', helperValidateLambdaWithNumber, 
          (2, self.ransacrtr, float, 'lambda num_matches: max(n.floor(num_matches*%s), 6)', 'This is not a valid value for the ratio of matched keypoints (0..1): ', 'RANSAC_dfun', (0, 1))),
      ('ICPMXIT',   helperValidateNumeric,      (2, self.icpiters, float, 'This is not a valid value for the maximum number of iterations of the ICP algorithm (1..10000): ',       'ICP_maxiter', (0, 10000)))
      ])
    
    self.allValidations = [validationScaling, validationRotParams, validationRotRANSACParams, validationPhaseCorrelation, validationPhaseCorrelationSP, validationOtherParams]
    
    #self.alltabs.ChangeSelection(5)
    
    #Bind the EVT_CLOSE event to closeWindow(), because something I did (maybe adding threads, I don't know for sure) ruined the default behaviour
    self.Bind(wx.EVT_CLOSE, self.closeWindow)

  def closeWindow(self, event):
    """Close the app"""
    plt.close()
    self.Destroy() #This will close the app window.
    wx.GetApp().ExitMainLoop()

  def msgbox(self, message):
    """safe execution of msgbox, because it may interfere with the event handling from the matplotlib figure"""
    if self.canvas is not None and self.canvas.HasCapture():
      self.canvas.ReleaseMouse()
    msgbox(message)
  
  def _onSize(self, event):
    """machinery for matplotlib integration"""
    self._resizeflag = True

  def _onIdle(self, evt):
    """machinery for matplotlib integration"""
    if self._resizeflag:
        self._resizeflag = False
        self._SetSize()

  def _SetSize(self, pixels = None):
      """
      machinery for matplotlib integration
      This method can be called to force the Plot to be a desired size, which defaults to
      the ClientSize of the panel
      """
      if not pixels:
          pixels = self.visupanel.GetClientSize()
      self.canvas.SetSize(pixels)
      self.figure.set_size_inches(pixels[0]/self.figure.get_dpi(), pixels[1]/self.figure.get_dpi())
      self.canvas.gui_repaint()#(drawDC=wx.PaintDC(self.canvas))

  def switchOptions(self, event):
    """generic event handler for checkbox/radiobutton enable/disable switching"""
    switchSpec = None
    if type(event) in (tuple, list):
      switchSpec = event
    else:
      obj = event.GetEventObject()
      for spec in self.switchspecs:
        if obj==spec[0]:
          switchSpec = spec
          break
      if switchSpec is None:
        raise Exception('This should not happen. Object: '+str(obj))
    boolfun, widgetlist = switchSpec[1:]    
    val = boolfun()
    for x in widgetlist:
      if type(x)!=tuple: #regular widget
        x.Enable(val)
      else: #nested switch spec
        if not val: #disable all nested specs
          #copy the tuple, but inconditionally disable the widgets
          subspec = list(x)
          subspec[1] = lambda: False
        else:
          subspec = x
        #execute the nested spec
        self.switchOptions(subspec)
  
  def doClearLog(self, event):
    """event handler to clean the log window"""
    self.logctrl.SetValue('')

  def showRectangles(self, event=None):  
    """event handler to display the rectangles in the matplotlib window"""
    rectangles = self.rectangles
    if rectangles is None:
      return
    ax = self.ax
    ax.clear()
    ax.hold(True)
    Xb, Yb, Zb = fake3dEqualAxes(ax, rectangles[:,:,0], rectangles[:,:,1], rectangles[:,:,2])
    for rectangle, color in it.izip(rectangles, h.C.COLORS01):
      ax.add_collection3d(Poly3DCollection([rectangle], facecolors=color))
    #ax.add_collection3d(Poly3DCollection(rectangles, facecolors=h.C.COLORS01))
    if self.rectanglelabels.GetValue():
      for idx, rectangle in enumerate(rectangles):
        center = rectangle.mean(axis=0)
        ax.text(center[0], center[1], center[2]+10, str(idx))
    for idx, (fun, bound) in enumerate(zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), (Xb, Yb, Zb))):
       fun(min(np.min(bound), np.min(rectangles[:,:,idx])),max(np.max(bound), np.max(rectangles[:,:,idx])))
       #fun(np.min(rectangles[:,:,idx]),np.max(rectangles[:,:,idx]))
    self.canvas.draw()    
  
  def OnResult(self, event):
    """event handler to receive notifications from worker threads.
    This is a bad mess and a different function should be factored
    out for each event type"""
    data = event.data
    if data[0]=='draw': #draw rectangles in the matplotlib window
      #get processed flags and rectangles
      processed  = data[1]
      rectangles = data[2]
      #reorder them according to the processing order, to honor the colors and the indexes
      processed  = processed[self.rt.processOrder]
      rectangles = rectangles[self.rt.processOrder]
      #take only those already processed
      self.rectangles = rectangles[processed]
      self.showRectangles()
    elif data[0]=='endRegistration': #registration has finished
      ok    = data[1] and self.rt.finished
      abort = data[2]
      self.computeButton.Enable(True)
      self.abortButton.Enable(False)
      self.computeOutputButton.Enable(True)
      self.alltabs.ChangeSelection(4)
      if abort:
        self.mystdout.write('\n\nABORTED')
      else:
        self.mystdout.write('\n\nFINISHED')
        if ok:
          self.msgbox('Registration finished. If there are errors or warnings, they can be inspected in the log window')
        else:
          self.msgbox('Registration process failed')
      self.worker = None
    elif data[0]=='endOutput': #output script has been finished
      if not data[2]:
        #error
        self.msgbox(data[3])
        self.alltabs.ChangeSelection(6)
        self.outputscript.SetFocus()
      else:
        ok = True
        deferredlines, globs = data[1]
        for line, compiled, locs in deferredlines:
          try:
            exec(compiled, globs, locs)
          except:
            self.msgbox('There was an error while trying to evaluate this deferred output command (any previous one has been executed):\n<%s>\nError:\n%s' % (line, traceback.format_exc()))
            ok = False
            break
        if ok:
          self.msgbox(data[3])
      self.computeButton.Enable(True)
      self.computeOutputButton.Enable(True)
      self.worker=None
    else:
      #should never happen...
      self.msgbox('unknown result event:\ntype=%s\nvalue: <%s>' % (str(type(data)), str(data)))
    
  def resetFileList(self):
    """helper to reset the list of files"""
    self.filelistctrl.ClearAll()
    self.filelistctrl.InsertColumn(0, '#')
    self.filelistctrl.InsertColumn(1, 'File Name')
    self.filelistctrl.InsertColumn(2, 'Path')
    self.filelistctrl.SetColumnWidth(0, -1)
    self.filelistctrl.SetColumnWidth(1, 300)
    self.filelistctrl.SetColumnWidth(2, 50)
  
  def doOpenFolderBut(self, event):
    """open a new working directory"""
    openDirDialog(self.openfoldertxt, "Select working directory", wx.DD_CHANGE_DIR)
  
  def changeOpenDir(self, event):
    """set the working directory"""
    dr = self.validateOpenDir()
    if dr[0]:
      os.chdir(dr[1])
    else:
      self.msgbox(dr[1])
      
      
  def doProcessFileNames(self, event):
    """make sure that file names are correct"""
    files = processMultiline(self.inputfilesctrl.GetValue())
    nofiles = [f for f in files if not op.isfile(f)]
    ok = len(nofiles)==0
    if ok:
      self.resetFileList()
      for idx, af in enumerate(files):
        self.filelistctrl.InsertStringItem(idx, str(idx))
        af = op.abspath(af)
        d, f = op.split(af)
        self.filelistctrl.SetStringItem(idx, 1, f)
        self.filelistctrl.SetStringItem(idx, 2, d)
    else:
      self.msgbox('The following ARE NOT files:\n'+'\n'.join(nofiles))
  
  def doSelectFiles(self, event):
    """Select files with a dialog"""
    result = genericFileDialog('Select Input Files', self.openfoldertxt.GetValue(), 
                               '', self.filetypes, wx.FD_OPEN | wx.FD_MULTIPLE)
    if result is not None:
      self.inputfilesctrl.SetValue('\n'.join(sorted(result)))
      self.doProcessFileNames(event)
  
  
  def validateOpenDir(self):
    """exactly what it says in the tin"""
    dr = self.openfoldertxt.GetValue()
    if len(dr)==0:
      return (False, 'The working directory has not been set')
    if dr[-1]!=os.sep:
      dr += os.sep
    if op.isdir(dr):
      return (True, dr)
    else:
      return (False, 'This is not a valid working directory: \n'+dr, (0, self.openfoldertxt))
  
  def validateInputFiles(self):
    """exactly what it says in the tin"""
    num = self.filelistctrl.GetItemCount()
    if num>1:
      files = [op.join(self.filelistctrl.GetItem(i, 2).GetText(), self.filelistctrl.GetItem(i, 1).GetText()) for i in xrange(num)]
      return (True, files)
    else:
      return (False, 'There must be at least two validated file names...', (0, self.inputfilesctrl))
  
  def validateConf(self, conf):  
    """validate all non-disabled configuration parameters"""
    for dovalidationfun, closures in self.allValidations:
      if dovalidationfun(): #test if the parameters are enabled
        #for each parameter, validate it
        for name, fun, args in closures:
          ret = fun(*args)
          if ret[0]:
            #print "setting parameter %s to %s" % (ret[2], str(ret[1]))
            setattr(conf, ret[2], ret[1])
          else:
            return ret
    #return (False, 'XKCD', (2, self.numpeakspc))
    return (True, conf)

  def validateProcessSpecification(self, files):
    """get the process specification"""
    if self.usegridmode.GetValue():
      ret = helperValidateNumeric(0, self.nrows, int, 'This is not a valid value for the number of rows: ', '')
      if not ret[0]: return ret
      nrows = ret[1]
      ret = helperValidateNumeric(0, self.ncols, int, 'This is not a valid value for the number of columns: ', '')
      if not ret[0]: return ret
      ncols = ret[1]
      if self.gridcomplete.GetValue():
        totalNum = None
      else:
        totalNum = len(files)
        if totalNum>(nrows*ncols):
          return (False, 'The number of files is bigger than the grid size %d*%d=%d' % (nrows, ncols, nrows*ncols), (0, self.inputfilesctrl))
      ret = helperRadioBox(0, self.fillmode, 
                           'Please select a fill mode (1)', 
                           'Please select a valid fill mode (1)', ['byrow', 'bycol'], '')
      if not ret[0]: return ret
      fillmode = ret[1]
      ret = helperRadioBox(0, self.gridmode, 
                           'Please select a fill mode (2)', 
                           'Please select a valid fill mode (2)', ['snake', 'grid'], '')
      if not ret[0]: return ret
      gridmode = ret[1]
      ret = helperRadioBox(0, self.registerorder, 
                           'Please select a register order', 
                           'Please select a valid register order', ['zigzag', 'straight'], '')
      if not ret[0]: return ret
      registerorder = ret[1]
      processSpecification = h.makeGridProcessSpecification(nrows, ncols, fillmode=fillmode, gridmode=gridmode, 
                                                          totalNum=totalNum, registerOrder=registerorder)
      #print processSpecification
      return (True, processSpecification)
    else:
      return (True, None)
  
  def validateALL(self, conf):
    """validate all input parameters"""
    #validate working directory
    wrkdir = self.validateOpenDir()
    if not wrkdir[0]: return wrkdir
    #validate input files
    files  = self.validateInputFiles()
    if not files[0]: return files
    #validate process specification
    processSpecification = self.validateProcessSpecification(files)
    if not processSpecification[0]: return processSpecification
    #validate configuration options
    conf = self.validateConf(conf)
    if not conf[0]: return conf
    #return all validated parameters
    return (True, conf[1], wrkdir[1], files[1], processSpecification[1])
  
  def processNotValidated(self, result):
    if len(result)>2:
      self.alltabs.ChangeSelection(result[2][0])
      result[2][1].SetFocus()
    self.msgbox(result[1])
  
  def doAbort(self, event):
    self.worker.abort()
    self.msgbox('This button sends a signal to abort the registration. The process will be aborted when it can process the signal, which may take several minutes if it is in the middle of a lengthy sub-computation.')
  
  def doCompute(self, event):
    """registering time!!!!"""
    conf = h.ConfigurationRegister()
    result = self.validateALL(conf)
    if not result[0]:
      self.processNotValidated(result)
    else:
      conf, wrkdir, files, processSpecification = result[1:]
      conf.copyHeightmaps = False
      conf.RANSAC_fitplane_saferatio = 1.0
      lines = processMultiline(self.freeformparams.GetText(), removeComments=True, doStrip=False)
      for line in lines:
        try:
          exec(line)
        except:
          self.msgbox('There was an error while trying to evaluate this freeform command:\n<%s>\nError:\n%s' % (line, traceback.format_exc()))
          self.alltabs.ChangeSelection(3)
          self.freeformparams.SetFocus()
          return
      self.computeButton.Enable(False)
      self.computeOutputButton.Enable(False)
      self.abortButton.Enable(True)
      self.logctrl.SetValue('')
      self.alltabs.ChangeSelection(4)
      self.rt = h.RegisterTools(savePath=wrkdir, originalLogHandles=[self.mystdout])
      self.rectangles = None #reset rectangles
      closure = [doComputeFun, self.rt, files, processSpecification, conf, wrkdir, {}]
#      closure = [h.simpleComputeRegistration, files,
#                 {'originalLogHandles':[self.mystdout], 
#                 'forceNew':True, 'processSpecification':processSpecification,
#                 'path':wrkdir, 'conf':conf}]
      self.worker = WorkerThread(self, closure, 'endRegistration')
      self.logctrl.SetFocus()

  def executeOutputStatements(self, notifyFun, abortFun, wrkdir, rt):
    """helper function for the closure to execute output script statements in a separate thread.
    It is somewhat complex because lines prefixed with -> are not executed here, but they are
    compiled and packaged to be executed in the parent wxApp thread, because in Windows, matplotlib
    figures have to be declared in the parent thread (they close automagically when the thread
    in which they were generated is closed). This is useful for rt.showImages(...) statements
    in the output script, which generate matplotlib figures"""
    lines = processMultiline(self.outputscript.GetText(), removeEmpty=False, removeComments=False, doStrip=False)
    deferredexec = []
    for linenum, line in enumerate(lines, 1):
      try:
        striped = line.strip()
        if (striped=='') or striped.startswith('#'):
          continue
        if line.startswith('->'): #deferred execution
          compiled = compile(line[2:], '<statement %d>' % linenum, 'exec')
          context = locals()
          #maybe we want to remove self from locals? but then, the docs say to NOT modify this dictionary. Maybe making a copy of the dictoionary, then removing self... too much work...
          deferredexec.append((line, compiled, context))
        else:
          compiled = compile(line, '<statement %d>' % linenum, 'exec')
          exec(compiled)# in globals(), locals()
      except:
        notifyFun(('endOutput', [], False, 'There was an error while trying to evaluate this output command (in the line %d, any previous one has been executed) for the registration in the working directory (%s):\n<%s>\nError:\n%s' % (linenum, wrkdir, line, traceback.format_exc())))
        return
    globalContext = globals()
    globalContext = {}
    notifyFun(('endOutput', (deferredexec, globalContext), True, 'All output commands were executed without errors'))

  def doOutput(self, event):
    "event handler to set up the execution of the output script"""
    wrkdir = self.validateOpenDir()
    if not wrkdir[0]:
      self.processNotValidated(wrkdir)
      return
    else:
      wrkdir = wrkdir[1]
    try:
      rt = h.RegisterTools(savePath=wrkdir, originalLogHandles=[self.mystdout])
      if not op.isfile(rt.files['state']):
        self.msgbox('To process output commands, the directory %s must have been used to process a registration. A key file is missing. Maybe you intended to write another directory, or some files were deleted?' % wrkdir)
        return
      rt.loadVars(None)
      if not rt.finished:
        self.msgbox('To process output commands, the directory %s must have been used to process a registration. It seems like the registration was initiated, but then aborted. We cannot execute output commands in this directory' % wrkdir)
        return
    except:
      self.msgbox('To process output commands, the directory %s must have been used to process a registration. It seems like the registration was initiated, but some problem arised. Maybe it was aborted, or some file is missing or corrupted. Error from the registration engine: \n%s' % (wrkdir, traceback.format_exc()))
      return
    self.computeButton.Enable(False)
    self.computeOutputButton.Enable(False)
    closure = [self.executeOutputStatements, wrkdir, rt, {}]
    self.worker = WorkerThread(self, closure, None)


########################################################################
### BOILERPLATE CODE TO START THE APP ###
########################################################################

# end of class MyFrame
class MyApp(wx.App):
    def OnInit(self):
        #wx.InitAllImageHandlers() #remove annoying warning message
        frame_1 = AllFrameConcrete(None, wx.ID_ANY, "") #start the beast
        self.SetTopWindow(frame_1)
        frame_1.Show()
        return 1

# end of class MyApp

def main():
    try:
      gettext.install("app") # replace with the appropriate catalog name

      app = MyApp(0)
      app.MainLoop()
    except:
      print 'An exception propagated outside of Wxpython: '
      traceback.print_exc()
      terminate()
    

if __name__ == "__main__":
  main()
