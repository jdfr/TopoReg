#To use the library in Windows, you can either install all dependencies or use winpython

#if you just want to use the GUI, the easiest route is to use Winpython. 
#However, winpython is like 1.5GB in a fresh install, way too much.
#You can reduce the app footprint by using cx_freeze (you can install golkhe's
#cx_freeze wheel in wxpython http://www.lfd.uci.edu/~gohlke/pythonlibs/ )
#This script drives cx_freeze to generate a standalone app.

#launchApplication.bat can be used to start the frozen application.

##use with this command line: python setupapp.py build

import sys
from cx_Freeze import setup, Executable
import glob

prefix1 = 'C:\\Users\\jd\\TopoReg\\TopoRegWithWinPython\\WinPython-64bit-2.7.9.3\\python-2.7.9.amd64\\Lib\\site-packages\\scipy\\special\\'
prefix2 = r"C:\Users\jd\TopoReg\TopoRegWithWinPython\WinPython-64bit-2.7.9.3\python-2.7.9.amd64\Lib\site-packages\numpy\core" "\\"
prefix3 = r"C:\Users\jd\TopoReg\TopoRegWithWinPython\WinPython-64bit-2.7.9.3\python-2.7.9.amd64" "\\"

incl = lambda prf, fil: (prf+fil, fil)

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {"packages": ["os",
                                  "mpl_toolkits.mplot3d",
                                  "scipy",
                                  ],
                                  "excludes": ["tkinter"],
                                  "include_files": [
                                       #fix some weird error to freeze scipy
                                       incl(prefix1, '_ufuncs.pyd'),
                                       incl(prefix2, 'libmmd.dll'),
                                       incl(prefix2, 'libifcoremd.dll'),
                                       #WinPython's MVS08 runtime, just in case
                                       incl(prefix3, 'msvcm90.dll'),
                                       incl(prefix3, 'msvcp90.dll'),
                                       incl(prefix3, 'msvcr90.dll'),
                                       #add our own dependencies (opencv is already taken care of, as it is used as a python library)
                                       incl('', 'help.html'),
                                       incl('', 'icptool.exe')
                                       ]+[incl('', x) for x in glob.glob('pcl_*.dll')]
                                  }

# GUI applications require a different base on Windows (the default is for a
# console application).
base = None
#if sys.platform == "win32":
#    base = "Win32GUI"

setup(  name = "TopoReg",
        version = "0.1",
        description = "Topographic image registration",
        options = {"build_exe": build_exe_options},
        executables = [Executable("wxgui.py", base=base)])