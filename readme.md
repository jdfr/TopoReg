TopoReg
=======

TopoReg is a library / application in Python 2.7.X and some bits of Cython and C++. It is designed to facilitate the registration for PLU files from a SensoFar confocal microscope (an old model which does not have any facilities for heightmap stitching, and has manual controls for sample motion in XY), although adapting it to accept any kind of topographic image data should be quite straightforward. It is designed to be able to register many files, although it does not perform global registration (i.e. the files are registered sequentially; optionally it can be given hints about the registration order so errors are minimized).

TopoReg grew "organically" from strawlab's python-pcl bindings, which were modified in the process (ironically, it cannot use them in Windows). It was developed in debian sid (almost indentical to debian jessie at the time of main development). Here are the instructions to build it from repos and source where needed. No instructions for other linux distros are provided, but it should be straightforward to build it in these. It can also be compiled on windows provided we do not actually use the python-pcl bindings (because Python 2.7.X in Windows is compiled in the ancient MSVS2008, while PCL requires later versions because C++11). For that reason, in Windows, PCL functionality is encapsulated in a small standalone application that is called from the library / main python application.

A standalone binary package for windows is available as a release.

Build instructions
------------------

Sadly, I developed this before becoming educated into the necessity of single-step build processes, so the build process is not automated. Here, we describe the instructions for Linux: 

* first, we need to pull our dependencies: installing spyder (for development) from the repos should pull almost everything we need in the python side: numpy, scipy, matplotlib, cython...

* we also need to install wxpython (in debian it is based upon the GTK port of wxwidgets) and wxGlade is useful to do most of the GUI's boilerplate automatically.

* we use opencv. Specifically, we also use the nonfree module, which is a mess to install in debian, although a stackoverflow.com contributor has written fairly easy instructions to build it ( http://stackoverflow.com/questions/18561910/opencv-python-cant-use-surf-sift#answer-27694433 ). We also modified slightly one of the sources of the nonfree module (sift.cpp), which should be replaced in the source tree of OpenCV's nonfree module (either the upstream tarball or in the debian source package that we modified to include the nonfree module) if we compile from source. The modification to sift.cpp is to enable the algorithm to accept images whose color components are in floating point format.

* we use PCL 1.7, available from the repos. Just install and enjoy. If you have to build it from source, godspeed.

* to compile strawlab's python-pcl bindings, just run the setup.py file (command "python setup.py build_ext --inplace") or the Makefile from the source directory.

* we also need to compile a custom cython extension with the command "python accumsetup.py build_ext --inplace"

* if we want to change the gui, we need launch wxglade to make the modification and rewrite wxsimpleapp.py

* and I think that with all this, we are ready to go.

For Windows it is mostly the same, except that we cannot build python-pcl bindings, so we have to compile the small icptool application with cmake (after installing PCL 1.7).

Brief description of the sources
--------------------------------

* wxgui.py: user interface. Contains many snippets adapted from a long list of third-party sources (wxpython multithreading, wxpython stdout redirection, wxpython StyledTextCtrl configuration, matplotlib integration in wxpython apps, etc.).

* wxsimpleapp.py: boilerplate user interface code generate by wxglade from wxsimpleapp.wxg

* accum.pyx: cython extension to speed up some array routines

* write3d.py: routines to write output files in PLY format (point clouds), can be open with meshlab

* tifffile.py: to write tiff files (scipy depends upon PIL/Pillow, whose tiff support is not very good or inexistent, depending on the version). Adapted from third-party code (Golkhe's python modules, http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html ).

* stl.py: to write stl files, can be open with meshlab. Adapted from third-party code (python-stl v1.3.3: https://github.com/WoLpH/numpy-stl/ ).

* simpleRegister.py: simple script to perform a registration

* simpleAnalyzer.py: simple script to analyze the results of a registration process

* setupapp.py: script to freeze an executable from wxgui.py (to be used in windows with winpython)

* setup.py: script to compile strawlab's python-pcl bindings

* conf.py: part of the configuration files for the python-pcl bindings

* readme.rst.python-pcl and LICENSE.python-pcl: came with the python-pcl bindings

* accumsetup.py: script to compile a custom cython extension to speed up some array processing routines

* rotations.py: computational geometry routines. Some routines are adapted from third-party code (Golkhe's python modules, http://www.lfd.uci.edu/~gohlke/code/transformations.py.html ).

* register_images.py: superresolution registration with FFT upsampling. Adapted from third-party code https://github.com/keflavich/image_registration

* ransac.py: generic RANSAC implementation. Adapted from third-party code http://scipy.github.io/old-wiki/pages/Cookbook/RANSAC

* pluconv.py: code to read PLU files from our confocal device. Adapted from third-party code (gwyddion, originally in C).

* imreg.py: code to do phase correlation. Adapted from third-party code (Fiji's stitching plugin, originally in Java).

* heightmaps.py: main file, containing the registration and analysis API.

* help.odt: source file for the help displayed in the user interface. It has to be saved as help.html to be displayed.

* pcl: directory containing the sources for the python-pcl bindings. Thoroughly modified to be used by heightmaps.py

* icptool: directory with the source for a command line application to expose PCL's ICP algorithm to heightmaps.py in Windows, where it is impractical to compile the PCL bindings for python 2.7

Above, "Adapted from third-party code" means a variety of things, such as "copied verbatim", "slightly modified", "inspired", "completely transformed", "translated from another language", or any combination thereof

Author
------

Jose David Fernández Rodríguez

License
-------

Boost License (see LICENSE.txt), to the extent it is compatible with the licenses of the software I used to build this together.