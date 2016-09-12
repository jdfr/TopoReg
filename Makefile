all: pcl/_pcl.so pcl/registration.so

#pcl/_pcl.so: pcl/_pcl.pxd pcl/_pcl.pyx setup.py pcl/pcl_defs.pxd \
#             pcl/minipcl.cpp pcl/indexing.hpp pcl/cpd_impl.cpp pcl/cpd_impl.h
#	python setup.py build_ext --inplace

pcl/_pcl.so: pcl/_pcl.pxd pcl/_pcl.pyx setup.py pcl/pcl_defs.pxd \
             pcl/minipcl.cpp pcl/adhoc.cpp pcl/indexing.hpp
	python setup.py build_ext --inplace

#pcl/cpd_impl.o: pcl/cpd_impl.c pcl/cpd_impl.h
#	x86_64-linux-gnu-gcc -c -fPIC -o pcl/cpd_impl.o pcl/cpd_impl.c

pcl/registration.so: setup.py pcl/_pcl.pxd pcl/pcl_defs.pxd \
                      pcl/registration.pyx pcl/myndt.cpp
	python setup.py build_ext --inplace

test: pcl/_pcl.so tests/test.py
	nosetests -s

clean:
	rm -rf build
	rm -f pcl/_pcl.cpp pcl/registration.cpp pcl/_pcl.so

doc: pcl.so conf.py readme.rst
	sphinx-build -b singlehtml -d build/doctrees . build/html

showdoc: doc
	gvfs-open build/html/readme.html
