#ifndef _MYNDT_H_
#define _MYNDT_H_

#include <pcl/point_types.h>

double executeNDT(pcl::PointCloud<pcl::PointXYZ>::Ptr &source, pcl::PointCloud<pcl::PointXYZ>::Ptr &target, //pcl::PointCloud<pcl::PointXYZ> *output,
                bool useepsilon, double epsilon, 
                bool usestep, double step, 
                bool useresolution, double resolution,
                bool useiters, double maxiters,
                double *vec);

bool registerWithNormals(int npoints1, double *xyz1, double *normals1, int npoints2, double *xyz2, double *normals2, double *vec,
                         bool usemaxiter, int maxiter, 
                         bool usemaxcorrdist, double maxcorrdist, 
                         bool usetepsilon, double tepsilon, 
                         bool usefepsilon, double fepsilon);
#endif
