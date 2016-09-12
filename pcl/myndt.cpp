#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/icp.h>


#include <Eigen/Dense>

#include "myndt.h"

//#include <math.h>

#include <stdio.h>

bool registerWithNormals(int npoints1, double *xyz1, double *normals1, int npoints2, double *xyz2, double *normals2, double *vec,
                         bool usemaxiter, int maxiter, 
                         bool usemaxcorrdist, double maxcorrdist, 
                         bool usetepsilon, double tepsilon, 
                         bool usefepsilon, double fepsilon) {
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud1 (new pcl::PointCloud<pcl::PointNormal>);
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud2 (new pcl::PointCloud<pcl::PointNormal>);
  pcl::PointCloud<pcl::PointNormal>::Ptr final  (new pcl::PointCloud<pcl::PointNormal>);
  pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal>::Matrix4 transf;

  printf("doing cloud 1\n");
  cloud1->width = npoints1;
  cloud1->height = 1;
  cloud1->is_dense = false;
  cloud1->points.resize(npoints1);
  for (int i=0; i<npoints1; i++) {
    cloud1->points[i].x = *xyz1++;
    cloud1->points[i].y = *xyz1++;
    cloud1->points[i].z = *xyz1++;
    cloud1->points[i].normal[0] = *normals1++;
    cloud1->points[i].normal[1] = *normals1++;
    cloud1->points[i].normal[2] = *normals1++;
  }
  
  printf("doing cloud 2\n");
  cloud2->width = npoints2;
  cloud2->height = 1;
  cloud2->is_dense = false;
  cloud2->points.resize(npoints2);
  for (int i=0; i<npoints2; i++) {
    cloud2->points[i].x = *xyz2++;
    cloud2->points[i].y = *xyz2++;
    cloud2->points[i].z = *xyz2++;
    cloud2->points[i].normal[0] = *normals2++;
    cloud2->points[i].normal[1] = *normals2++;
    cloud2->points[i].normal[2] = *normals2++;
  }

  printf("after doing clouds\n");
  pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> icp;

  if (usemaxiter) {
    icp.setMaximumIterations(maxiter);
  }
  if (usemaxcorrdist) {
    icp.setMaxCorrespondenceDistance(maxcorrdist);
  }
  if (usetepsilon) {
    icp.setTransformationEpsilon(tepsilon);
  }
  if (usefepsilon) {
    icp.setTransformationEpsilon(fepsilon);
  }

  
  printf("hola10\n");
  icp.setInputCloud(cloud1);
  icp.setInputTarget(cloud2);

  printf("hola11\n");
  icp.align(*final);
  printf("hola12\n");

  transf = icp.getFinalTransformation();
  for (int k=0;k<16;k++) {
    vec[k] = transf.data()[k];
  }

  return icp.hasConverged();

}

double executeNDT(pcl::PointCloud<pcl::PointXYZ>::Ptr &source, pcl::PointCloud<pcl::PointXYZ>::Ptr &target, //pcl::PointCloud<pcl::PointXYZ> *output,
                bool useepsilon, double epsilon, 
                bool usestep, double step, 
                bool useresolution, double resolution,
                bool useiters, double maxiters, double *vec) {
  // Initializing Normal Distributions Transform (NDT).
  pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
  pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>::Matrix4 transf;
  pcl::PointCloud<pcl::PointXYZ> output;
//  Eigen::Matrix4f identity;
//
//  for (int k=0;k<16;k++) {
//    identity.data()[k] = 0.0;
//  }
//  for (int k=0;k<16;k+=5) {
//    identity.data()[k] = 1.0;
//  }

  // Setting scale dependent NDT parameters
  if (useepsilon) {
    // Setting minimum transformation difference for termination condition.
    //ndt.setTransformationEpsilon (0.01);
    ndt.setTransformationEpsilon (epsilon);
  }
  if (usestep) {
    // Setting maximum step size for More-Thuente line search.
    //ndt.setStepSize (0.1);
    ndt.setStepSize (step);
  }
  if (useresolution) {
    //Setting Resolution of NDT grid structure (VoxelGridCovariance).
    //ndt.setResolution (1.0);
    ndt.setResolution (resolution);
  }
  if (useiters) {
    // Setting max number of registration iterations.
    //ndt.setMaximumIterations (35);
    ndt.setMaximumIterations (maxiters);
  }
  

  // Setting point cloud to be aligned.
  ndt.setInputSource (source);
  // Setting point cloud to be aligned to.
  ndt.setInputTarget (target);
  
  printf("hola11\n");
  ndt.align (output);//, identity);//, init_guess);
  printf("hola2\n");
  
  transf = ndt.getFinalTransformation();
  for (int k=0;k<16;k++) {
    vec[k] = transf.data()[k];
  }

  return ndt.getTransformationProbability();
  
}


