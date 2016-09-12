#include <stdio.h>
#include <fcntl.h>
#include <io.h>


//#include <iostream>
//#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>


//VERY SIMPLE PROGRAM, READ INPUT PARAMETERS AND CLOUDS FROM STDIN IN BINARY MODE; WRITE RESULTS TO STDOUT IN BINARY MODE ALSO
int main (int argc, char** argv)
{

   int result, numpointsA, numpointsB, max_iter, has_converged;
   float point[3];
   double fitnessScore;
   Eigen::Matrix4f transf;

   // Set "stdin" to have binary mode:
   result = _setmode( _fileno( stdin ), _O_BINARY );
   if( result == -1 )
      return -1;
   // Set "stdout" to have binary mode:
   result = _setmode( _fileno( stdout ), _O_BINARY );
   if( result == -1 )
      return -2;
  
  fread(&numpointsA, sizeof(numpointsA), 1, stdin);
  fread(&numpointsB, sizeof(numpointsB), 1, stdin);
  fread(&max_iter,   sizeof(max_iter),   1, stdin);
  
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZ>);

  // Fill in the CloudIn data
  cloud_in->width    = numpointsA;
  cloud_in->height   = 1;
  cloud_in->is_dense = false;
  cloud_in->points.resize (cloud_in->width * cloud_in->height);
  for (size_t i = 0; i < cloud_in->points.size (); ++i)
  {
    fread(point, sizeof(float), 3, stdin);
    cloud_in->points[i].x = point[0];
    cloud_in->points[i].y = point[1];
    cloud_in->points[i].z = point[2];
  }
  
  // Fill in the CloudOut data
  cloud_out->width    = numpointsB;
  cloud_out->height   = 1;
  cloud_out->is_dense = false;
  cloud_out->points.resize (cloud_out->width * cloud_out->height);
  for (size_t i = 0; i < cloud_out->points.size (); ++i)
  {
    fread(point, sizeof(float), 3, stdin);
    cloud_out->points[i].x = point[0];
    cloud_out->points[i].y = point[1];
    cloud_out->points[i].z = point[2];
  }
  
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  
  if (max_iter>=0) {
    icp.setMaximumIterations(max_iter);
  }
  
  icp.setInputCloud(cloud_in);
  icp.setInputTarget(cloud_out);
  
  pcl::PointCloud<pcl::PointXYZ> Final;
  
  icp.align(Final);
  
  has_converged = icp.hasConverged();
  fitnessScore  = icp.getFitnessScore();
  transf        = icp.getFinalTransformation();
  
  fwrite(&has_converged, sizeof(has_converged),  1, stdout);
  fwrite(&fitnessScore,  sizeof(fitnessScore),   1, stdout);
  fwrite(transf.data(),  sizeof(float),         16, stdout);
  
  return (0);
}
