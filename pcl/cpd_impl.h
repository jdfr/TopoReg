#ifndef _CPD_IMPL_H_
#define _CPD_IMPL_H_

double cpd_comp_P(double* x,double* y, double sigma2,double outlier,double* P1,double* Pt1,double* Px,int N,int M,int D,double *P,double *temp_x);

void cpd_comp_correspondence(double* x,double* y, double sigma2,double outlier,double* Pc,int N,int M,int D, double*P, double*P1);

#endif