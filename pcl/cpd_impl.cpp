#include <math.h>

//#include <stdio.h>

#include "cpd_impl.h"

#define	max(A, B)	((A) > (B) ? (A) : (B))
#define	min(A, B)	((A) < (B) ? (A) : (B))


///////////////
double cpd_comp_P(double* x,double* y, double sigma2,double outlier,double* P1,double* Pt1,double* Px,int N,int M,int D,double *P,double *temp_x)

{
  int		n, m, d;
  double	ksig, diff, razn, outlier_tmp, sp;
  double E=0.0;
  
  
  ksig = -2.0 * sigma2;
  outlier_tmp=(outlier*M*pow (-ksig*3.14159265358979,0.5*D))/((1-outlier)*N); 
 /* printf ("ksig = %lf\n", *sigma2);*/
  /* outlier_tmp=*outlier*N/(1- *outlier)/M*(-ksig*3.14159265358979); */
  
  
  for (n=0; n < N; n++) {
  //  printf("%d/%d\n", n,N);
      
      sp=0;
      for (m=0; m < M; m++) {
          razn=0;
          for (d=0; d < D; d++) {
             diff=*(x+n+d*N)-*(y+m+d*M);  diff=diff*diff;
             razn+=diff;
          }
          
          *(P+m)=exp(razn/ksig);
          sp+=*(P+m);
      }
      
      sp+=outlier_tmp;
      *(Pt1+n)=1-outlier_tmp/ sp;
      
      for (d=0; d < D; d++) {
       *(temp_x+d)=*(x+n+d*N)/ sp;
      }
         
      for (m=0; m < M; m++) {
         
          *(P1+m)+=*(P+m)/ sp;
          
          for (d=0; d < D; d++) {
          *(Px+m+d*M)+= *(temp_x+d)**(P+m);
          }
          
      }
      
   E +=  -log(sp);     
  }
  E +=D*N*log(sigma2)/2;
    
  
  return E;
}




void cpd_comp_correspondence(double* x,double* y, double sigma2,double outlier,double* Pc,int N,int M,int D, double*P, double*P1)

{
  int		n, m, d;
  double	ksig, diff, razn, outlier_tmp,temp_x,sp;
  
  
  ksig = -2.0 * (sigma2+1e-3);
  outlier_tmp=(outlier*M*pow (-ksig*3.14159265358979,0.5*D))/((1-outlier)*N); 
  if (outlier_tmp==0) outlier_tmp=1e-10;
  
  
 /* printf ("ksig = %lf\n", *sigma2);*/
  
  
  for (n=0; n < N; n++) {
      sp=0;
      for (m=0; m < M; m++) {
          razn=0;
          for (d=0; d < D; d++) {
              diff=*(x+n+d*N)-*(y+m+d*M);  diff=diff*diff;
              razn+=diff;
          }
          
          *(P+m)=exp(razn/ksig);
          sp+=*(P+m);
          
      }
      
      sp+=outlier_tmp;
      
      
      for (m=0; m < M; m++) {
          
          temp_x=*(P+m)/ sp;
          
          if (n==0)
          {*(P1+m)= *(P+m)/ sp;
          *(Pc+m)=n+1;};
          
          if (temp_x > *(P1+m))
          {
              *(P1+m)= *(P+m)/ sp;
              *(Pc+m)=n+1;
          }
    
              
      }
      
  }

   
  return;
}