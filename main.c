#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include "Timer.h"

#define JACOBI       0
#define GAUSS_SEIDEL 1
#define SOR          2

#define METHOD       0

#define FLOAT8

#ifdef FLOAT8
#define real      double
#define FABS( x ) fabs( x )
#define SIN( x )  sin( x )
#define COS( x )  cos( x )
#define EPSILON   __DBL_EPSILON__
#else
#define real      float
#define FABS( x ) fabsf( x )
#define SIN( x )  sinf( x )
#define COS( x )  cosf( x )
#define EPSILON   __FLT_EPSILON__
#endif

#define SQR( x )  ( (x) * (x) )

void **calloc_2d_array (size_t nr, size_t nc, size_t size);


int
main ()
{
 Timer_t Timer;

 int itr = 0;

/**/
 FILE *fptr = fopen("potential","w");

#if ( METHOD == SOR )
/* overrelaxation parameter(1<w<2)*/
  real w = 1.1;
  real correction;
#endif

/*number of grids*/
  const int Nx = 128;
  const int Ny = 128;

/*size of computational domain*/
  const size_t Lx = 1;
  const size_t Ly = 1;

/*memory allocation*/
  real **Mass           = (real **) calloc_2d_array (Nx, Ny, sizeof (real));
  real **Potential      = (real **) calloc_2d_array (Nx, Ny, sizeof (real));
# if ( METHOD == JACOBI )
  real **Potential_New  = (real **) calloc_2d_array (Nx, Ny, sizeof (real));
# endif
/*exact solution*/
  real **ExactPotential = (real **) calloc_2d_array (Nx, Ny, sizeof (real));
/*relative error between exact and numerical solution*/
  real **RelativeError  = (real **) calloc_2d_array (Nx, Ny, sizeof (real));


  real Error = 0.0;
  real Threshold = (real)(EPSILON);



/*size of grid*/
  const real dx = Lx/(real)(Nx-1);
  const real dy = Ly/(real)(Ny-1);

/* Give a mass distrbution*/
#pragma omp parallel for collapse(2)
  for(int i=0; i<Nx ;i++)
  for(int j=0; j<Ny ;j++)
  {
    real x=i*dx;
    real y=j*dy;

    Mass[i][j] = (float)2.0*x*(y-(float)1.0)*(y-(float)2*x+x*y+(float)2)*exp(x-y);
  }

/* Give BC condition along x-direction*/
#pragma omp parallel for
  for (int x = 0; x<Nx; x++)
  {
    Potential     [x][Ny-1] = (real)0.0;
    Potential     [x][   0] = (real)0.0;
    ExactPotential[x][Ny-1] = Potential[x][Ny-1];
    ExactPotential[x][   0] = Potential[x][   0];
  }

/* Give BC condition along y-direction*/
#pragma omp parallel for
  for (int y = 0; y<Ny; y++)
  {
    Potential     [   0][y] = (real)0.0;
    Potential     [Nx-1][y] = (real)0.0;
    ExactPotential[   0][y] = Potential[   0][y];
    ExactPotential[Nx-1][y] = Potential[Nx-1][y];
  }

/*initial guess*/
#pragma omp parallel for collapse(2)
  for(int x=1; x<Nx-1 ;x++)
  for(int y=1; y<Ny-1 ;y++)
    Potential[x][y] = (real)0.5;

  Timer.Start();

/* perform relaxation */
  do{
            itr++;
     
#           if ( METHOD == SOR )  /*Successive Overrelaxation*/

            /* update odd cells */
#           pragma omp parallel for collapse(2) reduction(+:Error) private(correction)
            for(int x=1; x<Nx-1 ;x++)
            for(int y=1; y<Ny-1 ;y++)
            {
               if ( (x+y)%2 == 1 )
               {
                   correction = (real)0.25 * w * (   Potential[x+1][y  ]
                                                   + Potential[x-1][y  ]
                                                   + Potential[x  ][y+1]
                                                   + Potential[x  ][y-1] - dx*dy * Mass[x][y] -(real)4.0 * Potential[x][y] );

                  /*sum of error*/
                  Error += FABS( correction/Potential[x][y] );
   
                  /*update*/
                  Potential[x][y] += correction;
               }
            }


            /* update even cells */
#           pragma omp parallel for collapse(2) reduction(+:Error) private(correction)
            for(int x=1; x<Nx-1 ;x++)
            for(int y=1; y<Ny-1 ;y++)
            {
               if ( (x+y)%2 == 0 )
               {
                  correction = (real)0.25 * w * ( Potential[x+1][y  ]
                                                + Potential[x-1][y  ]
                                                + Potential[x  ][y+1]
                                                + Potential[x  ][y-1] - dx*dy * Mass[x][y] -(real)4.0 * Potential[x][y] );

                  /*sum of error*/
                  Error += FABS( correction/Potential[x][y] );
   
                  /*update*/
                  Potential[x][y] += correction;
	       	   }
            }


#           elif ( METHOD == GAUSS_SEIDEL )


            for(int x=1; x<Nx-1 ;x++)
            for(int y=1; y<Ny-1 ;y++)
            {
               /*Gauss-Seidal*/
               real delta = (real)0.25*(   Potential[x+1][y  ]   
                                         + Potential[x-1][y  ]   
                                         + Potential[x  ][y+1] 
                                         + Potential[x  ][y-1] - dx*dy * Mass[x][y] );    

               /*sum of error*/
               Error += FABS( delta - Potential[x][y] );                                                                            


               Potential[x][y] = delta;
            }


#           elif ( METHOD == JACOBI )


            for(int x=1; x<Nx-1 ;x++)
            for(int y=1; y<Ny-1 ;y++)
            {
               Potential_New[x][y] = (real)0.25*(   Potential[x+1][y  ] 
                                                  + Potential[x-1][y  ] 
                                                  + Potential[x  ][y+1] 
                                                  + Potential[x  ][y-1] - dx*dy * Mass[x][y] );     
      
      
            /*sum of error*/
              Error += FABS( Potential_New[x][y] - Potential[x][y] );
           
            }
      
            /*memory copy*/
            for(int x=0; x<Nx-1 ;x++)
            for(int y=0; y<Ny-1 ;y++)
              Potential[x][y] = Potential_New[x][y];
      

#           endif


         /*calculate L-1 norm error*/
         Error /= (real)((Nx-2)*(Ny-2));


  }while( Error >= Threshold );

  Timer.Stop();


/* exact solution */
/*--- Ref. https://math.stackexchange.com/questions/1251117/analytic-solution-to-poisson-equation */

real Bmn, L1Error, ExactValue;

L1Error = (real)0.0;

#pragma omp parallel for reduction(+:L1Error) collapse(2)
for(int i=0; i<Nx ;i++)
for(int j=0; j<Ny ;j++)
{
     real x=i*dx;
     real y=j*dy;

     ExactPotential[i][j] = x*y*((float)1-x)*((float)1-y)*exp(x-y);

     /*calculate relative error between exact and numerical solution*/
     RelativeError[i][j] = (real)1.0 - ExactPotential[i][j] / Potential[i][j];

     /*L1-norm error between exact and numerical solution*/
     if ( RelativeError[i][j] == RelativeError[i][j] )   L1Error += FABS( RelativeError[i][j] );
}


L1Error /= (real)((Nx-2)*(Ny-2));

/*output data*/
   
   /*header*/
   fprintf(fptr, "#L1Error: %20.16e\n", L1Error);
   fprintf(fptr, "#Elapsed Time: %15.8e\n", Timer.GetValue());
   fprintf(fptr, "#iterations: %d\n", itr);
   fprintf(fptr, "#========================================================\n");
   fprintf( fptr, "%13s  %14s  %14s %16s %20s %20s",
            "#x[1]", "y[2]", "Mass[3]", "Potential[4]", "ExactPotential[5]", "RelativeError[6]\n");

   /*data*/
#  pragma omp parallel for collapse(2) ordered
   for(int x=0; x<Nx ;x++)
   for(int y=0; y<Ny ;y++)
      #pragma omp ordered
      fprintf(fptr, "%10.7e   %10.7e   %10.7e   %10.7e   %10.7e   %10.7e\n",
         x*dx, y*dy, Mass[x][y], Potential[x][y], ExactPotential[x][y], RelativeError[x][y]);


  return 0;
}


void **
calloc_2d_array (size_t nr, size_t nc, size_t size)
{
  void **array;
  size_t i;
  if ((array = (void **) calloc (nr, sizeof (void *))) == NULL)
    {
      printf ("[calloc_2d] failed to allocate mem for %d pointers\n", (int) nr);
      return NULL;
    }
  if ((array[0] = (void *) calloc (nr * nc, size)) == NULL)
    {
      printf ("[calloc_2d] failed to allocate memory (%d X %d of size %d)\n",
	      (int) nr, (int) nc, (int) size);
      free ((void *) array);
      return NULL;
    }
  for (i = 1; i < nr; i++)
    {
      array[i] = (void *) ((unsigned char *) array[0] + i * nc * size);
    }
  return array;
}
