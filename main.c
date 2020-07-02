/*======================================================================*/
// Purpose       : Poisson solver with SOR, Gauss-Seidel and Jacobi scheme
// Parallelism   : OpenMP, OpenMPI
// Date          : 11 June 2019
// exact solution: https://math.stackexchange.com/questions/1251117/analytic-solution-to-poisson-equation
/*======================================================================*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <stdbool.h>
#include "Timer.h"

#define JACOBI       0
#define GAUSS_SEIDEL 1
#define SOR          2

#define METHOD       SOR /* JACOBI, GAUSS_SEIDEL, or SOR */
#define NUM_THREADS 8
#define FLOAT8

#define NUM_NX 256

#define MPI

#ifdef FLOAT8
#define real      double
#define FABS( x ) fabs( x )
#define SIN( x )  sin( x )
#define COS( x )  cos( x )
#define EPSILON   __DBL_EPSILON__
#define MPI_MYREAL  MPI_DOUBLE
#else
#define real      float
#define FABS( x ) fabsf( x )
#define SIN( x )  sinf( x )
#define COS( x )  cosf( x )
#define EPSILON   __FLT_EPSILON__
#define MPI_MYREAL  MPI_FLOAT
#endif

#define SQR( x )  ( (x) * (x) )

void **calloc_2d_array (size_t nr, size_t nc, size_t size);


int main ( int argc, char *argv[]  )
{
  Timer_t Timer;

  int itr = 0;

  /* Create files */
  FILE *fptr  = fopen("Potential","w");


  /*Number of grids*/
  const int Nx = NUM_NX;
  const int Ny = Nx;

# if ( METHOD == SOR )
  /* Overrelaxation parameter(1<w<2)*/
  real w, correction;

  switch ( Nx )
  {
    case 32:
     w=(real) 1.8200;
     break;
    case 64:
     w=(real) 1.9200;
     break;
    case 128:
     w=(real) 1.9525;
     break;
    case 256:
     w=(real) 1.9767;
     break;
  }
# endif


  /*Size of computational domain*/
  const size_t Lx = 1;
  const size_t Ly = 1;


  /* Initial MPI */
# ifdef MPI
  int MyRank, NRank;
  MPI_Init( &argc, &argv );
  MPI_Comm_rank( MPI_COMM_WORLD, &MyRank );
  MPI_Comm_size( MPI_COMM_WORLD, &NRank );
# endif

  /* Size of array */
  int DimArryX;
# ifdef MPI
  DimArryX     = Nx/2+1;
# else
  DimArryX     = Nx;
# endif

  /*Memory allocation*/
  real **Mass            = (real **) calloc_2d_array (DimArryX, Ny, sizeof (real));
  real **Potential       = (real **) calloc_2d_array (DimArryX, Ny, sizeof (real));


# if ( METHOD == JACOBI )
  real **Potential_New   = (real **) calloc_2d_array (Nx, Ny, sizeof (real));
# endif

  real  **ExactPotential, **RelativeError;


# ifdef MPI
  if ( MyRank == 0 )
# endif
  {
    /* Exact solution*/
    ExactPotential = (real **) calloc_2d_array (Nx, Ny, sizeof (real));

    /* Relative error between exact and numerical solution*/
    RelativeError  = (real **) calloc_2d_array (Nx, Ny, sizeof (real));
  }

  real Error = 0.0;
  real Threshold = (real)(100*EPSILON);


  /*size of grid*/
  const real dx = Lx/(real)(Nx-1);
  const real dy = Ly/(real)(Ny-1);

  /* mass distribution */
  for(int i=0; i<DimArryX  ;i++) // DimArryX = Nx/2+1
  for(int j=0; j<Ny        ;j++)
  {
    real x, y;

#   ifdef MPI
    if      ( MyRank == 0 )  x =         i  * dx;
    else if ( MyRank == 1 )  x = (Nx/2-1+i) * dx;
#   else
    x = i * dx;
#   endif

    y = j * dy;

    Mass[i][j] = (real)2.0*x*(y-(real)1.0)*(y-(real)2*x+x*y+(real)2)*exp(x-y);
  }

  /* BC condition along x-direction*/
  for (int i = 0; i<DimArryX; i++) // DimArryX = Nx/2+1
  {
    Potential[i][Ny-1]     = (real)0.0;
    Potential[i][   0]     = (real)0.0; 
   
#   ifdef MPI 
    if ( MyRank == 0 )
#   endif
    {
      ExactPotential[i][Ny-1]   = 0.0;
      ExactPotential[i][   0]   = 0.0;
    }
  }

  /* BC condition along y-direction*/
  for (int j = 1; j<Ny-1; j++)
  {
#   ifdef MPI
    if ( MyRank == 0 )
      Potential[   0][j] = (real)0.0;

    if ( MyRank == 1 )
      Potential[Nx/2][j] = (real)0.0;
#   endif

#   ifdef MPI
    if ( MyRank == 0 )
#   endif
    {
      ExactPotential[   0][j]     = 0.0;
      ExactPotential[Nx-1][j]     = 0.0;
    }
  }

  /*initial guess*/
  if ( MyRank == 0 )
  {
    for(int i=1; i<Nx/2+1 ;i++)
    for(int j=1; j<Ny-1   ;j++)
    {
      Potential[i][j] = 1.0;
    }
  }
  else
  {
    for(int i=0; i<Nx/2 ;i++)
    for(int j=1; j<Ny-1 ;j++)
      Potential[i][j] = (real)1.0;
  }


  bool Stop0, Stop1, Stop01, Stop10, Stop;

# ifdef MPI
  if ( MyRank == 0 )
  {
    Stop0   = false;
    Stop10  = false;
  }
  else if ( MyRank == 1 )
  {
    Stop1   = false;
    Stop01  = false;
  }

# endif


  /* Start timer */
  Timer.Start();

  /* Perform relaxation */
  do
  {
            Error=(real)0.0;
            itr++;
     
#           if ( METHOD == SOR )  /*Successive Overrelaxation*/

           /* update odd cells */
#           pragma omp parallel for collapse(2) reduction(+:Error) private(correction) num_threads(NUM_THREADS)
            for(int i=1; i<Nx/2 ;i++)
            for(int j=1; j<Ny-1 ;j++)
            {
               if ( (i+j)%2 == 1 )
               {
                   correction = (real)0.25 * w * (   Potential[i+1][j  ]
                                                   + Potential[i-1][j  ]
                                                   + Potential[i  ][j+1]
                                                   + Potential[i  ][j-1] - dx*dy * Mass[i][j] - (real)4.0 * Potential[i][j] );

                  /*sum of error*/
                  Error += FABS( correction/Potential[i][j] );
   
                  /*update*/
                  Potential[i][j] += correction;
               }
            }

            /* MPI synchronization */
#           ifdef MPI
            MPI_Barrier(MPI_COMM_WORLD);
#           endif

            /* Swap data on both sides of the dividing line between rank 0 and rank 1 */
#           ifdef MPI
            int Tag = itr;

            if ( MyRank == 0 )
            {
               MPI_Send( Potential[Nx/2-1], Ny, MPI_MYREAL, 1, Tag,   MPI_COMM_WORLD );
               MPI_Recv( Potential[Nx/2]  , Ny, MPI_MYREAL, 1, Tag+1, MPI_COMM_WORLD, MPI_STATUSES_IGNORE );
            }
            else if ( MyRank == 1 )
            {
               MPI_Recv( Potential[0]     , Ny, MPI_MYREAL, 0, Tag,   MPI_COMM_WORLD, MPI_STATUSES_IGNORE );
               MPI_Send( Potential[1]     , Ny, MPI_MYREAL, 0, Tag+1, MPI_COMM_WORLD );
            }
#           endif

            /* update even cells */
#           pragma omp parallel for collapse(2) reduction(+:Error) private(correction) num_threads(NUM_THREADS)
            for(int i=1; i<Nx/2 ;i++)
            for(int j=1; j<Ny-1 ;j++)
            {
               if ( (i+j)%2 == 0 )
               {
                  correction = (real)0.25 * w * ( Potential[i+1][j  ]
                                                + Potential[i-1][j  ]
                                                + Potential[i  ][j+1]
                                                + Potential[i  ][j-1] - dx*dy * Mass[i][j] -(real)4.0 * Potential[i][j] );

                  /*sum of error*/
                  Error += FABS( correction/Potential[i][j] );
   
                  /*update*/
                  Potential[i][j] += correction;
                }
            }

            /* MPI synchronization */
#           ifdef MPI
            MPI_Barrier(MPI_COMM_WORLD);
#           endif

            /* Swap data on both sides of the dividing line between rank 0 and rank 1 */
#           ifdef MPI
            Tag =Tag+10;

            if ( MyRank == 0 )
            {
               MPI_Send( Potential[Nx/2-1], Ny, MPI_MYREAL, 1, Tag,   MPI_COMM_WORLD );
               MPI_Recv( Potential[Nx/2]  , Ny, MPI_MYREAL, 1, Tag+1, MPI_COMM_WORLD, MPI_STATUSES_IGNORE );
            }
            else if ( MyRank == 1 )
            {
               MPI_Recv( Potential[0]     , Ny, MPI_MYREAL, 0, Tag,   MPI_COMM_WORLD, MPI_STATUSES_IGNORE );
               MPI_Send( Potential[1]     , Ny, MPI_MYREAL, 0, Tag+1, MPI_COMM_WORLD );
            }
#           endif

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
#        ifdef MPI
         Error /= (real)( (Nx/2) * (Ny-2) );

         if (  MyRank == 0 )
         {
           if ( Error < Threshold ) Stop0 = true;
           MPI_Send( &Stop0,   1, MPI_C_BOOL, 1, 9999,  MPI_COMM_WORLD );
           MPI_Recv( &Stop10,  1, MPI_C_BOOL, 1, 99999, MPI_COMM_WORLD, MPI_STATUSES_IGNORE );
         } 
         if (  MyRank == 1 )
         {
           if ( Error < Threshold ) Stop1 = true;
           MPI_Send( &Stop1,   1, MPI_C_BOOL, 0, 99999, MPI_COMM_WORLD );
           MPI_Recv( &Stop01,  1, MPI_C_BOOL, 0, 9999,  MPI_COMM_WORLD, MPI_STATUSES_IGNORE );
         }


         if ( MyRank == 0 )        Stop = Stop0 && Stop10;
         else if ( MyRank == 1 )   Stop = Stop1 && Stop01;
#        else
         Error /= (real)( (Nx-2) * (Ny-2) );
         Stop = Error < Threshold;
#        endif

         if (MyRank == 0) printf("itr=%d\n", itr);

  }while( !Stop );




  Timer.Stop();

  /* allocate memory to store final potential */
# ifdef MPI
  real **Potential_All = (real **) calloc_2d_array (Nx, Ny, sizeof (real));
  real **Mass_All      = (real **) calloc_2d_array (Nx, Ny, sizeof (real));
# endif

  /* copy data from `Potential` on rank0 to `Potential_All` on rank0 */
  if ( MyRank == 0 )
  {
    for (int i=0;i<Nx/2;i++)
    for (int j=0;j<Ny  ;j++)
    {
      Potential_All[i][j] = Potential[i][j];
      Mass_All     [i][j] = Mass     [i][j];
    }
  }

  /* copy data from `Potential` on rank1 to `Potential_All` on rank0 */
  if ( MyRank == 1 )
  {
    MPI_Send( &Potential[0][0],        (Nx/2)*Ny, MPI_MYREAL, 0, 123, MPI_COMM_WORLD );
    MPI_Send( &Mass     [0][0],        (Nx/2)*Ny, MPI_MYREAL, 0, 124, MPI_COMM_WORLD );
  }

  if ( MyRank == 0 )
  {
    MPI_Recv( &Potential_All[Nx/2][0], (Nx/2)*Ny, MPI_MYREAL, 1, 123, MPI_COMM_WORLD, MPI_STATUSES_IGNORE );
    MPI_Recv( &Mass_All     [Nx/2][0], (Nx/2)*Ny, MPI_MYREAL, 1, 124, MPI_COMM_WORLD, MPI_STATUSES_IGNORE );
  }


  /* free memory */
  free((void*)Mass[0]);
  free((void*)Mass);
  free((void*)Potential[0]);
  free((void*)Potential);


  if ( MyRank == 0 )
  {
    
    /* ================= compare numerical solution with exact solution ================= */
    real L1Error;
    
    L1Error = (real)0.0;
    
    for(int i=0; i<Nx ;i++)
    for(int j=0; j<Ny ;j++)
    {
         real x=i*dx;
         real y=j*dy;
    
         ExactPotential[i][j] = x*y*((real)1-x)*((real)1-y)*exp(x-y);
    
         /*calculate relative error between exact and numerical solution*/
         RelativeError[i][j] = (real)1.0 - ExactPotential[i][j] / Potential_All[i][j];
    
         /*L1-norm error between exact and numerical solution*/
         if ( RelativeError[i][j] == RelativeError[i][j] )   L1Error += FABS( RelativeError[i][j] );
    }
    
    
    L1Error /= (real)((Nx-2)*(Ny-2));
    
    /* ================= output data ================= */
       
    /*header*/
    fprintf(fptr, "#cells/sec:    %20.3e\n",(double)(Nx*Ny)/Timer.GetValue());
    fprintf(fptr, "#number of threads: %20d\n", NUM_THREADS);
    fprintf(fptr, "#L1Error:      %20.16e\n", L1Error);
    fprintf(fptr, "#Elapsed Time: %20.8e\n", Timer.GetValue());
    fprintf(fptr, "#iterations: %20d\n", itr);
    fprintf(fptr, "#========================================================\n");
    fprintf( fptr, "%13s  %14s %16s %16s %20s %20s\n",
             "#x[1]", "y[2]", "Mass[3]", "Potential_All[4]", "ExactPotential[5]", "RelativeError[6]" );
    
    /*data*/
    for(int x=0; x<Nx ;x++)
    for(int y=0; y<Ny ;y++)
       fprintf(fptr, "%10.7e   %10.7e  %10.7e  %10.7e   %10.7e   %10.7e\n",
          x*dx, y*dy, Mass_All[x][y], Potential_All[x][y], ExactPotential[x][y], RelativeError[x][y] );
  
  
  
    /* free memory */
    free((void*)Mass_All[0]);
    free((void*)Potential_All[0]);
  }

  
  return 0;
}

void **calloc_2d_array (size_t nr, size_t nc, size_t size)
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
