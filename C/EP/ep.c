/*
 *
 * Copyright (c) 2011 - 2014
 *   University of Houston System and Oak Ridge National Laboratory.
 * 
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 
 * o Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * 
 * o Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 * 
 * o Neither the name of the University of Houston System, Oak Ridge
 *   National Laboratory nor the names of its contributors may be used to
 *   endorse or promote products derived from this software without specific
 *   prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <shmem.h>

#include "type.h"
#include "npbparams.h"
#include "randdp.h"
#include "timers.h"
#include "print_results.h"

#define MAX(X,Y)  (((X) > (Y)) ? (X) : (Y))

#define MK        16
#define MM        (M - MK)
#define NN        (1 << MM)
#define NK        (1 << MK)
#define NQ        10
#define EPSILON   1.0e-8
#define A         1220703125.0
#define S         271828183.0

static double x[2 * NK];
static double qq[NQ];
static double q[NQ];
double sx, sy;
double psx, psy;
double an;
double timer1, timer2;

long pSync[_SHMEM_BCAST_SYNC_SIZE];
double pWrk[_SHMEM_REDUCE_SYNC_SIZE];

int
main (int argc, char *argv[])
{
  double Mops, t1, t2, t3, t4, x1, x2;
  double tm, tt, gc;
  double sx_verify_value, sy_verify_value, sx_err, sy_err;
  int np;
  int i, ik, kk, l, k, nit;
  int k_offset, j;
  int me, npes, kstart, kend, kl;
  int bs;
  logical verified, timers_enabled;

  double dum[3] = { 1.0, 1.0, 1.0 };
  char size[16];

  start_pes (0);
  me = _my_pe ();
  npes = _num_pes ();

  for (i = 0; i < _SHMEM_BCAST_SYNC_SIZE; i += 1)
    {
      pSync[i] = _SHMEM_SYNC_VALUE;
    }
  shmem_barrier_all ();
  FILE *fp;
  if ((fp = fopen ("timer.flag", "r")) == NULL)
    {
      timers_enabled = false;
    }
  else
    {
      timers_enabled = true;
      fclose (fp);
    }

  //--------------------------------------------------------------------
  //  Because the size of the problem is too large to store in a 32-bit
  //  integer for some classes, we put it into a string (for printing).
  //  Have to strip off the decimal point put in there by the floating
  //  point print statement (internal file)
  //--------------------------------------------------------------------
  if (me == 0)
    {
      sprintf (size, "%15.0lf", pow (2.0, M + 1));
      j = 14;
      if (size[j] == '.')
	j--;
      size[j + 1] = '\0';
      printf
	("\n\n NAS Parallel Benchmarks (NPB3.3-OpenSHMEM-C) - EP Benchmark\n");
      printf ("\n Number of random numbers generated: %15s\n", size);
      printf ("\n Number of available threads:          %13d\n", npes);
      fflush (stdout);

      verified = false;
    }

  //--------------------------------------------------------------------
  //  Compute the number of "batches" of random number pairs generated 
  //  per processor. Adjust if the number of processors does not evenly 
  //  divide the total number
  //--------------------------------------------------------------------

  np = NN;

  //--------------------------------------------------------------------
  //  Call the random number generator functions and initialize
  //  the x-array to reduce the effects of paging on the timings.
  //  Also, call all mathematical functions that are used. Make
  //  sure these initializations cannot be eliminated as dead code.
  //--------------------------------------------------------------------

  vranlc (0, &dum[0], dum[1], &dum[2]);
  dum[0] = randlc (&dum[1], dum[2]);

  for (i = 0; i < 2 * NK; i++)
    {
      x[i] = -1.0e99;
    }

  Mops = log (sqrt (fabs (MAX (1.0, 1.0))));

  //{
  timer_clear (0);
  if (timers_enabled)
    timer_clear (1);
  if (timers_enabled)
    timer_clear (2);
  //}
  timer_start (0);

  t1 = A;
  vranlc (0, &t1, A, x);

  //--------------------------------------------------------------------
  //  Compute AN = A ^ (2 * NK) (mod 2^46).
  //--------------------------------------------------------------------

  t1 = A;

  for (i = 0; i < MK + 1; i++)
    {
      t2 = randlc (&t1, t1);
    }

  an = t1;
  tt = S;
  gc = 0.0;
  sx = 0.0;
  sy = 0.0;
  for (i = 0; i < NQ; i++)
    {
      q[i] = 0.0;
    }

  //--------------------------------------------------------------------
  //  Each instance of this loop may be performed independently. We compute
  //  the k offsets separately to take into account the fact that some nodes
  //  have more numbers to generate than others
  //--------------------------------------------------------------------

  k_offset = -1;

  for (i = 0; i < NQ; i++)
    {
      qq[i] = 0.0;
    }

  bs = np / npes;
  if (bs * npes == np)
    {
      //can be evenly devided
      kstart = bs * me;
      kend = kstart + bs;
    }
  else
    {
      //can not be evenly devided
      kl = np - bs * npes;
      if (me < kl)
	{
	  kstart = bs * me;
	  kend = kstart + bs;
	}
      else
	{
	  kstart = kl * bs + (bs + 1) * (me - kl);
	  kend = kstart + bs + 1;
	}
    }
  for (k = kstart; k < kend; k++)
    {
      //kk = k_offset + k; 
      kk = k;
      t1 = S;
      t2 = an;
      // Find starting seed t1 for this kk.

      for (i = 1; i <= 100; i++)
	{
	  ik = kk / 2;
	  if ((2 * ik) != kk)
	    t3 = randlc (&t1, t2);
	  if (ik == 0)
	    break;
	  t3 = randlc (&t2, t2);
	  kk = ik;
	}

      //-----------------------------------------------------------
      //  Compute uniform pseudorandom numbers.
      //-----------------------------------------------------------
      if (timers_enabled)
	timer_start (2);
      vranlc (2 * NK, &t1, A, x);
      if (timers_enabled)
	timer_stop (2);

      //----------------------------------------------------------
      //  Compute Gaussian deviates by acceptance-rejection method and
      //  tally counts in concentri//square annuli.  This loop is not 
      //  vectorizable. 
      //---------------------------------------------------------
      if (timers_enabled)
	timer_start (1);

      for (i = 0; i < NK; i++)
	{
	  x1 = 2.0 * x[2 * i] - 1.0;
	  x2 = 2.0 * x[2 * i + 1] - 1.0;
	  t1 = x1 * x1 + x2 * x2;
	  if (t1 <= 1.0)
	    {
	      t2 = sqrt (-2.0 * log (t1) / t1);
	      t3 = (x1 * t2);
	      t4 = (x2 * t2);
	      l = MAX (fabs (t3), fabs (t4));
	      qq[l] = qq[l] + 1.0;
	      sx = sx + t3;
	      sy = sy + t4;
	    }
	}

      if (timers_enabled)
	timer_stop (1);
    }

  for (i = 0; i < NQ; i++)
    {
      q[i] = qq[i];
    }

  timer1 = timer_read (1);
  timer2 = timer_read (2);
  shmem_barrier_all ();
  shmem_double_sum_to_all (&timer1, &timer1, 1, 0, 0, npes, pWrk, pSync);
  shmem_double_sum_to_all (&timer2, &timer2, 1, 0, 0, npes, pWrk, pSync);
  shmem_double_sum_to_all (&psx, &sx, 1, 0, 0, npes, pWrk, pSync);
  shmem_double_sum_to_all (&psy, &sy, 1, 0, 0, npes, pWrk, pSync);
  shmem_barrier_all ();

  //=============
  if (me == 0)
    {
      timer1 /= npes;
      timer2 /= npes;
      sx = psx;
      sy = psy;
      for (i = 0; i < NQ; i++)
	{
	  gc = gc + q[i];
	}

      timer_stop (0);
      tm = timer_read (0);

      nit = 0;
      verified = true;
      if (M == 24)
	{
	  sx_verify_value = -3.247834652034740e+3;
	  sy_verify_value = -6.958407078382297e+3;
	}
      else if (M == 25)
	{
	  sx_verify_value = -2.863319731645753e+3;
	  sy_verify_value = -6.320053679109499e+3;
	}
      else if (M == 28)
	{
	  sx_verify_value = -4.295875165629892e+3;
	  sy_verify_value = -1.580732573678431e+4;
	}
      else if (M == 30)
	{
	  sx_verify_value = 4.033815542441498e+4;
	  sy_verify_value = -2.660669192809235e+4;
	}
      else if (M == 32)
	{
	  sx_verify_value = 4.764367927995374e+4;
	  sy_verify_value = -8.084072988043731e+4;
	}
      else if (M == 36)
	{
	  sx_verify_value = 1.982481200946593e+5;
	  sy_verify_value = -1.020596636361769e+5;
	}
      else if (M == 40)
	{
	  sx_verify_value = -5.319717441530e+05;
	  sy_verify_value = -3.688834557731e+05;
	}
      else
	{
	  verified = false;
	}
      if (verified)
	{
	  sx_err = fabs ((sx - sx_verify_value) / sx_verify_value);
	  sy_err = fabs ((sy - sy_verify_value) / sy_verify_value);
	  verified = ((sx_err <= EPSILON) && (sy_err <= EPSILON));
	}

      Mops = pow (2.0, M + 1) / tm / 1000000.0;

      printf ("\nEP Benchmark Results:\n\n");
      printf ("CPU Time =%10.4lf\n", tm);
      printf ("N = 2^%5d\n", M);
      printf ("No. Gaussian Pairs = %15.0lf\n", gc);
      printf ("Sums = %25.15lE %25.15lE\n", sx, sy);
      printf ("Counts: \n");
      for (i = 0; i < NQ; i++)
	{
	  printf ("%3d%15.0lf\n", i, q[i]);
	}

      print_results ("EP", CLASS, M + 1, 0, 0, nit,
		     tm, Mops,
		     "Random numbers generated",
		     verified, NPBVERSION, COMPILETIME, CS1,
		     CS2, CS3, CS4, CS5, CS6, CS7);

      if (timers_enabled)
	{
	  if (tm <= 0.0)
	    tm = 1.0;
	  tt = timer_read (0);
	  printf ("\nTotal time:     %9.3lf (%6.2lf)\n", tt, tt * 100.0 / tm);
	  //tt = timer_read(1);
	  tt = timer1;
	  printf ("Gaussian pairs: %9.3lf (%6.2lf)\n", tt, tt * 100.0 / tm);
	  tt = timer2;
	  printf ("Random numbers: %9.3lf (%6.2lf)\n", tt, tt * 100.0 / tm);
	}
    }
  return 0;
}
