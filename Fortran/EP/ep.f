c  Copyright (c) 2011 - 2014
c    University of Houston System and Oak Ridge National Laboratory.
c  
c  All rights reserved.
c  
c  Redistribution and use in source and binary forms, with or without
c  modification, are permitted provided that the following conditions
c  are met:
c  
c  o Redistributions of source code must retain the above copyright notice,
c    this list of conditions and the following disclaimer.
c  
c  o Redistributions in binary form must reproduce the above copyright
c    notice, this list of conditions and the following disclaimer in the
c    documentation and/or other materials provided with the distribution.
c  
c  o Neither the name of the University of Houston System, Oak Ridge
c    National Laboratory nor the names of its contributors may be used to
c    endorse or promote products derived from this software without specific
c    prior written permission.
c  
c  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
c  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
c  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
c  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
c  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
c  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
c  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
c  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
c  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
c  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
c  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
c  

c---------------------------------------------------------------------
      program EMBAR
c---------------------------------------------------------------------
C
c   This is the MPI version of the APP Benchmark 1,
c   the "embarassingly parallel" benchmark.
c
c
c   M is the Log_2 of the number of complex pairs of uniform (0, 1) random
c   numbers.  MK is the Log_2 of the size of each batch of uniform random
c   numbers.  MK can be set for convenience on a given system, since it does
c   not affect the results.

      implicit none

      include 'npbparams.h'
      include 'mpinpb.h'
      include 'mpp/shmem.fh'

      integer num_pes, my_pe
      double precision Mops, epsilon, a, s, t1, t2, t3, t4, x1, 
     >                 x2, q, x,  an, tt, gc, dum(3),
     >                 timer_read
c      double precision sx, sy, tm
      double precision, save::sx 
      double precision, save::sy 
      double precision, save::tm 
      double precision sx_verify_value, sy_verify_value, sx_err, sy_err
      integer          mk, mm, nn, nk, nq, np, ierr, node, no_nodes, 
     >                 i, ik, kk, l, k, nit, ierrcode, no_large_nodes,
     >                 np_add, k_offset, j
      logical          verified, timers_enabled
      parameter       (timers_enabled = .false.)
      external         randlc, timer_read
      double precision randlc, qq
      character*14     size

      parameter (mk = 16, mm = m - mk, nn = 2 ** mm,
     >           nk = 2 ** mk, nq = 10, epsilon=1.d-8,
     >           a = 1220703125.d0, s = 271828183.d0)
      common/storage/ x(2*nk), q(0:nq-1), qq(10000)
      data             dum /1.d0, 1.d0, 1.d0/
      integer, dimension(SHMEM_BCAST_SYNC_SIZE), save :: psync
      double precision, dimension(SHMEM_REDUCE_MIN_WRKDATA_SIZE),
     > save :: pwrk

c      call mpi_init(ierr)
c      call mpi_comm_rank(MPI_COMM_WORLD,node,ierr)
c      call mpi_comm_size(MPI_COMM_WORLD,no_nodes,ierr)
      call start_pes(0)
      no_nodes = num_pes()
      node = my_pe()

      psync = SHMEM_SYNC_VALUE
      call shmem_barrier_all

      root = 0

      if (.not. convertdouble) then
         dp_type = MPI_DOUBLE_PRECISION
      else
         dp_type = MPI_REAL
      endif

      if (node.eq.root)  then

c   Because the size of the problem is too large to store in a 32-bit
c   integer for some classes, we put it into a string (for printing).
c   Have to strip off the decimal point put in there by the floating
c   point print statement (internal file)

          write(*, 1000)
          write(size, '(f13.0)' ) 2.d0**(m+1)
          do j =14,1,-1
             if (size(j:j) .eq. '.') size(j:j) = ' '
          end do
          write (*,1001) size
          write(*, 1003) no_nodes

 1000 format(/,' NAS Parallel Benchmarks 3.2 -- EP Benchmark',/)
 1001     format(' Number of random numbers generated: ', a15)
 1003     format(' Number of active processes:         ', i13, /)

      endif

      verified = .false.

c   Compute the number of "batches" of random number pairs generated 
c   per processor. Adjust if the number of processors does not evenly 
c   divide the total number

      np = nn / no_nodes
      no_large_nodes = mod(nn, no_nodes)
      if (node .lt. no_large_nodes) then
         np_add = 1
      else
         np_add = 0
      endif
      np = np + np_add

      if (np .eq. 0) then
         write (6, 1) no_nodes, nn
 1       format ('Too many nodes:',2i6)
c         call mpi_abort(MPI_COMM_WORLD,ierrcode,ierr)
         stop
      endif

c   Call the random number generator functions and initialize
c   the x-array to reduce the effects of paging on the timings.
c   Also, call all mathematical functions that are used. Make
c   sure these initializations cannot be eliminated as dead code.

      call vranlc(0, dum(1), dum(2), dum(3))
      dum(1) = randlc(dum(2), dum(3))
      do 5    i = 1, 2*nk
         x(i) = -1.d99
 5    continue
      Mops = log(sqrt(abs(max(1.d0,1.d0))))

c---------------------------------------------------------------------
c      Synchronize before placing time stamp
c---------------------------------------------------------------------
c      call mpi_barrier(MPI_COMM_WORLD, ierr)
      call shmem_barrier_all
      
      call timer_clear(1)
      call timer_clear(2)
      call timer_clear(3)
      call timer_start(1)

      t1 = a
      call vranlc(0, t1, a, x)

c   Compute AN = A ^ (2 * NK) (mod 2^46).

      t1 = a

      do 100 i = 1, mk + 1
         t2 = randlc(t1, t1)
 100  continue

      an = t1
      tt = s
      gc = 0.d0
      sx = 0.d0
      sy = 0.d0

      do 110 i = 0, nq - 1
         q(i) = 0.d0
 110  continue

c   Each instance of this loop may be performed independently. We compute
c   the k offsets separately to take into account the fact that some nodes
c   have more numbers to generate than others

      if (np_add .eq. 1) then
         k_offset = node * np -1
      else
         k_offset = no_large_nodes*(np+1) + (node-no_large_nodes)*np -1
      endif

      do 150 k = 1, np
         kk = k_offset + k 
         t1 = s
         t2 = an

c        Find starting seed t1 for this kk.

         do 120 i = 1, 100
            ik = kk / 2
            if (2 * ik .ne. kk) t3 = randlc(t1, t2)
            if (ik .eq. 0) goto 130
            t3 = randlc(t2, t2)
            kk = ik
 120     continue

c        Compute uniform pseudorandom numbers.
 130     continue

         if (timers_enabled) call timer_start(3)
         call vranlc(2 * nk, t1, a, x)
         if (timers_enabled) call timer_stop(3)

c        Compute Gaussian deviates by acceptance-rejection method and 
c        tally counts in concentric square annuli.  This loop is not 
c        vectorizable. 

         if (timers_enabled) call timer_start(2)

         do 140 i = 1, nk
            x1 = 2.d0 * x(2*i-1) - 1.d0
            x2 = 2.d0 * x(2*i) - 1.d0
            t1 = x1 ** 2 + x2 ** 2
            if (t1 .le. 1.d0) then
               t2   = sqrt(-2.d0 * log(t1) / t1)
               t3   = (x1 * t2)
               t4   = (x2 * t2)
               l    = max(abs(t3), abs(t4))
               q(l) = q(l) + 1.d0
               sx   = sx + t3
               sy   = sy + t4
            endif
 140     continue

         if (timers_enabled) call timer_stop(2)

 150  continue


      call shmem_barrier_all();
c      call mpi_allreduce(sx, x, 1, dp_type,
c     >                   MPI_SUM, MPI_COMM_WORLD, ierr)
      call shmem_real8_sum_to_all(x,sx,1,0,0,no_nodes,pwrk,psync)
      sx = x(1)
c      call mpi_allreduce(sy, x, 1, dp_type,
c     >                   MPI_SUM, MPI_COMM_WORLD, ierr)
      call shmem_real8_sum_to_all(x,sy,1,0,0,no_nodes,pwrk,psync)
      sy = x(1)
c      call mpi_allreduce(q, x, nq, dp_type,
c     >                   MPI_SUM, MPI_COMM_WORLD, ierr)
      call shmem_real8_sum_to_all(x,q,nq,0,0,no_nodes,pwrk,psync)

      do i = 1, nq
         q(i-1) = x(i)
      enddo

      do 160 i = 0, nq - 1
        gc = gc + q(i)
 160  continue

      call timer_stop(1)
      tm  = timer_read(1)

c      call mpi_allreduce(tm, x, 1, dp_type,
c     >                   MPI_MAX, MPI_COMM_WORLD, ierr)
      call shmem_real8_max_to_all(x,tm,1,0,0,no_nodes,pwrk,psync)
      tm = x(1)

      if (node.eq.root) then
         nit=0
         verified = .true.
         if (m.eq.24) then
            sx_verify_value = -3.247834652034740D+3
            sy_verify_value = -6.958407078382297D+3
         elseif (m.eq.25) then
            sx_verify_value = -2.863319731645753D+3
            sy_verify_value = -6.320053679109499D+3
         elseif (m.eq.28) then
            sx_verify_value = -4.295875165629892D+3
            sy_verify_value = -1.580732573678431D+4
         elseif (m.eq.30) then
            sx_verify_value =  4.033815542441498D+4
            sy_verify_value = -2.660669192809235D+4
         elseif (m.eq.32) then
            sx_verify_value =  4.764367927995374D+4
            sy_verify_value = -8.084072988043731D+4
         elseif (m.eq.36) then
            sx_verify_value =  1.982481200946593D+5
            sy_verify_value = -1.020596636361769D+5
         else
            verified = .false.
         endif
         if (verified) then
            sx_err = abs((sx - sx_verify_value)/sx_verify_value)
            sy_err = abs((sy - sy_verify_value)/sy_verify_value)
            verified = ((sx_err.le.epsilon) .and. (sy_err.le.epsilon))
         endif
         Mops = 2.d0**(m+1)/tm/1000000.d0

         write (6,11) tm, m, gc, sx, sy, (i, q(i), i = 0, nq - 1)
 11      format ('EP Benchmark Results:'//'CPU Time =',f10.4/'N = 2^',
     >           i5/'No. Gaussian Pairs =',f15.0/'Sums = ',1p,2d25.15/
     >           'Counts:'/(i3,0p,f15.0))

         call print_results('EP', class, m+1, 0, 0, nit, npm, 
     >                      no_nodes, tm, Mops, 
     >                      'Random numbers generated', 
     >                      verified, npbversion, compiletime, cs1,
     >                      cs2, cs3, cs4, cs5, cs6, cs7)

      endif

      if (timers_enabled .and. (node .eq. root)) then
          print *, 'Total time:     ', timer_read(1)
          print *, 'Gaussian pairs: ', timer_read(2)
          print *, 'Random numbers: ', timer_read(3)
      endif

c      call mpi_finalize(ierr)

      end
