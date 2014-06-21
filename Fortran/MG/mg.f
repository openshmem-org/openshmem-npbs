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
      program mg_shmem
c---------------------------------------------------------------------

      implicit none

      include 'mpinpb.h'
      include 'globals.h'
      include 'mpp/shmem.fh'

c---------------------------------------------------------------------------c
c k is the current level. It is passed down through subroutine args
c and is NOT global. it is the current iteration
c---------------------------------------------------------------------------c

      integer num_pes, my_pe
      integer k, it
      
      external timer_read
      double precision tinit, mflops, timer_read
      double precision, save::t0
      double precision, save::t
      double precision, save::ttotal

c---------------------------------------------------------------------------c
c These arrays are in common because they are quite large
c and probably shouldn't be allocated on the stack. They
c are always passed as subroutine args. 
c---------------------------------------------------------------------------c

      double precision u(nr),v(nv),r(nr),a(0:3),c(0:3)
      common /noautom/ u,v,r   

      double precision rnm2, old2, oldu, epsilon
      double precision, save::rnmu
      integer n1, n2, n3, nn
      integer, save :: nit
      double precision verify_value, err
      logical verified

      integer ierr,i, fstatus
      integer T_bench, T_init
      parameter (T_bench=1, T_init=2)
      double precision, dimension(SHMEM_REDUCE_MIN_WRKDATA_SIZE),
     > save :: pwrk
      integer, dimension(SHMEM_BCAST_SYNC_SIZE), save :: psync1
      integer, dimension(SHMEM_BCAST_SYNC_SIZE), save :: psync2

      call start_pes(0)
      nprocs = num_pes()
      me = my_pe()
      psync1 = SHMEM_SYNC_VALUE
      psync2 = SHMEM_SYNC_VALUE

      root = 0

      call timer_clear(T_bench)
      call timer_clear(T_init)

      call shmem_barrier_all()

      call timer_start(T_init)


c---------------------------------------------------------------------
c Read in and broadcast input data
c---------------------------------------------------------------------

      if( me .eq. root )then
         write (*, 1000) 

         open(unit=7,file="mg.input", status="old", iostat=fstatus)
         if (fstatus .eq. 0) then
            write(*,50) 
 50         format(' Reading from input file mg.input')
            read(7,*) lt
            read(7,*) nx(lt), ny(lt), nz(lt)
            read(7,*) nit
            read(7,*) (debug_vec(i),i=0,7)
         else
            write(*,51) 
 51         format(' No input file. Using compiled defaults ')
            lt = lt_default
            nit = nit_default
            nx(lt) = nx_default
            ny(lt) = ny_default
            nz(lt) = nz_default
            do i = 0,7
               debug_vec(i) = debug_default
            end do
         endif
      endif

      call shmem_broadcast4(lt, lt, 1, 0, 0, 0, nprocs, psync1)
      call shmem_broadcast4(nit, nit, 1, 0, 0, 0, nprocs, psync2)
      call shmem_broadcast4(nx(lt), nx(lt), 1, 0, 0, 0, nprocs, psync1)
      call shmem_broadcast4(ny(lt), ny(lt), 1, 0, 0, 0, nprocs, psync2)
      call shmem_broadcast4(nz(lt), nz(lt), 1, 0, 0, 0, nprocs, psync1)
      call shmem_broadcast4(debug_vec(0), debug_vec(0), 8, 0, 0, 0, 
     >                      nprocs, psync2)

      if ( (nx(lt) .ne. ny(lt)) .or. (nx(lt) .ne. nz(lt)) ) then
         Class = 'U' 
      else if( nx(lt) .eq. 32 .and. nit .eq. 4 ) then
         Class = 'S'
      else if( nx(lt) .eq. 128 .and. nit .eq. 4 ) then
         Class = 'W'
      else if( nx(lt) .eq. 256 .and. nit .eq. 4 ) then  
         Class = 'A'
      else if( nx(lt) .eq. 256 .and. nit .eq. 20 ) then
         Class = 'B'
      else if( nx(lt) .eq. 512 .and. nit .eq. 20 ) then  
         Class = 'C'
      else if( nx(lt) .eq. 1024 .and. nit .eq. 50 ) then  
         Class = 'D'
      else
         Class = 'U'
      endif

c---------------------------------------------------------------------
c  Use these for debug info:
c---------------------------------------------------------------------
c     debug_vec(0) = 1 !=> report all norms
c     debug_vec(1) = 1 !=> some setup information
c     debug_vec(1) = 2 !=> more setup information
c     debug_vec(2) = k => at level k or below, show result of resid
c     debug_vec(3) = k => at level k or below, show result of psinv
c     debug_vec(4) = k => at level k or below, show result of rprj
c     debug_vec(5) = k => at level k or below, show result of interp
c     debug_vec(6) = 1 => (unused)
c     debug_vec(7) = 1 => (unused)
c---------------------------------------------------------------------
      a(0) = -8.0D0/3.0D0 
      a(1) =  0.0D0 
      a(2) =  1.0D0/6.0D0 
      a(3) =  1.0D0/12.0D0
      
      if(Class .eq. 'A' .or. Class .eq. 'S'.or. Class .eq.'W') then
c---------------------------------------------------------------------
c     Coefficients for the S(a) smoother
c---------------------------------------------------------------------
         c(0) =  -3.0D0/8.0D0
         c(1) =  +1.0D0/32.0D0
         c(2) =  -1.0D0/64.0D0
         c(3) =   0.0D0
      else
c---------------------------------------------------------------------
c     Coefficients for the S(b) smoother
c---------------------------------------------------------------------
         c(0) =  -3.0D0/17.0D0
         c(1) =  +1.0D0/33.0D0
         c(2) =  -1.0D0/61.0D0
         c(3) =   0.0D0
      endif
      lb = 1
      k  = lt

      call setup(n1,n2,n3,k)
      call zero3(u,n1,n2,n3)
      call zran3(v,n1,n2,n3,nx(lt),ny(lt),k)

      call norm2u3(v,n1,n2,n3,rnm2,rnmu,nx(lt),ny(lt),nz(lt))

      if( me .eq. root )then
         write (*, 1001) nx(lt),ny(lt),nz(lt), Class
         write (*, 1002) nit

 1000 format(//,' NAS Parallel Benchmarks 3.2 -- MG Benchmark', /)
 1001    format(' Size: ', i4, 'x', i4, 'x', i4, '  (class ', A, ')' )
 1002    format(' Iterations: ', i3)
 1003    format(' Number of processes: ', i5)
         if (nprocs .ne. nprocs_compiled) then
           write (*, 1004) nprocs_compiled
           write (*, 1005) nprocs
 1004      format(' WARNING: compiled for ', i5, ' processes ')
 1005      format(' Number of active processes: ', i5, /)
         else
           write (*, 1003) nprocs
         endif
      endif

      call resid(u,v,r,n1,n2,n3,a,k)
      call norm2u3(r,n1,n2,n3,rnm2,rnmu,nx(lt),ny(lt),nz(lt))
      old2 = rnm2
      oldu = rnmu

c---------------------------------------------------------------------
c     One iteration for startup
c---------------------------------------------------------------------
      call mg3P(u,v,r,a,c,n1,n2,n3,k)
      call resid(u,v,r,n1,n2,n3,a,k)
      call setup(n1,n2,n3,k)
      call zero3(u,n1,n2,n3)
      call zran3(v,n1,n2,n3,nx(lt),ny(lt),k)

      call shmem_barrier_all()

      call timer_stop(T_init)
      call timer_start(T_bench)

      call resid(u,v,r,n1,n2,n3,a,k)
      call norm2u3(r,n1,n2,n3,rnm2,rnmu,nx(lt),ny(lt),nz(lt))
      old2 = rnm2
      oldu = rnmu

      do  it=1,nit
         call mg3P(u,v,r,a,c,n1,n2,n3,k)
         call resid(u,v,r,n1,n2,n3,a,k)
      enddo


      call norm2u3(r,n1,n2,n3,rnm2,rnmu,nx(lt),ny(lt),nz(lt))

      call timer_stop(T_bench)

      t0 = timer_read(T_bench)
      tinit = timer_read(T_init)

      call shmem_barrier_all()
      call shmem_real8_max_to_all(t,t0,1,0,0,nprocs,pwrk,psync1)
      call shmem_real8_sum_to_all(ttotal,t0,1,0,0,nprocs,pwrk,psync1)

      verified = .FALSE.
      verify_value = 0.0
      if( me .eq. root )then
         write( *,'(/A,F15.3,A/)' ) 
     >        ' Initialization time: ',tinit, ' seconds'
         write(*,100)
 100     format(' Benchmark completed ')

         epsilon = 1.d-8
         if (Class .ne. 'U') then
         if(Class.eq.'S') then
            verify_value = 0.530770700573d-04
         elseif(Class.eq.'W') then
            verify_value = 0.646732937534d-05
         elseif(Class.eq.'A') then
            verify_value = 0.243336530907d-05
         elseif(Class.eq.'B') then
            verify_value = 0.180056440136d-05
         elseif(Class.eq.'C') then
            verify_value = 0.570673228574d-06
         elseif(Class.eq.'D') then
            verify_value = 0.158327506043d-09
         endif

         err = abs( rnm2 - verify_value ) / verify_value
         if( err .le. epsilon ) then
               verified = .TRUE.
               write(*, 200)
               write(*, 201) rnm2
               write(*, 202) err
 200           format(' VERIFICATION SUCCESSFUL ')
 201           format(' L2 Norm is ', E20.12)
 202           format(' Error is   ', E20.12)
            else
               verified = .FALSE.
               write(*, 300) 
               write(*, 301) rnm2
               write(*, 302) verify_value
 300           format(' VERIFICATION FAILED')
 301           format(' L2 Norm is             ', E20.12)
 302           format(' The correct L2 Norm is ', E20.12)
            endif
         else
            verified = .FALSE.
            write (*, 400)
            write (*, 401)
 400        format(' Problem size unknown')
 401        format(' NO VERIFICATION PERFORMED')
         endif

         nn = nx(lt)*ny(lt)*nz(lt)

         if( t .ne. 0. ) then
            mflops = 58.*nit*nn*1.0D-6 /t
         else
            mflops = 0.0
         endif

         call print_results('MG', class, nx(lt), ny(lt), nz(lt), 
     >                      nit, nprocs_compiled, nprocs, t,
     >                      mflops, '          floating point', 
     >                      verified, npbversion, compiletime,
     >                      cs1, cs2, cs3, cs4, cs5, cs6, cs7)
         write (*, 6) ttotal
 6       format(' RUn Time in seconds = ',12x, f12.2)



      endif
 600  format( i4, 2e19.12)

      end

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      subroutine setup(n1,n2,n3,k)

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      implicit none

      include 'mpinpb.h'
      include 'globals.h'

      integer  is1, is2, is3, ie1, ie2, ie3
      common /grid/ is1,is2,is3,ie1,ie2,ie3

      integer n1,n2,n3,k
      integer dx, dy, log_p, d, i, j

      integer ax, next(3),mi(3,10),mip(3,10)
      integer ng(3,10)
      integer idi(3), pi(3), idin(3,-1:1)
      integer s, dir,ierr

      do  j=-1,1,1
         do  d=1,3
            msg_type(d,j) = 100*(j+2+10*d)
         enddo
      enddo

      ng(1,lt) = nx(lt)
      ng(2,lt) = ny(lt)
      ng(3,lt) = nz(lt)
      do  ax=1,3
         next(ax) = 1
         do  k=lt-1,1,-1
            ng(ax,k) = ng(ax,k+1)/2
         enddo
      enddo
 61   format(10i4)
      do  k=lt,1,-1
         nx(k) = ng(1,k)
         ny(k) = ng(2,k)
         nz(k) = ng(3,k)
      enddo

      log_p  = log(float(nprocs)+0.0001)/log(2.0)
      dx     = log_p/3
      pi(1)  = 2**dx
      idi(1) = mod(me,pi(1))

      dy     = (log_p-dx)/2
      pi(2)  = 2**dy
      idi(2) = mod((me/pi(1)),pi(2))

      pi(3)  = nprocs/(pi(1)*pi(2))
      idi(3) = me/(pi(1)*pi(2))

      do  k = lt,1,-1
         dead(k) = .false.
         do  ax = 1,3
            take_ex(ax,k) = .false.
            give_ex(ax,k) = .false.

            mi(ax,k) = 2 + 
     >           ((idi(ax)+1)*ng(ax,k))/pi(ax) -
     >           ((idi(ax)+0)*ng(ax,k))/pi(ax)
            mip(ax,k) = 2 + 
     >           ((next(ax)+idi(ax)+1)*ng(ax,k))/pi(ax) -
     >           ((next(ax)+idi(ax)+0)*ng(ax,k))/pi(ax) 

            if(mip(ax,k).eq.2.or.mi(ax,k).eq.2)then
               next(ax) = 2*next(ax)
            endif

            if( k+1 .le. lt )then
               if((mip(ax,k).eq.2).and.(mi(ax,k).eq.3))then
                  give_ex(ax,k+1) = .true.
               endif
               if((mip(ax,k).eq.3).and.(mi(ax,k).eq.2))then
                  take_ex(ax,k+1) = .true.
               endif
            endif
         enddo

         if( mi(1,k).eq.2 .or. 
     >        mi(2,k).eq.2 .or. 
     >        mi(3,k).eq.2      )then
            dead(k) = .true.
         endif
         m1(k) = mi(1,k)
         m2(k) = mi(2,k)
         m3(k) = mi(3,k)

         do  ax=1,3
            idin(ax,+1) = mod( idi(ax) + next(ax) + pi(ax) , pi(ax) )
            idin(ax,-1) = mod( idi(ax) - next(ax) + pi(ax) , pi(ax) )
         enddo
         do  dir = 1,-1,-2
            nbr(1,dir,k) = idin(1,dir) + pi(1)
     >           *(idi(2)      + pi(2)
     >           * idi(3))
            nbr(2,dir,k) = idi(1)      + pi(1)
     >           *(idin(2,dir) + pi(2)
     >           * idi(3))
            nbr(3,dir,k) = idi(1)      + pi(1)
     >           *(idi(2)      + pi(2)
     >           * idin(3,dir))
         enddo
      enddo

      k = lt
      is1 = 2 + ng(1,k) - ((pi(1)  -idi(1))*ng(1,lt))/pi(1)
      ie1 = 1 + ng(1,k) - ((pi(1)-1-idi(1))*ng(1,lt))/pi(1)
      n1 = 3 + ie1 - is1
      is2 = 2 + ng(2,k) - ((pi(2)  -idi(2))*ng(2,lt))/pi(2)
      ie2 = 1 + ng(2,k) - ((pi(2)-1-idi(2))*ng(2,lt))/pi(2)
      n2 = 3 + ie2 - is2
      is3 = 2 + ng(3,k) - ((pi(3)  -idi(3))*ng(3,lt))/pi(3)
      ie3 = 1 + ng(3,k) - ((pi(3)-1-idi(3))*ng(3,lt))/pi(3)
      n3 = 3 + ie3 - is3


      ir(lt)=1
      do  j = lt-1, 1, -1
         ir(j)=ir(j+1)+m1(j+1)*m2(j+1)*m3(j+1)
      enddo


      if( debug_vec(1) .ge. 1 )then
         if( me .eq. root )write(*,*)' in setup, '
         if( me .eq. root )write(*,*)' me   k  lt  nx  ny  nz ',
     >        ' n1  n2  n3 is1 is2 is3 ie1 ie2 ie3'
         do  i=0,nprocs-1
            if( me .eq. i )then
               write(*,9) me,k,lt,ng(1,k),ng(2,k),ng(3,k),
     >              n1,n2,n3,is1,is2,is3,ie1,ie2,ie3
 9             format(15i4)
            endif
            call shmem_barrier_all()
         enddo
      endif
      if( debug_vec(1) .ge. 2 )then
         do  i=0,nprocs-1
            if( me .eq. i )then
               write(*,*)' '
               write(*,*)' processor =',me
               do  k=lt,1,-1
                  write(*,7)k,idi(1),idi(2),idi(3),
     >                 ((nbr(d,j,k),j=-1,1,2),d=1,3),
     >                 (mi(d,k),d=1,3)
               enddo
 7             format(i4,'idi=',3i4,'nbr=',3(2i4,'  '),'mi=',3i4,' ')
               write(*,*)'idi(s) = ',(idi(s),s=1,3)
               write(*,*)'dead(2), dead(1) = ',dead(2),dead(1)
               do  ax=1,3
                  write(*,*)'give_ex(ax,2)= ',give_ex(ax,2)
                  write(*,*)'take_ex(ax,2)= ',take_ex(ax,2)
               enddo
            endif
            call shmem_barrier_all()
         enddo
      endif

      k = lt

      return
      end

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      subroutine mg3P(u,v,r,a,c,n1,n2,n3,k)

c---------------------------------------------------------------------
c---------------------------------------------------------------------

c---------------------------------------------------------------------
c     multigrid V-cycle routine
c---------------------------------------------------------------------
      implicit none

      include 'mpinpb.h'
      include 'globals.h'

      integer n1, n2, n3, k
      double precision u(nr),v(nv),r(nr)
      double precision a(0:3),c(0:3)

      integer j

c---------------------------------------------------------------------
c     down cycle.
c     restrict the residual from the find grid to the coarse
c---------------------------------------------------------------------

      do  k= lt, lb+1 , -1
         j = k-1
         call rprj3(r(ir(k)),m1(k),m2(k),m3(k),
     >        r(ir(j)),m1(j),m2(j),m3(j),k)
      enddo

      k = lb
c---------------------------------------------------------------------
c     compute an approximate solution on the coarsest grid
c---------------------------------------------------------------------
      call zero3(u(ir(k)),m1(k),m2(k),m3(k))
      call psinv(r(ir(k)),u(ir(k)),m1(k),m2(k),m3(k),c,k)

      do  k = lb+1, lt-1     
          j = k-1
c---------------------------------------------------------------------
c        prolongate from level k-1  to k
c---------------------------------------------------------------------
         call zero3(u(ir(k)),m1(k),m2(k),m3(k))
         call interp(u(ir(j)),m1(j),m2(j),m3(j),
     >               u(ir(k)),m1(k),m2(k),m3(k),k)
c---------------------------------------------------------------------
c        compute residual for level k
c---------------------------------------------------------------------
         call resid(u(ir(k)),r(ir(k)),r(ir(k)),m1(k),m2(k),m3(k),a,k)
c---------------------------------------------------------------------
c        apply smoother
c---------------------------------------------------------------------
         call psinv(r(ir(k)),u(ir(k)),m1(k),m2(k),m3(k),c,k)
      enddo
 200  continue
      j = lt - 1
      k = lt
      call interp(u(ir(j)),m1(j),m2(j),m3(j),u,n1,n2,n3,k)
      call resid(u,v,r,n1,n2,n3,a,k)
      call psinv(r,u,n1,n2,n3,c,k)

      return
      end

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      subroutine psinv( r,u,n1,n2,n3,c,k)

c---------------------------------------------------------------------
c---------------------------------------------------------------------

c---------------------------------------------------------------------
c     psinv applies an approximate inverse as smoother:  u = u + Cr
c
c     This  implementation costs  15A + 4M per result, where
c     A and M denote the costs of Addition and Multiplication.  
c     Presuming coefficient c(3) is zero (the NPB assumes this,
c     but it is thus not a general case), 2A + 1M may be eliminated,
c     resulting in 13A + 3M.
c     Note that this vectorizes, and is also fine for cache 
c     based machines.  
c---------------------------------------------------------------------
      implicit none

      include 'globals.h'

      integer n1,n2,n3,k
      double precision u(n1,n2,n3),r(n1,n2,n3),c(0:3)
      integer i3, i2, i1

      double precision r1(m), r2(m)
      
      do i3=2,n3-1
         do i2=2,n2-1
            do i1=1,n1
               r1(i1) = r(i1,i2-1,i3) + r(i1,i2+1,i3)
     >                + r(i1,i2,i3-1) + r(i1,i2,i3+1)
               r2(i1) = r(i1,i2-1,i3-1) + r(i1,i2+1,i3-1)
     >                + r(i1,i2-1,i3+1) + r(i1,i2+1,i3+1)
            enddo
            do i1=2,n1-1
               u(i1,i2,i3) = u(i1,i2,i3)
     >                     + c(0) * r(i1,i2,i3)
     >                     + c(1) * ( r(i1-1,i2,i3) + r(i1+1,i2,i3)
     >                              + r1(i1) )
     >                     + c(2) * ( r2(i1) + r1(i1-1) + r1(i1+1) )
c---------------------------------------------------------------------
c  Assume c(3) = 0    (Enable line below if c(3) not= 0)
c---------------------------------------------------------------------
c    >                     + c(3) * ( r2(i1-1) + r2(i1+1) )
c---------------------------------------------------------------------
            enddo
         enddo
      enddo

c---------------------------------------------------------------------
c     exchange boundary points
c---------------------------------------------------------------------
      call comm3(u,n1,n2,n3,k)

      if( debug_vec(0) .ge. 1 )then
         call rep_nrm(u,n1,n2,n3,'   psinv',k)
      endif

      if( debug_vec(3) .ge. k )then
         call showall(u,n1,n2,n3)
      endif

      return
      end

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      subroutine resid( u,v,r,n1,n2,n3,a,k )

c---------------------------------------------------------------------
c---------------------------------------------------------------------

c---------------------------------------------------------------------
c     resid computes the residual:  r = v - Au
c
c     This  implementation costs  15A + 4M per result, where
c     A and M denote the costs of Addition (or Subtraction) and 
c     Multiplication, respectively. 
c     Presuming coefficient a(1) is zero (the NPB assumes this,
c     but it is thus not a general case), 3A + 1M may be eliminated,
c     resulting in 12A + 3M.
c     Note that this vectorizes, and is also fine for cache 
c     based machines.  
c---------------------------------------------------------------------
      implicit none

      include 'globals.h'

      integer n1,n2,n3,k
      double precision u(n1,n2,n3),v(n1,n2,n3),r(n1,n2,n3),a(0:3)
      integer i3, i2, i1
      double precision u1(m), u2(m)

      do i3=2,n3-1
         do i2=2,n2-1
            do i1=1,n1
               u1(i1) = u(i1,i2-1,i3) + u(i1,i2+1,i3)
     >                + u(i1,i2,i3-1) + u(i1,i2,i3+1)
               u2(i1) = u(i1,i2-1,i3-1) + u(i1,i2+1,i3-1)
     >                + u(i1,i2-1,i3+1) + u(i1,i2+1,i3+1)
            enddo
            do i1=2,n1-1
               r(i1,i2,i3) = v(i1,i2,i3)
     >                     - a(0) * u(i1,i2,i3)
c---------------------------------------------------------------------
c  Assume a(1) = 0      (Enable 2 lines below if a(1) not= 0)
c---------------------------------------------------------------------
c    >                     - a(1) * ( u(i1-1,i2,i3) + u(i1+1,i2,i3)
c    >                              + u1(i1) )
c---------------------------------------------------------------------
     >                     - a(2) * ( u2(i1) + u1(i1-1) + u1(i1+1) )
     >                     - a(3) * ( u2(i1-1) + u2(i1+1) )
            enddo
         enddo
      enddo

c---------------------------------------------------------------------
c     exchange boundary data
c---------------------------------------------------------------------
      call comm3(r,n1,n2,n3,k)

      if( debug_vec(0) .ge. 1 )then
         call rep_nrm(r,n1,n2,n3,'   resid',k)
      endif

      if( debug_vec(2) .ge. k )then
         call showall(r,n1,n2,n3)
      endif

      return
      end

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      subroutine rprj3( r,m1k,m2k,m3k,s,m1j,m2j,m3j,k )

c---------------------------------------------------------------------
c---------------------------------------------------------------------

c---------------------------------------------------------------------
c     rprj3 projects onto the next coarser grid, 
c     using a trilinear Finite Element projection:  s = r' = P r
c     
c     This  implementation costs  20A + 4M per result, where
c     A and M denote the costs of Addition and Multiplication.  
c     Note that this vectorizes, and is also fine for cache 
c     based machines.  
c---------------------------------------------------------------------
      implicit none

      include 'mpinpb.h'
      include 'globals.h'

      integer m1k, m2k, m3k, m1j, m2j, m3j,k
      double precision r(m1k,m2k,m3k), s(m1j,m2j,m3j)
      integer j3, j2, j1, i3, i2, i1, d1, d2, d3, j

      double precision x1(m), y1(m), x2,y2


      if(m1k.eq.3)then
        d1 = 2
      else
        d1 = 1
      endif

      if(m2k.eq.3)then
        d2 = 2
      else
        d2 = 1
      endif

      if(m3k.eq.3)then
        d3 = 2
      else
        d3 = 1
      endif

      do  j3=2,m3j-1
         i3 = 2*j3-d3
C        i3 = 2*j3-1
         do  j2=2,m2j-1
            i2 = 2*j2-d2
C           i2 = 2*j2-1

            do j1=2,m1j
              i1 = 2*j1-d1
C             i1 = 2*j1-1
              x1(i1-1) = r(i1-1,i2-1,i3  ) + r(i1-1,i2+1,i3  )
     >                 + r(i1-1,i2,  i3-1) + r(i1-1,i2,  i3+1)
              y1(i1-1) = r(i1-1,i2-1,i3-1) + r(i1-1,i2-1,i3+1)
     >                 + r(i1-1,i2+1,i3-1) + r(i1-1,i2+1,i3+1)
            enddo

            do  j1=2,m1j-1
              i1 = 2*j1-d1
C             i1 = 2*j1-1
              y2 = r(i1,  i2-1,i3-1) + r(i1,  i2-1,i3+1)
     >           + r(i1,  i2+1,i3-1) + r(i1,  i2+1,i3+1)
              x2 = r(i1,  i2-1,i3  ) + r(i1,  i2+1,i3  )
     >           + r(i1,  i2,  i3-1) + r(i1,  i2,  i3+1)
              s(j1,j2,j3) =
     >               0.5D0 * r(i1,i2,i3)
     >             + 0.25D0 * ( r(i1-1,i2,i3) + r(i1+1,i2,i3) + x2)
     >             + 0.125D0 * ( x1(i1-1) + x1(i1+1) + y2)
     >             + 0.0625D0 * ( y1(i1-1) + y1(i1+1) )
            enddo

         enddo
      enddo


      j = k-1
      call comm3(s,m1j,m2j,m3j,j)

      if( debug_vec(0) .ge. 1 )then
         call rep_nrm(s,m1j,m2j,m3j,'   rprj3',k-1)
      endif

      if( debug_vec(4) .ge. k )then
         call showall(s,m1j,m2j,m3j)
      endif

      return
      end

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      subroutine interp( z,mm1,mm2,mm3,u,n1,n2,n3,k )

c---------------------------------------------------------------------
c---------------------------------------------------------------------

c---------------------------------------------------------------------
c     interp adds the trilinear interpolation of the correction
c     from the coarser grid to the current approximation:  u = u + Qu'
c     
c     Observe that this  implementation costs  16A + 4M, where
c     A and M denote the costs of Addition and Multiplication.  
c     Note that this vectorizes, and is also fine for cache 
c     based machines.  Vector machines may get slightly better 
c     performance however, with 8 separate "do i1" loops, rather than 4.
c---------------------------------------------------------------------
      implicit none

      include 'mpinpb.h'
      include 'globals.h'

      integer mm1, mm2, mm3, n1, n2, n3,k
      double precision z(mm1,mm2,mm3),u(n1,n2,n3)
      integer i3, i2, i1, d1, d2, d3, t1, t2, t3

c note that m = 1037 in globals.h but for this only need to be
c 535 to handle up to 1024^3
c      integer m
c      parameter( m=535 )
      double precision z1(m),z2(m),z3(m)


      if( n1 .ne. 3 .and. n2 .ne. 3 .and. n3 .ne. 3 ) then

         do  i3=1,mm3-1
            do  i2=1,mm2-1

               do i1=1,mm1
                  z1(i1) = z(i1,i2+1,i3) + z(i1,i2,i3)
                  z2(i1) = z(i1,i2,i3+1) + z(i1,i2,i3)
                  z3(i1) = z(i1,i2+1,i3+1) + z(i1,i2,i3+1) + z1(i1)
               enddo

               do  i1=1,mm1-1
                  u(2*i1-1,2*i2-1,2*i3-1)=u(2*i1-1,2*i2-1,2*i3-1)
     >                 +z(i1,i2,i3)
                  u(2*i1,2*i2-1,2*i3-1)=u(2*i1,2*i2-1,2*i3-1)
     >                 +0.5d0*(z(i1+1,i2,i3)+z(i1,i2,i3))
               enddo
               do i1=1,mm1-1
                  u(2*i1-1,2*i2,2*i3-1)=u(2*i1-1,2*i2,2*i3-1)
     >                 +0.5d0 * z1(i1)
                  u(2*i1,2*i2,2*i3-1)=u(2*i1,2*i2,2*i3-1)
     >                 +0.25d0*( z1(i1) + z1(i1+1) )
               enddo
               do i1=1,mm1-1
                  u(2*i1-1,2*i2-1,2*i3)=u(2*i1-1,2*i2-1,2*i3)
     >                 +0.5d0 * z2(i1)
                  u(2*i1,2*i2-1,2*i3)=u(2*i1,2*i2-1,2*i3)
     >                 +0.25d0*( z2(i1) + z2(i1+1) )
               enddo
               do i1=1,mm1-1
                  u(2*i1-1,2*i2,2*i3)=u(2*i1-1,2*i2,2*i3)
     >                 +0.25d0* z3(i1)
                  u(2*i1,2*i2,2*i3)=u(2*i1,2*i2,2*i3)
     >                 +0.125d0*( z3(i1) + z3(i1+1) )
               enddo
            enddo
         enddo

      else

         if(n1.eq.3)then
            d1 = 2
            t1 = 1
         else
            d1 = 1
            t1 = 0
         endif
         
         if(n2.eq.3)then
            d2 = 2
            t2 = 1
         else
            d2 = 1
            t2 = 0
         endif
         
         if(n3.eq.3)then
            d3 = 2
            t3 = 1
         else
            d3 = 1
            t3 = 0
         endif
         
         do  i3=d3,mm3-1
            do  i2=d2,mm2-1
               do  i1=d1,mm1-1
                  u(2*i1-d1,2*i2-d2,2*i3-d3)=u(2*i1-d1,2*i2-d2,2*i3-d3)
     >                 +z(i1,i2,i3)
               enddo
               do  i1=1,mm1-1
                  u(2*i1-t1,2*i2-d2,2*i3-d3)=u(2*i1-t1,2*i2-d2,2*i3-d3)
     >                 +0.5D0*(z(i1+1,i2,i3)+z(i1,i2,i3))
               enddo
            enddo
            do  i2=1,mm2-1
               do  i1=d1,mm1-1
                  u(2*i1-d1,2*i2-t2,2*i3-d3)=u(2*i1-d1,2*i2-t2,2*i3-d3)
     >                 +0.5D0*(z(i1,i2+1,i3)+z(i1,i2,i3))
               enddo
               do  i1=1,mm1-1
                  u(2*i1-t1,2*i2-t2,2*i3-d3)=u(2*i1-t1,2*i2-t2,2*i3-d3)
     >                 +0.25D0*(z(i1+1,i2+1,i3)+z(i1+1,i2,i3)
     >                 +z(i1,  i2+1,i3)+z(i1,  i2,i3))
               enddo
            enddo
         enddo

         do  i3=1,mm3-1
            do  i2=d2,mm2-1
               do  i1=d1,mm1-1
                  u(2*i1-d1,2*i2-d2,2*i3-t3)=u(2*i1-d1,2*i2-d2,2*i3-t3)
     >                 +0.5D0*(z(i1,i2,i3+1)+z(i1,i2,i3))
               enddo
               do  i1=1,mm1-1
                  u(2*i1-t1,2*i2-d2,2*i3-t3)=u(2*i1-t1,2*i2-d2,2*i3-t3)
     >                 +0.25D0*(z(i1+1,i2,i3+1)+z(i1,i2,i3+1)
     >                 +z(i1+1,i2,i3  )+z(i1,i2,i3  ))
               enddo
            enddo
            do  i2=1,mm2-1
               do  i1=d1,mm1-1
                  u(2*i1-d1,2*i2-t2,2*i3-t3)=u(2*i1-d1,2*i2-t2,2*i3-t3)
     >                 +0.25D0*(z(i1,i2+1,i3+1)+z(i1,i2,i3+1)
     >                 +z(i1,i2+1,i3  )+z(i1,i2,i3  ))
               enddo
               do  i1=1,mm1-1
                  u(2*i1-t1,2*i2-t2,2*i3-t3)=u(2*i1-t1,2*i2-t2,2*i3-t3)
     >                 +0.125D0*(z(i1+1,i2+1,i3+1)+z(i1+1,i2,i3+1)
     >                 +z(i1  ,i2+1,i3+1)+z(i1  ,i2,i3+1)
     >                 +z(i1+1,i2+1,i3  )+z(i1+1,i2,i3  )
     >                 +z(i1  ,i2+1,i3  )+z(i1  ,i2,i3  ))
               enddo
            enddo
         enddo

      endif

      call comm3_ex(u,n1,n2,n3,k)

      if( debug_vec(0) .ge. 1 )then
         call rep_nrm(z,mm1,mm2,mm3,'z: inter',k-1)
         call rep_nrm(u,n1,n2,n3,'u: inter',k)
      endif

      if( debug_vec(5) .ge. k )then
         call showall(z,mm1,mm2,mm3)
         call showall(u,n1,n2,n3)
      endif

      return 
      end

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      subroutine norm2u3(r,n1,n2,n3,rnm2,rnmu,nx,ny,nz)

c---------------------------------------------------------------------
c---------------------------------------------------------------------

c---------------------------------------------------------------------
c     norm2u3 evaluates approximations to the L2 norm and the
c     uniform (or L-infinity or Chebyshev) norm, under the
c     assumption that the boundaries are periodic or zero.  Add the
c     boundaries in with half weight (quarter weight on the edges
c     and eighth weight at the corners) for inhomogeneous boundaries.
c---------------------------------------------------------------------
      implicit none

      include 'mpinpb.h'
      include 'mpp/shmem.fh'

      integer n1, n2, n3, nx, ny, nz
      double precision rnm2, rnmu, r(n1,n2,n3)
      double precision a
      double precision, save::s 
      double precision, save::ss
      integer i3, i2, i1, ierr

      integer n
      integer, dimension(SHMEM_BCAST_SYNC_SIZE), save :: psync1
      double precision, dimension(SHMEM_REDUCE_MIN_WRKDATA_SIZE),
     > save :: pwrk1
      integer, dimension(SHMEM_BCAST_SYNC_SIZE), save :: psync2
      double precision, dimension(SHMEM_REDUCE_MIN_WRKDATA_SIZE),
     > save :: pwrk2

      n = nx*ny*nz

      s=0.0D0
      rnmu = 0.0D0
      do  i3=2,n3-1
         do  i2=2,n2-1
            do  i1=2,n1-1
               s=s+r(i1,i2,i3)**2
               a=abs(r(i1,i2,i3))
               if(a.gt.rnmu)rnmu=a
            enddo
         enddo
      enddo

      psync1 = SHMEM_SYNC_VALUE
      psync2 = SHMEM_SYNC_VALUE
      call shmem_barrier_all();
      
      call shmem_real8_max_to_all(ss,rnmu,1,0,0,nprocs,pwrk1,psync1)
      rnmu = ss
      call shmem_real8_sum_to_all(ss,s,1,0,0,nprocs,pwrk2,psync2)
      s = ss
      rnm2=sqrt( s / float( n ))

      return
      end

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      subroutine rep_nrm(u,n1,n2,n3,title,kk)

c---------------------------------------------------------------------
c---------------------------------------------------------------------

c---------------------------------------------------------------------
c     report on norm
c---------------------------------------------------------------------
      implicit none

      include 'mpinpb.h'
      include 'globals.h'

      integer n1, n2, n3, kk
      double precision u(n1,n2,n3)
      character*8 title

      double precision rnm2, rnmu


      call norm2u3(u,n1,n2,n3,rnm2,rnmu,nx(kk),ny(kk),nz(kk))
      if( me .eq. root )then
         write(*,7)kk,title,rnm2,rnmu
 7       format(' Level',i2,' in ',a8,': norms =',D21.14,D21.14)
      endif

      return
      end

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      subroutine comm3(u,n1,n2,n3,kk)

c---------------------------------------------------------------------
c---------------------------------------------------------------------

c---------------------------------------------------------------------
c     comm3 organizes the communication on all borders 
c---------------------------------------------------------------------
      implicit none

      include 'mpinpb.h'
      include 'globals.h'

      integer n1, n2, n3, kk
      double precision u(n1,n2,n3)
      integer axis

      do  axis = 1, 3
       call shmem_barrier_all()
       if( .not. dead(kk) )then
               call ready( axis, -1 )
               call ready( axis, +1 )
       endif
               call shmem_barrier_all()
       if( .not. dead(kk) )then
               call give3( axis, -1, u, n1, n2, n3, kk )
               call give3( axis, +1, u, n1, n2, n3, kk )
       endif
               call shmem_barrier_all()
       if( .not. dead(kk) )then
               call take3( axis, -1, u, n1, n2, n3 )
               call take3( axis, +1, u, n1, n2, n3 )
       endif
      enddo

      if( dead(kk) )then
         call zero3(u,n1,n2,n3)
      endif
      call shmem_barrier_all()
      return
      end

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      subroutine comm3_ex(u,n1,n2,n3,kk)

c---------------------------------------------------------------------
c---------------------------------------------------------------------

c---------------------------------------------------------------------
c     comm3_ex  communicates to expand the number of processors
c---------------------------------------------------------------------
      implicit none

      include 'mpinpb.h'
      include 'globals.h'
      include 'mpp/shmem.fh'


      integer n1, n2, n3, kk
      double precision u(n1,n2,n3)
      integer axis


      do  axis = 1, 3
           call shmem_barrier_all()
            if( take_ex( axis, kk ) )then
               call ready( axis, -1 )
               call ready( axis, +1 )
            endif
            call shmem_barrier_all()
            if( give_ex( axis, kk ) )then
               call give3_ex( axis, -1, u, n1, n2, n3, kk )
               call give3_ex( axis, +1, u, n1, n2, n3, kk )
            endif
            call shmem_barrier_all()
            if( take_ex( axis, kk ) )then
               call take3_ex( axis, -1, u, n1, n2, n3)
               call take3_ex( axis, +1, u, n1, n2, n3)
            endif
      enddo

      return
      end

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      subroutine ready( axis, dir )

c---------------------------------------------------------------------
c---------------------------------------------------------------------

c---------------------------------------------------------------------
c     ready allocates a buffer to take in a message
c---------------------------------------------------------------------
      implicit none

      include 'mpinpb.h'
      include 'globals.h'

      integer axis, dir
      integer buff_id,buff_len,i,ierr

      buff_id = 3 + dir
      buff_len = nm2

      do  i=1,nm2
         buff(i,buff_id) = 0.0D0
      enddo


c---------------------------------------------------------------------
c     fake message request type
c---------------------------------------------------------------------
      msg_id(axis,dir,1) = msg_type(axis,dir) +1000*me

      return
      end


c---------------------------------------------------------------------
c---------------------------------------------------------------------

      subroutine give3( axis, dir, u, n1, n2, n3, k )

c---------------------------------------------------------------------
c---------------------------------------------------------------------

c---------------------------------------------------------------------
c     give3 sends border data out in the requested direction
c---------------------------------------------------------------------
      implicit none

      include 'mpinpb.h'
      include 'globals.h'

      integer axis, dir, n1, n2, n3, k, ierr
      double precision u( n1, n2, n3 )

      integer i3, i2, i1, buff_len,buff_id,buff_id1

      buff_id = 2 + dir 
      buff_id1 = 3 + dir
      buff_len = 0

      if( axis .eq.  1 )then
         if( dir .eq. -1 )then

            do  i3=2,n3-1
               do  i2=2,n2-1
                  buff_len = buff_len + 1
                  buff(buff_len,buff_id ) = u( 2,  i2,i3)
               enddo
            enddo

           call shmem_double_put(buff(1,buff_id1),buff(1,buff_id),
     >           buff_len, nbr( axis, dir, k ))


         else if( dir .eq. +1 ) then

            do  i3=2,n3-1
               do  i2=2,n2-1
                  buff_len = buff_len + 1
                  buff(buff_len, buff_id ) = u( n1-1, i2,i3)
               enddo
            enddo

            call shmem_double_put(buff(1,buff_id1),buff(1,buff_id),
     >           buff_len, nbr( axis, dir, k ))


         endif
      endif

      if( axis .eq.  2 )then
         if( dir .eq. -1 )then

            do  i3=2,n3-1
               do  i1=1,n1
                  buff_len = buff_len + 1
                  buff(buff_len, buff_id ) = u( i1,  2,i3)
               enddo
            enddo

           call shmem_double_put(buff(1,buff_id1),buff(1,buff_id),
     >           buff_len, nbr( axis, dir, k ))


         else if( dir .eq. +1 ) then

            do  i3=2,n3-1
               do  i1=1,n1
                  buff_len = buff_len + 1
                  buff(buff_len,  buff_id )= u( i1,n2-1,i3)
               enddo
            enddo

            call shmem_double_put(buff(1,buff_id1),buff(1,buff_id),
     >           buff_len, nbr( axis, dir, k ))

         endif
      endif

      if( axis .eq.  3 )then
         if( dir .eq. -1 )then

            do  i2=1,n2
               do  i1=1,n1
                  buff_len = buff_len + 1
                  buff(buff_len, buff_id ) = u( i1,i2,2)
               enddo
            enddo

            call shmem_double_put(buff(1,buff_id1),buff(1,buff_id),
     >           buff_len, nbr( axis, dir, k ))


         else if( dir .eq. +1 ) then

            do  i2=1,n2
               do  i1=1,n1
                  buff_len = buff_len + 1
                  buff(buff_len, buff_id ) = u( i1,i2,n3-1)
               enddo
            enddo

            call shmem_double_put(buff(1,buff_id1),buff(1,buff_id),
     >           buff_len, nbr( axis, dir, k ))


         endif
      endif

      return
      end

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      subroutine take3( axis, dir, u, n1, n2, n3 )

c---------------------------------------------------------------------
c---------------------------------------------------------------------

c---------------------------------------------------------------------
c     take3 copies in border data from the requested direction
c---------------------------------------------------------------------
      implicit none

      include 'mpinpb.h'
      include 'globals.h'
      include 'mpp/shmem.fh'

      integer axis, dir, n1, n2, n3
      double precision u( n1, n2, n3 )

      integer buff_id, indx

      integer status(mpi_status_size), ierr

      integer i3, i2, i1,temp


      buff_id = 3 + dir
      indx = 0

      if( axis .eq.  1 )then
         if( dir .eq. -1 )then

            do  i3=2,n3-1
               do  i2=2,n2-1
                  indx = indx + 1
                  u(n1,i2,i3) = buff(indx, buff_id )
               enddo
            enddo

         else if( dir .eq. +1 ) then

            do  i3=2,n3-1
               do  i2=2,n2-1
                  indx = indx + 1
                  u(1,i2,i3) = buff(indx, buff_id )
               enddo
            enddo

         endif
      endif

      if( axis .eq.  2 )then
         if( dir .eq. -1 )then

            do  i3=2,n3-1
               do  i1=1,n1
                  indx = indx + 1
                  u(i1,n2,i3) = buff(indx, buff_id )
               enddo
            enddo

         else if( dir .eq. +1 ) then

            do  i3=2,n3-1
               do  i1=1,n1
                  indx = indx + 1
                  u(i1,1,i3) = buff(indx, buff_id )
               enddo
            enddo

         endif
      endif

      if( axis .eq.  3 )then
         if( dir .eq. -1 )then

            do  i2=1,n2
               do  i1=1,n1
                  indx = indx + 1
                  u(i1,i2,n3) = buff(indx, buff_id )
               enddo
            enddo

         else if( dir .eq. +1 ) then

            do  i2=1,n2
               do  i1=1,n1
                  indx = indx + 1
                  u(i1,i2,1) = buff(indx, buff_id )
               enddo
            enddo

         endif
      endif

      return
      end

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      subroutine give3_ex( axis, dir, u, n1, n2, n3, k )

c---------------------------------------------------------------------
c---------------------------------------------------------------------

c---------------------------------------------------------------------
c     give3_ex sends border data out to expand number of processors
c---------------------------------------------------------------------
      implicit none

      include 'mpinpb.h'
      include 'globals.h'

      integer axis, dir, n1, n2, n3, k, ierr
      double precision u( n1, n2, n3 )

      integer i3, i2, i1, buff_len, buff_id,buff_id1

      buff_id = 2 + dir 
      buff_id1 = 3 + dir
      buff_len = 0

      if( axis .eq.  1 )then
         if( dir .eq. -1 )then

            do  i3=1,n3
               do  i2=1,n2
                  buff_len = buff_len + 1
                  buff(buff_len,buff_id ) = u( 2,  i2,i3)
               enddo
            enddo

            call shmem_double_put(buff(1,buff_id1),buff(1,buff_id),
     >           buff_len, nbr( axis, dir, k ))


         else if( dir .eq. +1 ) then

            do  i3=1,n3
               do  i2=1,n2
                  do  i1=n1-1,n1
                     buff_len = buff_len + 1
                     buff(buff_len,buff_id)= u(i1,i2,i3)
                  enddo
               enddo
            enddo

            call shmem_double_put(buff(1,buff_id1),buff(1,buff_id),
     >           buff_len, nbr( axis, dir, k ))


         endif
      endif

      if( axis .eq.  2 )then
         if( dir .eq. -1 )then

            do  i3=1,n3
               do  i1=1,n1
                  buff_len = buff_len + 1
                  buff(buff_len, buff_id ) = u( i1,  2,i3)
               enddo
            enddo

            call shmem_double_put(buff(1,buff_id1),buff(1,buff_id),
     >           buff_len, nbr( axis, dir, k ))

         else if( dir .eq. +1 ) then

            do  i3=1,n3
               do  i2=n2-1,n2
                  do  i1=1,n1
                     buff_len = buff_len + 1
                     buff(buff_len,buff_id )= u(i1,i2,i3)
                  enddo
               enddo
            enddo

            call shmem_double_put(buff(1,buff_id1),buff(1,buff_id),
     >           buff_len, nbr( axis, dir, k ))


         endif
      endif

      if( axis .eq.  3 )then
         if( dir .eq. -1 )then

            do  i2=1,n2
               do  i1=1,n1
                  buff_len = buff_len + 1
                  buff(buff_len, buff_id ) = u( i1,i2,2)
               enddo
            enddo

            call shmem_double_put(buff(1,buff_id1),buff(1,buff_id),
     >           buff_len, nbr( axis, dir, k ))


         else if( dir .eq. +1 ) then

            do  i3=n3-1,n3
               do  i2=1,n2
                  do  i1=1,n1
                     buff_len = buff_len + 1
                     buff(buff_len, buff_id ) = u( i1,i2,i3)
                  enddo
               enddo
            enddo

            call shmem_double_put(buff(1,buff_id1),buff(1,buff_id),
     >           buff_len, nbr( axis, dir, k ))


         endif
      endif

      return
      end

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      subroutine take3_ex( axis, dir, u, n1, n2, n3)

c---------------------------------------------------------------------
c---------------------------------------------------------------------

c---------------------------------------------------------------------
c     take3_ex copies in border data to expand number of processors
c---------------------------------------------------------------------
      implicit none

      include 'mpinpb.h'
      include 'globals.h'
      include 'mpp/shmem.fh'

      integer axis, dir, n1, n2, n3
      double precision u( n1, n2, n3 )

      integer buff_id, indx

      integer status(mpi_status_size) , ierr

      integer i3, i2, i1


      buff_id = 3 + dir
      indx = 0

      if( axis .eq.  1 )then
         if( dir .eq. -1 )then

            do  i3=1,n3
               do  i2=1,n2
                  indx = indx + 1
                  u(n1,i2,i3) = buff(indx, buff_id )
               enddo
            enddo

         else if( dir .eq. +1 ) then

            do  i3=1,n3
               do  i2=1,n2
                  do  i1=1,2
                     indx = indx + 1
                     u(i1,i2,i3) = buff(indx,buff_id)
                  enddo
               enddo
            enddo

         endif
      endif

      if( axis .eq.  2 )then
         if( dir .eq. -1 )then

            do  i3=1,n3
               do  i1=1,n1
                  indx = indx + 1
                  u(i1,n2,i3) = buff(indx, buff_id )
               enddo
            enddo

         else if( dir .eq. +1 ) then

            do  i3=1,n3
               do  i2=1,2
                  do  i1=1,n1
                     indx = indx + 1
                     u(i1,i2,i3) = buff(indx,buff_id)
                  enddo
               enddo
            enddo

         endif
      endif

      if( axis .eq.  3 )then
         if( dir .eq. -1 )then

            do  i2=1,n2
               do  i1=1,n1
                  indx = indx + 1
                  u(i1,i2,n3) = buff(indx, buff_id )
               enddo
            enddo

         else if( dir .eq. +1 ) then

            do  i3=1,2
               do  i2=1,n2
                  do  i1=1,n1
                     indx = indx + 1
                     u(i1,i2,i3) = buff(indx,buff_id)
                  enddo
               enddo
            enddo

         endif
      endif

      return
      end

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      subroutine comm1p( axis, u, n1, n2, n3, kk )

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      implicit none

      include 'mpinpb.h'
      include 'globals.h'

      integer axis, dir, n1, n2, n3
      double precision u( n1, n2, n3 )

      integer i3, i2, i1, buff_len,buff_id
      integer i, kk, indx

      dir = -1

      buff_id = 3 + dir
      buff_len = nm2

      do  i=1,nm2
         buff(i,buff_id) = 0.0D0
      enddo


      dir = +1

      buff_id = 3 + dir
      buff_len = nm2

      do  i=1,nm2
         buff(i,buff_id) = 0.0D0
      enddo

      dir = +1

      buff_id = 2 + dir 
      buff_len = 0

      if( axis .eq.  1 )then
         do  i3=2,n3-1
            do  i2=2,n2-1
               buff_len = buff_len + 1
               buff(buff_len, buff_id ) = u( n1-1, i2,i3)
            enddo
         enddo
      endif

      if( axis .eq.  2 )then
         do  i3=2,n3-1
            do  i1=1,n1
               buff_len = buff_len + 1
               buff(buff_len,  buff_id )= u( i1,n2-1,i3)
            enddo
         enddo
      endif

      if( axis .eq.  3 )then
         do  i2=1,n2
            do  i1=1,n1
               buff_len = buff_len + 1
               buff(buff_len, buff_id ) = u( i1,i2,n3-1)
            enddo
         enddo
      endif

      dir = -1

      buff_id = 2 + dir 
      buff_len = 0

      if( axis .eq.  1 )then
         do  i3=2,n3-1
            do  i2=2,n2-1
               buff_len = buff_len + 1
               buff(buff_len,buff_id ) = u( 2,  i2,i3)
            enddo
         enddo
      endif

      if( axis .eq.  2 )then
         do  i3=2,n3-1
            do  i1=1,n1
               buff_len = buff_len + 1
               buff(buff_len, buff_id ) = u( i1,  2,i3)
            enddo
         enddo
      endif

      if( axis .eq.  3 )then
         do  i2=1,n2
            do  i1=1,n1
               buff_len = buff_len + 1
               buff(buff_len, buff_id ) = u( i1,i2,2)
            enddo
         enddo
      endif

      do  i=1,nm2
         buff(i,4) = buff(i,3)
         buff(i,2) = buff(i,1)
      enddo

      dir = -1

      buff_id = 3 + dir
      indx = 0

      if( axis .eq.  1 )then
         do  i3=2,n3-1
            do  i2=2,n2-1
               indx = indx + 1
               u(n1,i2,i3) = buff(indx, buff_id )
            enddo
         enddo
      endif

      if( axis .eq.  2 )then
         do  i3=2,n3-1
            do  i1=1,n1
               indx = indx + 1
               u(i1,n2,i3) = buff(indx, buff_id )
            enddo
         enddo
      endif

      if( axis .eq.  3 )then
         do  i2=1,n2
            do  i1=1,n1
               indx = indx + 1
               u(i1,i2,n3) = buff(indx, buff_id )
            enddo
         enddo
      endif


      dir = +1

      buff_id = 3 + dir
      indx = 0

      if( axis .eq.  1 )then
         do  i3=2,n3-1
            do  i2=2,n2-1
               indx = indx + 1
               u(1,i2,i3) = buff(indx, buff_id )
            enddo
         enddo
      endif

      if( axis .eq.  2 )then
         do  i3=2,n3-1
            do  i1=1,n1
               indx = indx + 1
               u(i1,1,i3) = buff(indx, buff_id )
            enddo
         enddo
      endif

      if( axis .eq.  3 )then
         do  i2=1,n2
            do  i1=1,n1
               indx = indx + 1
               u(i1,i2,1) = buff(indx, buff_id )
            enddo
         enddo
      endif

      return
      end

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      subroutine comm1p_ex( axis, u, n1, n2, n3, kk )

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      implicit none

      include 'mpinpb.h'
      include 'globals.h'

      integer axis, dir, n1, n2, n3
      double precision u( n1, n2, n3 )

      integer i3, i2, i1, buff_len,buff_id
      integer i, kk, indx

      if( take_ex( axis, kk ) ) then

         dir = -1

         buff_id = 3 + dir
         buff_len = nm2

         do  i=1,nm2
            buff(i,buff_id) = 0.0D0
         enddo


         dir = +1

         buff_id = 3 + dir
         buff_len = nm2

         do  i=1,nm2
            buff(i,buff_id) = 0.0D0
         enddo


         dir = -1

         buff_id = 3 + dir
         indx = 0

         if( axis .eq.  1 )then
            do  i3=1,n3
               do  i2=1,n2
                  indx = indx + 1
                  u(n1,i2,i3) = buff(indx, buff_id )
               enddo
            enddo
         endif

         if( axis .eq.  2 )then
            do  i3=1,n3
               do  i1=1,n1
                  indx = indx + 1
                  u(i1,n2,i3) = buff(indx, buff_id )
               enddo
            enddo
         endif

         if( axis .eq.  3 )then
            do  i2=1,n2
               do  i1=1,n1
                  indx = indx + 1
                  u(i1,i2,n3) = buff(indx, buff_id )
               enddo
            enddo
         endif

         dir = +1

         buff_id = 3 + dir
         indx = 0

         if( axis .eq.  1 )then
            do  i3=1,n3
               do  i2=1,n2
                  do  i1=1,2
                     indx = indx + 1
                     u(i1,i2,i3) = buff(indx,buff_id)
                  enddo
               enddo
            enddo
         endif

         if( axis .eq.  2 )then
            do  i3=1,n3
               do  i2=1,2
                  do  i1=1,n1
                     indx = indx + 1
                     u(i1,i2,i3) = buff(indx,buff_id)
                  enddo
               enddo
            enddo
         endif

         if( axis .eq.  3 )then
            do  i3=1,2
               do  i2=1,n2
                  do  i1=1,n1
                     indx = indx + 1
                     u(i1,i2,i3) = buff(indx,buff_id)
                  enddo
               enddo
            enddo
         endif

      endif

      if( give_ex( axis, kk ) )then

         dir = +1

         buff_id = 2 + dir 
         buff_len = 0

         if( axis .eq.  1 )then
            do  i3=1,n3
               do  i2=1,n2
                  do  i1=n1-1,n1
                     buff_len = buff_len + 1
                     buff(buff_len,buff_id)= u(i1,i2,i3)
                  enddo
               enddo
            enddo
         endif

         if( axis .eq.  2 )then
            do  i3=1,n3
               do  i2=n2-1,n2
                  do  i1=1,n1
                     buff_len = buff_len + 1
                     buff(buff_len,buff_id )= u(i1,i2,i3)
                  enddo
               enddo
            enddo
         endif

         if( axis .eq.  3 )then
            do  i3=n3-1,n3
               do  i2=1,n2
                  do  i1=1,n1
                     buff_len = buff_len + 1
                     buff(buff_len, buff_id ) = u( i1,i2,i3)
                  enddo
               enddo
            enddo
         endif

         dir = -1

         buff_id = 2 + dir 
         buff_len = 0

         if( axis .eq.  1 )then
            do  i3=1,n3
               do  i2=1,n2
                  buff_len = buff_len + 1
                  buff(buff_len,buff_id ) = u( 2,  i2,i3)
               enddo
            enddo
         endif

         if( axis .eq.  2 )then
            do  i3=1,n3
               do  i1=1,n1
                  buff_len = buff_len + 1
                  buff(buff_len, buff_id ) = u( i1,  2,i3)
               enddo
            enddo
         endif

         if( axis .eq.  3 )then
            do  i2=1,n2
               do  i1=1,n1
                  buff_len = buff_len + 1
                  buff(buff_len, buff_id ) = u( i1,i2,2)
               enddo
            enddo
         endif

      endif

      do  i=1,nm2
         buff(i,4) = buff(i,3)
         buff(i,2) = buff(i,1)
      enddo

      return
      end

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      subroutine zran3(z,n1,n2,n3,nx,ny,k)

c---------------------------------------------------------------------
c---------------------------------------------------------------------

c---------------------------------------------------------------------
c     zran3  loads +1 at ten randomly chosen points,
c     loads -1 at a different ten random points,
c     and zero elsewhere.
c---------------------------------------------------------------------
      implicit none

      include 'mpinpb.h'
      include 'mpp/shmem.fh'

      integer  is1, is2, is3, ie1, ie2, ie3
      common /grid/ is1,is2,is3,ie1,ie2,ie3

      integer n1, n2, n3, k, nx, ny, ierr, i0, m0, m1
      double precision z(n1,n2,n3)

      integer mm, i1, i2, i3, d1, e1, e2, e3
      double precision x, a
      double precision xx, x0, x1, a1, a2, ai, power
      parameter( mm = 10,  a = 5.D0 ** 13, x = 314159265.D0)
      double precision ten( mm, 0:1 )
      double precision, save :: temp
      double precision, save :: best 
      integer i, j1( mm, 0:1 ), j2( mm, 0:1 ), j3( mm, 0:1 )
      integer, dimension( 0:3, mm, 0:1 ), save :: jg
      integer, save :: jg_temp(4)

      external randlc
      double precision randlc, rdummy

      integer, dimension(SHMEM_BCAST_SYNC_SIZE), save :: psync1
      integer, dimension(SHMEM_BCAST_SYNC_SIZE), save :: psync2
      double precision, dimension(SHMEM_REDUCE_MIN_WRKDATA_SIZE),
     > save :: pwrk1
      integer, dimension(SHMEM_REDUCE_MIN_WRKDATA_SIZE),
     > save :: pwrk2

      a1 = power( a, nx )
      a2 = power( a, nx*ny )

      call zero3(z,n1,n2,n3)

      i = is1-2+nx*(is2-2+ny*(is3-2))

      ai = power( a, i )
      d1 = ie1 - is1 + 1
      e1 = ie1 - is1 + 2
      e2 = ie2 - is2 + 2
      e3 = ie3 - is3 + 2
      x0 = x
      rdummy = randlc( x0, ai )
      do  i3 = 2, e3
         x1 = x0
         do  i2 = 2, e2
            xx = x1
            call vranlc( d1, xx, a, z( 2, i2, i3 ))
            rdummy = randlc( x1, a1 )
         enddo
         rdummy = randlc( x0, a2 )
      enddo

c---------------------------------------------------------------------
c       call comm3(z,n1,n2,n3)
c       call showall(z,n1,n2,n3)
c---------------------------------------------------------------------

c---------------------------------------------------------------------
c     each processor looks for twenty candidates
c---------------------------------------------------------------------
      do  i=1,mm
         ten( i, 1 ) = 0.0D0
         j1( i, 1 ) = 0
         j2( i, 1 ) = 0
         j3( i, 1 ) = 0
         ten( i, 0 ) = 1.0D0
         j1( i, 0 ) = 0
         j2( i, 0 ) = 0
         j3( i, 0 ) = 0
      enddo

      do  i3=2,n3-1
         do  i2=2,n2-1
            do  i1=2,n1-1
               if( z(i1,i2,i3) .gt. ten( 1, 1 ) )then
                  ten(1,1) = z(i1,i2,i3) 
                  j1(1,1) = i1
                  j2(1,1) = i2
                  j3(1,1) = i3
                  call bubble( ten, j1, j2, j3, mm, 1 )
               endif
               if( z(i1,i2,i3) .lt. ten( 1, 0 ) )then
                  ten(1,0) = z(i1,i2,i3) 
                  j1(1,0) = i1
                  j2(1,0) = i2
                  j3(1,0) = i3
                  call bubble( ten, j1, j2, j3, mm, 0 )
               endif
            enddo
         enddo
      enddo

      psync1 = SHMEM_SYNC_VALUE
      psync2 = SHMEM_SYNC_VALUE
      call shmem_barrier_all()

c---------------------------------------------------------------------
c     Now which of these are globally best?
c---------------------------------------------------------------------
      i1 = mm
      i0 = mm
      do  i=mm,1,-1

         best = z( j1(i1,1), j2(i1,1), j3(i1,1) )
         call shmem_real8_max_to_all(temp,best,1,0,0,nprocs,pwrk1,
     >                               psync1)
         best = temp
         if(best.eq.z(j1(i1,1),j2(i1,1),j3(i1,1)))then
            jg( 0, i, 1) = me
            jg( 1, i, 1) = is1 - 2 + j1( i1, 1 ) 
            jg( 2, i, 1) = is2 - 2 + j2( i1, 1 ) 
            jg( 3, i, 1) = is3 - 2 + j3( i1, 1 ) 
            i1 = i1-1
         else
            jg( 0, i, 1) = 0
            jg( 1, i, 1) = 0
            jg( 2, i, 1) = 0
            jg( 3, i, 1) = 0
         endif
         ten( i, 1 ) = best
         call shmem_int4_max_to_all(jg_temp,jg(0,i,1),4,0,0,nprocs,
     >                            pwrk2,psync2)
         jg( 0, i, 1) =  jg_temp(1)
         jg( 1, i, 1) =  jg_temp(2)
         jg( 2, i, 1) =  jg_temp(3)
         jg( 3, i, 1) =  jg_temp(4)

         best = z( j1(i0,0), j2(i0,0), j3(i0,0) )
         call shmem_real8_min_to_all(temp,best,1,0,0,nprocs,pwrk1,
     >                                psync1)
         best = temp
         if(best.eq.z(j1(i0,0),j2(i0,0),j3(i0,0)))then
            jg( 0, i, 0) = me
            jg( 1, i, 0) = is1 - 2 + j1( i0, 0 ) 
            jg( 2, i, 0) = is2 - 2 + j2( i0, 0 ) 
            jg( 3, i, 0) = is3 - 2 + j3( i0, 0 ) 
            i0 = i0-1
         else
            jg( 0, i, 0) = 0
            jg( 1, i, 0) = 0
            jg( 2, i, 0) = 0
            jg( 3, i, 0) = 0
         endif
         ten( i, 0 ) = best
         call shmem_int4_max_to_all(jg_temp,jg(0,i,0),4,0,0,nprocs,
     >                            pwrk2,psync2)
         jg( 0, i, 0) =  jg_temp(1)
         jg( 1, i, 0) =  jg_temp(2)
         jg( 2, i, 0) =  jg_temp(3)
         jg( 3, i, 0) =  jg_temp(4)

      enddo
      m1 = i1+1
      m0 = i0+1

c      if( me .eq. root) then
c         write(*,*)' '
c         write(*,*)' negative charges at'
c         write(*,9)(jg(1,i,0),jg(2,i,0),jg(3,i,0),i=1,mm)
c         write(*,*)' positive charges at'
c         write(*,9)(jg(1,i,1),jg(2,i,1),jg(3,i,1),i=1,mm)
c         write(*,*)' small random numbers were'
c         write(*,8)(ten( i,0),i=mm,1,-1)
c         write(*,*)' and they were found on processor number'
c         write(*,7)(jg(0,i,0),i=mm,1,-1)
c         write(*,*)' large random numbers were'
c         write(*,8)(ten( i,1),i=mm,1,-1)
c         write(*,*)' and they were found on processor number'
c         write(*,7)(jg(0,i,1),i=mm,1,-1)
c      endif
c 9    format(5(' (',i3,2(',',i3),')'))
c 8    format(5D15.8)
c 7    format(10i4)
      call shmem_barrier_all()   
      do  i3=1,n3
         do  i2=1,n2
            do  i1=1,n1
               z(i1,i2,i3) = 0.0D0
            enddo
         enddo
      enddo
      do  i=mm,m0,-1
         z( j1(i,0), j2(i,0), j3(i,0) ) = -1.0D0
      enddo
      do  i=mm,m1,-1
         z( j1(i,1), j2(i,1), j3(i,1) ) = +1.0D0
      enddo
      call comm3(z,n1,n2,n3,k)

c---------------------------------------------------------------------
c          call showall(z,n1,n2,n3)
c---------------------------------------------------------------------

      return 
      end

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      subroutine show_l(z,n1,n2,n3)

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      implicit none

      include 'mpinpb.h'

      integer n1,n2,n3,i1,i2,i3,ierr
      double precision z(n1,n2,n3)
      integer m1, m2, m3,i

      m1 = min(n1,18)
      m2 = min(n2,14)
      m3 = min(n3,18)

      write(*,*)'  '
      do  i=0,nprocs-1
         if( me .eq. i )then
            write(*,*)' id = ', me
            do  i3=1,m3
               do  i1=1,m1
                  write(*,6)(z(i1,i2,i3),i2=1,m2)
               enddo
               write(*,*)' - - - - - - - '
            enddo
            write(*,*)'  '
 6          format(6f15.11)
         endif
         call shmem_barrier_all()
      enddo

      return 
      end

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      subroutine showall(z,n1,n2,n3)

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      implicit none

      include 'mpinpb.h'

      integer n1,n2,n3,i1,i2,i3,i,ierr
      double precision z(n1,n2,n3)
      integer m1, m2, m3

      m1 = min(n1,18)
      m2 = min(n2,14)
      m3 = min(n3,18)

      write(*,*)'  '
      do  i=0,nprocs-1
         if( me .eq. i )then
            write(*,*)' id = ', me
            do  i3=1,m3
               do  i1=1,m1
                  write(*,6)(z(i1,i2,i3),i2=1,m2)
               enddo
               write(*,*)' - - - - - - - '
            enddo
            write(*,*)'  '
 6          format(15f6.3)
         endif
         call shmem_barrier_all()
      enddo

      return 
      end

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      subroutine show(z,n1,n2,n3)

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      implicit none

      include 'mpinpb.h'
      integer n1,n2,n3,i1,i2,i3,ierr,i
      double precision z(n1,n2,n3)

      write(*,*)'  '
      do  i=0,nprocs-1
         if( me .eq. i )then
            write(*,*)' id = ', me
            do  i3=2,n3-1
               do  i1=2,n1-1
                  write(*,6)(z(i1,i2,i3),i2=2,n1-1)
               enddo
               write(*,*)' - - - - - - - '
            enddo
            write(*,*)'  '
 6          format(8D10.3)
         endif
         call shmem_barrier_all()
      enddo

c     call comm3(z,n1,n2,n3)

      return 
      end

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      double precision function power( a, n )

c---------------------------------------------------------------------
c---------------------------------------------------------------------

c---------------------------------------------------------------------
c     power  raises an integer, disguised as a double
c     precision real, to an integer power
c---------------------------------------------------------------------
      implicit none

      double precision a, aj
      integer n, nj
      external randlc
      double precision randlc, rdummy

      power = 1.0D0
      nj = n
      aj = a
 100  continue

      if( nj .eq. 0 ) goto 200
      if( mod(nj,2) .eq. 1 ) rdummy =  randlc( power, aj )
      rdummy = randlc( aj, aj )
      nj = nj/2
      go to 100

 200  continue
      return
      end

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      subroutine bubble( ten, j1, j2, j3, m, ind )

c---------------------------------------------------------------------
c---------------------------------------------------------------------

c---------------------------------------------------------------------
c     bubble        does a bubble sort in direction dir
c---------------------------------------------------------------------
      implicit none

      include 'mpinpb.h'

      integer m, ind, j1( m, 0:1 ), j2( m, 0:1 ), j3( m, 0:1 )
      double precision ten( m, 0:1 )
      double precision temp
      integer i, j_temp

      if( ind .eq. 1 )then

         do  i=1,m-1
            if( ten(i,ind) .gt. ten(i+1,ind) )then

               temp = ten( i+1, ind )
               ten( i+1, ind ) = ten( i, ind )
               ten( i, ind ) = temp

               j_temp           = j1( i+1, ind )
               j1( i+1, ind ) = j1( i,   ind )
               j1( i,   ind ) = j_temp

               j_temp           = j2( i+1, ind )
               j2( i+1, ind ) = j2( i,   ind )
               j2( i,   ind ) = j_temp

               j_temp           = j3( i+1, ind )
               j3( i+1, ind ) = j3( i,   ind )
               j3( i,   ind ) = j_temp

            else 
               return
            endif
         enddo

      else

         do  i=1,m-1
            if( ten(i,ind) .lt. ten(i+1,ind) )then

               temp = ten( i+1, ind )
               ten( i+1, ind ) = ten( i, ind )
               ten( i, ind ) = temp

               j_temp           = j1( i+1, ind )
               j1( i+1, ind ) = j1( i,   ind )
               j1( i,   ind ) = j_temp

               j_temp           = j2( i+1, ind )
               j2( i+1, ind ) = j2( i,   ind )
               j2( i,   ind ) = j_temp

               j_temp           = j3( i+1, ind )
               j3( i+1, ind ) = j3( i,   ind )
               j3( i,   ind ) = j_temp

            else 
               return
            endif
         enddo

      endif

      return
      end

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      subroutine zero3(z,n1,n2,n3)

c---------------------------------------------------------------------
c---------------------------------------------------------------------

      implicit none

      include 'mpinpb.h'

      integer n1, n2, n3
      double precision z(n1,n2,n3)
      integer i1, i2, i3

      do  i3=1,n3
         do  i2=1,n2
            do  i1=1,n1
               z(i1,i2,i3)=0.0D0
            enddo
         enddo
      enddo

      return
      end


c----- end of program ------------------------------------------------
