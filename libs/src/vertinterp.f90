MODULE mo_interp
  implicit none
  private
  public :: interp, interp_fld

contains

  SUBROUTINE interp(psrc, ptgt, tgtlevs, weights, &
                  & ncom, nsrc, ntgt, nlevsrc, nlevtgt, chunk_size_in)
    use, intrinsic :: ieee_arithmetic, only : ieee_value, ieee_quiet_nan
    use, intrinsic :: iso_fortran_env, dp => real64, intk => int32
    use omp_lib

    integer(intk), intent(in), value  :: ncom, nsrc, ntgt
    integer(intk), intent(in), value  :: nlevsrc
    integer(intk), intent(in), value  :: nlevtgt

    real(dp), intent(in), dimension(nlevsrc,nsrc,ncom) :: psrc
    real(dp), intent(in), dimension(nlevtgt,ntgt,ncom) :: ptgt

    integer(intk), intent(out) :: tgtlevs(nlevtgt,ntgt,nsrc,ncom)
    real(dp),     intent(out) :: weights(nlevtgt,ntgt,nsrc,ncom)

    integer(intk), intent(in), value, optional :: chunk_size_in
    integer(intk) :: max_chunk_size = 1000
    integer(intk) :: actual_chunk_size

    integer :: icom,isrc,itgt,ksrc,ktgt
    integer :: ksrcstart

    integer(intk) :: num_threads, thread_id
    integer(intk) :: ndimomp, id_dimomp, n_chunks, nc, c_start, c_end
    integer(intk) :: id_st1, id_st2, id_st3
    integer(intk) :: id_en1, id_en2, id_en3

    real(dp) :: timer

    if (present(chunk_size_in)) then
        max_chunk_size = chunk_size_in
    endif


    timer = omp_get_wtime()

    !$OMP PARALLEL PRIVATE(thread_id, icom, isrc, itgt, id_st1, id_en1, id_st2, &
    !$OMP id_en2, id_st3, id_en3, nc, c_start, c_end, ksrcstart)

    thread_id = int(omp_get_thread_num(), intk)

    ! Master thread decides
    if (thread_id == 0) then
        num_threads = int(omp_get_num_threads(), intk)

        ! Decide which dimension to parallelize on
        if (ntgt > num_threads) then
            ndimomp = ntgt
            id_dimomp = 3
        else if (nsrc > num_threads) then
            ndimomp = nsrc
            id_dimomp = 2
        else if (ncom > num_threads) then
            ndimomp = ncom
            id_dimomp = 1
        else
            ndimomp = 1
            id_dimomp = 0
        endif

        ! Compute chunk size
        actual_chunk_size = min(ceiling(ndimomp * 1.0/num_threads), max_chunk_size)
        n_chunks = ceiling(ndimomp * 1.0/actual_chunk_size)
    end if

    !$OMP BARRIER

    ! Initialize for each potentially parallelizable dimension
    id_st1=1
    id_st2=1
    id_st3=1
    id_en1=ncom
    id_en2=nsrc
    id_en3=ntgt

    do nc=1,n_chunks
      if ((mod(nc,num_threads)) == thread_id) then
        c_start=(nc-1)*actual_chunk_size+1
        c_end = min(ndimomp,nc*actual_chunk_size)

        if (id_dimomp == 1) then
            id_st1 = c_start
            id_en1 = c_end
        else if (id_dimomp == 2) then
            id_st2 = c_start
            id_en2 = c_end
        else if (id_dimomp == 3) then
            id_st3 = c_start
            id_en3 = c_end
        else

        endif

        !write(*,*) "Thread ",thread_id," doing chunk ",nc,"/",n_chunks," on dimension",id_dimomp," from ",c_start," to ",c_end
        !write(*,*) "Thread ",thread_id," has dimensions 1(",id_st1,",",id_en1, &
        !&") - 2(",id_st2,",",id_en2,") - 3(",id_st3,",",id_en3,")"

        do icom=id_st1,id_en1
          do isrc=id_st2,id_en2
            associate(this_src => psrc(:,isrc,icom))
            do itgt=id_st3,id_en3

              associate(this_tgt => ptgt(:,itgt,icom), &
                      & this_weights => weights(:,itgt,isrc,icom), &
                      & this_tgtlevs => tgtlevs(:,itgt,isrc,icom))

                ksrcstart=2
tgtlevloop: do ktgt=1,nlevtgt

                  ! No interpolation cases
              if (this_tgt(ktgt) <= this_src(1)) then
                this_tgtlevs(ktgt) = 1
                this_weights(ktgt) = 0

                cycle
              end if

              if (this_tgt(ktgt) >= this_src(nlevsrc)) then
                this_tgtlevs(ktgt) = nlevsrc-1
                this_weights(ktgt) = 1
                cycle
              endif

 srclevloop:  do ksrc=ksrcstart,nlevsrc

                if (this_tgt(ktgt) < this_src(ksrc)) then
                    this_tgtlevs(ktgt) = (ksrc-1)
                    this_weights(ktgt) = (this_tgt(ktgt) - this_src(ksrc-1))/ &
                                       & (this_src(ksrc) - this_src(ksrc-1))
                    ksrcstart = ksrc
                    exit srclevloop
                endif

                ! Could not solve interpolation, e.g. if some of the src values are missing
                if (this_tgtlevs(ktgt) == 0) then
                  !write(*,*) "Warning!!! could not solve interpolation!!"
                  this_weights(ktgt) = ieee_value(this_weights(ktgt), ieee_quiet_nan)
                endif

              end do srclevloop
            end do tgtlevloop

          end associate
        end do
        end associate
      end do
     end do
     ! chunks loop
     end if
    end do
    !$OMP END PARALLEL

    !write(*,*)  "OMP time: ",omp_get_wtime()-timer,"s"

  END SUBROUTINE interp

  SUBROUTINE interp_fld(fsrc, fdst, tgtlevs, weights, &
                      & ncom, nsrc, ntgt, nlevsrc, nlevdst, chunk_size_in)
    use, intrinsic :: ieee_arithmetic, only : ieee_is_nan, ieee_value, ieee_quiet_nan
    use, intrinsic :: iso_fortran_env, dp => real64, intk => int32
    use omp_lib

    integer(intk), intent(in), value  :: ncom, nsrc, ntgt
    integer(intk), intent(in), value  :: nlevdst, nlevsrc

    real(dp), intent(in),  dimension(nlevsrc,nsrc,ncom) :: fsrc
    real(dp), intent(out), dimension(nlevdst,ntgt,nsrc,ncom) :: fdst

    integer(intk), intent(in), dimension(nlevdst,ntgt,ncom) :: tgtlevs
    real(dp),      intent(in), dimension(nlevdst,ntgt,ncom) :: weights


    integer(intk), intent(in), value, optional :: chunk_size_in
    integer(intk) :: max_chunk_size = 1000
    integer(intk) :: actual_chunk_size

    integer :: icom,isrc,itgt,k

    integer(intk) :: num_threads, thread_id
    integer(intk) :: ndimomp, id_dimomp, n_chunks, nc, c_start, c_end
    integer(intk) :: id_st1=1, id_st2=1, id_st3=1
    integer(intk) :: id_en1, id_en2, id_en3

    real(dp) :: timer


    if (present(chunk_size_in)) then
        max_chunk_size = chunk_size_in
    endif


    timer = omp_get_wtime()

    !$OMP PARALLEL PRIVATE(thread_id, icom, isrc, itgt, &
    !$OMP id_st1, id_en1, id_st2, id_en2, id_st3, id_en3, nc, c_start, c_end)
    thread_id = int(omp_get_thread_num(), intk)

    ! Master thread decides here
    if (thread_id == 0) then
        num_threads = int(omp_get_num_threads(), intk)

        ! Which dimension to parallelize on
        if (ncom > num_threads) then
            ndimomp = ncom
            id_dimomp = 1
        else if (nsrc > num_threads) then
            ndimomp = nsrc
            id_dimomp = 2
        else if (ntgt > num_threads) then
            ndimomp = ntgt
            id_dimomp = 3
        else
            ndimomp = 1
            id_dimomp = 0
        endif

        ! Determine chunk size
        actual_chunk_size = min(ceiling(ndimomp * 1.0/num_threads), max_chunk_size)
        n_chunks = ceiling(ndimomp *1.0/actual_chunk_size)
    end if

    !$OMP BARRIER

    ! Initialize for each potentially parallelizable dimension
    id_st1=1
    id_st2=1
    id_st3=1
    id_en1=ncom
    id_en2=nsrc
    id_en3=ntgt

    do nc=1,n_chunks
      if ((mod(nc,num_threads)) == thread_id) then
        c_start=(nc-1)*actual_chunk_size+1
        c_end = min(ndimomp,nc*actual_chunk_size)

        if (id_dimomp == 1) then
            id_st1 = c_start
            id_en1 = c_end
        else if (id_dimomp == 2) then
            id_st2 = c_start
            id_en2 = c_end
        else if (id_dimomp == 3) then
            id_st3 = c_start
            id_en3 = c_end
        else

        endif

        do icom=id_st1,id_en1
          do isrc=id_st2,id_en2
            associate(this_src => fsrc(:,isrc,icom))
            do itgt=id_st3,id_en3

              associate(this_dst => fdst(:,itgt,isrc,icom), &
                this_tgtlevs => tgtlevs(:,itgt,icom), &
                this_weights => weights(:,itgt,icom))

              do k=1,nlevdst
                ! Interpolation not defined - weights is NaN
                if (ieee_is_nan(this_weights(k)) .or. (this_tgtlevs(k) >= nlevsrc)) then
                  this_dst(k) = ieee_value(this_dst(k), ieee_quiet_nan)
                else
                  this_dst(k) = this_src(this_tgtlevs(k)+1)*this_weights(k) + &
                              & this_src(this_tgtlevs(k))*(1-this_weights(k))
                endif
              end do
              end associate
          end do
            end associate
        end do
      end do
     ! chunks loop
     end if
    end do
    !$OMP END PARALLEL
    !write(*,*)  "OMP time: ",omp_get_wtime()-timer,"s"
  END SUBROUTINE

  SUBROUTINE f_interp(psrc, ptgt, tgtlevs, weights, ncom, nsrc, ntgt, nlevsrc, nlevtgt, chunk_size_in) &
           & bind(C, name="interp")
    use iso_c_binding, dp => c_double, intk => c_int32_t

    integer(intk), intent(in), value  :: ncom, nsrc, ntgt
    integer(intk), intent(in), value  :: nlevsrc
    integer(intk), intent(in), value  :: nlevtgt

    real(dp), intent(in), dimension(nlevsrc,nsrc,ncom) :: psrc
    real(dp), intent(in), dimension(nlevtgt,ntgt,ncom) :: ptgt

    integer(intk), intent(out) :: tgtlevs(nlevtgt,ntgt,nsrc,ncom)
    real(dp),      intent(out) :: weights(nlevtgt,ntgt,nsrc,ncom)

    integer(intk), intent(in), value :: chunk_size_in


    call interp(psrc, ptgt, tgtlevs, weights, ncom, nsrc, ntgt, nlevsrc, nlevtgt, chunk_size_in)


  END SUBROUTINE f_interp

  SUBROUTINE f_interp_fld(fsrc, fdst, tgtlevs, weights, ncom, nsrc, ntgt, nlevsrc, nlevtgt, chunk_size_in) &
           & bind(C,  name="interp_fld")
    use iso_c_binding, dp => c_double, intk => c_int32_t

    integer(intk), intent(in), value  :: ntgt, nsrc, ncom
    integer(intk), intent(in), value  :: nlevtgt, nlevsrc

    real(dp), intent(in),  dimension(nlevsrc,nsrc,ncom) :: fsrc
    real(dp), intent(out), dimension(nlevtgt,ntgt,nsrc,ncom) :: fdst

    integer(intk), intent(in) :: tgtlevs(nlevtgt,ntgt,nsrc,ncom)
    real(dp),      intent(in) :: weights(nlevtgt,ntgt,nsrc,ncom)


    integer(intk), intent(in), value :: chunk_size_in


    call interp_fld(fsrc, fdst, tgtlevs, weights, ncom, nsrc, ntgt, nlevsrc, nlevtgt, chunk_size_in)


  END SUBROUTINE f_interp_fld

END MODULE mo_interp
