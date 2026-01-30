MODULE mo_lut_tools
  implicit none
  private
  public :: get_bin_nearest, get_bin_bounds, get_flat_index, get_flexi_lutvals


contains
  FUNCTION get_flat_index(index_tuple, ndims, nbins, c_order) result(flat_index)
    use, intrinsic :: iso_fortran_env, realk => real32, db => real64, intk => int32

    integer(intk) :: ndims
    integer(intk), dimension(ndims) :: index_tuple, nbins

    logical :: c_order

    integer(intk) :: flat_index
    integer(intk) :: istart, istop, iincr, jdim, jmul

    if (c_order) then
      istart = ndims
      istop = 1
      iincr = -1
    else
      istart = 1
      istop = ndims
      iincr = 1
    ENDIF

    flat_index = 1
    jmul = 1
    do jdim=istart,istop,iincr
      flat_index = flat_index + (index_tuple(jdim) - 1)*jmul
      jmul = jmul * nbins(jdim)
    enddo

  END FUNCTION get_flat_index

  SUBROUTINE get_flexi_lutvals(lut_maps, nmaps, map_size, nwspeed, nspec, nvals, max_lut_bins, &
                             & aero_species, lut_spec_bins, nspecbins, val_out, &
                             & wspeeds, lut_wspeed_bins, nwspeedbins, chunk_size_in, c_order_in)
    use, intrinsic :: ieee_arithmetic, only : ieee_is_nan, ieee_value, ieee_quiet_nan
    use, intrinsic :: iso_fortran_env, realk => real32, db => real64, intk => int32

    use omp_lib

    ! (nmaps, nbins(m), nbins(m-1), ..., nbins(1)) = (nmaps, ntot_lut) as leading column
    real(realk), intent(in), dimension(nmaps, map_size) :: lut_maps

    ! map_size = prod nbins(i) for i=1,...,nwspeed+nspec
    integer(intk), intent(in), value :: nmaps, map_size, nwspeed, nspec, nvals, max_lut_bins

    ! (nspec, nvals) = (ntot_species) as leading column
    real(realk), intent(in), dimension(nspec, nvals) :: aero_species

    ! wspeed and spec as leading column
    real(realk), intent(in), dimension(max_lut_bins, nspec) :: lut_spec_bins
    integer(intk), intent(in), dimension(nspec) :: nspecbins

    real(realk), intent(out), dimension(nmaps, nvals) :: val_out

    ! If lut has vertical speed dimensions (supported up to 2)
    real(realk), intent(in), dimension(nwspeed, nvals), optional :: wspeeds
    real(realk), intent(in), dimension(max_lut_bins, nwspeed), optional :: lut_wspeed_bins
    integer(intk), intent(in), dimension(nwspeed), optional :: nwspeedbins

    integer(intk), intent(in), value, optional :: chunk_size_in
    logical, intent(in), value, optional :: c_order_in
    logical :: c_order

    integer(intk) :: chunk_size

    integer(intk), dimension(2, nwspeed) :: tgtbounds
    integer(intk), dimension(nspec) :: tgtbins
    integer(intk), dimension(4,nwspeed+nspec) :: lut_index_tuples

    integer(intk) :: lut_tot_dims
    integer(intk), dimension(nwspeed+nspec) :: lut_all_nbins

    integer(intk) :: i, i1, i2, j1, j2, idx_val

    real(realk) :: a1, a, w, w1, w2, z, z1, z2

    integer(intk) :: num_threads, thread_id
    integer(intk) :: n_chunks, nc, c_start, c_end
    real(db) :: timer

    logical :: interp_w, interp_z

    ! Assume c_order by default
    if (present(c_order_in)) then
      c_order = c_order_in
    else
      c_order = .true.
    endif

    chunk_size = 12000
    if (present(chunk_size_in)) then
      chunk_size = chunk_size_in
    endif
    n_chunks = ceiling(nvals * 1.0/chunk_size)

    timer = omp_get_wtime()

    ! check consistency of optional parameters
    if (nwspeed > 0) then
      if (.not. present(wspeeds)) then
        write(*,*) "Error: wspeeds must be provided if nwspeed > 0"
        stop
      end if
      if (.not. present(lut_wspeed_bins)) then
        write(*,*) "Error: lut_wspeed_bins must be provided if nwspeed > 0"
        stop
      end if
      if (.not. present(nwspeedbins)) then
        write(*,*) "Error: nwspeedbins must be provided if nwspeed > 0"
        stop
      end if
    end if

    ! Populate lut_all_nbins
    lut_tot_dims = nwspeed+nspec
    do i=1,nspec
      lut_all_nbins(nwspeed+i) = nspecbins(i)
      !lut_all_bins(1:nspecbins(i), nwspeed+i) = lut_spec_bins(1:nspecbins(i), i)
    end do
    if (nwspeed > 0) then
      do i=1,nwspeed
       lut_all_nbins(i) = nwspeedbins(i)
       !lut_all_bins(1:nwspeedbins(i), i) = lut_wspeed_bins(1:nwspeedbins(i), i)
      end do
    end if


    !$OMP PARALLEL PRIVATE(idx_val, num_threads, thread_id, tgtbins, tgtbounds, c_start, c_end, &
    !$OMP& i, nc, lut_index_tuples, a, w, w1, w2, z, z1, z2, &
    !$OMP& interp_w, interp_z, j1, j2, i1, i2)
    num_threads = int(omp_get_num_threads(), intk)
    thread_id = int(omp_get_thread_num(), intk)
    !write(*,*) "Using ",num_threads," threads."

    !do nc=1,n_chunks
    !  if ((mod(nc,num_threads)) == thread_id) then
    !    c_start=(nc-1)*chunk_size+1
    !    c_end = min(nvals,nc*chunk_size)
    !    do idx_val=c_start,c_end

    !$OMP DO SCHEDULE(DYNAMIC, chunk_size)
    do idx_val=1,nvals
      ! tgtbins(spec1, spec2, spec3, ..., specm)
      if (ieee_is_nan(aero_species(1, idx_val))) then
        val_out(:, idx_val) = ieee_value(val_out(1, idx_val), ieee_quiet_nan)
        cycle
      endif

      ! nearest neighbor for species
      call get_bin_nearest(aero_species(:, idx_val), lut_spec_bins, &
                          & nspecbins, max_lut_bins, nspec, &
                          & tgtbins)

      ! Populate tgtbins for aero species
      do i=1,max(1,2*nwspeed)
        lut_index_tuples(i,nwspeed+1:nwspeed+nspec) = tgtbins(:)
      end do

      !do i=1,nspec
      !  write(*,'(A,I0,A,I0,A,F7.4,A,I0,A,F7.4)') "IDX ",idx_val," aero",i,"= ",aero_species(i,idx_val)," bin ",tgtbins(i),"= ",lut_spec_bins(tgtbins(i),i)
      !end do


      ! linear interpolation for updraft speeds
      if (nwspeed > 0) then
        call get_bin_bounds(wspeeds(:, idx_val), lut_wspeed_bins, &
                          & nwspeedbins, max_lut_bins, nwspeed, &
                          & tgtbounds)

        ! First wspeed dimension
        i1 = tgtbounds(1,1)
        i2 = tgtbounds(2,1)

        lut_index_tuples(1:2,1) = tgtbounds(:,1)

      endif

      select case (nwspeed)
      case (0)
      val_out(:, idx_val) = lut_maps(:, get_flat_index(lut_index_tuples(1,:), lut_tot_dims, lut_all_nbins, c_order))

      case (1)

      associate( &
        fw1z1 => lut_maps(:, get_flat_index(lut_index_tuples(1,:), lut_tot_dims, lut_all_nbins, c_order)), &
      & fw2z1 => lut_maps(:, get_flat_index(lut_index_tuples(2,:), lut_tot_dims, lut_all_nbins, c_order)))

        interp_w = (i1 /= i2)
        if (interp_w) then

          w = wspeeds(1, idx_val)
          w1 = lut_wspeed_bins(i1,1)
          w2 = lut_wspeed_bins(i2,1)
          a = (w - w1)/(w2 - w1)

          val_out(:, idx_val) = a*fw2z1(:) + (1-a)*fw1z1(:)
        !no interpolation
        else
          val_out(:, idx_val) = fw1z1(:)
        endif

      end associate
      case (2)


      lut_index_tuples(3:4,1) = tgtbounds(:,1)
      j1=tgtbounds(1,2)
      j2=tgtbounds(2,2)
      lut_index_tuples(1:2,2) = j1
      lut_index_tuples(3:4,2) = j2

      w = wspeeds(1, idx_val)
      w1 = lut_wspeed_bins(i1,1)
      w2 = lut_wspeed_bins(i2,1)
      z = wspeeds(2, idx_val)
      z1 = lut_wspeed_bins(j1,2)
      z2 = lut_wspeed_bins(j2,2)

      associate( &
        fw1z1 => lut_maps(:, get_flat_index(lut_index_tuples(1,:), lut_tot_dims, lut_all_nbins, c_order)), &
      & fw2z1 => lut_maps(:, get_flat_index(lut_index_tuples(2,:), lut_tot_dims, lut_all_nbins, c_order)), &
      & fw1z2 => lut_maps(:, get_flat_index(lut_index_tuples(3,:), lut_tot_dims, lut_all_nbins, c_order)), &
      & fw2z2 => lut_maps(:, get_flat_index(lut_index_tuples(4,:), lut_tot_dims, lut_all_nbins, c_order)))

        interp_w = (i1 /= i2)
        interp_z = (j1 /= j2)
        if (interp_w) then
          !bilinear
          if (interp_z) then
            val_out(:, idx_val) = 1/((w2-w1)*(z2-z1))* &
            & (fw1z1(:)*(w2-w)*(z2-z) + fw1z2(:)*(w2-w)*(z-z1) + &
            &  fw2z1(:)*(w-w1)*(z2-z) + fw2z2(:)*(w-w1)*(z-z1))
          !linear
          else
            a = (w2 - w)/(w2 - w1)
            val_out(:, idx_val) = a*fw1z1(:) + (1-a)*fw2z1(:)
          endif
        !linear
        elseif (interp_z) then
          a = (z2 - z)/(z2 - z1)
          val_out(:, idx_val) = a*fw1z1(:) + (1-a)*fw1z2(:)
        else
        ! no need to interpolate
        val_out(:, idx_val) = fw1z1(:)
        endif
      end associate
      end select

    end do
    !$OMP END DO
    !$OMP END PARALLEL

    !write(*,*)  "OMP time: ",omp_get_wtime()-timer,"s"


  END SUBROUTINE get_flexi_lutvals

  SUBROUTINE get_bin_bounds(values, bins, nbins, max_lut_bins, nvals, tgtbounds)
    use, intrinsic :: ieee_arithmetic, only : ieee_is_nan
    use, intrinsic :: iso_fortran_env, realk => real32, intk => int32
    integer(intk), intent(in), value  :: max_lut_bins, nvals
    real(realk), intent(in), dimension(nvals) :: values
    integer(intk), intent(in), dimension(nvals) :: nbins
    real(realk), intent(in), dimension(max_lut_bins, nvals) :: bins
    integer(intk), intent(out), dimension(2, nvals) :: tgtbounds

    integer(intk), dimension(nvals) :: nearest_bins
    integer(intk) :: s

    call get_bin_nearest(values, bins, nbins, max_lut_bins, nvals, nearest_bins)
    do s=1,nvals
      if (ieee_is_nan(values(s))) then
         tgtbounds(:,s) = -1
         cycle
      endif

      if (values(s) < bins(nearest_bins(s),s)) then
        if (nearest_bins(s) == 1) then
          tgtbounds(:,s) = 1
        else
          tgtbounds(1,s) = nearest_bins(s)-1
          tgtbounds(2,s) = nearest_bins(s)
        endif
      else
        if (nearest_bins(s) == nbins(s)) then
          tgtbounds(:,s) = nbins(s)
        else
          tgtbounds(1,s) = nearest_bins(s)
          tgtbounds(2,s) = nearest_bins(s)+1
        endif
      endif
    end do

  END SUBROUTINE get_bin_bounds

  SUBROUTINE get_bin_nearest(values, bins, nbins, max_lut_bins, nvals, tgtbins)
    use, intrinsic :: ieee_arithmetic, only : ieee_is_nan
    use, intrinsic :: iso_fortran_env, realk => real32, intk => int32

    integer(intk), intent(in), value  :: max_lut_bins, nvals

    real(realk), intent(in), dimension(nvals) :: values
    integer(intk), intent(in), dimension(nvals) :: nbins
    real(realk), intent(in), dimension(max_lut_bins, nvals) :: bins

    integer(intk), intent(out), dimension(nvals) :: tgtbins

    real(realk) :: this_diff
    integer(intk) :: s
    integer(intk) :: b_lo,b_hi,b_mi

    do s=1,nvals
      ! no overhead compared to intrinsic
      if (ieee_is_nan(values(s))) then
         tgtbins(:) = -1
         exit
      endif

      associate(this_value => values(s), this_bins => bins(:,s), &
              & this_nbins => nbins(s), this_tgtbin => tgtbins(s))

        b_lo = 1
        b_hi = this_nbins

        if (this_value <= this_bins(b_lo)) then
          b_mi = b_lo
          b_hi = b_lo
        else if (this_value >= this_bins(b_hi)) then
          b_mi = b_hi
          b_lo = b_hi
        else
            ! Binary search
            do while (b_hi - b_lo > 1)
              b_mi = (b_hi + b_lo)/2
              this_diff = this_value - this_bins(b_mi)

              if (this_diff >= 0) then
                b_lo = b_mi
              endif

              if (this_diff <= 0) then
                b_hi = b_mi
              endif
            enddo
        endif

        ! Choose nearest
        if (b_hi > b_lo) then
          if (2*this_value-this_bins(b_lo)-this_bins(b_hi) < 0) then
            b_mi = b_lo
          else
            b_mi = b_hi
          endif
        endif

        this_tgtbin = b_mi

      end associate
    end do

  END SUBROUTINE get_bin_nearest

  SUBROUTINE f_get_flexi_lutvals(lut_maps, nmaps, map_size, nwspeed, nspec, nvals, max_lut_bins, &
                               & aero_species, lut_spec_bins, nspecbins, val_out, &
                               & wspeeds, lut_wspeed_bins, nwspeedbins, chunk_size_in, c_order_in) &
                               & bind(C,  name="get_flexi_lutvals")
    use iso_c_binding, realk => c_float, intk => c_int32_t
    use, intrinsic :: iso_fortran_env, db => real64

    ! Input dimensions
    integer(intk), intent(in), value :: nmaps, map_size, nwspeed, nspec, nvals, max_lut_bins

    ! Input arrays
    real(realk), intent(in), dimension(nmaps, map_size) :: lut_maps
    real(realk), intent(in), dimension(nspec, nvals) :: aero_species
    real(realk), intent(in), dimension(max_lut_bins, nspec) :: lut_spec_bins
    integer(intk), intent(in), dimension(nspec) :: nspecbins
    real(realk), intent(in), dimension(nwspeed, nvals) :: wspeeds
    real(realk), intent(in), dimension(max_lut_bins, nwspeed) :: lut_wspeed_bins
    integer(intk), intent(in), dimension(nwspeed) :: nwspeedbins

    ! Output array
    real(realk), intent(out), dimension(nmaps, nvals) :: val_out

    ! Optional parameters
    integer(intk), intent(in), value :: chunk_size_in
    logical(c_bool), intent(in), value :: c_order_in
    logical :: c_order
    c_order = c_order_in

    call get_flexi_lutvals(lut_maps, nmaps, map_size, nwspeed, nspec, nvals, max_lut_bins, &
                         & aero_species, lut_spec_bins, nspecbins, val_out, &
                         & wspeeds=wspeeds, lut_wspeed_bins=lut_wspeed_bins, &
                         & nwspeedbins=nwspeedbins, chunk_size_in=chunk_size_in, &
                         & c_order_in=c_order)

  END SUBROUTINE f_get_flexi_lutvals

  SUBROUTINE f_get_flexi_lutvals_nowspeed(lut_maps, nmaps, map_size, nwspeed, nspec, nvals, max_lut_bins, &
                               & aero_species, lut_spec_bins, nspecbins, val_out, &
                               & chunk_size_in, c_order_in) &
                               & bind(C,  name="get_flexi_lutvals_nowspeed")
    use iso_c_binding, realk => c_float, intk => c_int32_t
    use, intrinsic :: iso_fortran_env, db => real64

    ! Input dimensions
    integer(intk), intent(in), value :: nmaps, map_size, nwspeed, nspec, nvals, max_lut_bins

    ! Input arrays
    real(realk), intent(in), dimension(nmaps, map_size) :: lut_maps
    real(realk), intent(in), dimension(nspec, nvals) :: aero_species
    real(realk), intent(in), dimension(max_lut_bins, nspec) :: lut_spec_bins
    integer(intk), intent(in), dimension(nspec) :: nspecbins

    ! Output array
    real(realk), intent(out), dimension(nmaps, nvals) :: val_out

    ! Optional parameters
    integer(intk), intent(in), value :: chunk_size_in
    logical(c_bool), intent(in), value :: c_order_in
    logical :: c_order
    c_order = c_order_in

    call get_flexi_lutvals(lut_maps, nmaps, map_size, nwspeed, nspec, nvals, max_lut_bins, &
                         & aero_species, lut_spec_bins, nspecbins, val_out, &
                         & chunk_size_in=chunk_size_in, c_order_in=c_order)

  END SUBROUTINE f_get_flexi_lutvals_nowspeed


END MODULE mo_lut_tools
