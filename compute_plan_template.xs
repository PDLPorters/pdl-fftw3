MODULE = PDL::FFTW3 PACKAGE = PDL::FFTW3

IV
compute_plan( dims_ref, do_double_precision, is_real_fft, do_inverse_fft )
  SV*  dims_ref
  bool do_double_precision
  bool is_real_fft
  bool do_inverse_fft
CODE:
{
  // Given input and output matrices, this function computes the FFTW plan

  // PDL stores its data in the opposite dimension order from what FFTW wants. I
  // handle this by passing in the dimension counts backwards.
  AV* dims_av = (AV*)SvRV(dims_ref);
  int rank = av_len(dims_av) + 1;

  int dims_row_first[rank];
  for( int i=0; i<rank; i++)
    dims_row_first[i] = SvIV( *av_fetch( dims_av, rank-i-1, 0) );

  // I do a dummy alloc of memory so I can pass properly-aligned pointers to the
  // planner
  void* in_data = do_double_precision ?
    (void*) fftw_alloc_complex(16) :
    (void*)fftwf_alloc_complex(16);
  void* out_data = do_double_precision ?
    (void*) fftw_alloc_complex(16) :
    (void*)fftwf_alloc_complex(16);

  void* plan;
  if( !is_real_fft )
  {
    int direction = do_inverse_fft ? FFTW_BACKWARD : FFTW_FORWARD;

    // complex-complex FFT. Input/output have identical dimensions
    if( !do_double_precision )
      plan =
        fftwf_plan_dft( rank, dims_row_first,
                        (fftwf_complex*)in_data, (fftwf_complex*)out_data,
                        direction, FFTW_ESTIMATE);
    else
      plan =
        fftw_plan_dft( rank, dims_row_first,
                       (fftw_complex*)in_data, (fftw_complex*)out_data,
                       direction, FFTW_ESTIMATE);
  }
  else
  {
    // real-complex FFT. Input/output have different dimensions
    if( !do_double_precision)
    {
      if( !do_inverse_fft )
        plan =
          fftwf_plan_dft_r2c( rank, dims_row_first,
                              (float*)in_data, (fftwf_complex*)out_data,
                              FFTW_ESTIMATE );
      else
        plan =
          fftwf_plan_dft_c2r( rank, dims_row_first,
                              (fftwf_complex*)in_data, (float*)out_data,
                              FFTW_ESTIMATE );
    }
    else
    {
      if( !do_inverse_fft )
        plan =
          fftw_plan_dft_r2c( rank, dims_row_first,
                             (double*)in_data, (fftw_complex*)out_data,
                             FFTW_ESTIMATE );
      else
        plan =
          fftw_plan_dft_c2r( rank, dims_row_first,
                             (fftw_complex*)in_data, (double*)out_data,
                             FFTW_ESTIMATE );
    }
  }

  if(do_double_precision)
  {
    fftw_free (in_data);
    fftw_free (out_data);
  }
  else
  {
    fftwf_free(in_data);
    fftwf_free(out_data);
  }

  if( plan == NULL )
    XSRETURN_UNDEF;
  else
    RETVAL = PTR2IV(plan);
}
OUTPUT:
 RETVAL



bool
is_same_data( in, out )
  pdl* in
  pdl* out
CODE:
{
  RETVAL = in->data == out->data;
}
OUTPUT:
 RETVAL
