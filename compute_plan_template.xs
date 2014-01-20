MODULE = PDL::FFTW3 PACKAGE = PDL::FFTW3

IV
compute_plan( dims_ref, do_double_precision, is_real_fft, do_inverse_fft, in_pdl, out_pdl )
  SV*  dims_ref
  bool do_double_precision
  bool is_real_fft
  bool do_inverse_fft
  pdl* in_pdl
  pdl* out_pdl
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

  void* in_data = in_pdl->data;
  void* out_data = out_pdl->data;

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

  if( plan == NULL )
    XSRETURN_UNDEF;
  else
    RETVAL = PTR2IV(plan);
}
OUTPUT:
 RETVAL



int
is_same_data( in, out )
  pdl* in
  pdl* out
CODE:
{
  RETVAL = (in->data == out->data) ? 1 : 0;
}
OUTPUT:
 RETVAL


int
get_data_alignment( in )
  pdl* in
CODE:
{
  int alignment;

  alignment = ( PTR2UV(in->data) % (UVTYPE)16 == 0 ) ? 16 :
              ( PTR2UV(in->data) % (UVTYPE) 8 == 0 ) ?  8 :
              ( PTR2UV(in->data) % (UVTYPE) 4 == 0 ) ?  4 :
              ( PTR2UV(in->data) % (UVTYPE) 2 == 0 ) ?  2 : 1;

  RETVAL = alignment;
}
OUTPUT:
 RETVAL
