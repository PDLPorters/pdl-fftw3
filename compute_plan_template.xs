MODULE = PDL::FFTW3 PACKAGE = PDL::FFTW3

void *
compute_plan( dims_ref, do_double_precision, is_real_fft, do_inverse_fft, in, out )
  SV*  dims_ref
  bool do_double_precision
  bool is_real_fft
  bool do_inverse_fft
  pdl* in
  pdl* out
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

  // TODO if out is null, I should make a plan with a different pointer, maybe
  void* plan;

  if( !is_real_fft )
  {
    int direction = do_inverse_fft ? FFTW_BACKWARD : FFTW_FORWARD;

    // complex-complex FFT. Input/output have identical dimensions
    if( !do_double_precision )
    {
      // TODO check for the overwriting of the input when I do something fancier
      // than FFTW_ESTIMATE
      plan =
        fftwf_plan_dft( rank, dims_row_first,
                        (fftwf_complex*)in->data, (fftwf_complex*)out->data,
                        direction, FFTW_ESTIMATE);
    }
    else
    {
      // TODO check for the overwriting of the input when I do something fancier
      // than FFTW_ESTIMATE
      plan =
        fftw_plan_dft( rank, dims_row_first,
                       (fftw_complex*)in->data, (fftw_complex*)out->data,
                       direction, FFTW_ESTIMATE);
    }
  }
  else
  {
    // real-complex FFT. Input/output have different dimensions
    if( !do_double_precision)
    {
      if( !do_inverse_fft )
      {
        plan =
          fftwf_plan_dft_r2c( rank, dims_row_first,
                              (float*)in->data, (fftwf_complex*)out->data,
                              FFTW_ESTIMATE );
      }
      else
      {
        plan =
          fftwf_plan_dft_c2r( rank, dims_row_first,
                              (fftwf_complex*)in->data, (float*)out->data,
                              FFTW_ESTIMATE );
        // TODO check for the overwriting of the input when I do something fancier
      }
    }
    else
    {
      if( !do_inverse_fft )
      {
        plan =
          fftw_plan_dft_r2c( rank, dims_row_first,
                             (double*)in->data, (fftw_complex*)out->data,
                             FFTW_ESTIMATE );
      }
      else
      {
        plan =
          fftw_plan_dft_c2r( rank, dims_row_first,
                             (fftw_complex*)in->data, (double*)out->data,
                             FFTW_ESTIMATE );
        // TODO check for the overwriting of the input when I do something fancier
      }
    }
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
