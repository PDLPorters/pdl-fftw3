MODULE = PDL::FFTW3 PACKAGE = PDL::FFTW3

void *
compute_plan( rank, do_double_precision, do_inverse_fft, in, out )
  int rank
  bool do_double_precision
  bool do_inverse_fft
  pdl* in
  pdl* out
CODE:
{
  // Given input and output matrices, this function computes the FFTW plan

  // PDL stores its data in the opposite dimension order from what FFTW wants. I
  // handle this by passing in the dimension counts backwards. Furthermore, the
  // dimension indices start at 1 because dim0 is the (real,imag) dimension that
  // is implicit in FFTW
  int dims_row_first          [rank];


  // TODO if out is null, I should make a plan with a different pointer, maybe
  void* plan;


  int direction = do_inverse_fft ? FFTW_BACKWARD : FFTW_FORWARD;

  // complex-complex FFT. Input/output have identical dimensions
  for( int i=0; i<rank; i++)
    dims_row_first[i] = in->dims[ rank-i ];

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
