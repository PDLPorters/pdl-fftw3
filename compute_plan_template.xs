MODULE = PDL::FFTW3 PACKAGE = PDL::FFTW3

void *
compute_plan( rank, do_double_precision, in, out )
  int rank
  bool do_double_precision
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
  int in_dims_embed_row_first [rank];
  int out_dims_embed_row_first[rank];
  for( int i=0; i<rank; i++)
  {
    dims_row_first[i]           = in->dims[ rank-i ];

    // TODO support nembed stuff here. watch for out can be null
    in_dims_embed_row_first[i]  = in->dims[ rank-i ];
    out_dims_embed_row_first[i] = in->dims[ rank-i ];
  }

  // TODO if out is null, I should make a plan with a different pointer, maybe
  void* plan;
  if( !do_double_precision )
  {
    // TODO check for the overwriting of the input when I do something fancier
    // than FFTW_ESTIMATE
    plan =
      fftwf_plan_many_dft( rank, dims_row_first,

                           // just 1 transform
                           1,

                           // the data pointer, the nembed dimension counters, stride, dist,
                           (fftwf_complex*)in->data,  in_dims_embed_row_first,  1, 0,
                           (fftwf_complex*)out->data, out_dims_embed_row_first, 1, 0,

                           FFTW_FORWARD, FFTW_ESTIMATE);
  }
  else
  {
    // TODO check for the overwriting of the input when I do something fancier
    // than FFTW_ESTIMATE
    plan =
      fftw_plan_many_dft( rank, dims_row_first,

                          // just 1 transform
                          1,

                          // the data pointer, the nembed dimension counters, stride, dist,
                          (fftw_complex*)in->data,  in_dims_embed_row_first,  1, 0,
                          (fftw_complex*)out->data, out_dims_embed_row_first, 1, 0,

                          FFTW_FORWARD, FFTW_ESTIMATE);
  }

  RETVAL = PTR2IV(plan);
}
OUTPUT:
 RETVAL
