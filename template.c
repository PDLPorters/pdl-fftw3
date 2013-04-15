// This is the template used by PP to generate the FFTW routines. This is passed
// into pp_def() in the 'Code' key. Before this file is passed to pp_def, the
// following strings are replaced:
//
// RANK            This is the rank of the FFT being generated

#ifndef __TEMPLATE_ALREADY_INCLUDED__

/* the Linux kernel does something similar to assert at compile time */
#define static_assert(x) (void)( sizeof( int[ 1 - 2* !(x) ]) )

#define __TEMPLATE_ALREADY_INCLUDED__
#endif


{
  // make sure the PDL data type I'm using matches the FFTW data type
  static_assert( sizeof($GENERIC())*2 == sizeof($TFD(fftwf_,fftw_)complex) );

  // PP asked for these dimensions, so these should all be fine. If they aren't,
  // PP messed up somehow
  assert( $PDL(in) ->ndims == RANK+1 );
  assert( $PDL(out)->ndims == RANK+1 );
  assert( $PDL(in)->dims[0] == 2 );
  for( int i=0; i<RANK; i++ )
    assert( $PDL(in)->dims[i] == $PDL(out)->dims[i] );


  // PDL stores its data in the opposite dimension order from what FFTW wants. I
  // handle this by passing in the dimension counts backwards. Furthermore, the
  // dimension indices start at 1 because dim0 is the (real,imag) dimension that
  // is implicit in FFTW
  int dims_row_first[RANK];
  for( int i=0; i<RANK; i++)
    dims_row_first[i] = $PDL(in)->dims[ RANK-i ];

  $TFD(fftwf_,fftw_)plan p =
    $TFD(fftwf_,fftw_)plan_many_dft( RANK, dims_row_first,

                           // just 1 transform
                           1,

                           // the data pointer, the nembed dimension counters, stride, dist,
                           ($TFD(fftwf_,fftw_)complex*)$P(in),  dims_row_first, 1, 0,
                           ($TFD(fftwf_,fftw_)complex*)$P(out), dims_row_first, 1, 0,

                           FFTW_FORWARD, FFTW_ESTIMATE);

  // how do I return an error? this shouldn't be an assert
  assert( p );


  fprintf(stderr, "making a plan. pin, pout: (0x%p, 0x%p)\n", $P(in), $P(out));
  // how do I make sure these are aligned?

  fprintf(stderr, "running a dft. pin, pout: (0x%p, 0x%p)\n", $P(in), $P(out));
  // for( unsigned long long i=0; i<1000000; i++ )
  //   for( unsigned long long j=0; j<1000000; j++ );

  $TFD(fftwf_,fftw_)execute_dft( p,
                       ($TFD(fftwf_,fftw_)complex*)$P(in),
                       ($TFD(fftwf_,fftw_)complex*)$P(out) );

  $TFD(fftwf_,fftw_)destroy_plan(p);
}

