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

  $TFD(fftwf_,fftw_)plan plan = INT2PTR( $TFD(fftwf_,fftw_)plan, $COMP(plan));
  $TFD(fftwf_,fftw_)execute_dft( plan,
                                 ($TFD(fftwf_,fftw_)complex*)$P(in),
                                 ($TFD(fftwf_,fftw_)complex*)$P(out) );
}

