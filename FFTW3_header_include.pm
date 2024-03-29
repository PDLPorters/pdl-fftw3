# This file is included by FFTW3.pd


use PDL::Types;
use List::Util 'reduce';
use threads::shared;

# When I compute an FFTW plan, it goes here.
# This is :shared so that it can be used with Perl threads.
my %existingPlans :shared;

# these are for the unit tests
our $_Nplans = 0;
our $_last_do_double_precision;

# This file is included verbatim into the final module via pp_addpm()

# This is a function that sits between the user's call into this module and the
# PP-generated internals. Specifically, this function is called BEFORE any PDL
# threading happens. Here I make sure the FFTW plan exists, or if it doesn't, I
# make it. Thus the PP-based internals can safely assume that the plan exists
sub __fft_internal {
  my $thisfunction = shift;

  my ($do_inverse_fft, $is_real_fft, $is_native_output, $rank) = $thisfunction =~ /^(i?)(r?)(N?).*fft([0-9]+)/;

  # first I parse the variables. This is a very direct translation of what PP
  # does normally. Plan-creation has to be outside of PP, so I must re-do this
  # here
  my $Nargs = scalar @_;

  my ($in, $out);
  if ( $Nargs == 2 ) {
    # all variables on stack, read in output and temp vars
    ($in, $out) = map {defined $_ ? PDL::Core::topdl($_) : $_} @_;
  } elsif ( $Nargs == 1 ) {
    $in = PDL::Core::topdl $_[0];
    if ( $in->is_inplace ) {
      barf <<EOF if $is_real_fft;
$thisfunction: in-place real FFTs are not supported since the input/output types and data sizes differ.
Giving up.
EOF
      $out = $in;
      $in->set_inplace(0);
    } else {
      $out = PDL::null();
    }
  } else {
    barf( <<EOF );
$thisfunction must be given the input or the input and output as args.
Exactly 1 or 2 arguments are required. Instead I got $Nargs args. Giving up.
EOF
  }

  # make sure the in/out types match. Convert $in if needed. This needs to
  # happen before we instantiate $out (if it's null) to make sure we know the
  # type
  processTypes( $thisfunction, $is_native_output, \$in, \$out );

  # I now create an ndarray for the null output. Normally PP does this, but I need
  # to have the ndarray made to create plans. If I don't, the alignment may
  # differ between plan-time and run-time
  if ( $out->isnull ) {
    my @args = getOutArgs($in, $is_real_fft, $do_inverse_fft, $is_native_output);
    $out .= zeros(@args);
  }

  validateArguments( $rank, $is_real_fft, $do_inverse_fft, $is_native_output, $thisfunction, $in, $out );

  # I need to physical-ize the ndarrays before I make a plan. Again, normally PP
  # does this, but to make sure alignments match, I need to do this myself, now
  $in->make_physical;
  $out->make_physical;

  my $plan = getPlan( $thisfunction, $rank, $is_real_fft, $do_inverse_fft, $in, $out );
  barf "$thisfunction couldn't make a plan. Giving up\n" unless defined $plan;

  my $is_native = !$in->type->real; # native complex
  $is_native_output ||= !$out->type->real;
  # I now have the arguments and the plan. Go!
  my $internal_function = 'PDL::__';
  $internal_function .=
    ($is_native && !$is_real_fft) ? 'N' :
    !$is_real_fft ? '' :
    ($is_native && $do_inverse_fft) ? 'irN' :
    $do_inverse_fft ? 'ir' :
    ($is_native_output) ? 'rN' :
    'r';
  $internal_function .= "fft$rank";
  eval { no strict 'refs'; $internal_function->( $in, $out, $plan ) };
  barf $@ if $@;

  ($in->isa('PDL::Complex') && !($do_inverse_fft  && $is_real_fft))
    ? $out->complex : $out;
}

sub getOutArgs {
  my ($in, $is_real_fft, $do_inverse_fft, $is_native_output) = @_;

  my @dims = $in->dims;
  my $is_native = !$in->type->real;
  my $out_type = $in->type;

  if ( !$is_real_fft ) {
    # complex fft. Output is the same size as the input.
  } elsif ( !$do_inverse_fft ) {
    # forward real fft
    $dims[0] = int($dims[0]/2)+1;
    if ($is_native_output) {
      $out_type = typeWithComplexity(getPrecision($out_type), $is_native_output);
    } else {
      unshift @dims, 2;
    }
  } else {
    # backward real fft
    #
    # there's an ambiguity here. I want int($out->dim(0)/2) + 1 == $in->dim(1),
    # however this could mean that
    #  $out->dim(0) = 2*$in->dim(1) - 2
    # or
    #  $out->dim(0) = 2*$in->dim(1) - 1
    #
    # WITHOUT ANY OTHER INFORMATION, I ASSUME EVEN INPUT SIZES, SO I ASSUME
    #  $out->dim(0) = 2*$in->dim(1) - 2
    if ($is_native) {
      $out_type = ($out_type == cfloat) ? float : double;
    } else {
      shift @dims;
    }
    $dims[0] = 2*($dims[0]-1);
  }
  ($out_type, @dims);
}

sub validateArguments
{
  my ($rank, $is_real_fft, $do_inverse_fft, $is_native_output, $thisfunction, $in, $out) = @_;

  for my $arg ( $in, $out )
  {
    barf <<EOF unless defined $arg;
$thisfunction arguments must all be defined. If you want an auto-growing ndarray, use 'null' such as
$thisfunction( \$in, \$out = null )
Giving up.
EOF

    my $type = ref $arg;
    $type = 'scalar' unless defined $arg;
    barf <<EOF unless ref $arg && $arg->isa('PDL');
$thisfunction arguments must be of type 'PDL' (including 'PDL::Complex').
Instead I got an arg of type '$type'. Giving up.
EOF
  }

  # validate dimensionality of the ndarrays
  my @inout = ($in, $out);

  for my $iarg ( 0..1 )
  {
    my $arg = $inout[$iarg];

    if( $arg->isnull )
    {
      barf "$thisfunction: don't know what to do with a null input. Giving up";
    }

    if( !$is_real_fft )
    { validateArgumentDimensions_complex( $rank, $thisfunction, $arg); }
    else
    { validateArgumentDimensions_real( $rank, $do_inverse_fft, $is_native_output, $thisfunction, $iarg, $arg); }
  }

  # we have an explicit output ndarray we're filling in. Make sure the
  # input/output dimensions match up
  if ( !$is_real_fft )
  { matchDimensions_complex($thisfunction, $rank, $in, $out); }
  else
  { matchDimensions_real($thisfunction, $rank, $do_inverse_fft, $is_native_output, $in, $out); }
}

sub validateArgumentDimensions_complex
{
  my ( $rank, $thisfunction, $arg ) = @_;
  my $is_native = !$arg->type->real;

  # complex FFT. Identically-sized inputs/outputs
  barf <<EOF if !$is_native and $arg->dim(0) != 2;
$thisfunction must have dim(0) == 2 for non-native complex inputs and outputs.
This is the (real,imag) dimension. Giving up.
EOF

  my $dims_cmp = $arg->ndims - ($is_native ? 0 : 1);
  barf <<EOF if $dims_cmp < $rank;
Tried to compute a $rank-dimensional FFT, but an array has fewer than $rank dimensions.
Giving up.
EOF
}

sub validateArgumentDimensions_real {
  my ( $rank, $do_inverse_fft, $is_native_output, $thisfunction, $iarg, $arg ) = @_;
  my $is_native = !$arg->type->real; # native complex
#use Carp; use Test::More; diag "vAD_r ($arg)($is_native) ", $arg->info;

  # real FFT. Forward transform takes in real and spits out complex;
  # backward transform does the reverse
  if ( !$is_native && $arg->dim(0) != 2 ) {
    my ($verb, $var);
    if ( !$is_native_output && !$do_inverse_fft && $iarg == 1 ) {
      ($verb, $var) = qw(produces output);
    } elsif ( $do_inverse_fft && $iarg == 0 ) {
      ($verb, $var) = qw(takes input);
    }
    barf <<EOF if $verb;
$thisfunction $verb complex $var, so \$$var->dim(0) == 2 should be true,
but it's not (in @{[$arg->info]}: $arg). This is the (real,imag) dimension. Giving up.
EOF
  }

  my ($min_dimensionality, $var) = $rank;
  if( $iarg == 0 ) {
    # The input needs at least $rank dimensions. If this is a backward
    # transform, the input is complex, so it needs an extra dimension
    $min_dimensionality++ if $do_inverse_fft && !$is_native;
    $var = 'input';
  } else {
    # The output needs at least $rank dimensions. If this is a forward
    # transform, the output is complex, so it needs an extra dimension
    $min_dimensionality++ if !$do_inverse_fft && !$is_native_output;
    $var = 'output';
  }
  if ( $arg->ndims < $min_dimensionality ) {
    barf <<EOF;
$thisfunction: The $var needs at least $min_dimensionality dimensions, but
it has fewer. Giving up.
EOF
  }
}

sub matchDimensions_complex {
  my ($thisfunction, $rank, $in, $out) = @_;
  for my $idim (0..$rank) {
    if ( $in->dim($idim) != $out->dim($idim) ) {
      barf <<EOF;
$thisfunction was given input/output matrices of non-matching sizes.
Giving up.
EOF
    }
  }
}

sub matchDimensions_real {
  my ($thisfunction, $rank, $do_inverse_fft, $is_native_output, $in, $out) = @_;
  my ($varname1, $varname2, $var1, $var2);
  if ( !$do_inverse_fft ) {
    # Forward FFT. The input is real, the output is complex. $output->dim(0)
    # == 2, since that's the (real, imag) dimension. Furthermore,
    # $output->dim(1) should be int($input->dim(0)/2) + 1 (Section 2.4 of
    # the FFTW3 documentation)
    ($varname1, $varname2, $var1, $var2) = (qw(input output), $in, $out);
  } else {
    # Backward FFT. The input is complex, the output is real.
    ($varname1, $varname2, $var1, $var2) = (qw(output input), $out, $in);
  }
  my $is_native = !$var2->type->real || $is_native_output; # native complex
  barf <<EOF if int($var1->dim(0)/2) + 1 != $var2->dim($is_native ? 0 : 1);
$thisfunction: mismatched first dimension:
\$$varname2->dim(1) == int(\$$varname1->dim(0)/2) + 1 wasn't true.
$varname1: @{[$var1->info]}
$varname2: @{[$var2->info]}
Giving up.
EOF
  for my $idim (1..$rank-1) {
    if ( $var1->dim($idim) != $var2->dim($idim + ($is_native ? 0 : 1)) ) {
      barf <<EOF;
$thisfunction was given input/output matrices of non-matching sizes.
Giving up.
EOF
    }
  }
}

sub processTypes
{
  my ($thisfunction, $is_native_output, $in, $out) = @_;

  # types:
  #
  # Input and output types must match, and I can only really deal with float and
  # double. If given an output, I refuse to tweak the type of the output,
  # otherwise, I upgrade to float and then to double
  if( $$out->isnull ) {
    if( $$in->type < float ) {
      forceType( $in, (float) );
    }
  } else {
    # I'm given an output. Make sure this is of a type I can work with,
    # otherwise give up
    my $out_type = $$out->type;
    barf <<EOF if $out_type < float;
$thisfunction can only generate 'float' or 'double' output. You gave an output
of type '$out_type'. I can't change this so I give up
EOF
    my $in_type = $$in->type;
    my $in_precision = getPrecision($in_type);
    my $out_precision = getPrecision($out_type);
    return if $in_precision == $out_precision;
    forceType( $in, typeWithComplexity($out_precision, !$in_type->real) );
    forceType( $out, typeWithComplexity($out_precision, !$out_type->real) );
  }
}

sub typeWithComplexity {
  my ($precision, $complex) = @_;
  $complex ? ($precision == 1 ? cfloat : cdouble) :
    $precision == 1 ? float : double;
}

sub getPrecision {
  my ($type) = @_;
  ($type <= float || $type == cfloat) ? 1 : # float
  2; # double
}

sub forceType
{
  my ($x, $type) = @_;
  $$x = convert( $$x, $type ) unless $$x->type == $type;
}

sub getPlan
{
  my ($thisfunction, $rank, $is_real_fft, $do_inverse_fft, $in, $out) = @_;

  # I get the plan ID, check if I already have a plan, and make a new plan if I
  # don't already have one

  my @dims; # the dimensionality of the FFT
  if( !$is_real_fft )
  {
    # complex FFT
    @dims = $in->dims;
    shift @dims if $in->type->real; # ignore first dimension which is (real, imag)
  }
  elsif( !$do_inverse_fft )
  {
    # forward real FFT - the input IS the dimensionality
    @dims = $in->dims;
  }
  else
  {
    # backward real FFT
    # we're given an output, and this is the dimensionality
    @dims = $out->dims;
  }

  my $Nslices = reduce {$a*$b} splice(@dims, $rank);
  $Nslices = 1 unless defined $Nslices;

  my $do_double_precision = ($in->get_datatype == $PDL_F || $in->get_datatype == $PDL_CF)
    ? 0 : 1;
  $_last_do_double_precision = $do_double_precision;

  my $do_inplace = is_same_data( $in, $out );

  # I compute a single plan for the whole set of thread slices. I make a
  # worst-case plan, so I find the worst-aligned thread slice and plan off of
  # it. So if $Nslices>1 then the worst-case alignment is the worse of (1st,
  # 2nd) slices
  my $in_alignment  = get_data_alignment_pdl( $in );
  my $out_alignment = get_data_alignment_pdl( $out );
  my $stride_bytes  = ($do_double_precision ? 8 : 4) * reduce {$a*$b} @dims;
  if( $Nslices > 1 )
  {
    my $in_alignment_2nd  = get_data_alignment_int($in_alignment  + $stride_bytes);
    my $out_alignment_2nd = get_data_alignment_int($out_alignment + $stride_bytes);
    $in_alignment         = $in_alignment_2nd  if $in_alignment_2nd  < $in_alignment;
    $out_alignment        = $out_alignment_2nd if $out_alignment_2nd < $out_alignment;
  }

  my $planID = join('_',
                    $thisfunction,
                    $do_double_precision,
                    $do_inplace,
                    $in_alignment,
                    $out_alignment,
                    @dims);
  if ( !exists $existingPlans{$planID} )
  {
    lock(%existingPlans);
    $existingPlans{$planID} = compute_plan( \@dims, $do_double_precision, $is_real_fft, $do_inverse_fft,
                                            $in, $out, $in_alignment, $out_alignment );
    $_Nplans++;
  }

  return $existingPlans{$planID};
}
