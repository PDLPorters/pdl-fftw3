use PDL::Types;

# when I compute an FFTW plan, it goes here
my %existingPlans;
our $_Nplans = 0;

# This file is included verbatim into the final module via pp_addpm()

# This is a function that sits between the user's call into this module and the
# PP-generated internals. Specifically, this function is called BEFORE and PDL
# threading happens. Here I make sure the FFTW plan exists, or if it doesn't, I
# make it. Thus the PP-based internals can be guaranteed that the plan exists
sub __fft_with_rank
{
  my $rank = shift;

  # first I parse the variables. This is a very direct translation of what PP
  # does normally when no PMCode is given
  my $Nargs = scalar @_;

  my ($in, $out);
  if( $Nargs == 2 )
  {
    # all variables on stack, read in output and temp vars
    ($in, $out) = map {defined $_ ? PDL::Core::topdl($_) : $_} @_;
  }
  elsif( $Nargs == 1 )
  {
    $in = PDL::Core::topdl $_[0];
    $out = PDL::null();
  }
  else
  {
    barf( <<EOF );
fft$rank must be given the input or the input and output as args.
Exactly 1 or 2 arguments are required. Instead I got $Nargs args. Giving up.
EOF
  }


  validateArguments( $rank, $in, $out );
  my $plan = getPlan( $rank, $in, $out );
  barf "fft$rank couldn't make a plan. Giving up\n" unless defined $plan;

  # I now have the arguments and the plan. Go!
  my $internal_function = "PDL::__fft$rank";
  eval( $internal_function . '( $in, $out, $plan );' );
  barf "eval failed calling the internal FFT routine: $@" if $@;

  return $out;





  sub validateArguments
  {
    my ($rank, $in, $out) = @_;

    for my $arg ( $in, $out )
    {
      barf <<EOF unless defined $arg;
fft$rank arguments must all be defined. If you want an auto-growing piddle, use 'null' such as
 fft$rank( $in, $out = null )
Giving up.
EOF

      my $type = ref $arg;
      $type = 'scalar' unless defined $arg;

      barf <<EOF unless ref $arg && ref $arg eq 'PDL';
fft$rank arguments must be of type 'PDL'. Instead I got an arg of
type '$type'. Giving up.
EOF
    }

    # validate dimensionality of the piddles
    for my $arg ( $in, $out )
    {
      next if $arg->dim(0) == 0; # null is allowed for out

      barf <<EOF if $arg->dim(0) != 2;
fft$rank must have dim(0) == 2. This is the (real,imag) dimension.
Giving up.
EOF

      if ( $arg->ndims-1 < $rank )
      {
        barf <<EOF;
Tried to compute a $rank-dimensional FFT, but an array has fewer than $rank dimensions.
Giving up.
EOF
      }
    }

    if ( $out->dim(0) )
    {
      for my $idim(0..$rank)
      {
        if( $in->dim($idim) != $out->dim($idim) )
        {
          barf <<EOF;
fft$rank was given input/output matrices of non-matching sizes.
Giving up.
EOF
        }
      }

      if( $in->get_datatype != $out->get_datatype )
      {
        barf "fft$rank given inputs/outputs of mismatched types. Giving up.";
      }
    }
  }

  sub getPlan
  {
    my ($rank, $in, $out) = @_;

    # I get the plan ID, check if I already have a plan, and make a new plan if I
    # don't already have one

    my $dims   = [$in->dims]; # dims is the dimensionality of the FFT.
    shift @$dims;             # validateDimensions() made sure the dimensions are
    splice @$dims, $rank;     # valid

    # TODO support nembed stuff here
    my $in_dims_embed  = $dims;
    my $out_dims_embed = $out->dim(0) ? $dims : $dims;

    # TODO if not F then D? is this how it works?
    my $do_double_precision = $in->get_datatype != $PDL_F;

    my $planID = join('_', $rank, $do_double_precision, @$dims, @$in_dims_embed, @$out_dims_embed);
    if ( !exists $existingPlans{$planID} )
    {
      $existingPlans{$planID} = compute_plan( $rank, $do_double_precision, $in, $out );
      $_Nplans++;
    }

    return $existingPlans{$planID};
  }
}
