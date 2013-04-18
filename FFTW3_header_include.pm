use PDL::Types;

# when I compute an FFTW plan, it goes here
my %existingPlans;

# these are for the unit tests
our $_Nplans = 0;
our $_last_do_double_precision;

# This file is included verbatim into the final module via pp_addpm()

# This is a function that sits between the user's call into this module and the
# PP-generated internals. Specifically, this function is called BEFORE and PDL
# threading happens. Here I make sure the FFTW plan exists, or if it doesn't, I
# make it. Thus the PP-based internals can be guaranteed that the plan exists
sub __fft_with_rank
{
  my $rank         = shift;
  my $thisfunction = shift;

  my $do_inverse_fft = $thisfunction =~ /^i/;

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
    if( $in->is_inplace )
    {
      $out = $in;
      $in->set_inplace(0);
    }
    else
    {
      $out = PDL::null();
    }
  }
  else
  {
    barf( <<EOF );
$thisfunction must be given the input or the input and output as args.
Exactly 1 or 2 arguments are required. Instead I got $Nargs args. Giving up.
EOF
  }

  validateArguments( $rank, $thisfunction, $in, $out );
  processTypes( $thisfunction, \$in, \$out );
  my $plan = getPlan( $thisfunction, $rank, $do_inverse_fft, $in, $out );
  barf "$thisfunction couldn't make a plan. Giving up\n" unless defined $plan;

  # I now have the arguments and the plan. Go!
  my $internal_function = "PDL::__fft$rank";
  eval( $internal_function . '( $in, $out, $plan );' );
  barf "eval failed calling the internal FFT routine: $@" if $@;

  return $out;





  sub validateArguments
  {
    my ($rank, $thisfunction, $in, $out) = @_;

    for my $arg ( $in, $out )
    {
      barf <<EOF unless defined $arg;
$thisfunction arguments must all be defined. If you want an auto-growing piddle, use 'null' such as
 $thisfunction( \$in, \$out = null )
Giving up.
EOF

      my $type = ref $arg;
      $type = 'scalar' unless defined $arg;

      barf <<EOF unless ref $arg && ref $arg eq 'PDL';
$thisfunction arguments must be of type 'PDL'. Instead I got an arg of
type '$type'. Giving up.
EOF
    }

    # validate dimensionality of the piddles
    for my $arg ( $in, $out )
    {
      next if $arg->isnull; # null is allowed for out

      barf <<EOF if $arg->dim(0) != 2;
$thisfunction must have dim(0) == 2. This is the (real,imag) dimension.
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

    if ( ! $out->isnull )
    {
      for my $idim(0..$rank)
      {
        if( $in->dim($idim) != $out->dim($idim) )
        {
          barf <<EOF;
$thisfunction was given input/output matrices of non-matching sizes.
Giving up.
EOF
        }
      }
    }
  }

  sub processTypes
  {
    my ($thisfunction, $in, $out) = @_;

    # types:
    #
    # Input and output types must match, and I can only really deal with float and
    # double. If given an output, I refuse to tweak the type of the output,
    # otherwise, I upgrade to float and then to double

    my $targetType;

    if( $$out->isnull )
    {
      # Output is generated. I thus only worry about the type of the input. If
      # It's not one of the types I like, upgrade to a float
      my $in_type  = $$in->type;
      $targetType = ( $in_type < float ) ? (float) : $in_type;

      forceType( $in, $targetType );
    }
    else
    {
      # I'm given an output. Make sure this is of a type I can work with,
      # otherwise give up

      my $out_type = $$out->type;

      barf <<EOF if $out_type < float;
$thisfunction can only generate 'float' or 'double' output. You gave an output
of type '$out_type'. I can't change this so I give up
EOF

      $targetType = ( $out_type < float ) ? (float) : $out_type;

      forceType( $in,  $targetType );
      forceType( $out, $targetType );
    }


    sub forceType
    {
      my ($x, $type) = @_;
      $$x = convert( $$x, $type ) unless $$x->type == $type;
    }
  }

  sub getPlan
  {
    my ($thisfunction, $rank, $do_inverse_fft, $in, $out) = @_;

    # I get the plan ID, check if I already have a plan, and make a new plan if I
    # don't already have one

    my $dims   = [$in->dims]; # dims is the dimensionality of the FFT.
    shift @$dims;             # validateDimensions() made sure the dimensions are
    splice @$dims, $rank;     # valid

    # TODO support nembed stuff here
    my $in_dims_embed  = $dims;
    my $out_dims_embed = $out->isnull ? $dims : $dims;

    my $do_double_precision = $in->get_datatype == $PDL_F ? 0 : 1;
    $_last_do_double_precision = $do_double_precision;

    my $do_inplace = is_same_data( $in, $out );
    my $planID = join('_',
                      $rank,
                      $do_double_precision,
                      $do_inverse_fft,
                      $do_inplace,
                      @$dims, @$in_dims_embed, @$out_dims_embed);
    if ( !exists $existingPlans{$planID} )
    {
      $existingPlans{$planID} = compute_plan( $rank, $do_double_precision, $do_inverse_fft, $in, $out );
      $_Nplans++;
    }

    return $existingPlans{$planID};
  }
}
