use strict;
use warnings;
use ExtUtils::MakeMaker;
use PDL::Core::Dev;

##############################
# Try to use Alien::FFTW3 - but if it's not present
# fall back to pkg-config.  This is so that 
# a Debian package won't have to include Alien::FFTW3.
my $cflags;
my $libs;
if( eval "require Alien::FFTW3" ) {

    ## Ensure at least version 3.3; die if we can't get it.
    Alien::FFTW3->VERSION(3.3);
    
    my $p = Alien::FFTW3->precision;
    unless($p->{'d'} and $p->{'f'}) {
	die "PDL::FFTW3 - needs both double-precision and single-precision fftw3 libraries\n\t(libfftw3 and libfftw3f). Alien::FFTW3 found only ".(join(",",keys %$p))."\n";
    }
    
    $cflags = Alien::FFTW3->cflags;
    $libs   = Alien::FFTW3->libs;
} else {
    ## we don't have Alien::FFTW3 - try to use pkg-config
    printf "Alien::FFTW3 not found.  Trying pkg-config...\n";
    $cflags = `pkg-config --cflags fftw3f fftw3`;
    $libs   = `pkg-config --libs   fftw3f fftw3`;

    unless ($cflags && $libs) {
	die "Couldn't find the FFTW3 installation (no Alien::FFTW3 - used POSIX pkg-config).\n    Giving up.\n";
    }
}


my @package = qw(FFTW3.pd FFTW3 PDL::FFTW3);
my %descriptor = pdlpp_stdargs(\@package);

$descriptor{VERSION_FROM}   = 'FFTW3.pd';
$descriptor{OPTIMIZE}  = '-O2';
$descriptor{CCFLAGS} .= '--std=gnu99';

# I support single and double precision FFTW calls, so both fftw and fftw3f
push @{$descriptor{LIBS}  },  $libs;

$descriptor{INC} = '' unless defined $descriptor{INC};
$descriptor{INC} .= $cflags;

$descriptor{depend} = { 'FFTW3.pm' => join(' ', qw(template_complex.c template_real.c
                                                   compute_plan_template.xs
                                                   FFTW3_header_include.pm)) };

# This Alien::FFTW3 requirement is actually optional. pkg-config is used if this
# isn't available
$descriptor{PREREQ_PM} = {'PDL'=>2.006,
                          'Alien::FFTW3'=>0};

$descriptor{NO_MYMETA} = 0;

$descriptor{NAME}   = "PDL::FFTW3";
$descriptor{AUTHOR} = "Dima Kogan <dima\@secretsauce.net>, Craig DeForest <deforest\@boulder.swri.edu>";
$descriptor{ABSTRACT} = "PDL interface to the Fastest Fourier Transform in the West";
$descriptor{META_ADD} = { resources => {
    homepage => 'http://github.com/dkogan/PDL-FFTW3',
    repository => 'git://gitub.com/dkogan/PDL-FFTW3.git',
    bugtracker => 'http://github.com/dkogan/PDL-FFTW3/issues'
			  }
};

WriteMakefile( %descriptor );

sub MY::postamble
{
  pdlpp_postamble(\@package);
}

