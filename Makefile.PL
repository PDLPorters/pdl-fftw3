use strict;
use warnings;
use ExtUtils::MakeMaker;
use PDL::Core::Dev;

my @package = qw(FFTW3.pd FFTW3 PDL::FFTW3);
my %descriptor = pdlpp_stdargs(\@package);

$descriptor{VERSION_FROM}   = 'FFTW3.pd';
$descriptor{OPTIMIZE}  = '-O2';
$descriptor{CCFLAGS} .= '--std=gnu99';

# I support single and double precision FFTW calls, so both fftw and fftw3f
push @{$descriptor{LIBS}  },  `pkg-config --libs fftw3f fftw3`;

$descriptor{INC} = '' unless defined $descriptor{INC};
$descriptor{INC} .= `pkg-config --cflags fftw3f fftw3`;

$descriptor{depend} = { 'FFTW3.pm' => join(' ', qw(template_complex.c template_real.c
                                                   compute_plan_template.xs
                                                   FFTW3_header_include.pm)) };


WriteMakefile( %descriptor );

sub MY::postamble
{
  pdlpp_postamble(\@package);
}

