#! /usr/bin/perl

die "chrom.sizes\tWindow_Size(bp)\tOutput_Name\n" unless $#ARGV == 2;
my($input, $WINDOW, $output) = @ARGV;
open(IN, "<$input") or die "Can't open $input for reading!\n";

#chr1	248956422
#chr2	242193529
#chr3	198295559
#chr4	190214555
#chr5	181538259

$line = "";
while($line = <IN>) {
	chomp($line);
	@SIZE = split(/\t/, $line);

	open(OUT, ">$output\_$SIZE[0]\_$WINDOW\.bed") or die "Can't open $output\_$SIZE[0]\_$WINDOW\bp for writing!\n";
	for($x = 0; $x <= $SIZE[1] - $WINDOW; $x += $WINDOW) {
		$START = $x;
		$STOP = $x + $WINDOW;
		print OUT "$SIZE[0]\t$START\t$STOP\t$SIZE[0]:$START-$STOP\t.\t.\n";

	}
	close OUT;
}
close IN;
