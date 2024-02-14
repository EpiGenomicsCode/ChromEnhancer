#! /usr/bin/perl

die "BED_File\tWindow_distance(bp)\tWindow_Size(bp)\tOutput_File\n" unless $#ARGV == 3;
my($input, $WINDOW, $SIZE, $output) = @ARGV;
open(IN, "<$input") or die "Can't open $input for reading!\n";
open(OUT, ">$output") or die "Can't open $output for reading!\n";

while($line = <IN>) {
	chomp($line);
	@array = split(/\t/, $line);
	$START = $array[1];
	$STOP = $array[2];
	# Frameshift window upstream window size
	$newSTART = $START - $WINDOW - $SIZE;
	$newSTOP = $newSTART + $SIZE;

	$ID = "$array[0]\:$newSTART\-$newSTOP";
	if($START >= 0) { print OUT "$array[0]\t$newSTART\t$newSTOP\t$ID\t0\t.\n"; }

	# Frameshift window downstream window size
	$newSTART = $STOP + $WINDOW;
	$newSTOP = $newSTART + $SIZE;

        $ID = "$array[0]\:$newSTART\-$newSTOP";
	print OUT "$array[0]\t$newSTART\t$newSTOP\t$ID\t0\t.\n";
}
close IN;
close OUT;

#sacCer3 genome size
#1.2163423E7
