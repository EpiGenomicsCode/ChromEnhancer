#! /usr/bin/perl

srand(1);

die "#_Rows\t%_True\tOutput_File" unless $#ARGV == 2;
my($ROWS, $PROB, $output) = @ARGV;
open(OUT, ">$output") or die "Can't open $output for reading!\n";

for($x = 0; $x < $ROWS; $x++) {
#	$LABEL = int(rand(2));
	if(rand(1) < $PROB) { $LABEL = 1; }
	else { $LABEL = 0; }
	print OUT "chrZ\t$x\t$x\tchrZ:$x\t$LABEL\t.\n";
}

