#! /usr/bin/perl

die "BED_File\tOutput_File\n" unless $#ARGV == 1;
my($input, $output) = @ARGV;
open(IN, "<$input") or die "Can't open $input for reading!\n";
open(OUT, ">$output") or die "Can't open $output for reading!\n";

while($line = <IN>) {
	chomp($line);
	@array = split(/\t/, $line);
	$START = $array[1];
	$STOP = $array[2];
	if($array[3] eq "") { $array[3] = "$array[0]:$START\-$STOP"; }
	$START -= 100;
	$STOP -= 100;
	for($x = 0; $x <= 10; $x++) {
		if($START >= 0) { print OUT "$array[0]\t$START\t$STOP\t$array[3]\t1\t.\n"; }
		$START += 20;
		$STOP += 20;
	}
}
close IN;
close OUT;

#sacCer3 genome size
#1.2163423E7
