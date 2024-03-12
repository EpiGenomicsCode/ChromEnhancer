#! /usr/bin/perl

die "BED_File\tOutput_File\n" unless $#ARGV == 1;
my($input, $output) = @ARGV;
open(IN, "<$input") or die "Can't open $input for reading!\n";
open(OUT, ">$output") or die "Can't open $output for writing!\n";

while($line = <IN>) {
	chomp($line);
	next if($line =~ "track");
	@array = split(/\t/, $line);
	print OUT int($array[4]),"\n";
}
close IN;
close OUT;
