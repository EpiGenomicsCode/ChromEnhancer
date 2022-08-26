#! /usr/bin/perl

die "BED_File\tScore\tOutput_File_Name\n" unless $#ARGV == 2;
my($input, $score, $output) = @ARGV;
open(IN, "<$input") or die "Can't open $input for reading!\n";
open(OUT, ">$output") or die "Can't open $output for writing!\n";

while($line = <IN>) {
	chomp($line);
	next if($line =~ "track");
	@array = split(/\t/, $line);
	$array[4] = $score;
	print OUT join("\t", @array),"\n";
}
close IN;
close OUT;
