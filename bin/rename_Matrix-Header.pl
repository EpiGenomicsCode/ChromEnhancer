#! /usr/bin/perl

die "Input_File\tOutput_File\n" unless $#ARGV == 1;
my($input, $output) = @ARGV;
open(IN, "<$input") or die "Can't open $input for reading!\n";
open(OUT, ">$output") or die "Can't open $output for writing!\n";
		
$line = "";
while($line = <IN>) {
	chomp($line);
	@array = split(/\t/, $line);
	if($array[1] eq "Sum") { print OUT "\t$input\n"; }
	else { print OUT $line,"\n"; }
}
close IN;
close OUT;
