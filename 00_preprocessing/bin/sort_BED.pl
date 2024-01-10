#! /usr/bin/perl

die "BED_File\tOutput_File_Name\n" unless $#ARGV == 1;
my($input, $output) = @ARGV;
open(IN, "<$input") or die "Can't open $input for reading!\n";
open(OUT, ">$output") or die "Can't open $output for writing!\n";

@ARRAY = ();
while($line = <IN>) {
	chomp($line);
	@array = split(/\t/, $line);
	push(@ARRAY, {chr => $array[0], start => $array[1], line => $line});
}
close IN;
@TEMP = sort { $$a{'start'} <=> $$b{'start'} } @ARRAY;
@SORT = sort { $$b{'chr'} cmp $$a{'chr'} } @TEMP;

$currentChr = "";
for($x =0 ; $x <= $#SORT; $x++) {
        print OUT $SORT[$x]{'line'},"\n";
}
close OUT;
