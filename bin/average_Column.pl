#! /usr/bin/perl/

die "Input_File\tOutput_File\n" unless $#ARGV == 1;
my($input, $output) = @ARGV;
open(IN, "<$input") or die "Can't open $input for reading!\n";
open(OUT, ">$output") or die "Can't open $output for reading!\n";

$SUM = $COUNT = 0;

while($line = <IN>) {
	chomp($line);
	@array = split(/\t/, $line);
	next if($array[0] eq "");
	$SUM += $array[1];
	$COUNT++;
}
close IN;
$AVG = $SUM / $COUNT;
print OUT "$input\t$AVG\n";
close OUT;

