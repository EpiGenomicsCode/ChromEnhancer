#! /usr/bin/perl

die "chromtrack_File\tOutput_File\n" unless $#ARGV == 1;
my($input, $output) = @ARGV;
if ($input =~ /.gz$/) { open(IN, "gunzip -c $input |") || die "cant open pipe to $input"; }
else { open(IN, "<$input") or die "Can't open $input for reading!\n"; }

$MAX = 0;
while($line = <IN>) {
	chomp($line);
	@array = split(/\s+/, $line);
	for($x = 0; $x <= $#array; $x++) {
		if($array[$x] > $MAX) { $MAX = $array[$x]; }
	}
}
close IN;

if ($input =~ /.gz$/) { open(IN, "gunzip -c $input |") || die "cant open pipe to $input"; }
else { open(IN, "<$input") or die "Can't open $input for reading!\n"; }
open(OUT, ">$output") or die "Can't open $output for writing!\n";

while($line = <IN>) {
        chomp($line);
        @array = split(/\s+/, $line);
        for($x = 0; $x <= $#array; $x++) {
		if($array[$x] != 0) { $array[$x] = sprintf("%.4f", $array[$x]/$MAX); }
		#$array[$x] /= $MAX;
        }
	print OUT join(" ", @array),"\n";
}
close IN;
close OUT;
