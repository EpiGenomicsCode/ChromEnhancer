#! /usr/bin/perl

die "CDT.GZ_File\tScaling_Factor\tOutput_File\n" unless $#ARGV == 2;
my($input, $scale, $output) = @ARGV;
if ($input =~ /.gz$/) { open(IN, "gunzip -c $input |") || die "cant open pipe to $input"; }
else { open(IN, "<$input") or die "Can't open $input for reading!\n"; }
open(OUT, ">$output") or die "Can't open $output for writing!\n";

$lineCount = 0;
while($line = <IN>) {
	chomp($line);
	next if($line =~ "YORF");
	@fullArray = split(/\t/, $line);
	@ARRAY = ((0) x 100);
	$currentIndex = 0;
	$currentCount = 0;
	$currentSum = 0;
	for($x = 2; $x <= $#fullArray; $x++) {
		$currentSum += $fullArray[$x];
		if($currentCount > 9) { 
			$ARRAY[$currentIndex] = ($currentSum * $scale);
			$currentIndex++;
			$currentCount = $currentSum = 0;
		}
		$currentCount++;
	}
	$ARRAY[$currentIndex] = ($currentSum * $scale);
	print OUT join(" ", @ARRAY),"\n";
	$lineCount++;
#	if($lineCount < 100){ print join(" ", @ARRAY),"\n"; }
}
close IN;
close OUT;
