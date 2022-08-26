#! /usr/bin/perl

die "BED_File\tOutput_File_Name\n" unless $#ARGV == 1;
my($input, $output) = @ARGV;
open(IN, "<$input") or die "Can't open $input for reading!\n";

@ARRAY = ();
while($line = <IN>) {
	chomp($line);
	@array = split(/\t/, $line);
	push(@ARRAY, {line => $line});
}
close IN;

$LIMIT = 500000;
$INDEX = $counter = 0;
open(OUT, ">$output\_$INDEX\.bed") or die "Can't open $output\_$INDEX\.bed for writing!\n";
for($x = 0; $x <= $#ARRAY; $x++) {
        print OUT $ARRAY[$x]{'line'},"\n";
	$counter++;
	if($counter == $LIMIT) {
		$counter = 0;
		$INDEX++;
		close OUT;
		open(OUT, ">$output\_$INDEX\.bed") or die "Can't open $output\_$INDEX\.bed for writing!\n";
	}
}
close OUT;
