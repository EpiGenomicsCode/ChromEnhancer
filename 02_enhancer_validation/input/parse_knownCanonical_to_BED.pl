#! /usr/bin/perl

die "TAB_File\tWindow_Size(bp)\tOutput_BED\n" unless $#ARGV == 2;
my($input, $SIZE, $output) = @ARGV;
open(IN, "<$input") or die "Can't open $input for reading!\n";
open(OUT, ">$output") or die "Can't open $output for writing!\n";

# #filter: not (refSeqStatus.status = 'Inferred') and refSeqStatus.mol = 'mRNA'
# #hg38.knownCanonical.chrom	hg38.knownCanonical.chromStart	hg38.knownCanonical.chromEnd	hg38.kgXref.geneSymbol	hg38.refGene.strand	hg38.refGene.score
# chr1	169849630	169893896	SCYL3	-	0
# chr1	169795039	169854080	FIRRM	+	0
# chr1	27612063	27635185	FGR	-	0
# chr1	196652042	196747504	CFH	+	0
# chr1	24356998	24413782	STPG1	-	0
# chr1	24415801	24472976	NIPAL3	+	0
# chr1	33007985	33036883	AK2	-	0

$line = "";
while($line = <IN>) {
	chomp($line);
	next if($line =~ "track" || $line =~ "#");
	@array = split(/\t/, $line);
	next if($array[0] =~ "_");
	$START = $array[1] - ($SIZE / 2);
	$STOP = $START + $SIZE;
	if($array[4] eq "-") {
		$START = $array[2] - ($SIZE / 2);
		$STOP = $START + $SIZE;
	}
	print OUT "$array[0]\t$START\t$STOP\t$array[3]\t0\t$array[4]\n";
}
close IN;
close OUT;

