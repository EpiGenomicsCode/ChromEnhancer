#!/usr/bin/perl

die "chromHMM-TAB_File\tOutput_File\n" unless $#ARGV == 1;
my($input, $output) = @ARGV;
open(IN, "<$input") or die "Can't open $input for reading!\n";
open(OUT, ">$output") or die "Can't open $output for writing!\n";


# Define the labels to count
my @labels = qw(EnhA1 EnhA2 EnhBiv EnhG1 EnhG2 EnhWk Het Quies ReprPC ReprPCWk TssA TssBiv TssFlnk TssFlnkD TssFlnkU Tx TxWk ZNF/Rpts);

# Initialize counts for each label
my %label_counts;
foreach my $label (@labels) {
    $label_counts{$label} = 0;
}

# Total count for calculating percentages
my $total_count = 0;

# Process each line in the file
while (my $line = <IN>) {
    chomp $line;
    my @columns = split /\s+/, $line;

    # Increment the count for the label in column 4
    if (defined $columns[3] && exists $label_counts{$columns[3]}) {
        $label_counts{$columns[3]}++;
        $total_count++;
    }
}

# Close the file
close IN;

# Print the counts for each label
foreach my $label (@labels) {
    my $count = $label_counts{$label};
    my $percentage = $total_count > 0 ? ($count / $total_count) * 100 : 0;
    printf OUT "$label\t$percentage\n";
}
