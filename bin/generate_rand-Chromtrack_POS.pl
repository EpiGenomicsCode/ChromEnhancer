#!/usr/bin/perl

# Check if the correct number of command-line arguments are provided
if (@ARGV != 2) {
    die "Usage: $0 <input_file> <output_file>\n";
}

# Extract input and output filenames from command-line arguments
my ($input_file, $output_file) = @ARGV;

# Open input file for reading
open(my $fh_in, "<", $input_file) or die "Can't open $input_file: $!";

# Open output file for writing
open(my $fh_out, ">", $output_file) or die "Can't create $output_file: $!";

# Process each line of the input file
while (my $line = <$fh_in>) {
    chomp($line);

    # Split the line into columns
    my @columns = split(/\s+/, $line);

    # Extract the 4th column value
    my $fourth_column = $columns[4];

    # Generate a vector based on the value of the 4th column
    my @vector;
    if ($fourth_column == 0) {
        @vector = (0) x 100; # Fill with zeros
    } elsif ($fourth_column == 1) {
        @vector = map { rand(1) } (1..100); # Fill with random numbers between 0 and 1
    } else {
        die "Invalid value in the 4th column: $fourth_column";
    }

    # Output the vector as a tab-delimited row
    print $fh_out join(" ", @vector), "\n";
}

# Close input and output files
close($fh_in);
close($fh_out);

print "Output written to $output_file\n";

