# Script Table of Contents

## matrix_values_to_binary.py
```
usage: matrix_values_to_binary.py [-h] -i input_fn -o output_fn [-t value]

Rewrite matrix of decimals as matrix of 0/1 values.

options:
  -h, --help            show this help message and exit
  -i input_fn, --input input_fn
                        the tab-delimited file of column values (first column is id column)
  -o output_fn, --output output_fn
                        the tab-delimited file of rounded column values
  -t value, --threshold value
                        the value threshold for calling 0/1
```

## merge_and_replace_columns.py
```
usage: merge_and_replace_columns.py [-h] -i input_fn -o output_fn -a value -b value [--intersect | --union]

Merge and replace two columns in a tab-delimited file.

options:
  -h, --help            show this help message and exit
  -i input_fn, --input input_fn
                        the tab-delimited file
  -o output_fn, --output output_fn
                        the tab-delimited output file
  -a value, --col-a value
                        the first 0-indexed column to merge
  -b value, --col-b value
                        the second 0-indexed column to merge
  --intersect
  --union
```
