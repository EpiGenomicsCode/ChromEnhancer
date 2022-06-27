
# TODO:

1. Figure out Data
    * Big oof, not sure yet will need to look deeper into Bichrom construct data
2. Build Network
    * A Transformer of some kind
    * Something like: https://github.com/yifengtao/genome-transformer
    * Or maybe: https://towardsdatascience.com/bringing-bert-to-the-field-how-to-predict-gene-expression-from-corn-dna-9287af91fcf8
3. Run Network
    * Not Yet
4. Collect Results

# Resources

1. Download the container: docker build -t <tag_name> .
2. Run the container: docker run -it --rm --gpus all --name pytorch -v $PWD:/work <tag_name> 
3. start jupyter notebook: jupyter notebook --ip 0.0.0.0 --no-browser --allow-root