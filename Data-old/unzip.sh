# Remove all the folders 
rm -rf 220802_DATA 220803_CelllineDATA 230124_CHR-Data_Sequence 

# unzip all folder in the current directory
for f in *.zip; do unzip "$f"; done


# go into each folder and untar all the files in the folder then remove the file
for f in */; do
    cd $f
    for g in *.tar.gz; do tar -xzvf "$g"; done
    rm *.tar.gz
    for g in *.tar; do tar -xvf "$g"; done
    rm *.tar
    # if there is a subfolder in the folder then go into the subfolder and untar all the files
    for g in */; do
        cd $g
        for h in *.gz; do gzip -d "$h"; done
        cd ..
    done
    cd ..
done

# create the distribution folders
for Chrom in "A549" "MCF7" "HepG2" "K562"; do
    # tar all files in the subdirectories with Chrom in the name and don't have .tar in the name
    tar --exclude="*gz" -czvf $Chrom.tar.gz */*/$Chrom*
done





