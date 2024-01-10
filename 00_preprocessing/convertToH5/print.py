import h5py
import glob

def print_all_datasets(filename):
    # Open the h5 file
    with h5py.File(filename, 'r') as f:
        # Print out all dataset names in the file
        for dataset_name in f.keys():
            print(f"\tdataset:{dataset_name}\tshape:{f[dataset_name].shape}")

def print_all_groups(filename):
    # Open the h5 file
    with h5py.File(filename, 'r') as f:
        # Recursively print out all groups in the file
        def print_groups(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"\tgroup:{name}")
                obj.visititems(print_groups)
        
        f.visititems(print_groups)


files = glob.glob("./*/*h5")

for file in files:
    print(file)
    print_all_datasets(file)
    print_all_groups(file)
