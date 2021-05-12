import torchaudio
import torch
import os

def load_wav(file_path):
    x, samplerate = torchaudio.load_wav(file_path)

    return x

def pickle_dataset(root):
    """
    Files have to be in folders like this:
    root/ -> class1/
            class2/
            ...
            classN/

    Save a pickled file for each class in `pickled` folder. 
    Each pickled file contains audio features of size (#files, length_file, #features).
    """

    classes = os.listdir(root)
    if 'pickled' in classes:
        classes.remove('pickled')
    else:
        os.mkdir(os.path.join(root, 'pickled'))

    for classname in classes:
        print(f"Pickling {classname}")
        filenames = os.listdir(os.path.join(root, classname))
        features = [ load_wav(os.path.join(root, classname, filename))
                for filename in filenames if filename.endswith('.wav')]
        features = torch.cat(features)

        torch.save(features, os.path.join(root, 'pickled', f"{classname}.pt"))
    print("Done")