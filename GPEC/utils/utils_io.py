import pickle
import os

def save_dict(dictionary, path):
    with open(path, 'wb') as handle:
        pickle.dump(dictionary, handle, protocol = pickle.HIGHEST_PROTOCOL)

def load_dict(path):
    with open(path, 'rb') as handle:
        dictionary = pickle.load(handle)
    return dictionary

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def print_args(args):
    for x, y in vars(args).items():
        print('{:<16} : {}'.format(x, y))

def chdir_script(file):
    '''
    Changes current directory to that of the current python script

    args:
        file: "__file__"
    '''
    abspath = os.path.abspath(file)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

def get_filedir(file):
    '''
    returns directory path of current file

    args:
        file: "__file__"
    '''
    abspath = os.path.abspath(file)
    dname = os.path.dirname(abspath)
    return dname

def list_fonts():
    '''
    print fonts available in system
    '''
    import matplotlib.font_manager
    fpaths = matplotlib.font_manager.findSystemFonts()

    for i in fpaths:
        f = matplotlib.font_manager.get_font(i)
        print(f.family_name)


def dict_to_argparse(dictionary):
    '''
    converts a dictionary of variables and values to argparse format

    input:
        dictionary
    return:
        argparse object
    '''
    import argparse
    parser = argparse.ArgumentParser()
    for k, v in dictionary.items():
        parser.add_argument('--' + k, default = v)

    args, unknown = parser.parse_known_args()
    return args
