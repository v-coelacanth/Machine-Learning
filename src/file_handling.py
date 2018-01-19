import os
# import shutil
import scikits.audiolab
# import sunau
import scipy
import csv
import numpy as np
# from sklearn.naive_bayes import GaussianNB
from scikits.talkbox.features import mfcc
# from glob import glob
import librosa
from librosa import feature as lf


def get_files_in_directory(some_directory):
    """Returns list of files in some_directory."""

    """ If some_directory is a directory, returns a list of files in the
        directory.  Otherwise, complains and exits.
    """
    """ Input variable: at string. """
    """ Output variable: a list of strings. """

    # print "some_directory is", some_directory
    if os.path.isdir(some_directory):
        return os.listdir(some_directory)
    else:
        print "Directory ", some_directory, " not found.  Program exiting."


def create_directory(some_directory):
    """Create directory some_directory."""
    """
        Create the directory some_directory.  If this directory already
        exists, remove it and all its subdirectories and then create it
        again.
    """
    """ Input variable: string """
    """ Output variables: nothing """

    if os.path.isdir(some_directory):
        print "Directory", some_directory, "already exists.  Overwriting."
        shutil.rmtree(some_directory)
        os.mkdir(some_directory)
    else:
        os.mkdir(some_directory)
    return

def convert_string_class_to_int( classification):
    """Return the index of classification in the list of class_names. """
    """ Identify the classification, a string, with an analogous integer
        representing the classification.
        """
    """ Input variable: string """
    """ Output variable: int """

    class_names = ['blues', 'classical', 'country', 'disco', 'hiphop',
        'jazz','metal', 'pop', 'reggae', 'rock']

    if classification in class_names:
        return class_names.index(classification)
    else:
        print classification, " is not a legit classification"
        exit(-1)

def create_new_filename(filename, new_extension):
    """Return filename with new_extension subbed for current file extension."""
    """ It is likely that filename will be the entire path. This doesn't
        matter.  The function will still just sub the new_extension for the
        old extension in the filename. """
    """ Input variables:
            filename - string
            new_extension - string  """
    """ Returns: string """

    first_part, extension = os.path.splitext(filename)
    print "first_part is", first_part
    print "old_extenision is", extension
    print "new filename is", first_part + "." + new_extension
    return first_part + "." + new_extension

def clean_file_list(file_list, extension):
    """ Returns list of all files from file_list with extension 'extension'."""
    """ Step through file_list, removing all filenames from file_list that
        do not have the specified extension.   Return the cleaned list.  """

    for file in file_list:
        if not file.endswith(extension):
            # print "Warning: file", file, " does not have extension", \
            #      extension + ". It will be ignored."
            file_list.remove(file)
    return file_list

def clean_directory_list(directory_list):
    """Return all filenames that are directories from directory_list. """
    """ Remove all filenames that are not directories from directory_list.
        Return revised directory list.
    """
    """ Input variable: a list of strings. """
    """ Output variables: a list of strings. """
    for directory in directory_list:
        if not os.path.isdir(directory):
            # print "Warning: file", directory, "is not a directory."
            # print "It will be ignored."
            directory_list.remove(directory)
    return directory_list
