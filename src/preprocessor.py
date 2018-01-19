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
from file_handling import *
from parser_demo import *


def extract_features(in_directory, out_file, compute_mfcc, compute_tempo,
              compute_contrast, compute_crossing_rate, compute_fft,
              fft_count, max_instead_of_mean, add_class):
    """Get specified features of files in in_directory. Write to out_file """

    """ Note: in_directory is a directory of directories of genres. """
    """ For each .au file in in_directory's subdirectories, compute the features
        of the file.  If add_classification is True, append its classification
        to all_features.  Write the features (and possible classification to one
        row of the .csv out_file.
    """
    """ Data types expected:
        The variables
                  compute_mfcc, compute_tempo, compute_contrast,
                  compute_crossing_rate, compute_fft, max_instead_of_mean
        are all Booleans.
        If the flag compute_{whatever} is True, extract that feature. Otherwise,
        don't.
        If max_instead_of_mean is True, the mfcc feature extraction will use
        the max of columns instead of the mean of columns.

        The variable
            in_directory
        is a string that contains the full path to the directory
        whose subdirectories contain the .au files.

        The variable
            out_file
        is a string that contains the full path to the file to which we'll
        write the extracted features.  It should be the name of a .csv file.
    """


    counter = 0

    """ Create output directory. """

    """ This just gets the names of the directories, without paths.
        We need the actual paths.  """

    directory_list = os.walk(in_directory)
    directory_list = [dir[0] for dir in directory_list]

    """ VANESSA, THIS IS A PATCH  """
    """ if len(directory_list) > 1:
        directory_list.remove(in_directory) """

    """ Check whether everything in directory_list is actually a directory.
        Discard those that are not.  """

    for directory in directory_list:
        # print "Directories are", directory_list
        file_list = get_files_in_directory(directory)

        """ The classification is the directory name. """
        if add_class:
            classification = os.path.basename(directory)
        else:
            classification = '/validation'

        """ Remove files that don't have .au for the file extension from the list.
        There should just be system files. """

        file_list = clean_file_list(file_list, ".au")

        """ Open the .csv file and get it ready for writing. """

        with open(out_file, 'wb') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            for filename in file_list:
                if filename.endswith(".au"):  # < 10 for testing only
                    path_to_filename = in_directory + classification + \
                          "/" + filename

                    data,  fs,  enc = scikits.audiolab.auread(path_to_filename)
                    counter += 1
                    """ Compute the MFCC."""
                    if compute_mfcc:
                        # data,  fs,  enc = scikits.audiolab.auread(path_to_filename)
                        # print "Adding mfcc to features."
                        ceps, mspec, spec = mfcc(data)

                        """ We are assuming the start and end of each sample may be
                            less genre-specific that the middle.  Discard the first
                            10 percent and last 10 percent.  """
                        middle_of_ceps = abs(ceps[int(len(ceps) * 0.1):
                                                int(len(ceps) * 0.9)])

                        if max_instead_of_mean:
                            extracted_features = np.max(middle_of_ceps, axis = 0)
                        else:
                            extracted_features = np.mean(middle_of_ceps, axis = 0)
                    # print "after compute_mfcc, length of extracted_features is", len(extracted_features)

                    if compute_fft:
                        # print "Computing FFT. fft_count is", fft_count
                        # print "Adding fft to features"
                        fft_features = abs(scipy.fft(data)[:fft_count])
                        """ If feature array already exists, append fft's to
                            it.  Otherwise, create it.
                        """
                        try:
                            if len(extracted_features) >= 1:
                                extracted_features = np.append(extracted_features, fft_features)
                        except:
                            extracted_features = fft_features
                        # print "after fft, len of extracted_features is", len(extracted_features)

                    if compute_tempo or compute_contrast \
                                     or compute_crossing_rate:
                        """ Compute the mean tempo and add as feature. """
                        y, sr = librosa.load(path_to_filename)
                        if compute_tempo:
                            # print "adding tempo to features"
                            tempo, beat_frames = librosa.beat.beat_track(y=y,
                                                                         sr=sr)
                            extracted_features = np.append(extracted_features, tempo)
                            # print "after compute_tempo, len of extracted_features is", len(extracted_features)
                        """ Compute the contrast and add as feature. """
                        if compute_contrast:
                            # print "adding contrat to features"
                            S = np.abs(librosa.stft(y))
                            contrast = librosa.feature.spectral_contrast(S=S,
                                                                         sr=sr)
                            contrast = contrast[:, 0]
                            extracted_features = np.append(extracted_features, contrast)
                            # print "after contrast, len of extracted_features is", len(extracted_features)

                        if compute_crossing_rate:
                            # print "adding crossing_rate to features"
                            crossing_rate = lf.zero_crossing_rate(y)
                            crossing_rate = np.mean(crossing_rate)
                            extracted_features = np.append(extracted_features,
                                                      crossing_rate)

                    if add_class:
                            # print "adding class to features"
                            extracted_features = np.append(extracted_features,
                                  convert_string_class_to_int(classification))

                    """ Now append features for this file to matrix of
                        features for all files.  Make sure all elements of this
                        feature set are numbers. If not, do not add the feature
                        set. So, effectively, data with values of NaN or Inf is
                        discarded. """
                    if np.isnan(extracted_features).any() or \
                                np.isinf(extracted_features).any():
                        """ If data is invalid, don't add this line.  """
                        counter = counter - 1
                        print "Found line with NaN or inf.  Omitting."
                    elif counter == 1:    # First time through.  Create matrix.
                        all_features = np.asmatrix(extracted_features)
                    else:
                        try:
                            if len(all_features > 0):
                                all_features = np.append(all_features,
                                             np.asmatrix(extracted_features),
                                             axis=0)
                        except:
                            print "all_features does not exist"
                            all_features = np.asmatrix(extracted_features)
                    """ if counter >= 1:
                        print "ALL MFCC length is", len(all_features)
                        print "ALL MFCC shape is", all_features.shape """


    print "Writing out features to", out_file

    with open(out_file, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(all_features.tolist())


# def find_fft(in_directory, out_file, classification):
def find_fft(in_directory, out_file):
    """Get FFT of each file in in_directory and write to out_file """

    """ For each .au file in in_directory, compute the FFT of the file.  Append
        its classification and write the FFT and classification to one row of the
        .csv out_file.
    """
    counter = 0

    """ Create output directory.  The will overwrite any old version. """
    """  HEY VANESSA.   SHOULD YOU DO SOMETHING ABOUT THIS?  """
    #create_directory(out_directory)
    file_list = get_files_in_directory(in_directory)
    print "IN FFT file_list is", file_list
    """ Open the .csv file and get it ready for writing. """
    with open(out_file, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for filename in file_list:
            if filename.endswith(".au"):  # < 10 for testing only

                """ Compute the FFT and write the result to a file. """
                data,  fs,  enc = scikits.audiolab.auread(
                                  in_directory + "/" + filename)
                counter += 1
                print "FILENAME: ", filename
                # print "data is", data
                # print "length of data is", len(data)
                print "fs is", fs
                print "enc is", enc
                fft_features = abs(scipy.fft(data)[:1000])
                fft_features = np.append( fft_features, classification)
                print "type of fft_features", type(fft_features)
                print "features are", fft_features
                """ Take the au extension off the filename and add on the extension
                    fft.  Then write fft_features to
                    out_directory/revised_filename. """
                '''revised_filename = out_directory + "/" \
                                + create_new_filename(filename, "fft")
                print "REVISED filename is", revised_filename'''
                csvwriter.writerow(fft_features)


def main():

    # Blues is 0.  Classical is 1.
    # country is 2, disco is 3, hiphop is 4, jazz is 5, metal is 6
    # pop is 7, reggae is 8, rock is 9
    # class_names = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz',
    #               'metal', 'pop', 'reggae', 'rock']
    # print "length of class_names is", len(class_names)

    current_directory = os.path.dirname(os.path.realpath(__file__))
    parent_directory = os.path.dirname(current_directory)

    """ Determine which kind of preprocessor the user wants to run. """
    preprocessor_type, validation_only = get_args()

    if preprocessor_type == 'custom':
        """ Set up flags for the custom extractor. """
        compute_mfcc = True
        compute_tempo = True
        compute_contrast = True
        compute_crossing_rate = False
        compute_fft = True
        fft_count = 13
        max_instead_of_mean = True
        if not validation_only:  # Then preprocess the training data too.
            """ Run the extractor for the training data.   Output will end up in
                parent_directory/output/train.csv. """
            add_class = True
            """in_directory = "/Users/Vanessa/UNM School/Machine Learning/" + \
                            "Project 3/Code/data/genres/"""
            """out_file = "/Users/Vanessa/UNM School/Machine Learning/" + \
                       "Project 3/Code/output/validate_crossing.csv"""
            in_directory = parent_directory + "/data/genres/"
            print "in_directory for test data is", in_directory
            out_file = "../output/train.csv"
            print "Running extraction for test data.  This will take a while. "
            extract_features(in_directory, out_file, compute_mfcc, compute_tempo,
                             compute_contrast, compute_crossing_rate,
                             compute_fft, fft_count, max_instead_of_mean,
                             add_class)


        """ Run the extractor for the validation data. Output will end up in
            parent_directory/output/validate.csv. """
        add_class = False
        # in_directory = parent_directory + "/something/validation/"
        in_directory = parent_directory + "/something/"
        print "in_directory is", in_directory
        out_file = "../output/validate.csv"
        print "Running custom extraction for validation data."
        extract_features(in_directory, out_file, compute_mfcc, compute_tempo,
                         compute_contrast, compute_crossing_rate, compute_fft,
                         fft_count, max_instead_of_mean,
                         add_class)
    elif preprocessor_type == "mfcc":
        compute_mfcc = True
        compute_tempo = False
        compute_contrast = False
        compute_crossing_rate = False
        compute_fft = False
        fft_count = 0
        max_instead_of_mean = True
        if not validation_only:  # Then preprocess the training data too.
            """ Run the extractor for the training data.   Output will end up in
                ../output/train.csv. """
            add_class = True
            """in_directory = "/Users/Vanessa/UNM School/Machine Learning/" + \
                            "Project 3/Code/data/genres/"""
            """out_file = "/Users/Vanessa/UNM School/Machine Learning/" + \
                       "Project 3/Code/output/validate_crossing.csv"""
            in_directory = "../data/genres/"
            out_file = "../output/train.csv"
            print "Running extraction for MFCC test data.  This will take a while. "
            extract_features(in_directory, out_file, compute_mfcc, compute_tempo,
                      compute_contrast, compute_crossing_rate, compute_fft, fft_count,
                      max_instead_of_mean,
                      add_class)

        """ Run the extractor for the validation data. Output will end up in
            ../output/validate.csv. """
        add_class = False
        # in_directory = "../../something"
        # in_directory = parent_directory + "/something/validation/"
        in_directory = parent_directory + "/something/"
        out_file = "../output/validate.csv"
        print "Doing extraction for MFCC validation data."
        extract_features(in_directory, out_file, compute_mfcc, compute_tempo,
                  compute_contrast, compute_crossing_rate, compute_fft, fft_count,
                  max_instead_of_mean,
                  add_class)
    elif preprocessor_type == 'fft':
        compute_mfcc = False
        compute_tempo = False
        compute_contrast = False
        compute_crossing_rate = False
        compute_fft = True
        fft_count = 1000
        max_instead_of_mean = False

        if not validation_only: # Then process the training data.
            print "Doing FFT extraction for training set. This will take a while. "
            in_directory = "../data/genres/"
            out_file = "../output/train.csv"
            add_class = True
            # find_fft(in_directory, out_file, classification)
            # find_fft(in_directory, out_file)
            extract_features(in_directory, out_file, compute_mfcc, compute_tempo,
                      compute_contrast, compute_crossing_rate, compute_fft, \
                      fft_count, max_instead_of_mean, add_class)
        # in_directory = "../../something"
        add_class = False
        in_directory = parent_directory + "/something/"

        out_file = "../output/validate.csv"
        print "Doing extraction for FFT with validation data."
        extract_features(in_directory, out_file, compute_mfcc, compute_tempo,
                  compute_contrast, compute_crossing_rate, compute_fft, fft_count,
                  max_instead_of_mean,
                  add_class)
    else:
        print "Incorrect preprocessor type", preprocessor_type
        print "Program exiting"
        sys.exit(-1)


    """
    add_class = False

    if add_class == False:
        in_directory = "/Users/Vanessa/UNM School/Machine Learning/" + \
                       "Project 3/Code/something"
    else:
        in_directory = "/Users/Vanessa/UNM School/Machine Learning/" + \
                        "Project 3/Code/data/genres/"

    '''Get the directory where you want to put the preprocessed files.'''
    out_file = "/Users/Vanessa/UNM School/Machine Learning/" + \
               "Project 3/Code/output/validate_crossing.csv"

    compute_tempo = True
    compute_contrast = True
    compute_crossing_rate = True
    compute_fft = True
    fft_count = 13
    max_instead_of_mean = True
    extract_features(in_directory, out_file, compute_tempo,
              compute_contrast, compute_crossing_rate, compute_fft, fft_count,
              max_instead_of_mean,
              add_class)
"""
if __name__ == "__main__":
    main()
