

import argparse
import sys

def get_args():
    """Parse and return user's command line arguments."""
    """ Parse the user's chosen extraction type from the command line.  Also
        parse the optional argument --v which is a flag to skip extracting the
        test data.  Return the user's choice of processor as well as flag v
        that is set True if the user has chosen option --v and False otherwise.
        Return data types:  a string and a boolean
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("preprocessor_type",
                        help= "choices are fft, mfcc, and custom")
    parser.add_argument("--v",
                        help="only extract from the validation set" +
                        "not the training set", action = 'store_true')
    args = parser.parse_args()
    if args.v:
        print "Only extractinng from the validation set."
    else:
        args.v = False

    """ If the user does not choose a proper preprocssor type, complain
        and exit. """
    if not args.preprocessor_type in {"fft", "mfcc", "custom"}:
        print "\nChoose fft, mfcc, or custom as your preprocessor."
        print "Your choice", args.preprocessor_type, "is not a valid choice."
        print "Program exiting."
        sys.exit(-1)

    # print "returning", args.preprocessor_type, args.v
    return args.preprocessor_type, args.v

def main():

    get_args()

if __name__ == "__main__":
    main()
