import argparse
import os
import shutil
from zipfile import ZipFile


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", type=str, help="Team name for the submission.", default="GRT")
    parser.add_argument("--track", type=str, choices=['constrained', 'unconstrained'], default='constrained')
    parser.add_argument("--desc", type=str, help="Short description of the model + explainer", default="QE")
    parser.add_argument("-e", "--explainer", type=str, required=True, help="Dir where the explanations are saved")
    parser.add_argument("-s", "--save", default='submission.zip', help="Submission zip filename")
    args = parser.parse_args()
    dname = '.'
    fname = args.save

    # copy sentence-level predictions, and mt and src explanations to the informed folder
    shutil.copy(os.path.join(args.explainer, 'sentence_scores.txt'), 'sentence.submission')
    shutil.copy(os.path.join(args.explainer, 'aggregated_mt_scores.txt'), 'target.submission')
    shutil.copy(os.path.join(args.explainer, 'aggregated_source_scores.txt'), 'source.submission')

    # create metadata.txt
    # The first line contains your team name.
    # The second line must be either constrained or unconstrained, indicating the submission track.
    # The third line contains a short description (2-3 sentences) of the system you used to generate the results.
    f = open('metadata.txt', 'w')
    f.write(args.team + '\n')
    f.write(args.track + '\n')
    f.write(args.desc)
    f.close()

    # save .zip
    print('Saving submission to {}'.format(fname))
    obj = ZipFile(fname, 'w')
    obj.write('sentence.submission')
    obj.write('target.submission')
    obj.write('source.submission')
    obj.write('metadata.txt')
    obj.close()
