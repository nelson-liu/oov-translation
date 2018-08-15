#!/usr/bin/env bash
set -e

# This script runs an OOV solver on USC HPC. It'll probably only
# work if you're a member of the lc_dmm group / can access Nelson's
# working directory at /home/nlg-05/nelsonfl/

export LC_ALL=en_US.utf8
export LANG=$LC_ALL

# Parse Arguments
# File with OOVs to translate, one per line.
SOURCEPATH=$1
echo "Path to OOV file: ${SOURCEPATH}"

# Path to the model to use to translate OOVs.
MODELPATH=$2
echo "Path to saved model: ${MODELPATH}"

# Path to write output file (<source>\t<predicted translation>) to.
OUTPUTPATH=$3
echo "Path to write output file: ${OUTPUTPATH}"

# OPTIONAL: Number of processes to use to predict in parallel, if supported.
DEFAULT_NJOBS=1
NJOBS=${4:-$DEFAULT_NJOBS}
echo "Number of processes to use to predict in parallel, if supported: ${NJOBS}"

# Set up
# Save the current path variable
OLD_PATH=$PATH
# Prepend Nelson's Python installation to the PATH, and activate
# the oov environment.
export PATH=/home/nlg-05/nelsonfl/miniconda3/bin/:$PATH
echo "Activating OOV Python environment"
source activate oov

# CD to the OOV folder, since some code uses relative paths.
echo "Running OOV model"
pushd .
RELATIVESOURCEPATH=`realpath --relative-to /home/nlg-05/nelsonfl/Github/nfliu_oov "${SOURCEPATH}"`
RELATIVEMODELPATH=`realpath --relative-to /home/nlg-05/nelsonfl/Github/nfliu_oov "${MODELPATH}"`
RELATIVEOUTPUTPATH=`realpath --relative-to /home/nlg-05/nelsonfl/Github/nfliu_oov "${OUTPUTPATH}"`
cd /home/nlg-05/nelsonfl/Github/nfliu_oov
# Run the model to predict translations for OOVs.
python /home/nlg-05/nelsonfl/Github/nfliu_oov/scripts/run/run_oov_solver.py \
       --predict_from_path="${RELATIVEMODELPATH}" \
       --eval_path="${RELATIVESOURCEPATH}" \
       --eval_output_path="${RELATIVEOUTPUTPATH}.tmp" \
       --n_jobs="${NJOBS}"
popd

echo "Producing output file with proper format at ${RELATIVEOUTPUTPATH}"
# Paste together the source file and tmp output (file with one predicted
# translation per line) to generate a final output file
paste "${SOURCEPATH}" "${OUTPUTPATH}.tmp" > "${OUTPUTPATH}"

echo "Cleaning up."
# Clean up
# Remove the tmp output file.
rm "${OUTPUTPATH}.tmp"
# Deactivate the environment.
source deactivate
# Reset the PATH.
export PATH=$OLDPATH
