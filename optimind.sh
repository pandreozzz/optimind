#!/bin/bash
#
# Launch OptimiNd
# Submits batch jobs for LUT tuning, statistics computation, and ND calculation
#
# Usage:
#   ./launch_script.sh -f config.json [OPTIONS]
#   ./launch_script.sh config.json [OPTIONS]
#
# Options:
#   -f, --file-config PATH    Path to JSON configuration file (required)
#   -m, --mem GB              Memory per job in GB (default: 96)
#   -c, --cpus N              Number of CPUs per job (default: 32)
#   -p, --procs N             Number of parallel processes (default: 8)
#   -z, --zero-year YEAR      Base year for array indexing (default: 2000)
#   -i, --ini-year N          Initial year index (default: 3)
#   -e, --end-year N          End year index (default: 20)
#   -t, --no-tune             Skip tuning step
#   -s, --no-stats            Skip statistics computation
#   -n, --no-nd               Skip ND computation
#   -x, --noexec
#   -h, --help                Show this help message
#

set -euo pipefail

SCRIPT_DIR=$(realpath "$(dirname "$0")")

BASE_LOGDIR="${BASE_LOGDIR:-"${SCRIPT_DIR}/logs"}"

BASH_EXEC="${BASH_EXEC:-/bin/bash}"

# Python-ware
PYTHON_EXEC="${PYTHON_EXEC:-python3}"
TUNE_MODULE="pyoptimind.tune"
STATS_MODULE="pyoptimind.stats"
ND_MODULE="pyoptimind.nd"

# Work directory
WORKDIR="${SCRIPT_DIR}/workdir"
mkdir -p "$WORKDIR"

# Default parameters
# For doing e.g. the 1950s, set zero_year=1950, ini_year=1, end_year=9
INI_YEAR=3
END_YEAR=20
ZERO_YEAR=2000
# Default optimal for 1/1 degrees, 3-hourly. Follow optimal for 1.5/1.5 degrees and 3/3 degrees resolution.
JOBMEM_GB=96 #64 12
NUM_CPUS=32 #24 8
NUM_PROCS=8 #6 4

# Feature switches
TUNE_SWITCH=true
STATS_SWITCH=true
ND_SWITCH=true

CONFIG_PATH=""
POSITIONAL_ARGS=()

# Help message
print_help() {
    sed -n '/^#/!q;/^#!/!p' "$0" | sed 's/^ *# *//'
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      print_help
      exit 0
      ;;
    -f|--file-config)
      CONFIG_PATH="${2%/}"
      shift 2
      ;;
    -m|--mem)
      JOBMEM_GB="$2"
      shift 2
      ;;
    -c|--cpus)
      NUM_CPUS="$2"
      shift 2
      ;;
    -p|--procs)
      NUM_PROCS="$2"
      shift 2
      ;;
    -z|--zero-year)
      ZERO_YEAR="$2"
      shift 2
      ;;
    -i|--ini-year)
      INI_YEAR="$2"
      shift 2
      ;;
    -e|--end-year)
      END_YEAR="$2"
      shift 2
      ;;
    -t|--no-tune)
      TUNE_SWITCH=false
      shift
      ;;
    -s|--no-stats)
      STATS_SWITCH=false
      shift
      ;;
    -n|--no-nd)
      ND_SWITCH=false
      shift
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done


# Validate config path
if [ -z "$CONFIG_PATH" ]; then
    if [ ${#POSITIONAL_ARGS[@]} -eq 1 ]; then
        CONFIG_PATH="${POSITIONAL_ARGS[0]}"
    else
        echo "   Error: Configuration file path must be provided"
        echo "   Use: -f|--file-config PATH or as positional argument"
        exit 1
    fi
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "  Error: Configuration file not found: $CONFIG_PATH"
    exit 1
fi

CONFIG_FILE="${CONFIG_PATH##*/}"
CONFIG_NAME="${CONFIG_FILE%.*}" #same as CONFIG_FILE, without extension
CONFIG_BASEDIR="${CONFIG_PATH%/*}" #where the CONFIG_FILE is located
TUNE_TYPE="${CONFIG_BASEDIR##*/}"

# Setup logging
LOGDIR="${BASE_LOGDIR}/${TUNE_TYPE}/logs_${CONFIG_NAME}"
mkdir -p "$LOGDIR"

# Create temporary work directory
TMP_DIRECTORY=$(mktemp -d "${WORKDIR}/tune_lut_XXXXXXXXXX")
trap "rm -rf $TMP_DIRECTORY" EXIT

echo "---------------------------------------------------------------"
echo "  tune-lut Launch Script"
echo "---------------------------------------------------------------"
echo ""
echo "Configuration:"
echo "   Config file: $CONFIG_PATH"
echo "   Tuning type: $TUNE_TYPE"
echo "   Config name: $CONFIG_NAME"
echo ""
echo "Job Parameters:"
echo "   Memory: ${JOBMEM_GB} GB"
echo "   CPUs: $NUM_CPUS"
echo "   Processes: $NUM_PROCS"
echo "   Year Range: $INI_YEAR to $END_YEAR (offset from $ZERO_YEAR)"
echo ""
echo "Steps to perform:"
echo "   Tuning: $([ "$TUNE_SWITCH" = true ] && echo 'yes' || echo 'no')"
echo "   Statistics: $([ "$STATS_SWITCH" = true ] && echo 'yes' || echo 'no')"
echo "   ND Computation: $([ "$ND_SWITCH" = true ] && echo 'yes' || echo 'no')"
echo ""
echo "Temporary Directory: $TMP_DIRECTORY"
echo "Log Directory: $LOGDIR"
echo ""
echo "---------------------------------------------------------------"
echo ""

# Display configuration file content
echo "Configuration file content:"
echo "---------------------------------------------------------------"
cat "$CONFIG_PATH"
echo "---------------------------------------------------------------"
echo ""

# Generate dummy wrapper scripts
WRAPPER_TUNE="${TMP_DIRECTORY}/wrapper_tune.sh"
WRAPPER_STATS="${TMP_DIRECTORY}/wrapper_stats.sh"
WRAPPER_ND="${TMP_DIRECTORY}/wrapper_nd.sh"

# Generate dummy scripts for tasks
cat << EOF > "$WRAPPER_TUNE"
#!/bin/bash
set -euo pipefail
this_year=\${SLURM_ARRAY_TASK_ID:- }
if [ \$this_year==" " ];
then
  this_year=\${1:- }
fi
this_year=\$((\$this_year+$ZERO_YEAR))
$PYTHON_EXEC -m "$TUNE_MODULE" --year \$this_year --config "$CONFIG_PATH" --logdir "$LOGDIR" --num-procs $NUM_PROCS
EOF
cat << EOF > "$WRAPPER_STATS"
#!/bin/bash
set -euo pipefail
$PYTHON_EXEC -m "$STATS_MODULE" "$LOGDIR" --ini-year $(($INI_YEAR+$ZERO_YEAR)) --end-year $(($END_YEAR+$ZERO_YEAR))
EOF
cat << EOF > "$WRAPPER_ND"
#!/bin/bash
set -euo pipefail
$PYTHON_EXEC -m "$ND_MODULE" \$((\$SLURM_ARRAY_TASK_ID+$ZERO_YEAR)) "$CONFIG_PATH" "$LOGDIR" "$JOBMEM_GB" "$NUM_PROCS"
EOF

if [ "$TUNE_SWITCH" = true ]; then
  echo "Tuning nd..."
  if command -v sbatch &> /dev/null; then
    sbatch --array=${INI_YEAR}-${END_YEAR}%20 --wait -t 06:00:00 -c $NUM_CPUS --chdir="$TMP_DIRECTORY" --mem="${JOBMEM_GB}GB" --output="${LOGDIR}/job_output_%a.txt" \
      --error="${LOGDIR}/job_output_%a.txt" "$WRAPPER_TUNE" &
    wait -n
    case $? in
      "0")
        echo "Done!"
        ;;
      "127")
        echo "Warning: there were no jobs   on wait!"
        ;;
      *)
        echo  "Tuning procedure failed for at least one job! Exiting."
        exit 1
        ;;
    esac
  else
    echo "SLURM not available. running locally."
    for ((this_year=INI_YEAR; this_year<=END_YEAR; this_year++));
    do
      echo "Launching year $ZERO_YEAR+$this_year..."
      $BASH_EXEC "$WRAPPER_TUNE" $this_year 2>&1 | tee "${LOGDIR}/job_output_${this_year}.txt"
    done
    if [ $? -ne 0 ]; then
      echo "Tuning procedure failed for at least one job! Exiting."
      exit 1
    else
      echo "Done!"
    fi
  fi

else
  echo "Skipping tuning!"
fi

if [ "$STATS_SWITCH" = true ]; then
  echo "Computing stats..."
  if command -v sbatch &> /dev/null; then
    sbatch --wait -t 00:05:00 -c 4 --chdir="$TMP_DIRECTORY" --mem="4GB" --output="${LOGDIR}/tune_stats_output.txt" --error="${LOGDIR}/tune_stats_output.txt" "$WRAPPER_STATS" &
    wait -n
    case $? in
      "0")
        echo "Done!"
        ;;
      "127")
        echo "Warning: there were no jobs on wait!"
        ;;
      *)
        echo "Failed computing stats! Exiting."
        exit 1
        ;;
    esac
  else
    echo "SLURM not available. running locally."
    $BASH_EXEC "$WRAPPER_STATS" 2>&1 | tee "${LOGDIR}/tune_stats_output.txt"
    if [ $? -ne 0 ]; then
      echo "Failed computing stats! Exiting."
      exit 1
    else
      echo "Done!"
    fi
  fi
else
echo "Skipping stats!"
fi

if [ "$ND_SWITCH" = true ]; then
  if command -v sbatch &> /dev/null; then
    echo "Computing tuned nd for all years..."
    sbatch --array=${INI_YEAR}-${END_YEAR}%20 --wait -t 01:00:00 -c $NUM_CPUS --chdir="$TMP_DIRECTORY" --mem="${JOBMEM_GB}GB" --output="${LOGDIR}/compute_nd_${ZERO_YEAR}+%a_output.txt" --error="${LOGDIR}/compute_nd_${ZERO_YEAR}+%a_output.txt" "$WRAPPER_ND" &
    wait -n
    case $? in
      "0")
        echo "Done!"
        ;;
      "127")
        echo "Warning: there were no jobs on wait!"
        ;;
      *)
        echo "Computing Nd failed for at least one job!!"
        exit 1
        ;;
    esac
  else
    echo "SLURM not available. running locally."
    for ((this_year=INI_YEAR; this_year<=END_YEAR; this_year++));
    do
      echo "Launching year $ZERO_YEAR+$this_year..."
      SLURM_ARRAY_TASK_ID=$this_year $BASH_EXEC "$WRAPPER_ND" 2>&1 | tee "${LOGDIR}/compute_nd_${ZERO_YEAR}+${this_year}_output.txt"
      if [ $? -ne 0 ]; then
        echo "Computing Nd failed for year $((this_year+ZERO_YEAR))!"
        exit 1
      fi
    done
  fi
else
  echo "Skipping tuned nd computation!"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  All steps completed successfully!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "  Results available at: $LOGDIR"
echo ""
