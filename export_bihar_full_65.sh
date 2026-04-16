#!/usr/bin/env bash
set -euo pipefail

PROJECT="${1:-weather-analysis-461411}"

cd "$(dirname "$0")"

# Bihar already has legacy Jun-Sep bands locally. This starts only the missing
# Jan-May and Oct-Dec exports, which can later be merged into a full Jan-Dec
# stack without re-exporting the existing 25/20 bands.
python3 export_missing_months_state_data.py \
  --state "Bihar" \
  --project "${PROJECT}" \
  --years "2020,2021,2022,2023,2024" \
  --months "1,2,3,4,5,10,11,12" \
  2>&1 | tee export_bihar_missing_months.log
