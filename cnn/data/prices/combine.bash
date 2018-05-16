#/usr/bin/env bash


OUTPUT_CSV=combined.csv

rm "${OUTPUT_CSV}" || true
ls -rt ./*.csv | while read DIRN; do
  if [ ! -f "${OUTPUT_CSV}" ]; then
    cat "${DIRN}" >> "${OUTPUT_CSV}"
  else  
    tail -n+2 "${DIRN}" >> "${OUTPUT_CSV}"
  fi
done

