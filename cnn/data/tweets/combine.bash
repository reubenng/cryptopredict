#/usr/bin/env bash


OUTPUT_CSV=combined.csv

rm "${OUTPUT_CSV}" || true
ls -rt ./*.csv | while read DIRN; do
  cat "${DIRN}" >> "${OUTPUT_CSV}"
done

