#!/bin/bash

for doy in 182 262 #178 #{2..366..1}
do
cp job_script_template_perccalc.pbs job_script.pbs
sed -i 's|doy_sel|doy_sel='${doy}'|g' job_script.pbs
sed -i 's|perc_day0|perc_day'${doy}'|g' job_script.pbs
qsub job_script.pbs
done
