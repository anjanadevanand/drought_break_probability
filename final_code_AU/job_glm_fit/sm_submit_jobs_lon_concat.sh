#!/bin/bash

glm_dir=/g/data/w97/ad9701/p_prob_analysis/model_output/awra/GLM_results_model4/

for iWeek in 4
do

# for start_lat in -19.025
# for start_lat in -11.025 -13.025 -15.025 -17.025 -19.025 -21.025
# for start_lat in -23.025 -25.025 -27.025 -29.025 -31.025 -33.025 
# for start_lat in -35.025 -37.025 -39.025 -41.025 -43.025
# for start_lat in -17.025                    #submitted to n81 on March 31
# TO DO: for start_lat in -19.025 -21.025 -23.025 -25.025 -27.025 -29.025 
# TO DO: for start_lat in -31.025 -33.025 -35.025 -37.025 -39.025 -41.025 -43.025 #-9.025
# for start_lat in -23.025 -25.025 -27.025 -29.025 -31.025 -33.025 -35.025 -37.025 -39.025 -41.025 -43.025

for start_lat in -23.025 -27.025 -29.025 -31.025 -33.025 -35.025 #-43.025 #-9.025
do
echo $start_lat

cp job_script_lon_concat_template.pbs job_script_lon_concat.pbs
sed -i 's/iWeek=6/iWeek='${iWeek}'/g' job_script_lon_concat.pbs
sed -i 's/start_lat/start_lat='${start_lat}'/g' job_script_lon_concat.pbs
sed -i 's|glm_dir|glm_dir='${glm_dir}'|g' job_script_lon_concat.pbs
qsub job_script_lon_concat.pbs

done
done
