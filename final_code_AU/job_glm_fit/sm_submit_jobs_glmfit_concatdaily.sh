#!/bin/bash

#files containing the identified soil mositure droughts (binary) and the associated required thresholds of P-E-Q 
#these file names are relative to 'main_dir' specified in the code

sm_events_files=sm_droughts/events_[1-2]*.nc
sm_deficit_files=sm_droughts/sm_deficits_[1-2]*.nc
glm_dir=/g/data/w97/ad9701/p_prob_analysis/model_output/awra/GLM_results_model4/

#sm_deficit_files=sm_droughts/deficits_timeseries_subset/sm_events_[1-2]*.nc
#glm_dir=GLM_results_model4_subset/sm_deficits_rollData_drought_timeseries/
#splitting jobs into three to speed up results: 1911 to 1947, 1948 to 1983, 1984 to 2020
#not subsetting time

#start_yr=1911
end_yr=2020

#splitting jobs by space 2deg x 2deg grid boxes. The .025 in lat-lons is used to ensure no overlap between adjascent spatial slices

for iWeek in 8 #4 #12 #8 #4 #8 12 #2 #6 12 #2 #6 12
do

#for start_lat in -35.025 #-31.025 -33.025

#for start_lat in -11.025 -13.025 -15.025 -17.025 -19.025 
#for start_lat in -21.025 -23.025 -25.025 -27.025 -29.025 -31.025
for start_lat in -33.025 -35.025 -37.025
#for start_lat in -39.025 -41.025 -43.025

#for start_lat in -27.025 -29.025 -31.025 -33.025 -35.025 -37.025 -39.025 -41.025 -43.025
#for start_lat in -25.025
# TO DO: for start_lat in -19.025 -21.025 -23.025 -25.025 -27.025 -29.025 
# TO DO: for start_lat in -31.025 -33.025 -35.025 -37.025 -39.025 -41.025 -43.025 #-9.025
# for start_lat in -23.025 -25.025 -27.025 -29.025 -31.025 -33.025 -35.025 -37.025 -39.025 -41.025 -43.025

#for start_lat in -41.025 #-43.025 #-9.025
do
#for start_lon in 123.025
for start_lon in 111.025 113.025 115.025 117.025 119.025 121.025 123.025 125.025 127.025 129.025 131.025 133.025 135.025 137.025 139.025 141.025 143.025 145.025 147.025 149.025 151.025 153.025 #111.025
do

# subset time later if needed
#1990 #1975 2000 #1911 1940 1969 1998 #1911 #1940 1969 1998 #1911 #1940 1969 1998
#echo ${start_yr}
#end_yr=$((${start_yr}+24))
#if (( end_yr > 2020 )); then
#end_yr=2020
#fi
#end_yr=2020
#echo ${end_yr}
#end_yr=start_yr

end_lat=$(awk "BEGIN{ print $start_lat - 2 }")
end_lon=$(awk "BEGIN{ print $start_lon + 2 }")

if (( $(echo "$start_lat > -10" |bc -l) )); then end_lat=-11.025; fi
if (( $(echo "$start_lon < 112" |bc -l) )); then end_lon=113.025; fi

echo $start_lon
echo $end_lon
echo $start_lat
echo $end_lat

cp job_script_template_sm_allyr.pbs job_script.pbs
sed -i 's/iWeek=6/iWeek='${iWeek}'/g' job_script.pbs
#sed -i 's/start_yr=1900/start_yr='${start_yr}'/g' job_script.pbs
sed -i 's/end_yr=1900/end_yr='${end_yr}'/g' job_script.pbs

sed -i 's/start_lat/start_lat='${start_lat}'/g' job_script.pbs
sed -i 's/end_lat/end_lat='${end_lat}'/g' job_script.pbs
sed -i 's/start_lon/start_lon='${start_lon}'/g' job_script.pbs
sed -i 's/end_lon/end_lon='${end_lon}'/g' job_script.pbs

sed -i 's/glm_6_0000/glm_'${iWeek}'_'${start_lat}'_'${start_lon}'/g' job_script.pbs
sed -i 's|glm_dir|glm_dir='${glm_dir}'|g' job_script.pbs
sed -i 's|sm_events_files|sm_events_files='${sm_events_files}'|g' job_script.pbs
sed -i 's|sm_deficit_files|sm_deficit_files='${sm_deficit_files}'|g' job_script.pbs
jid1=$(qsub job_script.pbs)

cp job_script_concat_template.pbs job_script_concat.pbs
sed -i 's/iWeek=6/iWeek='${iWeek}'/g' job_script_concat.pbs
sed -i 's/start_yr=1900/start_yr=1911/g' job_script_concat.pbs     #*******Fixed start_yr to 1911******
sed -i 's/end_yr=1900/end_yr='${end_yr}'/g' job_script_concat.pbs
 
sed -i 's/start_lat/start_lat='${start_lat}'/g' job_script_concat.pbs
sed -i 's/end_lat/end_lat='${end_lat}'/g' job_script_concat.pbs
sed -i 's/start_lon/start_lon='${start_lon}'/g' job_script_concat.pbs
sed -i 's/end_lon/end_lon='${end_lon}'/g' job_script_concat.pbs
 
sed -i 's/glm_6_0000/glm_'${iWeek}'_'${start_lat}'_'${start_lon}'/g' job_script_concat.pbs
sed -i 's|glm_dir|glm_dir='${glm_dir}'|g' job_script_concat.pbs
jid2=$(qsub -W depend=afterok:$jid1 job_script_concat.pbs)
#qsub job_script_concat.pbs

done
done
done
