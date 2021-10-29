import sys
import numpy as np

new_path = '/home/566/ad9701/drought_probability/'
if new_path not in sys.path:
    sys.path.append(new_path)
import my_data_process_funcs as dprocess
    
# creating information required for weekly accumulations
week_names = list(np.arange(2, 13, 2))
ndays = list(np.arange(14, 13*14, 14))
start_day = ['1911-01-02'] * len(week_names)

end_day = [#'2020-05-31',   #1
           '2020-05-24',   #2
           #'2020-05-31',   #3
           '2020-05-24',   #4
           '2020-05-10',   #6
           '2020-04-26',   #8
           '2020-03-29',   #10
           '2020-03-29'    #12
          ]
time_chunk = [#364,   #1 
              378,   #2
              #378,   #3
              336,   #4
              378,   #6
              336,   #8
              350,   #10
              336    #12
             ]

day_len_dict = dict(zip(week_names, ndays))
start_day_dict = dict(zip(week_names, start_day))
end_day_dict = dict(zip(week_names, end_day))
time_chunk_dict = dict(zip(week_names, time_chunk))

data_names = ['day_len', 'start_day', 'end_day', 'time_chunk']
values = [day_len_dict, start_day_dict, end_day_dict, time_chunk_dict]
allWeek_dict = dict(zip(data_names, values))

allWeek_allSets = {}
for i in range(2, 13, 2):
    field = 'week' + str(i) + '_sets'
    allWeek_allSets.update({field: dprocess.create_week_sets(nWeek = i, allWeek_dict = allWeek_dict)})