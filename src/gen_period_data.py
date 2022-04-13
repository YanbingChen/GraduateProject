import shutil

import numpy as np
from timeline_mapping import TimelineMap
from log_reader import LogReader, LogRecord

from robust_period_drive import robust_period_drive_multiprocess

import json
import os

'''
    gen_period_data.py
    Steps for all periodicity-detection-module
    1. Read log file from LOG_FILE
    2. Feed log for period detection algorithm (specified by PERIOD_DRIVE)
    3. Output result
'''
# ------------ Params --------------
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--logfile', default='../out/zeek_logs/Summary.tsv',
                    help='Input log files. ')
parser.add_argument('--outfile', default='../out/period_output/SummaryWithPeriod.tsv',
                    help='Output file with period fields appended')

parser.add_argument('--tempLogPath', default='../out/period_output',
                    help='Temporary log output path. Generated during running.')

args = parser.parse_args()

LOG_FILE = args.logfile
out_file = args.outfile
temp_path = args.tempLogPath
INTEREST_FIELDS = ["id.orig_h", "id.resp_h"]
PERIOD_DRIVE = robust_period_drive_multiprocess

# ------ Pre process & Timeline mapping ------
logreader = LogReader(LOG_FILE, INTEREST_FIELDS)
tm = TimelineMap()
for i in logreader:
    tm.insert(i.unique_id(), i.get_timestamp())

event_period_flag, event_period_data = \
    PERIOD_DRIVE(logreader, tm, path=temp_path, cpu_worker_num=16)

# print(event_period_data)
with open(os.path.join(temp_path, "period_data.json"), 'w') as f:
    json.dump(event_period_data, f)
with open(os.path.join(temp_path, "period_flag.json"), 'w') as f:
    json.dump(event_period_flag, f)


# Load period data from file
with open(os.path.join(temp_path, "period_flag.json"), 'r') as f:
    event_period_flag = json.load(f)
with open(os.path.join(temp_path, "period_data.json"), 'r') as f:
    event_period_data = json.load(f)

hash_fields = logreader.get_hash_fields()
period_exist_flag = False

with open(LOG_FILE, "r") as log_f, \
        open(out_file, "w") as out_f:
    for line in log_f:
        line = line[:-1]
        if line.startswith("#fields"):
            fields = line.split("\t")
            if fields[-1] == "period_length4":
                period_exist_flag = True
                break
            # Process fields
            out_f.write(line + "\tperiod\tperiod_length0\tperiod_length1\tperiod_length2"
                               "\tperiod_length3\tperiod_length4\n")
        elif not line.startswith("#"):
            # Process lines
            linesplit = line.split('\t')
            unique_id = LogRecord.gen_uniqueID(linesplit, hash_fields)
            out_f.write(line + "\t")
            period_flag = event_period_flag[unique_id]
            out_f.write(str(period_flag))
            out_f.write("\t")
            periods = event_period_data[unique_id]
            for i in range(5):
                if i < len(periods):
                    out_f.write(str(periods[i]))
                else:
                    out_f.write(str(0.0))

                if i != 4:
                    out_f.write("\t")
            out_f.write("\n")


