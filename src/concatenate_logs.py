import os
from parse import parse
import numpy as np

# Process data, summarize & format tsv, output
class log_concatenator():
    def __init__(self, fields, types, records):
        self.datapoints = records
        self.fields = fields
        self.types = types

    def output_file(self, out_file_path):
        with open(out_file_path, "w") as out_file:
            out_file.write(self.fields)
            out_file.write(self.types)
            vm_ids = list(self.datapoints.keys())
            vm_ids.sort()
            for vm_id in vm_ids:
                out_file.write(self.datapoints[vm_id])

    @staticmethod
    def read_conn_logs(log_path):
        log_files = next(os.walk(log_path))[2]
        records = {}
        field_line = None
        type_line = None

        for file_name in log_files:
            if not file_name.startswith("conn.log"):
                continue
            with open(os.path.join(log_path, file_name), "r") as file_obj:
                for line in file_obj:
                    if not line.startswith("#"):
                        timestamp = float(line.split("\t")[0])
                        records[timestamp] = line
                    else:
                        if field_line is None and line.startswith("#fields"):
                            field_line = line
                        elif type_line is None and line.startswith("#types"):
                            type_line = line

        return field_line, type_line, records

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--logpath', default='../out/zeek_logs',
                        help='The path for all log files. Logs must be named with \"conn.log\" prefix')
    parser.add_argument('--outfile', default='../out/zeek_logs/Summary.tsv',
                        help='The concatenated output file.')

    args = parser.parse_args()

    log_path = args.logpath
    out_file = args.outfile

    fields, types, records = log_concatenator.read_conn_logs(log_path)
    vm_log = log_concatenator(fields, types, records)

    vm_log.output_file(out_file)

    # vm_log.plot_logs()