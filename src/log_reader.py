from functools import total_ordering
import csv

'''
    LogRecord 它做了什么：
    1 记录一条指定数据
    2 提取timestamp, uid
    3 计算unique_id
    4 提供 记录条目内和记录条目之间的基本操作（比大小， 获取指定field，获取长度）
'''

@total_ordering
class LogRecord:
    values = []
    hash_elements = []
    timestamp = 0.0

    def __init__(self, timestamp, values, hash_elements=[]):
        assert isinstance(timestamp, float), "[LogRecord] Argument 'timestamp' type must be Float."
        assert isinstance(values, list), "[LogRecord] Argument 'values' type must be List."
        self.values = values
        self.timestamp = timestamp
        self.hash_elements = hash_elements

    def get_timestamp(self):
        return self.timestamp

    def __getitem__(self, item):
        assert isinstance(item, int), "[LogRecord] Argument 'item' type must be INT."
        return self.values[item]

    def __setitem__(self, key, value):
        assert isinstance(key, int), "[LogRecord] Argument 'key' type must be INT."
        assert isinstance(key, str), "[LogRecord] Argument 'value' type must be String."
        self.values[key] = value

    def unique_id(self):        # unique_id 相同 代表 我们定义事件相同
        return hash(self)

    def uid(self):              # uid 相同 代表 是同一个 Connection
        return self.values[1]

    def add_value(self, newValue):
        self.values.append(newValue)

    def __hash__(self):
        hash_elements = self.values
        if len(self.hash_elements) != 0:
            new_elements = []
            for i, val in enumerate(self.hash_elements):
                new_elements.append(self.values[val])
            hash_elements = new_elements

        h = 0x35254173
        for val in hash_elements:
            h ^= hash(val)
        return h

    def __len__(self):
        return len(self.values)

    def __str__(self):
        s = ""
        for i, val in enumerate(self.values):
            s += val
            if (i != len(self.values) - 1):
                s += ", "
        return s

    # Comparison
    def __eq__(self, other):
        assert isinstance(other, LogRecord), "[LogRecord] Comparison only between LogRecords."
        return self.timestamp == other.get_timestamp()

    def __lt__(self, other):
        assert isinstance(other, LogRecord), "[LogRecord] Comparison only between LogRecords."
        return self.timestamp < other.get_timestamp()


class LogReader:
    fields = {}
    records = []

    def __init__(self, path, hash_rule=[]):
        with open(path, 'r') as tsvfile:
            line = tsvfile.readline()
            while len(line) != 0 and not line.startswith("#fields"):
                line = tsvfile.readline()

            # Record fields
            if (len(line) == 0):
                raise TypeError

            split_line = line.split("\t", 1)
            line = split_line[1]
            split_line = line.split("\n", 1)
            line = split_line[0]
            split_line = line.split("\t")
            self.factory_id = len(split_line)
            for i, field in enumerate(split_line):
                self.fields[field] = i

            line = tsvfile.readline()
            while line.startswith("#"):
                line = tsvfile.readline()

            # Set hash_rule
            if len(hash_rule) != 0:
                new_rule = []
                for rule in hash_rule:
                    new_rule.append(self.fields[rule])
                hash_rule = new_rule

            while len(line) != 0:
                if not line.startswith("#"):
                    split_line = line.split("\n", 1)
                    line = split_line[0]
                    sp_line = line.split("\t")
                    ts = float(sp_line[0])
                    lr = LogRecord(ts, sp_line, hash_rule)
                    self.records.append(lr)

                line = tsvfile.readline()
                # print(line)

        # Sort records
        self.records.sort()

    def __len__(self):
        return len(self.records)

    def __str__(self):
        s = ""
        for i, val in enumerate(self):
            s += str(val)
            if i != len(self) - 1:
                s += '\n'
        return s

    def __iter__(self):
        return iter(self.records)

    # def __next__(self):
    #     if self.iter_m < len(self.records):
    #         rec = self.records[self.iter_m]
    #         self.iter_m += 1
    #         return rec
    #     else:
    #         raise StopIteration

    def __getitem__(self, item):
        return self.records[item]

    def get_fields(self):
        return self.fields

    def add_field(self, newKey):
        result = self.factory_id
        self.fields[newKey] = self.factory_id
        self.factory_id += 1
        return result

if __name__ == "__main__":
    # Template log reading
    lr = LogReader("../zeek/conn.log", ["id.orig_h", "id.resp_h"])
    for i in lr:
        print(i)
        print(i.unique_id())
