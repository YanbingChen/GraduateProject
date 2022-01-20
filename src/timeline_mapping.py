import numpy as np

class TimelineMap:
    data_points = []
    events = {}     # Dictionary like: {"-5145503496319726845": 1, "-586275386119677184": 2, ...}
    events_count = {}
    interval = [float("inf"), float("-inf")]

    def __init__(self, slice_num=1000):
        self.slice_num = slice_num
        self.id_factory = 1

    '''
        event_id: （我们定义的 相同事件 应有相同的event_id）
    '''
    def insert(self, event_id, ts):
        if not event_id in self.events:
            self.events[event_id] = self.id_factory
            self.events_count[self.id_factory] = 0
            self.id_factory += 1

        event_id = self.events[event_id]
        datapoint = [event_id, ts]
        self.data_points.append(datapoint)
        self.events_count[event_id] += 1

        if ts > self.interval[1]:
            self.interval[1] = ts
        if ts < self.interval[0]:
            self.interval[0] = ts

    def get_event_num(self):
        return len(self.events)

    def get_event_count(self, event_id):
        if event_id in self.events:
            event_id = self.events[event_id]
        else:
            return -1

        count = self.events_count[event_id]
        return count

    '''
        从 timeline_map 中提取 event_id 指定的事件，生成长度为slice_num的timeline,
        其中，事件发生的次数被限制在 max_dp_num 以内
        :return time_map: 生成的timeline map
                precision: timeline的精度 
                max_dp_num: timeline所包含事件的个数
    '''
    def resolve(self, slice_num=0, max_dp_num=0, event_id=[]):
        if slice_num == 0:
            slice_num = self.slice_num
        if max_dp_num == 0:
            max_dp_num = int(self.slice_num / 2)

        data_points = []
        interval = [float("inf"), float("-inf")]

        # 如果用户指定 event_id 则只生成由指定 event_id 组成的 timeline map
        if len(event_id) != 0:
            new_id = []
            for val in event_id:
                new_id.append(self.events[val])
            event_id = new_id
            for point in self.data_points:
                if point[0] in event_id:
                    data_points.append(point)
                    if interval[0] > point[1]:
                        interval[0] = point[1]
                    if interval[1] < point[1]:
                        interval[1] = point[1]
        else:
            data_points = self.data_points.copy()
            interval = self.interval.copy()

        # 对选出的数据进行排序
        data_points.sort(key=lambda x: x[1])        # Sort by timestamp

        # Truncate
        if len(data_points) > max_dp_num:
            del data_points[max_dp_num:]
            interval[1] = data_points[max_dp_num-1][1]
        else:
            max_dp_num = len(data_points)

        # Calculate precision
        precision = (interval[1] - interval[0]) / (slice_num - 1)
        if precision == 0.0:
            precision = 0.001
        element_num = slice_num
        offset = interval[0]

        # Construct output timeMap
        time_map = np.zeros(element_num)
        for point in data_points:
            ts = point[1]
            index = int((ts - offset) / precision)
            time_map[index] += 1.0

        return time_map, precision, max_dp_num
