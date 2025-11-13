import pyarrow.parquet as pq
import duckdb
import pandas
import os

class I24MotionData:
    trajectories_all_db_prefix = "trajectories_all"
    trajectories_subset_db_prefix = "trajectories_subset"
    road_lane_lookup = {
        1: [-1, -2, -3, -4],
        2: [-1, -2, -3, -4]
    }
    def __init__(self, road_id, timestamp_min, timestamp_max, s_min, s_max, data_folder="road_data/"):
        self.data_folder = data_folder
        self.road_id = road_id
        self.timestamp_min = timestamp_min
        self.timestamp_max = timestamp_max
        self.s_min = s_min
        self.s_max = s_max
        self.conn = duckdb.connect()
        self.preloadAllData()
        self.preloadSubset()

    def preloadAllData(self):
        for lane in self.road_lane_lookup[self.road_id]:
            lane_str = str(lane) if lane >= 0 else "neg"+str(abs(lane))
            ref = f"{self.trajectories_all_db_prefix}_{self.road_id}_{lane_str}"
            file_path = os.path.join(self.data_folder, f"road{self.road_id}lane{lane}.parquet")
            self.conn.execute(f""" 
                CREATE VIEW {ref} AS
                SELECT * FROM parquet_scan('{file_path}')             
            """)

    def preloadSubset(self):
        for lane in self.road_lane_lookup[self.road_id]:
            lane_str = str(lane) if lane >= 0 else "neg"+str(abs(lane))
            source_ref = f"{self.trajectories_all_db_prefix}_{self.road_id}_{lane_str}"
            dest_ref = f"{self.trajectories_subset_db_prefix}_{self.road_id}_{lane_str}"
            self.directQueryEdieBoxDB(source_ref, dest_ref, self.timestamp_min, self.timestamp_max, self.s_min, self.s_max)

    def directQueryEdieBoxDF(self, source_ref, timestamp_min, timestamp_max, s_min, s_max):
        q = f"""
        SELECT time, x, y, length, width, height, class, id, s, t
        FROM {source_ref}
        WHERE time BETWEEN ? AND ?
        AND s BETWEEN ? AND ?
        """
        return self.conn.execute(q, [timestamp_min, timestamp_max, s_min, s_max]).fetch_df()
    
    def directQueryEdieBoxDB(self, source_ref, dest_ref, timestamp_min, timestamp_max, s_min, s_max):
        q = f"""
        CREATE TABLE {dest_ref} AS
        SELECT time, x, y, length, width, height, class, id, s, t
        FROM {source_ref}
        WHERE time BETWEEN ? AND ?
        AND s BETWEEN ? AND ?
        """
        return self.conn.execute(q, [timestamp_min, timestamp_max, s_min, s_max])
    
    def queryEdieBoxSubset(self, timestamp_min, timestamp_max, s_min, s_max):
        result = {}
        for lane in self.road_lane_lookup[self.road_id]:
            lane_str = str(lane) if lane >= 0 else "neg"+str(abs(lane))
            source_ref = f"{self.trajectories_subset_db_prefix}_{self.road_id}_{lane_str}"
            result[lane] = self.directQueryEdieBoxDF(source_ref, timestamp_min, timestamp_max, s_min, s_max)
        return result