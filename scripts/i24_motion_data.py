import pyarrow.parquet as pq
import duckdb
import pandas
import os
import sys

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
    
    def getVehicleTrajectoryDF(self, source_ref, id):
        q = f"""
        SELECT time, x, y, length, width, height, class, id, s, t
        FROM {source_ref}
        WHERE id = ?
        ORDER BY time ASC
        """
        return self.conn.execute(q, [id]).fetch_df()
    
    def getVehicleTrajectory(self, lane, id):
        lane_str = str(lane) if lane >= 0 else "neg"+str(abs(lane))
        source_ref = f"{self.trajectories_subset_db_prefix}_{self.road_id}_{lane_str}"
        return self.getVehicleTrajectoryDF(source_ref, id)
    
    def queryEdieBoxSubset(self, timestamp_min, timestamp_max, s_min, s_max, ignore_ids = None):
        result = {}
        for lane in self.road_lane_lookup[self.road_id]:
            lane_str = str(lane) if lane >= 0 else "neg"+str(abs(lane))
            source_ref = f"{self.trajectories_subset_db_prefix}_{self.road_id}_{lane_str}"
            result[lane] = self.directQueryEdieBoxDF(source_ref, timestamp_min, timestamp_max, s_min, s_max)
            # Each unique id needs at least two rows corresponding to it so we can estimate velocity and justifiably say it's a reliable track
            unique_ids = list(result[lane]["id"].unique())
            for id in unique_ids:
                subset = result[lane][result[lane]["id"] == id]
                if (len(subset) < 2):
                    result[lane] = result[lane][result[lane]["id"] != id]
            if ignore_ids is not None:
                for ignore_id in ignore_ids:
                    result[lane] = result[lane][result[lane]["id"] != ignore_id]
        return result
    
