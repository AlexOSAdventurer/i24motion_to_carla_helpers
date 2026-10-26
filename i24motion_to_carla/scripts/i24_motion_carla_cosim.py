import i24_motion_data

class I24MotionCARLACoSim:
    initial_visible_time_max_difference = 0.1 # Seconds
    ghost_time_max_difference = 1.0 # Seconds
    desired_time_max_difference = 1.0 # Seconds
    desired_s_max_difference = 20.0 # Meters
    preload_time_window = 600.0
    preload_s_window = 500
    visible_window = 150 # Meters
    ghost_window = 50 # Meters
    simulation_time = 100.0
    delta_t = 0.01 # Timestep - 0.01 seconds as per CARLA
    
    def __init__(self, hero_road, hero_lane, desired_time, desired_s):
        self.vehicles_to_completely_ignore = []
        self.hero_road = hero_road
        self.hero_state = None
        self.visible_state = None
        self.ghost_state = None
        self.current_timestamp = None
        self.internal_update_state = True
        self.real_data = i24_motion_data.I24MotionData(hero_road, desired_time - self.preload_time_window, desired_time + self.preload_time_window, desired_s - self.preload_s_window, desired_s + self.preload_s_window)
        self.load_hero(hero_road, hero_lane, desired_time, desired_s)
        self.load_visible()
        self.load_ghosts()
        self.internal_update_state = False

    def generate_vehicle_state_from_row(self, row, lane, current_timestamp=None):
        feet_to_meters = 0.3048
        if current_timestamp is None:
            current_timestamp = self.current_timestamp
        estimated_velocity = self.estimate_vehicle_velocity_from_real(int(row["id"]), lane, float(row["time"]))
        estimated_s = float(row["s"]) + ((current_timestamp - float(row["time"])) * estimated_velocity)
        return {
            "id": int(row["id"]),
            "class": int(row["class"]),
            "length": float(row["length"]) * feet_to_meters,
            "width": float(row["width"]) * feet_to_meters,
            "time": current_timestamp,
            "s": estimated_s,
            "t": float(row["t"]),
            "velocity": estimated_velocity,
            "lane": lane,
            "road": self.hero_road
        }
    
    def generate_updated_vehicle_state_from_carla(self, vehicle_data):
        new_time = self.current_timestamp
        estimated_velocity = self.estimate_vehicle_velocity_from_real(int(vehicle_data["id"]), int(vehicle_data["lane"]), new_time)
        return {
            "id": int(vehicle_data["id"]),
            "class": int(vehicle_data["class"]),
            "length": float(vehicle_data["length"]),
            "width": float(vehicle_data["width"]),
            "time": new_time,
            "s": float(vehicle_data["s"]),
            "t": float(vehicle_data["t"]),
            "velocity": estimated_velocity,
            "lane": int(vehicle_data["lane"]),
            "road": self.hero_road
        }
    
    def get_vehicle_trajectory_from_real(self, id, lane):
        return self.real_data.get_vehicle_trajectory(lane, id)
    
    def estimate_vehicle_velocity_from_real(self, id, lane, current_time):
        trajectory = self.get_vehicle_trajectory_from_real(id, lane)
        trajectory_time_sorted = trajectory.sort_values(by=['time'], ascending=True)
        if len(trajectory_time_sorted) < 2:
            return float('nan')
        current_index = int(trajectory_time_sorted['time'].searchsorted(current_time))
        if (current_index >= len(trajectory_time_sorted)):
            current_index = len(trajectory_time_sorted) - 1
        if (current_index > 0):
            return float(trajectory_time_sorted.iloc[current_index]["s"] - trajectory_time_sorted.iloc[current_index - 1]["s"]) / float(trajectory_time_sorted.iloc[current_index]["time"] - trajectory_time_sorted.iloc[current_index - 1]["time"])
        return float(trajectory_time_sorted.iloc[current_index + 1]["s"] - trajectory_time_sorted.iloc[current_index]["s"]) / float(trajectory_time_sorted.iloc[current_index + 1]["time"] - trajectory_time_sorted.iloc[current_index]["time"])
    
    def estimate_ghost_vehicle_velocity(self, vehicle_data):
        # Replay its velocity for as long as we have it from the trajectory data - when that expires, we will simply discard the ghost and assume whatever its replacement becomes
        # This keeps us as data driven as possible
        return self.estimate_vehicle_velocity_from_real(self, vehicle_data["id"], vehicle_data["lane"], self.current_timestamp)

    def load_hero(self, hero_road, hero_lane, desired_time, desired_s):
        potential_heroes = self.real_data.query_edie_box_subset(desired_time - self.desired_time_max_difference, desired_time + self.desired_time_max_difference, desired_s - self.desired_s_max_difference, desired_s + self.desired_s_max_difference)[hero_lane]
        if len(potential_heroes) == 0:
            raise Exception(f"No avaiable heroes with {hero_road} and {hero_lane} and {desired_time} and {desired_s}")
        potential_heroes["time_delta"] = (potential_heroes["time"] - desired_time).abs()
        potential_heroes["s_delta"] = (potential_heroes["s"] - desired_s).abs()
        potential_heroes_sorted = potential_heroes.sort_values(by=['time_delta', 's_delta'], ascending=True)
        selected_hero = potential_heroes_sorted.iloc[0]
        original_hero_state = self.generate_vehicle_state_from_row(selected_hero, hero_lane, float(selected_hero["time"]))
        self.hero_state = original_hero_state
        self.current_timestamp = original_hero_state["time"]

    def get_current_visible_window(self):
        return self.current_timestamp - self.initial_visible_time_max_difference, self.current_timestamp + self.initial_visible_time_max_difference, self.hero_state["s"] - self.visible_window, self.hero_state["s"] + self.visible_window
    
    def get_current_behind_ghost_window(self):
        return self.current_timestamp - self.ghost_time_max_difference, self.current_timestamp + self.ghost_time_max_difference, self.hero_state["s"] - self.visible_window - self.ghost_window, self.hero_state["s"] - self.visible_window
    
    def get_current_ahead_ghost_window(self):
        return self.current_timestamp - self.ghost_time_max_difference, self.current_timestamp + self.ghost_time_max_difference, self.hero_state["s"] + self.visible_window, self.hero_state["s"] + self.visible_window + self.ghost_window
    
    def get_lanes(self):
        return self.real_data.road_lane_lookup[self.hero_road]
    
    def get_visible_ids(self):
        result = {}
        for lane in self.get_lanes():
            result[lane] = [id for id in self.visible_state[lane]]
        return result
    
    def get_visible_ids_flat(self):
        visible_ids = self.get_visible_ids()
        return sum([visible_ids[lane] for lane in self.get_lanes()], [])
    
    def get_ghost_ids(self):
        result = {
            "behind": {},
            "ahead": {}
        }
        for position in self.ghost_state:
            for lane in self.get_lanes():
                result[position][lane] = [id for id in self.ghost_state[position][lane]]
        return result
    
    def get_visible_data(self):
        return self.visible_state
    
    def get_ghost_data(self):
        return self.ghost_state
    
    def get_hero_data(self):
        return self.hero_state
    
    def get_lowest_behind_ghost_vehicle(self, lane):
        lane_data = self.ghost_state["behind"][lane]
        vehicle_ids = list(lane_data.keys())
        if len(vehicle_ids) == 0:
            return None
        lowest_vehicle = lane_data[vehicle_ids[0]]
        for entry in vehicle_ids[1:]:
            current_entry = lane_data[entry]
            if (current_entry["s"] < lowest_vehicle["s"]):
                lowest_vehicle = current_entry
        return lowest_vehicle
    
    def get_highest_ahead_ghost_vehicle(self, lane):
        lane_data = self.ghost_state["ahead"][lane]
        vehicle_ids = list(lane_data.keys())
        if len(vehicle_ids) == 0:
            return None
        highest_vehicle = lane_data[vehicle_ids[0]]
        for entry in vehicle_ids[1:]:
            current_entry = lane_data[entry]
            if (current_entry["s"] > highest_vehicle["s"]):
                highest_vehicle = current_entry
        return highest_vehicle
    
    def get_lowest_ahead_ghost_vehicle(self, lane):
        lane_data = self.ghost_state["ahead"][lane]
        vehicle_ids = list(lane_data.keys())
        if len(vehicle_ids) == 0:
            return None
        lowest_vehicle = lane_data[vehicle_ids[0]]
        for entry in vehicle_ids[1:]:
            current_entry = lane_data[entry]
            if (current_entry["s"] < lowest_vehicle["s"]):
                lowest_vehicle = current_entry
        return lowest_vehicle
    
    def get_lowest_behind_visible_vehicle(self, lane):
        lane_data = self.visible_state[lane]
        vehicle_ids = list(lane_data.keys())
        if len(vehicle_ids) == 0:
            return None
        lowest_vehicle = lane_data[vehicle_ids[0]]
        for entry in vehicle_ids[1:]:
            current_entry = lane_data[entry]
            if (current_entry["s"] < lowest_vehicle["s"]):
                lowest_vehicle = current_entry
        return lowest_vehicle
    
    def get_highest_ahead_visible_vehicle(self, lane):
        lane_data = self.visible_state[lane]
        vehicle_ids = list(lane_data.keys())
        if len(vehicle_ids) == 0:
            return None
        highest_vehicle = lane_data[vehicle_ids[0]]
        for entry in vehicle_ids[1:]:
            current_entry = lane_data[entry]
            if (current_entry["s"] > highest_vehicle["s"]):
                highest_vehicle = current_entry
        return highest_vehicle

    def load_visible(self):
        self.visible_state = {}
        for lane in self.get_lanes():
            self.visible_state[lane] = {}
        min_timestamp, max_timestamp, min_s, max_s = self.get_current_visible_window()
        potential_visibles = self.real_data.query_edie_box_subset(min_timestamp, max_timestamp, min_s, max_s, [self.hero_state["id"]])
        for lane in self.get_lanes():
            potential_visibles_lane_sorted = potential_visibles[lane].sort_values(by=["time"], ascending=True)
            uniques = list(potential_visibles_lane_sorted["id"].unique())
            for unique in uniques:
                unique_vehicle_data = potential_visibles_lane_sorted[potential_visibles_lane_sorted["id"] == unique].copy()
                unique_vehicle_data["time_delta"] = (unique_vehicle_data["time"] - self.current_timestamp).abs()
                unique_vehicle_data_sorted = unique_vehicle_data.sort_values(by=["time_delta"], ascending=True)
                unique_vehicle_data_row = unique_vehicle_data_sorted.iloc[0]
                #self.visible_state[lane][int(unique)] = self.generate_vehicle_state_from_row(unique_vehicle_data_row, lane)
                candidate = self.generate_vehicle_state_from_row(unique_vehicle_data_row, lane)
                self.register_new_visible_vehicle(candidate)
    
    def load_ghosts(self, ignore_ids=None):
        self.ghost_state = {
            "behind": {},
            "ahead": {}
        }
        for lane in self.get_lanes():
            self.ghost_state["behind"][lane] = {}
            self.ghost_state["ahead"][lane] = {}
        self.load_ghosts_behind(ignore_ids)
        self.load_ghosts_ahead(ignore_ids)

    def load_ghosts_behind(self, ignore_ids=None):
        min_timestamp, max_timestamp, min_s, max_s = self.get_current_behind_ghost_window()
        visible_ids_merged = self.get_visible_ids_flat()
        potential_ghosts = self.real_data.query_edie_box_subset(min_timestamp, max_timestamp, min_s, max_s, visible_ids_merged)
        for lane in self.get_lanes():
            potential_ghosts_lane_sorted = potential_ghosts[lane].sort_values(by=["time"], ascending=True)
            uniques = list(potential_ghosts_lane_sorted["id"].unique())
            uniques_filtered = uniques
            if ignore_ids is not None:
                uniques_filtered = []
                for unique in uniques:
                    if unique not in ignore_ids:
                        uniques_filtered.append(unique)
            for unique in uniques_filtered:
                unique_vehicle_data = potential_ghosts_lane_sorted[potential_ghosts_lane_sorted["id"] == unique].copy()
                unique_vehicle_data["time_delta"] = (unique_vehicle_data["time"] - self.current_timestamp).abs()
                unique_vehicle_data_sorted = unique_vehicle_data.sort_values(by=["time_delta"], ascending=True)
                unique_vehicle_data_row = unique_vehicle_data_sorted.iloc[0]
                new_ghost = self.generate_vehicle_state_from_row(unique_vehicle_data_row, lane)
                self.register_new_ghost_vehicle(new_ghost)

    def load_ghosts_ahead(self, ignore_ids=None):
        min_timestamp, max_timestamp, min_s, max_s = self.get_current_ahead_ghost_window()
        visible_ids = self.get_visible_ids()
        visible_ids_merged = sum([visible_ids[lane] for lane in visible_ids], [])
        potential_ghosts = self.real_data.query_edie_box_subset(min_timestamp, max_timestamp, min_s, max_s, visible_ids_merged)
        for lane in self.get_lanes():
            potential_ghosts_lane_sorted = potential_ghosts[lane].sort_values(by=["time"], ascending=True)
            uniques = list(potential_ghosts_lane_sorted["id"].unique())
            uniques_filtered = uniques
            if ignore_ids is not None:
                uniques_filtered = []
                for unique in uniques:
                    if unique not in ignore_ids:
                        uniques_filtered.append(unique)
            for unique in uniques_filtered:
                unique_vehicle_data = potential_ghosts_lane_sorted[potential_ghosts_lane_sorted["id"] == unique].copy()
                unique_vehicle_data["time_delta"] = (unique_vehicle_data["time"] - self.current_timestamp).abs()
                unique_vehicle_data_sorted = unique_vehicle_data.sort_values(by=["time_delta"], ascending=True)
                unique_vehicle_data_row = unique_vehicle_data_sorted.iloc[0]
                new_ghost = self.generate_vehicle_state_from_row(unique_vehicle_data_row, lane)
                self.register_new_ghost_vehicle(new_ghost)

    def tick(self, new_hero_state, new_visible_states):
        self.current_timestamp += self.delta_t
        self.internal_update_state = True
        self.update_hero_vehicle_via_carla(new_hero_state)
        self.update_visible_vehicles_via_carla(new_visible_states)
        self.update_visible_vehicles_via_ghosts()
        self.move_ghost_vehicles_and_update_roster()
        self.internal_update_state = False

    def update_hero_vehicle_via_carla(self, new_hero_state):
        hero_state_processed = self.generate_updated_vehicle_state_from_carla(new_hero_state)
        hero_state_processed["velocity"] = float(new_hero_state["velocity"])
        self.hero_state = hero_state_processed

    def update_visible_vehicles_via_carla(self, new_visible_states):
        self.visible_state = {}
        for lane in self.get_lanes():
            self.visible_state[lane] = {}
        for lane in self.get_lanes():
            for vehicle_id in new_visible_states[lane]:
                vehicle_data = new_visible_states[lane][vehicle_id]
                # Did we slip past the behind ghost cell?
                behind_ghost_window = self.get_current_behind_ghost_window()
                ahead_ghost_window = self.get_current_ahead_ghost_window()
                if (vehicle_data["s"] < behind_ghost_window[2]):
                    print(f"WARNING: Threw away {vehicle_data} because it was visible but then slipped behind the behind ghost cell!")
                    #self.vehicles_to_completely_ignore.append(vehicle_id)
                    #continue # Throw away this vehicle from now on
                # Did we slip into the behind ghost cell?
                elif (vehicle_data["s"] < behind_ghost_window[3]):
                    self.register_new_ghost_vehicle(vehicle_data)
                # Did we slip into the ahead ghost cell?
                elif (vehicle_data["s"] > ahead_ghost_window[2]):
                    self.register_new_ghost_vehicle(vehicle_data)
                # Did we slip past the ahead ghost cell?
                elif (vehicle_data["s"] > ahead_ghost_window[3]):
                    print(f"WARNING: Threw away {vehicle_data} because it was visible but then slipped ahead the ahead ghost cell!")
                    #self.vehicles_to_completely_ignore.append(vehicle_id)
                    #continue # Throw away this vehicle from now on
                # Vehicle still in visible region
                else:
                    new_data = self.generate_updated_vehicle_state_from_carla(vehicle_data)
                    self.register_new_visible_vehicle(new_data, True)

    def update_visible_vehicles_with_ghost_selection(self, ghost_data):
        visible_window = self.get_current_visible_window()
        for lane in self.get_lanes():
            for id in ghost_data[lane]:
                candidate = ghost_data[lane][id]
                candidate_new_s = candidate["s"] + (candidate["velocity"] * (self.current_timestamp - candidate["time"]))
                print(f"Candidate visible {candidate} which is a ghost has a projected {candidate_new_s} position with this window {visible_window}")
                if (candidate_new_s > visible_window[2]) and (candidate_new_s < visible_window[3]):
                    if (self.check_if_candidate_visible_no_overlap_with_current_visible(lane, candidate)):
                        candidate["s"] = candidate_new_s
                        self.register_new_visible_vehicle(candidate)
                    else:
                        print(f"WARNING: Threw away {candidate} visible vehicle because it overlapped with the other visible vehicles!")
                        # No need to remove. Will be dealt with when we reload the ghost data.

    def update_visible_vehicles_via_ghosts(self):
        # For each ghost vehicle, we will estimate its projected future position with a simple change to s.
        # Then, we will see if they fall under the visible region. If so, attempt to admit them, as long as geometry permits it.
        # Check behind vehicles
        self.update_visible_vehicles_with_ghost_selection(self.ghost_state["behind"])
        # Check ahead vehicles
        self.update_visible_vehicles_with_ghost_selection(self.ghost_state["ahead"])

    def check_vehicle_bounding_box_no_overlap(self, vehicle1, vehicle2):
        vehicle_first = vehicle1 if (vehicle1["s"] < vehicle2["s"]) else vehicle2
        vehicle_second = vehicle1 if (vehicle1["s"] > vehicle2["s"]) else vehicle2 
        vehicle_first_min, vehicle_first_max = vehicle_first["s"], vehicle_first["s"] + vehicle_first["length"]
        vehicle_second_min, vehicle_second_max = vehicle_second["s"], vehicle_second["s"] + vehicle_second["length"]

        return (vehicle_first_min < vehicle_second_min) and (vehicle_first_max < vehicle_second_min)
    
    def check_if_candidate_ghost_no_overlap_with_current_ghosts(self, ghost_data, candidate):
        for id in ghost_data:
            if not self.check_vehicle_bounding_box_no_overlap(ghost_data[id], candidate):
                return False
        return True
    
    def check_if_init_visible_no_overlap_with_current_visible(self, lane, candidate):
        for id in self.visible_state[lane]:
            if not self.check_vehicle_bounding_box_no_overlap(self.visible_state[lane][id], candidate):
                return False
        return True
    
    def check_if_candidate_visible_no_overlap_with_current_visible(self, lane, candidate):
        # Create a 1d bounding box for the lane that covers the backmost visible vehicle to the frontmost one. We cannot breach this.
        lowest_vehicle = self.get_lowest_behind_visible_vehicle(lane)
        highest_vehicle = self.get_highest_ahead_visible_vehicle(lane)
        if lowest_vehicle is None:
            return True
        result = ((candidate["s"] + candidate["length"]) < lowest_vehicle["s"]) or (candidate["s"] > (highest_vehicle["s"] + highest_vehicle["length"]))
        if not result:
            print(f"WARNING: Threw away {candidate} because it was in an invalid visible position with respect to {lowest_vehicle} and {highest_vehicle}.\n")
        return result
    
    def register_new_visible_vehicle(self, vehicle_data, ignore_invalid_visible_cell_position=False):
        if vehicle_data["id"] in self.vehicles_to_completely_ignore:
            print(f"WARNING: Threw away visible {vehicle_data} because it was marked as a vehicle to ignore.")
            return
        visible_window = self.get_current_visible_window()
        # Are we inside?
        if (vehicle_data["s"] > visible_window[2]) and (vehicle_data["s"] < visible_window[3]):
            if ignore_invalid_visible_cell_position or self.check_if_init_visible_no_overlap_with_current_visible(vehicle_data["lane"], vehicle_data):
                self.visible_state[vehicle_data["lane"]][vehicle_data["id"]] = vehicle_data
            else:
                print(f"WARNING: Threw away {vehicle_data} because it was in an invalid visible position")
                #self.vehicles_to_completely_ignore.append(vehicle_data["id"]) # Permanently throw away
        else:
            print(f"WARNING: Threw away {vehicle_data} because it wasn't in a valid visible position")
            #self.vehicles_to_completely_ignore.append(vehicle_data["id"]) # Permanently throw away

    def register_new_ghost_vehicle(self, vehicle_data, ignore_invalid_ghost_cell_position=False):
        if vehicle_data["id"] in self.vehicles_to_completely_ignore:
            #print(f"WARNING: Threw away ghost {vehicle_data} because it was marked as a vehicle to ignore.")
            return
        behind_ghost_window = self.get_current_behind_ghost_window()
        ahead_ghost_window = self.get_current_ahead_ghost_window()
        # Are we in the behind ghost cell?
        if (vehicle_data["s"] >= behind_ghost_window[2]) and (vehicle_data["s"] <= behind_ghost_window[3]):
            if ignore_invalid_ghost_cell_position or self.check_if_candidate_ghost_no_overlap_with_current_ghosts(self.ghost_state["behind"][vehicle_data["lane"]], vehicle_data):
                self.ghost_state["behind"][vehicle_data["lane"]][vehicle_data["id"]] = vehicle_data
            else:
                print(f"WARNING: Threw away {vehicle_data} because it was in an invalid behind ghost cell position")
                #self.vehicles_to_completely_ignore.append(vehicle_data["id"]) # Permanently throw away
        elif (vehicle_data["s"] >= ahead_ghost_window[2]) and (vehicle_data["s"] <= ahead_ghost_window[3]):
            if ignore_invalid_ghost_cell_position or self.check_if_candidate_ghost_no_overlap_with_current_ghosts(self.ghost_state["ahead"][vehicle_data["lane"]], vehicle_data):
                self.ghost_state["ahead"][vehicle_data["lane"]][vehicle_data["id"]] = vehicle_data
            else:
                print(f"WARNING: Threw away {vehicle_data} because it was in an invalid ahead ghost cell position")
                #self.vehicles_to_completely_ignore.append(vehicle_data["id"]) # Permanently throw away
        else:
            print(f"WARNING: Threw away {vehicle_data} because it wasn't in a valid ghost position")
            # Don't permanently pitch it. It might appear in a useful spot later.
            #self.vehicles_to_completely_ignore.append(vehicle_data["id"]) # Permanently throw away

    def move_ghost_vehicles_and_update_roster(self):
        # We simply reload what the valid ghost region is from the data available.
        # This implicitly moves and deletes the vehicles in the roster as they disappear
        # The only difference is that we ignore vehicles currently in the visible range. CARLA manages them.
        self.load_ghosts(self.get_visible_ids_flat())