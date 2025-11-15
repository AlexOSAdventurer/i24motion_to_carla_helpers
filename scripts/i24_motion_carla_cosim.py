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
    delta_t = 0.05 # Timestep - 0.05 seconds as per CARLA
    
    def __init__(self, hero_road, hero_lane, desired_time, desired_s):
        self.vehicles_to_completely_ignore = []
        self.hero_road = hero_road
        self.hero_state = None
        self.visible_state = None
        self.ghost_state = None
        self.current_timestamp = None
        self.internal_update_state = True
        self.real_data = i24_motion_data.I24MotionData(hero_road, desired_time - self.preload_time_window, desired_time + self.preload_time_window, desired_s - self.preload_s_window, desired_s + self.preload_s_window)
        self.loadHero(hero_road, hero_lane, desired_time, desired_s)
        self.loadVisible()
        self.loadGhosts()
        self.internal_update_state = False

    def generateVehicleStateFromRow(self, row, lane, current_timestamp=None):
        feet_to_meters = 0.3048
        if current_timestamp is None:
            current_timestamp = self.current_timestamp
        estimated_velocity = self.estimateVehicleVelocityFromReal(int(row["id"]), lane, float(row["time"]))
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
            "lane": lane
        }
    
    def generateUpdatedVehicleStateFromCARLA(self, vehicle_data):
        new_time = self.current_timestamp
        estimated_velocity = self.estimateVehicleVelocityFromReal(int(vehicle_data["id"]), int(vehicle_data["lane"]), new_time)
        return {
            "id": int(vehicle_data["id"]),
            "class": int(vehicle_data["class"]),
            "length": float(vehicle_data["length"]),
            "width": float(vehicle_data["width"]),
            "time": new_time,
            "s": float(vehicle_data["s"]),
            "t": float(vehicle_data["t"]),
            "velocity": estimated_velocity,
            "lane": int(vehicle_data["lane"])
        }
    
    def getVehicleTrajectoryFromReal(self, id, lane):
        return self.real_data.getVehicleTrajectory(lane, id)
    
    def estimateVehicleVelocityFromReal(self, id, lane, current_time):
        trajectory = self.getVehicleTrajectoryFromReal(id, lane)
        trajectory_time_sorted = trajectory.sort_values(by=['time'], ascending=True)
        if len(trajectory_time_sorted) < 2:
            return float('nan')
        current_index = int(trajectory_time_sorted['time'].searchsorted(current_time))
        if (current_index >= len(trajectory_time_sorted)):
            current_index = len(trajectory_time_sorted) - 1
        if (current_index > 0):
            return float(trajectory_time_sorted.iloc[current_index]["s"] - trajectory_time_sorted.iloc[current_index - 1]["s"]) / float(trajectory_time_sorted.iloc[current_index]["time"] - trajectory_time_sorted.iloc[current_index - 1]["time"])
        return float(trajectory_time_sorted.iloc[current_index + 1]["s"] - trajectory_time_sorted.iloc[current_index]["s"]) / float(trajectory_time_sorted.iloc[current_index + 1]["time"] - trajectory_time_sorted.iloc[current_index]["time"])
    
    def estimateGhostVehicleVelocity(self, vehicle_data):
        # Replay its velocity for as long as we have it from the trajectory data - when that expires, we will simply discard the ghost and assume whatever its replacement becomes
        # This keeps us as data driven as possible
        return self.estimateVehicleVelocityFromReal(self, vehicle_data["id"], vehicle_data["lane"], self.current_timestamp)

    def loadHero(self, hero_road, hero_lane, desired_time, desired_s):
        potential_heroes = self.real_data.queryEdieBoxSubset(desired_time - self.desired_time_max_difference, desired_time + self.desired_time_max_difference, desired_s - self.desired_s_max_difference, desired_s + self.desired_s_max_difference)[hero_lane]
        if len(potential_heroes) == 0:
            raise Exception(f"No avaiable heroes with {hero_road} and {hero_lane} and {desired_time} and {desired_s}")
        potential_heroes["time_delta"] = (potential_heroes["time"] - desired_time).abs()
        potential_heroes["s_delta"] = (potential_heroes["s"] - desired_s).abs()
        potential_heroes_sorted = potential_heroes.sort_values(by=['time_delta', 's_delta'], ascending=True)
        selected_hero = potential_heroes_sorted.iloc[0]
        original_hero_state = self.generateVehicleStateFromRow(selected_hero, hero_lane, float(selected_hero["time"]))
        self.hero_state = original_hero_state
        self.current_timestamp = original_hero_state["time"]

    def getCurrentVisibleWindow(self):
        return self.current_timestamp - self.initial_visible_time_max_difference, self.current_timestamp + self.initial_visible_time_max_difference, self.hero_state["s"] - self.visible_window, self.hero_state["s"] + self.visible_window
    
    def getCurrentBehindGhostWindow(self):
        return self.current_timestamp - self.ghost_time_max_difference, self.current_timestamp + self.ghost_time_max_difference, self.hero_state["s"] - self.visible_window - self.ghost_window, self.hero_state["s"] - self.visible_window
    
    def getCurrentAheadGhostWindow(self):
        return self.current_timestamp - self.ghost_time_max_difference, self.current_timestamp + self.ghost_time_max_difference, self.hero_state["s"] + self.visible_window, self.hero_state["s"] + self.visible_window + self.ghost_window
    
    def getLanes(self):
        return self.real_data.road_lane_lookup[self.hero_road]
    
    def getVisibleIds(self):
        result = {}
        for lane in self.getLanes():
            result[lane] = [id for id in self.visible_state[lane]]
        return result
    
    def getVisibleIdsFlat(self):
        visible_ids = self.getVisibleIds()
        return sum([visible_ids[lane] for lane in self.getLanes()], [])
    
    def getGhostIds(self):
        result = {
            "behind": {},
            "ahead": {}
        }
        for position in self.ghost_state:
            for lane in self.getLanes():
                result[position][lane] = [id for id in self.ghost_state[position][lane]]
        return result
    
    def getVisibleData(self):
        return self.visible_state
    
    def getGhostData(self):
        return self.ghost_state
    
    def getHeroData(self):
        return self.hero_state
    
    def getLowestBehindGhostVehicle(self, lane):
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
    
    def getHighestAheadGhostVehicle(self, lane):
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
    
    def getLowestBehindVisibleVehicle(self, lane):
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
    
    def getHighestAheadVisibleVehicle(self, lane):
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

    def loadVisible(self):
        self.visible_state = {}
        for lane in self.getLanes():
            self.visible_state[lane] = {}
        min_timestamp, max_timestamp, min_s, max_s = self.getCurrentVisibleWindow()
        potential_visibles = self.real_data.queryEdieBoxSubset(min_timestamp, max_timestamp, min_s, max_s, [self.hero_state["id"]])
        for lane in self.getLanes():
            potential_visibles_lane_sorted = potential_visibles[lane].sort_values(by=["time"], ascending=True)
            uniques = list(potential_visibles_lane_sorted["id"].unique())
            for unique in uniques:
                unique_vehicle_data = potential_visibles_lane_sorted[potential_visibles_lane_sorted["id"] == unique].copy()
                unique_vehicle_data["time_delta"] = (unique_vehicle_data["time"] - self.current_timestamp).abs()
                unique_vehicle_data_sorted = unique_vehicle_data.sort_values(by=["time_delta"], ascending=True)
                unique_vehicle_data_row = unique_vehicle_data_sorted.iloc[0]
                #self.visible_state[lane][int(unique)] = self.generateVehicleStateFromRow(unique_vehicle_data_row, lane)
                candidate = self.generateVehicleStateFromRow(unique_vehicle_data_row, lane)
                self.registerNewVisibleVehicle(candidate)
    
    def loadGhosts(self, ignore_ids=None):
        self.ghost_state = {
            "behind": {},
            "ahead": {}
        }
        for lane in self.getLanes():
            self.ghost_state["behind"][lane] = {}
            self.ghost_state["ahead"][lane] = {}
        self.loadGhostsBehind(ignore_ids)
        self.loadGhostsAhead(ignore_ids)

    def loadGhostsBehind(self, ignore_ids=None):
        min_timestamp, max_timestamp, min_s, max_s = self.getCurrentBehindGhostWindow()
        visible_ids_merged = self.getVisibleIdsFlat()
        potential_ghosts = self.real_data.queryEdieBoxSubset(min_timestamp, max_timestamp, min_s, max_s, visible_ids_merged)
        for lane in self.getLanes():
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
                new_ghost = self.generateVehicleStateFromRow(unique_vehicle_data_row, lane)
                self.registerNewGhostVehicle(new_ghost)

    def loadGhostsAhead(self, ignore_ids=None):
        min_timestamp, max_timestamp, min_s, max_s = self.getCurrentAheadGhostWindow()
        visible_ids = self.getVisibleIds()
        visible_ids_merged = sum([visible_ids[lane] for lane in visible_ids], [])
        potential_ghosts = self.real_data.queryEdieBoxSubset(min_timestamp, max_timestamp, min_s, max_s, visible_ids_merged)
        for lane in self.getLanes():
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
                new_ghost = self.generateVehicleStateFromRow(unique_vehicle_data_row, lane)
                self.registerNewGhostVehicle(new_ghost)

    def tick(self, new_hero_state, new_visible_states):
        self.current_timestamp += self.delta_t
        self.internal_update_state = True
        self.updateHeroVehicleViaCARLA(new_hero_state)
        self.updateVisibleVehiclesViaCARLA(new_visible_states)
        self.updateVisibleVehiclesViaGhosts()
        self.moveGhostVehiclesAndUpdateRoster()
        self.internal_update_state = False

    def updateHeroVehicleViaCARLA(self, new_hero_state):
        hero_state_processed = self.generateUpdatedVehicleStateFromCARLA(new_hero_state)
        hero_state_processed["velocity"] = float(new_hero_state["velocity"])
        self.hero_state = hero_state_processed

    def updateVisibleVehiclesViaCARLA(self, new_visible_states):
        visible_state_updated = {}
        for lane in self.getLanes():
            visible_state_updated[lane] = {}
        for lane in self.getLanes():
            for vehicle_id in new_visible_states[lane]:
                vehicle_data = new_visible_states[lane][vehicle_id]
                # Did we slip past the behind ghost cell?
                behind_ghost_window = self.getCurrentBehindGhostWindow()
                ahead_ghost_window = self.getCurrentAheadGhostWindow()
                if (vehicle_data["s"] < behind_ghost_window[2]):
                    print(f"WARNING: Threw away {vehicle_data} because it was visible but then slipped behind the behind ghost cell!")
                    self.vehicles_to_completely_ignore.append(vehicle_id)
                    continue # Throw away this vehicle from now on
                # Did we slip into the behind ghost cell?
                elif (vehicle_data["s"] < behind_ghost_window[3]):
                    self.registerNewGhostVehicle(vehicle_data)
                # Did we slip into the ahead ghost cell?
                elif (vehicle_data["s"] > ahead_ghost_window[2]):
                    self.registerNewGhostVehicle(vehicle_data)
                # Did we slip past the ahead ghost cell?
                elif (vehicle_data["s"] > ahead_ghost_window[3]):
                    print(f"WARNING: Threw away {vehicle_data} because it was visible but then slipped ahead the ahead ghost cell!")
                    self.vehicles_to_completely_ignore.append(vehicle_id)
                    continue # Throw away this vehicle from now on
                # Vehicle still in visible region
                else:
                    new_data = self.generateUpdatedVehicleStateFromCARLA(vehicle_data)
                    visible_state_updated[new_data["lane"]] = new_data

    def updateVisibleVehiclesWithGhostSelection(self, ghost_data):
        visible_window = self.getCurrentVisibleWindow()
        for lane in self.getLanes():
            for id in ghost_data[lane]:
                candidate = ghost_data[lane][id]
                candidate_new_s = candidate["s"] + (candidate["velocity"] * (self.current_timestamp - candidate["time"]))
                print(f"Candidate visible {candidate} which is a ghost has a projected {candidate_new_s} position with this window {visible_window}")
                if (candidate_new_s > visible_window[2]) and (candidate_new_s < visible_window[3]):
                    if (self.checkIfCandidateVisibleNoOverlapWithCurrentVisible(lane, candidate)):
                        candidate["s"] = candidate_new_s
                        self.registerNewVisibleVehicle(candidate)
                    else:
                        print(f"WARNING: Threw away {candidate} visible vehicle because it overlapped with the other visible vehicles!")
                        # No need to remove. Will be dealt with when we reload the ghost data.

    def updateVisibleVehiclesViaGhosts(self):
        # For each ghost vehicle, we will estimate its projected future position with a simple change to s.
        # Then, we will see if they fall under the visible region. If so, attempt to admit them, as long as geometry permits it.
        # Check behind vehicles
        self.updateVisibleVehiclesWithGhostSelection(self.ghost_state["behind"])
        # Check ahead vehicles
        self.updateVisibleVehiclesWithGhostSelection(self.ghost_state["ahead"])

    def checkVehicleBoundingBoxNoOverlap(self, vehicle1, vehicle2):
        vehicle_first = vehicle1 if (vehicle1["s"] < vehicle2["s"]) else vehicle2
        vehicle_second = vehicle1 if (vehicle1["s"] > vehicle2["s"]) else vehicle2 
        vehicle_first_min, vehicle_first_max = vehicle_first["s"], vehicle_first["s"] + vehicle_first["length"]
        vehicle_second_min, vehicle_second_max = vehicle_second["s"], vehicle_second["s"] + vehicle_second["length"]

        return (vehicle_first_min < vehicle_second_min) and (vehicle_first_max < vehicle_second_min)
    
    def checkIfCandidateGhostNoOverlapWithCurrentGhosts(self, ghost_data, candidate):
        for id in ghost_data:
            if not self.checkVehicleBoundingBoxNoOverlap(ghost_data[id], candidate):
                return False
        return True
    
    def checkIfInitVisibleNoOverlapWithCurrentVisible(self, lane, candidate):
        for id in self.visible_state[lane]:
            if not self.checkVehicleBoundingBoxNoOverlap(self.visible_state[lane][id], candidate):
                return False
        return True
    
    def checkIfCandidateVisibleNoOverlapWithCurrentVisible(self, lane, candidate):
        # Create a 1d bounding box for the lane that covers the backmost visible vehicle to the frontmost one. We cannot breach this.
        lowest_vehicle = self.getLowestBehindVisibleVehicle(lane)
        highest_vehicle = self.getHighestAheadVisibleVehicle(lane)
        if lowest_vehicle is None:
            return True
        result = ((candidate["s"] + candidate["length"]) < lowest_vehicle["s"]) or (candidate["s"] > (highest_vehicle["s"] + highest_vehicle["length"]))
        if not result:
            print(f"WARNING: Threw away {candidate} because it was in an invalid visible position with respect to {lowest_vehicle} and {highest_vehicle}.\n")
        return result
    
    def registerNewVisibleVehicle(self, vehicle_data, ignore_invalid_ghost_cell_position=False):
        if vehicle_data["id"] in self.vehicles_to_completely_ignore:
            print(f"WARNING: Threw away visible {vehicle_data} because it was marked as a vehicle to ignore.")
            return
        visible_window = self.getCurrentVisibleWindow()
        # Are we inside?
        if (vehicle_data["s"] > visible_window[2]) and (vehicle_data["s"] < visible_window[3]):
            if ignore_invalid_ghost_cell_position or self.checkIfInitVisibleNoOverlapWithCurrentVisible(vehicle_data["lane"], vehicle_data):
                self.visible_state[vehicle_data["lane"]][vehicle_data["id"]] = vehicle_data
            else:
                print(f"WARNING: Threw away {vehicle_data} because it was in an invalid visible position")
                self.vehicles_to_completely_ignore.append(vehicle_data["id"]) # Permanently throw away
        else:
            print(f"WARNING: Threw away {vehicle_data} because it wasn't in a valid visible position")
            self.vehicles_to_completely_ignore.append(vehicle_data["id"]) # Permanently throw away

    def registerNewGhostVehicle(self, vehicle_data, ignore_invalid_ghost_cell_position=False):
        if vehicle_data["id"] in self.vehicles_to_completely_ignore:
            print(f"WARNING: Threw away ghost {vehicle_data} because it was marked as a vehicle to ignore.")
            return
        behind_ghost_window = self.getCurrentBehindGhostWindow()
        ahead_ghost_window = self.getCurrentAheadGhostWindow()
        # Are we in the behind ghost cell?
        if (vehicle_data["s"] >= behind_ghost_window[2]) and (vehicle_data["s"] <= behind_ghost_window[3]):
            print("Checking behind!")
            if ignore_invalid_ghost_cell_position or self.checkIfCandidateGhostNoOverlapWithCurrentGhosts(self.ghost_state["behind"][vehicle_data["lane"]], vehicle_data):
                self.ghost_state["behind"][vehicle_data["lane"]][vehicle_data["id"]] = vehicle_data
            else:
                print(f"WARNING: Threw away {vehicle_data} because it was in an invalid behind ghost cell position")
                self.vehicles_to_completely_ignore.append(vehicle_data["id"]) # Permanently throw away
        elif (vehicle_data["s"] >= ahead_ghost_window[2]) and (vehicle_data["s"] <= ahead_ghost_window[3]):
            if ignore_invalid_ghost_cell_position or self.checkIfCandidateGhostNoOverlapWithCurrentGhosts(self.ghost_state["ahead"][vehicle_data["lane"]], vehicle_data):
                self.ghost_state["ahead"][vehicle_data["lane"]][vehicle_data["id"]] = vehicle_data
            else:
                print(f"WARNING: Threw away {vehicle_data} because it was in an invalid ahead ghost cell position")
                self.vehicles_to_completely_ignore.append(vehicle_data["id"]) # Permanently throw away
        else:
            print(f"WARNING: Threw away {vehicle_data} because it wasn't in a valid ghost position")
            # Don't permanently pitch it. It might appear in a useful spot later.
            #self.vehicles_to_completely_ignore.append(vehicle_data["id"]) # Permanently throw away

    def moveGhostVehiclesAndUpdateRoster(self):
        # We simply reload what the valid ghost region is from the data available.
        # This implicitly moves and deletes the vehicles in the roster as they disappear
        # The only difference is that we ignore vehicles currently in the visible range. CARLA manages them.
        self.loadGhosts(self.getVisibleIdsFlat())