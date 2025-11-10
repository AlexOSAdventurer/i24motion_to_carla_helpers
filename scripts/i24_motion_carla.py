import json
import numpy
import pandas
import carla
import time

class I24MotionCarlaSimulation:
    episode_length = 100.0
    mile_to_feet = 5280.0
    feet_to_meters = 0.3048
    waypoint_distance = 1.0 # Meters
    tick_step = 0.05
    lead_distance = 0.0
    vehicle_speed = 128.748 # kph

    def __init__(self, host, port, density_data_path, mapping_path, trajectory_output_path):
        self.host = host
        self.port = port
        self.tm_port = 8000
        self.client = None
        self.world = None
        self.carla_map = None
        self.current_time = 0.0
        self.road_cells = {}
        self.cars = []
        self.tm = None
        self.connectToHost()
        with open(density_data_path, "r") as f:
            self.density_data = json.load(f)
        with open(mapping_path, "r") as f:
            self.mapping = json.load(f)
        self.time_origin = self.density_data["time_start"]
        self.density_time_step = self.density_data["time_interval"]
        self.trajectory_output_path = trajectory_output_path
        self.trajectory_data = pandas.DataFrame(
            {
                "simulation_time": pandas.Series(dtype="float"),
                "vehicle_id": pandas.Series(dtype="int"),
                "road_id": pandas.Series(dtype="int"),
                "lane_id": pandas.Series(dtype="int"),
                "s": pandas.Series(dtype="float"),
                "velocity": pandas.Series(dtype="float"),
                "speed_limit_current": pandas.Series(dtype="float")
            }
        )

    def connectToHost(self):
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.tick_step
        settings.no_rendering_mode = True
        self.tm = self.client.get_trafficmanager(self.tm_port)
        #self.tm.set_global_distance_to_leading_vehicle(self.lead_distance)
        #self.tm.global_percentage_speed_difference(-400.0)
        self.tm.set_synchronous_mode(True)
        self.world.apply_settings(settings)
        self.world.tick()


    def launchSimulation(self):
        self.generateRoadCellMetadata()
        try:
            self.seedSimulation()
            while (self.current_time < self.episode_length):
                self.tick()
                print(self.current_time)
        finally:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

            print('\ndestroying %d vehicles' % len(self.cars))
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.cars])

            print(f"Saving {len(self.trajectory_data)} records to CSV at {self.trajectory_output_path}!")
            self.trajectory_data.to_csv(self.trajectory_output_path, index=False, header=True)

    def tick(self):
        current_time = time.time()
        self.world.tick()
        self.current_time += self.tick_step
        self.fetchData()
        self.updateVehicles()

    def updateVehicles(self):
        # If any vehicles are in the outgoing ghost zone, delete them for now
        vehicle_actors = self.world.get_actors(self.cars)
        batch = []
        for vehicle in vehicle_actors:
            actor_id = vehicle.id
            location = vehicle.get_location()
            closest_waypoint = self.carla_map.get_waypoint(location, project_to_road=True)
            road_id = closest_waypoint.road_id
            lane_id = closest_waypoint.lane_id
            cell_metadata = self.road_cells[road_id]
            outflow_ghost_cell = cell_metadata["outflow_ghost_cell"][lane_id]
            s = closest_waypoint.s
            if (s >= outflow_ghost_cell["s_start"]):
                print(f"Removing {actor_id} at {s} in {road_id}:{lane_id}")
                batch.append(carla.command.DestroyActor(actor_id))
                self.cars.pop(self.cars.index(actor_id))
        self.client.apply_batch_sync(batch, False)

        # For all remaining vehicles, make sure their TM is appropriately set.
        vehicle_actors = self.world.get_actors(self.cars)
        for vehicle in vehicle_actors:
            self.tm.distance_to_leading_vehicle(vehicle, self.lead_distance)
            self.tm.vehicle_percentage_speed_difference(vehicle, 0.0)
            self.tm.ignore_lights_percentage(vehicle,100)
            self.tm.ignore_signs_percentage(vehicle,100)
            self.tm.set_desired_speed(vehicle, self.vehicle_speed)

    def fetchData(self):
        print(len(self.cars))
        vehicle_actors = self.world.get_actors(self.cars)
        vehicle_ids = []
        road_ids = []
        lane_ids = []
        s_positions = []
        velocities = []
        simulation_time = []
        speed_limit_current = []
        for vehicle in vehicle_actors:
            velocity = vehicle.get_velocity().length()
            location = vehicle.get_location()
            closest_waypoint = self.carla_map.get_waypoint(location, project_to_road=True)
            vehicle_id = vehicle.id
            road_id = closest_waypoint.road_id
            lane_id = closest_waypoint.lane_id
            s = closest_waypoint.s
            speed_limit = vehicle.get_speed_limit()
            vehicle_ids.append(vehicle_id)
            road_ids.append(road_id)
            lane_ids.append(lane_id)
            s_positions.append(s)
            velocities.append(velocity)
            simulation_time.append(self.current_time)
            speed_limit_current.append(speed_limit)
        print(numpy.min(velocities), numpy.mean(velocities), numpy.max(velocities), numpy.std(velocities))
        print(numpy.min(speed_limit_current), numpy.mean(speed_limit_current), numpy.max(speed_limit_current), numpy.std(speed_limit_current))
        concat_df = pandas.DataFrame({
            "simulation_time": simulation_time,
            "vehicle_id": vehicle_ids,
            "road_id": road_ids,
            "lane_id": lane_ids,
            "s": s_positions,
            "velocity": velocities,
            "speed_limit": speed_limit_current
        })
        print(f"Recording {len(concat_df)} records!")
        self.trajectory_data = pandas.concat([self.trajectory_data, concat_df])

    def calculateDensityTime(self, sim_time):
        time_index = int(sim_time // self.density_time_step)
        density_time = (time_index * self.density_time_step) + self.time_origin
        density_time_string = str(round(density_time, 0))
        return density_time_string
    
    def calculateCurrentDensityTime(self):
        return self.calculateDensityTime(self.current_time)
    
    def getDensityInformation(self, sim_time, density_marker, density_road_id, lane_id):
        return self.density_data["data"][str(density_road_id)][str(lane_id)][str(density_marker)][self.calculateDensityTime(sim_time)]
    
    def getCurrentDensityInformation(self, density_marker, density_road_id, lane_id):
        return self.getDensityInformation(self.current_time, density_marker, density_road_id, lane_id)
        
    @staticmethod
    def westboundMarkerToS(marker, origin_marker, origin_meter):
        return (((origin_marker - marker) * I24MotionCarlaSimulation.mile_to_feet) * I24MotionCarlaSimulation.feet_to_meters) + origin_meter
    
    @staticmethod
    def eastboundMarkerToS(marker, origin_marker, origin_meter):
        return (((marker - origin_marker) * I24MotionCarlaSimulation.mile_to_feet) * I24MotionCarlaSimulation.feet_to_meters) + origin_meter
    
    @staticmethod
    def convertMileDeltaToMeters(delta):
        return (delta * I24MotionCarlaSimulation.mile_to_feet) * I24MotionCarlaSimulation.feet_to_meters
    
    def createCellDefinition(self, road_id, road_entry_mapping, waypoints, marker_region):
        result = {}
        for lane in range(1, road_entry_mapping["lanes"] + 1):
            lane_data = {}
            lane_id = int(lane) * -1 # Negative lane ids since it is all forward lanes for us here
            back_marker = marker_region[0]
            front_marker = marker_region[1]
            lane_data["back_marker"] = back_marker
            lane_data["front_marker"] = front_marker

            if road_entry_mapping["direction"] == "eastbound":
                lane_data["density_marker"] = str(round(back_marker, 2))
                s_start = I24MotionCarlaSimulation.eastboundMarkerToS(back_marker, road_entry_mapping["origin_marker"], road_entry_mapping["origin_meter"])
                s_end = I24MotionCarlaSimulation.eastboundMarkerToS(front_marker, road_entry_mapping["origin_marker"], road_entry_mapping["origin_meter"])            
            elif road_entry_mapping["direction"] == "westbound":
                lane_data["density_marker"] = str(round(front_marker, 2))
                s_start = I24MotionCarlaSimulation.westboundMarkerToS(back_marker, road_entry_mapping["origin_marker"], road_entry_mapping["origin_meter"])
                s_end = I24MotionCarlaSimulation.westboundMarkerToS(front_marker, road_entry_mapping["origin_marker"], road_entry_mapping["origin_meter"])
            lane_data["s_start"] = s_start
            lane_data["s_end"] = s_end
            lane_data["waypoints"] = [waypoint for waypoint in waypoints if (waypoint.road_id == road_id) and (waypoint.lane_id == lane_id) and (waypoint.s >= s_start) and (waypoint.s <= s_end)]
            lane_data["density_road_id"] = road_entry_mapping["direction"]

            result[lane_id] = lane_data       
        return result
    
    def generateRoadCellMetadata(self):
        all_waypoints = self.carla_map.generate_waypoints(self.waypoint_distance)
        metadata = {}
        for road in self.mapping:
            metadata_entry = {}
            road_id = int(road)
            metadata_entry["direction"] = self.mapping[road]["direction"]
            metadata_entry["inflow_ghost_cell"] = self.createCellDefinition(road_id, self.mapping[road], all_waypoints, self.mapping[road]["inflow_ghost_region"])
            metadata_entry["outflow_ghost_cell"] = self.createCellDefinition(road_id, self.mapping[road], all_waypoints, self.mapping[road]["outflow_ghost_region"])
            metadata_entry["normal_cells"] = {}
            for m in numpy.arange(self.mapping[road]["normal_region"][0], self.mapping[road]["normal_region"][1], self.mapping[road]["longitudinal_occupancy_cell_delta_miles"]):
                m = round(m, 2)
                m_final = round(m + self.mapping[road]["longitudinal_occupancy_cell_delta_miles"], 2)
                metadata_entry["normal_cells"][m] = self.createCellDefinition(road_id, self.mapping[road], all_waypoints, [m, m_final])
            metadata[road_id] =  metadata_entry
        self.road_cells = metadata

    def generateVehiclesInCell(self, cell_info, number_of_vehicles, fail_count=0):
        fail_attempts_limit = 3
        if (fail_count >= fail_attempts_limit):
            print(f"Failed to spawn {number_of_vehicles} in {cell_info}")
            return
        vehicle_blueprints = [bp for bp in self.world.get_blueprint_library().filter('vehicle*') 
                              if ("nissan" in bp.id) or 
                              ("carlamotors" in bp.id) or 
                              ("dodge" in bp.id) or 
                              ("toyota" in bp.id) or
                              ("audi" in bp.id) or 
                              ("mercedes" in bp.id) or 
                              ("jeep" in bp.id) or 
                              ("cooper" in bp.id) or 
                              ("ford" in bp.id) or 
                              ("lincoln" in bp.id) or 
                              ("volkswagen" in bp.id) or 
                              ("mitsubishi" in bp.id) or 
                              ("tesla" in bp.id)]
        batch = []
        new_vehicles_list = []
        for i in range(number_of_vehicles):        
            selected_waypoint = numpy.random.choice(cell_info["waypoints"]).transform
            selected_waypoint.location.z += 1.0
            selected_blueprint = numpy.random.choice(vehicle_blueprints)
            batch.append(carla.command.SpawnActor(selected_blueprint, selected_waypoint)
                .then(carla.command.SetAutopilot(carla.command.FutureActor, True, self.tm_port)))
        
        for response in self.client.apply_batch_sync(batch, False):
            if response.error:
                print("Failed to spawn vehicle!")
            else:
                new_vehicles_list.append(response.actor_id)
        all_vehicle_actors = self.world.get_actors(new_vehicles_list)
        print(f"Requested {number_of_vehicles}, spawned {len(all_vehicle_actors)}")
        for new_id in new_vehicles_list:
            self.cars.append(new_id)
        for actor in all_vehicle_actors:
            actor.set_autopilot(True, self.tm_port)
            self.tm.distance_to_leading_vehicle(actor, self.lead_distance)
            self.tm.vehicle_percentage_speed_difference(actor, 0.0)
            self.tm.ignore_lights_percentage(actor,100)
            self.tm.ignore_signs_percentage(actor,100)
            #self.tm.set_desired_speed(actor, 0.0)

        # If we failed to spawn vehicles, try again
        failed_number_of_spawns = number_of_vehicles - len(all_vehicle_actors)
        if (failed_number_of_spawns > 0):
            self.generateVehiclesInCell(cell_info, failed_number_of_spawns, fail_count=fail_count+1)

    
    def seedSimulation(self):
        # Normal cells
        for road in self.road_cells:
            for m in self.road_cells[road]["normal_cells"]:
                for lane_id in self.road_cells[road]["normal_cells"][m]:
                    cell_metadata = self.road_cells[road]["normal_cells"][m][lane_id]
                    cell_i24_data = self.getCurrentDensityInformation(cell_metadata["density_marker"], cell_metadata["density_road_id"], lane_id)
                    self.generateVehiclesInCell(cell_metadata, cell_i24_data["vehicle_count"])
        
        # Ingoing ghost cells
        for road in self.road_cells:
            for lane_id in self.road_cells[road]["inflow_ghost_cell"]:
                cell_metadata = self.road_cells[road]["inflow_ghost_cell"][lane_id]
                cell_i24_data = self.getCurrentDensityInformation(cell_metadata["density_marker"], cell_metadata["density_road_id"], lane_id)
                self.generateVehiclesInCell(cell_metadata, cell_i24_data["vehicle_count"])

        # Outgoing ghost cells
        for road in self.road_cells:
            for lane_id in self.road_cells[road]["outflow_ghost_cell"]:
                cell_metadata = self.road_cells[road]["inflow_ghost_cell"][lane_id]
                cell_i24_data = self.getCurrentDensityInformation(cell_metadata["density_marker"], cell_metadata["density_road_id"], lane_id)
                self.generateVehiclesInCell(cell_metadata, cell_i24_data["vehicle_count"])

if __name__ == "__main__":
    sim = I24MotionCarlaSimulation("localhost", 2000, "final_density_data.json", "i24_motion_to_carla_mapping_adjusted_origin.json", "result4.csv")
    sim.launchSimulation()