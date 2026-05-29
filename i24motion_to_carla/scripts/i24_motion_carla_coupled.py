import json
import numpy
import pandas
import carla
import math
import time
from . import  i24_motion_carla_cosim
import matplotlib.pyplot as plt
from moviepy import VideoClip

def get_numpy_array_from_figure(fig):
    # Draw the canvas, cache the renderer
    fig.canvas.draw()
    
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    # Use tostring_rgb() for RGB or tostring_argb() for ARGB (need to roll axis)
    buf = numpy.frombuffer(bytes(fig.canvas.get_renderer().buffer_rgba()), dtype='uint8')
    
    # Reshape the buffer to a 3D array (H, W, 3)
    image_np = buf.reshape(h, w, 4)
    image_np = image_np[:, :, :3]
    return image_np

def compute_world_bounds(cosim, padding=0.0):
    return (cosim.hero_state["s"] - cosim.visible_window - cosim.ghost_window - padding, cosim.hero_state["s"] + cosim.visible_window + cosim.ghost_window + padding, -50, 0)

def bbox_corners_2d(vehicle_data):
    cx, cy = vehicle_data["s"], vehicle_data["t"]
    length, width = vehicle_data["length"], vehicle_data["width"] / 2.0
    return numpy.array([[cx + length, cy + width], [cx + length, cy - width], [cx, cy - width], [cx, cy + width]])

def make_video_from_snapshots(cosim, snapshots, bounds, fps, out_path):

    duration = len(snapshots) / fps

    def make_frame(t):
        frame_idx = int(t * fps)
        frame_idx = min(frame_idx, len(snapshots) - 1)
        frame = snapshots[frame_idx]
        xmin, xmax, ymin, ymax = bounds[frame_idx]

        fig, ax = plt.subplots(figsize=(18, 6))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-20.0, 0.0)
        ax.set_aspect("equal")
        ax.set_title(f"Frame {frame_idx}")

        # Draw each vehicle bbox
        for car in frame:
            corners = bbox_corners_2d(car)
            poly = plt.Polygon(corners, fill=False, linewidth=1.0)
            ax.add_patch(poly)
            ax.text(
                car["s"] + 0.5, car["t"],
                str(car["id"]),
                color="red",
                fontsize=8,
                ha="center", va="center",
            )

        # Optional: draw axes labels / grid
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, linewidth=0.2)

        # Plot ghost and visible regions
        #Behind ghost region
        ax.axvspan(xmin, xmin + cosim.ghost_window, ymin=0, ymax=1, color='yellow', alpha=0.7)

        #Visible region
        ax.axvspan(xmin + cosim.ghost_window, xmax - cosim.ghost_window, ymin=0, ymax=1, color='green', alpha=0.7)

        #Ahead ghost region
        ax.axvspan(xmax - cosim.ghost_window, xmax, ymin=0, ymax=1, color='yellow', alpha=0.7)

        fig.tight_layout()
        image = get_numpy_array_from_figure(fig)
        plt.close(fig)
        return image

    clip = VideoClip(make_frame, duration=duration)
    clip.write_videofile(out_path, fps=fps, codec="libx264")

class I24MotionCarlaSimulationCoupled:
    visible_lead_vehicle_follow_ghost_window = 80.0 # If 80.0 meters within the lead ghost window, mirror the ghost vehicle velocity
    episode_length = 30.0
    mile_to_feet = 5280.0
    feet_to_meters = 0.3048
    mps_to_kph = 3.6
    waypoint_distance = 0.1 # Meters
    tick_step = 0.01
    lead_distance = 2.0
    #hero_vehicle_speed = 128.748 # kph
    hero_vehicle_speed = 144.841 # kph
    default_speed_limit = 30.0 # kph
    spawn_z_offset = 0.75 # Meters
    vehicle_blueprints_selection = ['vehicle.audi.a2', 
                                    'vehicle.mercedes.coupe_2020', 
                                    'vehicle.dodge.charger_police', 
                                    'vehicle.audi.tt', 
                                    'vehicle.jeep.wrangler_rubicon', 
                                    'vehicle.mini.cooper_s', 
                                    'vehicle.mercedes.coupe', 
                                    'vehicle.dodge.charger_2020', 
                                    'vehicle.ford.ambulance', 
                                    'vehicle.lincoln.mkz_2020', 
                                    'vehicle.mini.cooper_s_2021', 
                                    'vehicle.ford.crown', 
                                    'vehicle.toyota.prius', 
                                    'vehicle.carlamotors.european_hgv', 
                                    'vehicle.carlamotors.carlacola', 
                                    'vehicle.nissan.patrol_2021', 
                                    'vehicle.dodge.charger_police_2020', 
                                    'vehicle.mercedes.sprinter', 
                                    'vehicle.audi.etron', 
                                    'vehicle.volkswagen.t2_2021', 
                                    'vehicle.carlamotors.firetruck', 
                                    'vehicle.ford.mustang', 
                                    'vehicle.volkswagen.t2', 
                                    'vehicle.mitsubishi.fusorosa', 
                                    'vehicle.tesla.model3', 
                                    'vehicle.tesla.cybertruck', 
                                    'vehicle.lincoln.mkz_2017', 
                                    'vehicle.nissan.patrol', 
                                    'vehicle.nissan.micra']
    vehicle_blueprints_bounds = {
        'vehicle.audi.a2': {
            "xmin": -1.8526612520217896,
            "xmax": 1.8527082204818726,
            "ymin": -0.8940306901931763,
            "ymax": 0.8946478366851807
        },  
        'vehicle.mercedes.coupe_2020': {
            "xmin": -2.339423894882202,
            "xmax": 2.334214925765991,
            "ymin": -0.9059094786643982,
            "ymax": 0.9059030413627625
        },  
        'vehicle.dodge.charger_police': {
            "xmin": -2.46346116065979,
            "xmax": 2.5107829570770264,
            "ymin": -1.0186110734939575,
            "ymax": 1.0197900533676147
        },  
        'vehicle.audi.tt': {
            "xmin": -2.0909781455993652,
            "xmax": 2.0902318954467773,
            "ymin": -0.9970653057098389,
            "ymax": 0.9970518350601196
        },  
        'vehicle.jeep.wrangler_rubicon': {
            "xmin": -1.9328800439834595,
            "xmax": 1.9333406686782837,
            "ymin": -0.9515625834465027,
            "ymax": 0.9536339640617371
        },  
        'vehicle.mini.cooper_s': {
            "xmin": -1.9028857946395874,
            "xmax": 1.9029144048690796,
            "ymin": -0.9852210283279419,
            "ymax": 0.985054612159729
        },  
        'vehicle.mercedes.coupe': {
            "xmin": -2.5134027004241943,
            "xmax": 2.513374090194702,
            "ymin": -1.0765365362167358,
            "ymax": 1.0750097036361694
        },  
        'vehicle.dodge.charger_2020': {
            "xmin": -2.5092527866363525,
            "xmax": 2.498572587966919,
            "ymin": -0.9408262968063354,
            "ymax": 0.9407956600189209
        },  
        'vehicle.ford.ambulance': {
            "xmin": -3.471363067626953,
            "xmax": 2.894279956817627,
            "ymin": -1.1736520528793335,
            "ymax": 1.1775223016738892
        },  
        'vehicle.lincoln.mkz_2020': {
            "xmin": -2.4525094032287598,
            "xmax": 2.4398722648620605,
            "ymin": -0.9183293581008911,
            "ymax": 0.9183839559555054
        },   
        'vehicle.mini.cooper_s_2021': {
            "xmin": -2.307612657546997,
            "xmax": 2.245086431503296,
            "ymin": -1.048503041267395,
            "ymax": 1.0485690832138062
        },  
        'vehicle.ford.crown': {
            "xmin": -2.4974451065063477,
            "xmax": 2.8682336807250977,
            "ymin": -0.9003520607948303,
            "ymax": 0.9003720879554749
        },  
        'vehicle.toyota.prius': {
            "xmin": -2.2548015117645264,
            "xmax": 2.258721113204956,
            "ymin": -1.0037702322006226,
            "ymax": 1.0030442476272583
        },  
        'vehicle.carlamotors.european_hgv': {
            "xmin": -3.9551165103912354,
            "xmax": 3.9805939197540283,
            "ymin": -1.4402295351028442,
            "ymax": 1.45085871219635
        },  
        'vehicle.carlamotors.carlacola': {
            "xmin": -2.601931571960449,
            "xmax": 2.6019067764282227,
            "ymin": -1.3134857416152954,
            "ymax": 1.3135038614273071
        },  
        'vehicle.nissan.patrol_2021': {
            "xmin": -2.7545599937438965,
            "xmax": 2.8112688064575195,
            "ymin": -1.0749708414077759,
            "ymax": 1.0749961137771606
        },  
        'vehicle.dodge.charger_police_2020': {
            "xmin": -2.5092551708221436,
            "xmax": 2.728259325027466,
            "ymin": -0.9648845195770264,
            "ymax": 0.9648749828338623
        },  
        'vehicle.mercedes.sprinter': {
            "xmin": -2.9681336879730225,
            "xmax": 2.947056531906128,
            "ymin": -0.9900767803192139,
            "ymax": 0.9983561038970947
        },  
        'vehicle.audi.etron': {
            "xmin": -2.4314372539520264,
            "xmax": 2.42427134513855,
            "ymin": -1.016370177268982,
            "ymax": 1.0163863897323608
        },  
        'vehicle.volkswagen.t2_2021': {
            "xmin": -2.104430913925171,
            "xmax": 2.3377530574798584,
            "ymin": -0.8876193761825562,
            "ymax": 0.8869459629058838
        },  
        'vehicle.carlamotors.firetruck': {
            "xmin": -4.4874677658081055,
            "xmax": 3.9805736541748047,
            "ymin": -1.440228819847107,
            "ymax": 1.4508594274520874
        },  
        'vehicle.ford.mustang': {
            "xmin": -2.326622724533081,
            "xmax": 2.390902280807495,
            "ymin": -0.9474157094955444,
            "ymax": 0.9474111795425415
        },  
        'vehicle.volkswagen.t2': {
            "xmin": -2.2388687133789062,
            "xmax": 2.241568088531494,
            "ymin": -1.0352046489715576,
            "ymax": 1.0341105461120605
        },  
        'vehicle.mitsubishi.fusorosa': {
            "xmin": -5.581599235534668,
            "xmax": 4.6910858154296875,
            "ymin": -2.0929453372955322,
            "ymax": 1.8512064218521118
        },  
        'vehicle.tesla.model3': {
            "xmin": -2.3666815757751465,
            "xmax": 2.425097942352295,
            "ymin": -1.0817289352416992,
            "ymax": 1.0817210674285889
        },  
        'vehicle.tesla.cybertruck': {
            "xmin": -3.136770248413086,
            "xmax": 3.1367831230163574,
            "ymin": -1.1948214769363403,
            "ymax": 1.19475257396698
        },  
        'vehicle.lincoln.mkz_2017': {
            "xmin": -2.4467976093292236,
            "xmax": 2.454885721206665,
            "ymin": -1.0641733407974243,
            "ymax": 1.0641509294509888
        }, 
        'vehicle.nissan.patrol': {
            "xmin": -2.3596363067626953,
            "xmax": 2.244873523712158,
            "ymin": -0.9656212329864502,
            "ymax": 0.9659717082977295
        },  
        'vehicle.nissan.micra': {
            "xmin": -1.8166637420654297,
            "xmax": 1.8167121410369873,
            "ymin": -0.9216124415397644,
            "ymax": 0.9235013127326965
        }
    }
    
    def __init__(self, host, port, coupler, mapping, trajectory_output_path, bev_video_output_path):
        self.host = host
        self.port = port
        self.tm_port = 8000
        self.client = None
        self.world = None
        self.carla_map = None
        self.hero_state = None
        self.visible_states = None
        self.coupler = coupler
        self.cosim_snapshot = []
        self.cosim_bounds = []
        self.current_timestamp = 0.0
        self.connectToHost()
        self.mapping = mapping
        self.time_origin = self.coupler.getHeroData()["time"]
        self.trajectory_output_path = trajectory_output_path
        self.bev_video_output_path = bev_video_output_path
        self.trajectory_data = pandas.DataFrame(
            {
                "simulation_time": pandas.Series(dtype="float"),
                "vehicle_id": pandas.Series(dtype="int"),
                "road_id": pandas.Series(dtype="int"),
                "lane_id": pandas.Series(dtype="int"),
                "s": pandas.Series(dtype="float"),
                "velocity": pandas.Series(dtype="float"),
                "speed_limit": pandas.Series(dtype="float"),
                "hero_status": pandas.Series(dtype="int")
            }
        )
        self.spawn_points = self.loadSpawnPointsForRoad()
        self.vehicle_blueprints = self.getVehicleBlueprints()

    def connectToHost(self):
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.tick_step
        settings.no_rendering_mode = False
        self.tm = self.client.get_trafficmanager(self.tm_port)
        self.tm.set_random_device_seed(42)
        self.tm.set_global_distance_to_leading_vehicle(self.lead_distance)
        self.tm.global_percentage_speed_difference(0.0)
        self.tm.set_synchronous_mode(True)
        self.world.apply_settings(settings)
        self.world.tick()

    def loadSpawnPointsForRoad(self):
        spawn_points = {}
        waypoints = self.carla_map.generate_waypoints(self.waypoint_distance)
        for road_str in self.mapping:
            road_id = int(road_str)
            spawn_points[road_id] = {}
            for lane_count in range(1, self.mapping[road_str]["lanes"] + 1):
                lane_id = lane_count * -1
                presorted_waypoints = [waypoint for waypoint in waypoints if (waypoint.road_id == road_id) and (waypoint.lane_id == lane_id)]
                sorted_waypoints = sorted(presorted_waypoints, key=lambda waypoint: waypoint.s)
                spawn_points[road_id][lane_id] = sorted_waypoints
        return spawn_points

    def convertCarlaLocationToOpenDRIVELocation(self, location, blueprint_id):
        # Assume no lateral movement - we don't compute t
        closest_waypoint = self.carla_map.get_waypoint(location, project_to_road=True)
        s_delta = self.vehicle_blueprints_bounds[blueprint_id]["xmin"] if blueprint_id is not None else 0.0
        return {
            "road_id": closest_waypoint.road_id,
            "lane_id": closest_waypoint.lane_id,
            "s": closest_waypoint.s + s_delta,
        }
    
    def getSpawnWaypointForOpenDRIVELocation(self, opendrive_location):
        # Assume no lateral movement - we don't compute t
        available_waypoints = self.spawn_points[opendrive_location["road_id"]][opendrive_location["lane_id"]]
        s_index = int(opendrive_location["s"] // self.waypoint_distance)
        return available_waypoints[s_index]
    
    def getVehicleBlueprints(self):
        vehicle_blueprints = {bp.id: bp for bp in self.world.get_blueprint_library().filter('vehicle*') 
                              if bp.id in self.vehicle_blueprints_selection}
        return vehicle_blueprints
    
    def generateOpenDRIVELocationFromState(self, cosim_vehicle_state):
        return {
            "road_id": cosim_vehicle_state["road_id"],
            "lane_id": cosim_vehicle_state["lane_id"],
            "s": cosim_vehicle_state["s"]
        }
    
    def getEligibleBlueprintsFromLengthAndWidth(self, cosim_vehicle_state):
        eligible_blueprints = []
        for blueprint_id in self.vehicle_blueprints_selection:
            if (self.getBlueprintLength(blueprint_id) <= cosim_vehicle_state["length"]):
                eligible_blueprints.append(blueprint_id)
        # If for some reason we get a bad length and width that doesn't comply with anything we have, we have to just use fake dimensions instead that are still reasonable.
        if (len(eligible_blueprints) == 0):
            print(f"Falsifying the dimensions of {cosim_vehicle_state} so we can spawn it!")
            cosim_vehicle_state["length"] = 4.0
            cosim_vehicle_state["width"] = 2.5
            return self.getEligibleBlueprintsFromLengthAndWidth(cosim_vehicle_state)
        return eligible_blueprints
    
    def spawnVehicleFromCoSIM(self, cosim_vehicle_state, blueprint_id = None):
        selected_blueprint_id = numpy.random.choice(self.getEligibleBlueprintsFromLengthAndWidth(cosim_vehicle_state)) if blueprint_id is None else blueprint_id
        opendrive_location = self.offsetOpenDRIVELocationByBlueprint(self.generateOpenDRIVELocationFromState(cosim_vehicle_state), selected_blueprint_id)
        spawn_waypoint = self.getSpawnWaypointForOpenDRIVELocation(opendrive_location).transform
        selected_blueprint = self.vehicle_blueprints[selected_blueprint_id]
        spawn_waypoint.location.z += self.spawn_z_offset
        batch = []
        batch.append(carla.command.SpawnActor(selected_blueprint, spawn_waypoint)
            .then(carla.command.SetAutopilot(carla.command.FutureActor, False, self.tm_port)))
        
        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                pass
            else:
                return response.actor_id
        return None
    
    
    def offsetOpenDRIVELocationByBlueprint(self, opendrive_location, blueprint_id):
        return {
            "road_id": opendrive_location["road_id"],
            "lane_id": opendrive_location["lane_id"],
            "s": opendrive_location["s"] - self.vehicle_blueprints_bounds[blueprint_id]["xmin"]
        }
    
    def offsetSByBlueprintForCoSIM(self, s, blueprint_id):
        return s + self.vehicle_blueprints_bounds[blueprint_id]["xmin"]
    
    def getVehicleBoundingBox(self, vehicle_actor_id):
        actor = self.world.get_actors([vehicle_actor_id])[0]
        return actor.bounding_box.get_local_vertices()
    
    def despawnCARLAVehicle(self, vehicle_state):
        self.client.apply_batch_sync([carla.command.DestroyActor(vehicle_state["carla_actor_id"])])

    def pushVehicle(self, vehicle_state, speed):
        vehicle_id = vehicle_state["carla_actor_id"]
        vehicle_actor = self.world.get_actors([vehicle_id])[0]
        actor_transform = vehicle_actor.get_transform()
        mass = vehicle_actor.get_physics_control().mass
        yaw = math.radians(actor_transform.rotation.yaw)
        actor_transform.rotation.roll = 0.0
        actor_transform.rotation.pitch = 0.0
        actor_transform.location.z += 0.1
        batch = [
            carla.command.ApplyTransform(vehicle_id, actor_transform)
        ]
        self.client.apply_batch_sync(batch, False)
        batch = [
            carla.command.ApplyImpulse(vehicle_id, carla.Vector3D(x=mass * speed * math.cos(yaw),
                                            y=mass * speed * math.sin(yaw),
                                            z=0.0))
            
        ]
        self.client.apply_batch_sync(batch, False)
        batch = []
        batch.append(carla.command.SetAutopilot(vehicle_id, True, self.tm_port))
        self.client.apply_batch_sync(batch, False)
        print(f"At time {self.world.get_snapshot().timestamp} Pushed {vehicle_state} by {speed} with mass of {mass} and speed of {speed} and transform of {actor_transform}")
    
    def spawnHeroVehicle(self, blueprint_id=None):
        cosim_data = self.coupler.getHeroData()
        actor_id = self.spawnVehicleFromCoSIM(cosim_data, blueprint_id)
        print("hero", cosim_data["id"])
        if actor_id is None:
            print("Failed to spawn hero!")
            print(cosim_data)
            return
        new_hero_state = {
            "carla_actor_id": actor_id,
            "cosim_data": cosim_data,
            "just_spawned": True,
            "pushed": False
        }
        self.hero_state = new_hero_state
        #self.pushVehicle(new_hero_state, cosim_data["velocity"])

    def spawnVisibleVehicle(self, cosim_data, blueprint_id=None):
        actor_id = self.spawnVehicleFromCoSIM(cosim_data, blueprint_id)
        print("visible", cosim_data["id"])
        if actor_id is None:
            print("Failed to spawn visible!")
            print(cosim_data)
            return
        new_visible_state = {
            "carla_actor_id": actor_id,
            "cosim_data": cosim_data,
            "just_spawned": True,
            "pushed": False
        }
        self.visible_states[cosim_data["id"]] = new_visible_state
        #self.pushVehicle(new_visible_state, cosim_data["velocity"])

    def despawnHeroVehicle(self):
        self.despawnCARLAVehicle(self.hero_state)
        self.hero_state = None

    def despawnVisibleVehicle(self, visible_state):
        self.despawnCARLAVehicle(visible_state)
        self.visible_states.pop(visible_state["cosim_data"]["id"], None)
    
    def resetVisibleStates(self):
        self.visible_states = {}

    def getBlueprintLength(self, blueprint_id):
        return self.vehicle_blueprints_bounds[blueprint_id]["xmax"] - self.vehicle_blueprints_bounds[blueprint_id]["xmin"]
    
    def getBlueprintWidth(self, blueprint_id):
        return self.vehicle_blueprints_bounds[blueprint_id]["ymax"] - self.vehicle_blueprints_bounds[blueprint_id]["ymin"]

    def getVehicleLength(self, vehicle_state):
        vehicle_actor = self.world.get_actors([vehicle_state["carla_actor_id"]])[0]
        blueprint_id = vehicle_actor.type_id
        return self.getBlueprintLength(blueprint_id)
    
    def getVehicleWidth(self, vehicle_state):
        vehicle_actor = self.world.get_actors([vehicle_state["carla_actor_id"]])[0]
        blueprint_id = vehicle_actor.type_id
        return self.getBlueprintWidth(blueprint_id)

    def rebuildCoSIMVehicleState(self, vehicle_state):
        vehicle_actor = self.world.get_actors([vehicle_state["carla_actor_id"]])[0]
        location = vehicle_actor.get_location()
        closest_waypoint = self.carla_map.get_waypoint(location, project_to_road=True)
        vehicle_road_id = closest_waypoint.road_id
        vehicle_lane_id = closest_waypoint.lane_id
        return {
            "id": vehicle_state["cosim_data"]["id"],
            "class": vehicle_state["cosim_data"]["class"],
            "length": self.getVehicleLength(vehicle_state),
            "width": self.getVehicleWidth(vehicle_state),
            "time": vehicle_state["cosim_data"]["time"],
            "s": self.offsetSByBlueprintForCoSIM(closest_waypoint.s, vehicle_actor.type_id),
            "t": (vehicle_lane_id * 3.5) + 1.75, # Hack kludge to estimate
            "velocity": vehicle_actor.get_velocity().length(),
            "lane_id": vehicle_lane_id,
            "road_id": vehicle_road_id
        }
    def rebuildCoSIMHeroState(self):
        return self.rebuildCoSIMVehicleState(self.hero_state)

    def rebuildCoSIMVisibleState(self):
        cosim_visible_states = {}
        road_str = str(self.coupler.hero_road)
        for lane_count in range(1, self.mapping[road_str]["lanes"] + 1):
            lane_id = lane_count * -1
            cosim_visible_states[lane_id] = {}
        for cosim_id in self.visible_states:
            rebuilt_state = self.rebuildCoSIMVehicleState(self.visible_states[cosim_id])
            cosim_visible_states[rebuilt_state["lane_id"]][rebuilt_state["id"]] = rebuilt_state
        return cosim_visible_states

    def spawnAndDespawnVisibleVehiclesFromCoSIM(self):
        cosim_visible_vehicles = self.coupler.getVisibleData()
        self.resetVisibleStates()
        for lane in cosim_visible_vehicles:
            for id in cosim_visible_vehicles[lane]:
                self.spawnVisibleVehicle(cosim_visible_vehicles[lane][id])

    def updateVisibleVehiclesFromCoSIM(self):
        cosim_visible_vehicles = self.coupler.getVisibleData()
        cosim_visible_vehicles_ids = self.coupler.getVisibleIdsFlat()
        for lane in cosim_visible_vehicles:
            for id in cosim_visible_vehicles[lane]:
                if id not in self.visible_states: #New vehicle!
                    self.spawnVisibleVehicle(cosim_visible_vehicles[lane][id])
        current_visible_state_ids = list(self.visible_states.keys())
        for id in current_visible_state_ids:
            if id not in cosim_visible_vehicles_ids:
                self.despawnVisibleVehicle(self.visible_states[id])

    def updateHeroAndVisibleVehiclesFromTM(self):
        vehicle_actor = self.world.get_actors([self.hero_state["carla_actor_id"]])[0]
        self.tm.set_desired_speed(vehicle_actor, self.hero_vehicle_speed)
        #self.tm.set_desired_speed(vehicle_actor, 70.0)
        cosim_lead_vehicles = {}
        cosim_nominal_vehicle_actor_ids = []
        road_str = str(self.coupler.hero_road)
        for lane_count in range(1, self.mapping[road_str]["lanes"] + 1):
            lane_id = lane_count * -1
            potential_cosim_vehicles = [self.visible_states[cosim_id] for cosim_id in self.visible_states if self.visible_states[cosim_id]["cosim_data"]["lane_id"] == lane_id]
            potential_cosim_vehicles_sorted = sorted(potential_cosim_vehicles, key=lambda entry: entry["cosim_data"]["s"], reverse=True)
            cosim_lead_vehicles[lane_id] = []
            for entry in potential_cosim_vehicles_sorted:
                if (entry["carla_actor_id"] != self.hero_state["carla_actor_id"]) \
                and (entry["cosim_data"]["s"] >= (self.coupler.getCurrentAheadGhostWindow()[2] - self.visible_lead_vehicle_follow_ghost_window)):
                    cosim_lead_vehicles[lane_id].append(entry)
                else:
                    cosim_nominal_vehicle_actor_ids.append(entry["carla_actor_id"])
        for actor_id in cosim_nominal_vehicle_actor_ids:
            cosim_actor = self.world.get_actors([actor_id])[0]
            self.tm.set_desired_speed(cosim_actor, self.hero_vehicle_speed)
        for lane_count in range(1, self.mapping[road_str]["lanes"] + 1):
            lane_id = lane_count * -1
            cosim_lane_lead_vehicles = cosim_lead_vehicles[lane_id]
            cosim_ghost_lead_state = self.coupler.getLowestAheadGhostVehicle(lane_id)
            cosim_ghost_lead_velocity = cosim_ghost_lead_state["velocity"] * self.mps_to_kph if cosim_ghost_lead_state is not None else self.coupler.getAheadLaneVelocity(lane_id)
            for entry in cosim_lane_lead_vehicles:
                cosim_actor = self.world.get_actors([entry["carla_actor_id"]])[0]
                self.tm.set_desired_speed(cosim_actor, cosim_ghost_lead_velocity)

    def updateCoSIM(self):
        hero_rebuilt = self.rebuildCoSIMHeroState()
        visible_rebuilt = self.rebuildCoSIMVisibleState()
        return (hero_rebuilt, visible_rebuilt)

    def updateCoSIMSnapshotData(self):
        vehicles = []
        hero_state = self.coupler.getHeroData()
        visible_state = self.coupler.getVisibleData()
        ghost_state = self.coupler.getGhostData()
        bounds = compute_world_bounds(self.coupler)
        vehicles.append(hero_state)
        for lane in visible_state:
            for id in visible_state[lane]:
                vehicles.append(visible_state[lane][id])

        for position in ghost_state:
            for lane in ghost_state[position]:
                for id in ghost_state[position][lane]:
                    vehicles.append(ghost_state[position][lane][id])
        self.cosim_snapshot.append(vehicles)
        self.cosim_bounds.append(bounds)

    def fetchData(self):
        self.updateCoSIMSnapshotData()
        all_cars = [self.hero_state["carla_actor_id"]]
        for cosim_id in self.visible_states:
            all_cars.append(self.visible_states[cosim_id]["carla_actor_id"])
        vehicle_actors = self.world.get_actors(all_cars)
        vehicle_ids = []
        road_ids = []
        lane_ids = []
        s_positions = []
        velocities = []
        simulation_time = []
        speed_limit_current = []
        hero_status = []
        for vehicle in vehicle_actors:
            velocity = vehicle.get_velocity().length()
            location = vehicle.get_location()
            closest_waypoint = self.carla_map.get_waypoint(location, project_to_road=True)
            if (location.distance(closest_waypoint.transform.location) > 3.0):
                print("skipping!")
                continue
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
            simulation_time.append(self.current_timestamp)
            speed_limit_current.append(speed_limit)
            if (self.hero_state["carla_actor_id"] == vehicle.id):
                hero_status.append(1)
            else:
                hero_status.append(0)
        print("Velocities ", numpy.mean(velocities), self.current_timestamp)
        #print("SPEED LIMIT ", numpy.min(speed_limit_current), numpy.mean(speed_limit_current), numpy.max(speed_limit_current), numpy.std(speed_limit_current))
        concat_df = pandas.DataFrame({
            "simulation_time": simulation_time,
            "vehicle_id": vehicle_ids,
            "road_id": road_ids,
            "lane_id": lane_ids,
            "s": s_positions,
            "velocity": velocities,
            "speed_limit": speed_limit_current,
            "hero_status": hero_status
        })
        #print(f"Recording {len(concat_df)} records!")
        self.trajectory_data = pandas.concat([self.trajectory_data, concat_df])

    def markHeroesAndVisiblesInPreviousStepAsSpawned(self):
        self.hero_state["just_spawned"] = False
        for cosim_id in self.visible_states:
            self.visible_states[cosim_id]["just_spawned"] = False

    def pushEligibleVehicles(self):
        if (not self.hero_state["just_spawned"]) and (not self.hero_state["pushed"]):
            self.pushVehicle(self.hero_state, self.hero_state["cosim_data"]["velocity"])
            self.hero_state["pushed"] = True
            self.tm.auto_lane_change(self.world.get_actors([self.hero_state["carla_actor_id"]])[0], False)
        for cosim_id in self.visible_states:
            if (not self.visible_states[cosim_id]["just_spawned"]) and (not self.visible_states[cosim_id]["pushed"]):
                self.pushVehicle(self.visible_states[cosim_id], self.visible_states[cosim_id]["cosim_data"]["velocity"])
                self.visible_states[cosim_id]["pushed"] = True

    def initializeSimulation(self):
        self.spawnHeroVehicle("vehicle.audi.a2") # Lets keep us at it an audi for now ;)!
        self.spawnAndDespawnVisibleVehiclesFromCoSIM()

    def runSimulationOver(self, t=1.0):
        starting_timestamp = self.current_timestamp
        hero_rebuilt, visible_rebuilt = None, None
        while (self.current_timestamp < (starting_timestamp + t)):
            self.updateHeroAndVisibleVehiclesFromTM()
            self.world.tick()
            self.fetchData()
            self.markHeroesAndVisiblesInPreviousStepAsSpawned()
            self.pushEligibleVehicles()
            hero_rebuilt, visible_rebuilt = self.updateCoSIM()
            self.current_timestamp += self.tick_step
        print(self.current_timestamp)
        return hero_rebuilt, visible_rebuilt
    
    def updateSimulation(self):
        self.updateVisibleVehiclesFromCoSIM()

    def destroySimulation(self):
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

            all_cars = [self.hero_state["carla_actor_id"]]
            for cosim_id in self.visible_states:
                all_cars.append(self.visible_states[cosim_id]["carla_actor_id"])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in all_cars])
            print(f"Despawned {len(all_cars)} cars!")
        except:
            pass

        #print(f"Saving {len(self.trajectory_data)} records to CSV at {self.trajectory_output_path}!")
        #self.trajectory_data.to_csv(self.trajectory_output_path, index=False, header=True)

        #print(f"Saving BEV video at {self.bev_video_output_path}")
        #make_video_from_snapshots(self.coupler, self.cosim_snapshot, self.cosim_bounds, fps=int(1.0 / self.tick_step), out_path=self.bev_video_output_path)