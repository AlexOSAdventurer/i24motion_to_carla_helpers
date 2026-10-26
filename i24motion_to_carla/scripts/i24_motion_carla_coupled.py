import json
import numpy
import pandas
import carla
import cv2
import imageio
import queue
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
    mile_to_feet = 5280.0
    feet_to_meters = 0.3048
    mps_to_kph = 3.6
    waypoint_distance = 0.1 # Meters
    tick_step = 1.0/30.0
    lead_distance = 2.0
    FPS = 30.0
    hero_vehicle_speed = 144.841 # kph
    spawn_z_offset = 0.25 # Meters
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
    
    def __init__(self, host, port, coupler, mapping, carla_video_output_prefix, record_videos=True):
        self.host = host
        self.port = port
        self.tm_port = 8000
        self.client = None
        self.world = None
        self.carla_map = None
        self.hero_state = None
        self.record_videos = record_videos
        self.hero_cameras = {}
        self.hero_camera_queues = {}
        self.hero_camera_writers = {}
        self.visible_states = None
        self.coupler = coupler
        self.cosim_snapshot = []
        self.cosim_bounds = []
        self.current_timestamp = 0.0
        self.connect_to_host()
        self.mapping = mapping
        self.time_origin = self.coupler.get_hero_data()["time"]
        self.carla_video_output_prefix = carla_video_output_prefix
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
        self.spawn_points = self.load_spawn_points_for_road()
        self.vehicle_blueprints = self.get_vehicle_blueprints()

    def connect_to_host(self):
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.substepping = True
        settings.max_substep_delta_time = self.tick_step / 10.0
        settings.max_substeps = 10
        settings.fixed_delta_seconds = self.tick_step
        settings.no_rendering_mode = False
        self.tm = self.client.get_trafficmanager(self.tm_port)
        self.tm.set_random_device_seed(42)
        numpy.random.seed(42)
        self.tm.set_global_distance_to_leading_vehicle(self.lead_distance)
        self.tm.global_percentage_speed_difference(0.0)
        self.tm.set_synchronous_mode(True)
        self.world.apply_settings(settings)
        self.world.tick()

    def load_spawn_points_for_road(self):
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

    def convert_carla_location_to_open_drive_location(self, location, blueprint_id):
        # Assume no lateral movement - we don't compute t
        closest_waypoint = self.carla_map.get_waypoint(location, project_to_road=True)
        s_delta = self.vehicle_blueprints_bounds[blueprint_id]["xmin"] if blueprint_id is not None else 0.0
        return {
            "road_id": closest_waypoint.road_id,
            "lane_id": closest_waypoint.lane_id,
            "s": closest_waypoint.s + s_delta,
        }
    
    def get_spawn_waypoint_for_open_drive_location(self, opendrive_location):
        # Assume no lateral movement - we don't compute t
        available_waypoints = self.spawn_points[opendrive_location["road_id"]][opendrive_location["lane_id"]]
        s_index = int(opendrive_location["s"] // self.waypoint_distance)
        return available_waypoints[s_index]
    
    def get_vehicle_blueprints(self):
        vehicle_blueprints = {bp.id: bp for bp in self.world.get_blueprint_library().filter('vehicle*') 
                              if bp.id in self.vehicle_blueprints_selection}
        return vehicle_blueprints
    
    def generate_open_drive_location_from_state(self, cosim_vehicle_state):
        return {
            "road_id": cosim_vehicle_state["road_id"],
            "lane_id": cosim_vehicle_state["lane_id"],
            "s": cosim_vehicle_state["s"]
        }

    def _spawn_side_for(self, hero_cosim_data, cosim_data) -> str:
        """Which bubble boundary this vehicle entered through.

        Used to refund the right flux memory if the insertion turns out to have
        failed, so a rejected spawn exactly undoes the decrement that paid for it.
        """
        centre = hero_cosim_data["s"]
        return "rear" if float(cosim_data["s"]) < centre else "front"
    
    def get_eligible_blueprints_from_length_and_width(self, cosim_vehicle_state):
        eligible_blueprints = []
        for blueprint_id in self.vehicle_blueprints_selection:
            if (self.get_blueprint_length(blueprint_id) <= cosim_vehicle_state["length"]):
                eligible_blueprints.append(blueprint_id)
        # If for some reason we get a bad length and width that doesn't comply with anything we have, we have to just use fake dimensions instead that are still reasonable.
        if (len(eligible_blueprints) == 0):
            #print(f"Falsifying the dimensions of {cosim_vehicle_state} so we can spawn it!")
            cosim_vehicle_state["length"] = 4.0
            cosim_vehicle_state["width"] = 2.5
            return self.get_eligible_blueprints_from_length_and_width(cosim_vehicle_state)
        return eligible_blueprints
    
    def select_largest_blueprint(self, blueprints):
        blueprints_sorted = sorted(blueprints, key=self.get_blueprint_length)
        return blueprints_sorted[-1]
    
    def spawn_vehicle_from_co_sim(self, cosim_vehicle_state, blueprint_id=None, tick=False):
        selected_blueprint_id = numpy.random.choice(self.get_eligible_blueprints_from_length_and_width(cosim_vehicle_state)) if blueprint_id is None else blueprint_id
        #selected_blueprint_id = self.select_largest_blueprint(self.get_eligible_blueprints_from_length_and_width(cosim_vehicle_state)) if blueprint_id is None else blueprint_id
        opendrive_location = self.offset_open_drive_location_by_blueprint(self.generate_open_drive_location_from_state(cosim_vehicle_state), selected_blueprint_id)
        spawn_waypoint = self.get_spawn_waypoint_for_open_drive_location(opendrive_location).transform
        selected_blueprint = self.vehicle_blueprints[selected_blueprint_id]
        spawn_waypoint.location.z += self.spawn_z_offset
        yaw = math.radians(spawn_waypoint.rotation.yaw)
        cosim_vehicle_state["temporary_velocity_push"] = (cosim_vehicle_state["velocity"]*math.cos(yaw), cosim_vehicle_state["velocity"]*math.sin(yaw), 0.0)
        batch = []
        batch.append(carla.command.SpawnActor(selected_blueprint, spawn_waypoint)
            .then(carla.command.ApplyTargetVelocity(carla.command.FutureActor, carla.Vector3D(*cosim_vehicle_state["temporary_velocity_push"]))))
        
        actor_id = None
        for response in self.client.apply_batch_sync(batch, tick):
            if response.error:
                print("Corresponding error: ", response.error)
            else:
                actor_id = response.actor_id
                batch = []
                batch.append(carla.command.SetAutopilot(actor_id, False, self.tm_port))
                self.client.apply_batch_sync(batch, False)
        return actor_id
    
    
    def offset_open_drive_location_by_blueprint(self, opendrive_location, blueprint_id):
        return {
            "road_id": opendrive_location["road_id"],
            "lane_id": opendrive_location["lane_id"],
            "s": opendrive_location["s"] - self.vehicle_blueprints_bounds[blueprint_id]["xmin"]
        }
    
    def offset_s_by_blueprint_for_co_sim(self, s, blueprint_id):
        return s + self.vehicle_blueprints_bounds[blueprint_id]["xmin"]
    
    def get_vehicle_bounding_box(self, vehicle_actor_id):
        actor = self.world.get_actors([vehicle_actor_id])[0]
        return actor.bounding_box.get_local_vertices()
    
    def despawn_carla_vehicle(self, vehicle_state):
        self.client.apply_batch_sync([carla.command.DestroyActor(vehicle_state["carla_actor_id"])])

    def push_vehicle_impulse(self, vehicle_state, speed):
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
        #print(f"At time {self.world.get_snapshot().timestamp} Pushed {vehicle_state} by {speed} with mass of {mass} and speed of {speed} and transform of {actor_transform}")

    def push_vehicle(self, vehicle_state, speed):
        vehicle_id = vehicle_state["carla_actor_id"]
        vehicle_actor = self.world.get_actors([vehicle_id])[0]
        actor_transform = vehicle_actor.get_transform()
        actor_transform.rotation.roll = 0.0
        actor_transform.rotation.pitch = 0.0
        actor_transform.location.z += 0.1
        batch = [
            carla.command.ApplyTransform(vehicle_id, actor_transform),
            carla.command.SetAutopilot(vehicle_id, False, self.tm_port)
        ]
        self.client.apply_batch_sync(batch, False)
        vehicle_actor.enable_constant_velocity(
            carla.Vector3D(x=speed, y=0.0, z=0.0)
        )

        #print(f"At time {self.world.get_snapshot().timestamp} Pushed {vehicle_state} by {speed} and transform of {actor_transform}")
    
    def spawn_hero_vehicle(self, blueprint_id=None):
        cosim_data = self.coupler.get_hero_data()
        actor_id = self.spawn_vehicle_from_co_sim(cosim_data, blueprint_id, True)
        #print("hero", cosim_data["id"])
        if actor_id is None:
            #print("Failed to spawn hero!")
            #print(cosim_data)
            return
        new_hero_state = {
            "carla_actor_id": actor_id,
            "cosim_data": cosim_data,
            "just_spawned_cooldown": 1,
            "autopilot_enabled": False,
            "pushed": False
        }
        # Create topdown BEV view
        video_name = self.carla_video_output_prefix + "_bev.mp4"
        FPS = self.FPS
        HEIGHT = 100.0
        IMG_W, IMG_H, FOV = self.configure_topdown_camera(0.1, 400.0, 40.0, HEIGHT)
        print(IMG_W, IMG_H, FOV, HEIGHT)
        #IMG_W, IMG_H = 1024, 1024
        #FOV = 45.0
        #HEIGHT = 100.0
        bp_lib = self.world.get_blueprint_library()
        cam_bp = bp_lib.find('sensor.camera.rgb')  # or sensor.camera.rgb
        cam_bp.set_attribute('image_size_x', str(IMG_W))
        cam_bp.set_attribute('image_size_y', str(IMG_H))
        cam_bp.set_attribute('fov', str(FOV))
        cam_bp.set_attribute('sensor_tick', str(1.0/FPS))
        cam_transform = carla.Transform(
            carla.Location(x=0.0, z=HEIGHT),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
        )
        actor_list = self.world.get_actors()
        vehicle = actor_list.find(actor_id)

        camera = self.world.spawn_actor(
            cam_bp, cam_transform, attach_to=vehicle,
            attachment_type=carla.AttachmentType.Rigid,
        )
        
        if self.record_videos:
            self.hero_cameras[video_name] = camera
            self.hero_camera_queues[video_name] = queue.Queue()
            camera.listen(self.hero_camera_queues[video_name].put)
            self.hero_camera_writers[video_name] = imageio.get_writer(
                video_name, fps=FPS, codec="libx264", quality=8,
                macro_block_size=1,                      # don't auto-resize odd dimensions
                ffmpeg_params=["-pix_fmt", "yuv420p"],   # compatibility across players
            )

            # Create front-facing rear-based camera view
            video_name = self.carla_video_output_prefix + "_front_view.mp4"
            FPS = self.FPS
            IMG_W, IMG_H, FOV =  2560, 1440, 90
            bp_lib = self.world.get_blueprint_library()
            cam_bp = bp_lib.find('sensor.camera.rgb')  # or sensor.camera.rgb
            cam_bp.set_attribute('image_size_x', str(IMG_W))
            cam_bp.set_attribute('image_size_y', str(IMG_H))
            cam_bp.set_attribute('fov', str(FOV))
            cam_bp.set_attribute('sensor_tick', str(1.0/FPS))
            cam_transform = carla.Transform(
                carla.Location(x=-12.0, z=5.0),
                carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
            )
            actor_list = self.world.get_actors()
            vehicle = actor_list.find(actor_id)

            camera = self.world.spawn_actor(
                cam_bp, cam_transform, attach_to=vehicle,
                attachment_type=carla.AttachmentType.Rigid,
            )
            
            self.hero_cameras[video_name] = camera
            self.hero_camera_queues[video_name] = queue.Queue()
            camera.listen(self.hero_camera_queues[video_name].put)
            self.hero_camera_writers[video_name] = imageio.get_writer(
                video_name, fps=FPS, codec="libx264", quality=8,
                macro_block_size=1,                      # don't auto-resize odd dimensions
                ffmpeg_params=["-pix_fmt", "yuv420p"],   # compatibility across players
            )

            # Create rear-facing front-based camera view
            video_name = self.carla_video_output_prefix + "_rear_view.mp4"
            FPS = self.FPS
            IMG_W, IMG_H, FOV =  2560, 1440, 90
            bp_lib = self.world.get_blueprint_library()
            cam_bp = bp_lib.find('sensor.camera.rgb')  # or sensor.camera.rgb
            cam_bp.set_attribute('image_size_x', str(IMG_W))
            cam_bp.set_attribute('image_size_y', str(IMG_H))
            cam_bp.set_attribute('fov', str(FOV))
            cam_bp.set_attribute('sensor_tick', str(1.0/FPS))
            cam_transform = carla.Transform(
                carla.Location(x=12.0, z=5.0),
                carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0),
            )
            actor_list = self.world.get_actors()
            vehicle = actor_list.find(actor_id)

            camera = self.world.spawn_actor(
                cam_bp, cam_transform, attach_to=vehicle,
                attachment_type=carla.AttachmentType.Rigid,
            )
            
            self.hero_cameras[video_name] = camera
            self.hero_camera_queues[video_name] = queue.Queue()
            camera.listen(self.hero_camera_queues[video_name].put)
            self.hero_camera_writers[video_name] = imageio.get_writer(
                video_name, fps=FPS, codec="libx264", quality=8,
                macro_block_size=1,                      # don't auto-resize odd dimensions
                ffmpeg_params=["-pix_fmt", "yuv420p"],   # compatibility across players
            )

        self.hero_state = new_hero_state

        #self.pushVehicle(new_hero_state, cosim_data["velocity"])

    def spawn_visible_vehicle(self, cosim_data, blueprint_id=None):
        actor_id = self.spawn_vehicle_from_co_sim(cosim_data, blueprint_id)
        #print("visible", cosim_data["id"])
        if actor_id is None:
            print("Failed to spawn visible!")
            print(cosim_data)
            self.coupler.credit_lost_vehicle(int(cosim_data["lane_id"]), self._spawn_side_for(self.coupler.get_hero_data(), cosim_data))
            return
        new_visible_state = {
            "carla_actor_id": actor_id,
            "cosim_data": cosim_data,
            "just_spawned_cooldown": 1,
            "autopilot_enabled": False,
            "pushed": False,
        }
        self.visible_states[cosim_data["id"]] = new_visible_state
        #self.pushVehicle(new_visible_state, cosim_data["velocity"])

    def despawn_hero_vehicle(self):
        self.despawn_carla_vehicle(self.hero_state)
        self.hero_state = None

    def despawn_visible_vehicle(self, visible_state):
        self.despawn_carla_vehicle(visible_state)
        self.visible_states.pop(visible_state["cosim_data"]["id"], None)
    
    def reset_visible_states(self):
        self.visible_states = {}

    def get_blueprint_length(self, blueprint_id):
        return self.vehicle_blueprints_bounds[blueprint_id]["xmax"] - self.vehicle_blueprints_bounds[blueprint_id]["xmin"]
    
    def get_blueprint_width(self, blueprint_id):
        return self.vehicle_blueprints_bounds[blueprint_id]["ymax"] - self.vehicle_blueprints_bounds[blueprint_id]["ymin"]

    def get_vehicle_length(self, vehicle_state):
        vehicle_actor = self.world.get_actors([vehicle_state["carla_actor_id"]])[0]
        blueprint_id = vehicle_actor.type_id
        return self.get_blueprint_length(blueprint_id)
    
    def get_vehicle_width(self, vehicle_state):
        vehicle_actor = self.world.get_actors([vehicle_state["carla_actor_id"]])[0]
        blueprint_id = vehicle_actor.type_id
        return self.get_blueprint_width(blueprint_id)

    def rebuild_co_sim_vehicle_state(self, vehicle_state):
        vehicle_actors = self.world.get_actors([vehicle_state["carla_actor_id"]])
        if (len(vehicle_actors) == 0):
            return None
        vehicle_actor = vehicle_actors[0]
        location = vehicle_actor.get_location()
        closest_waypoint = self.carla_map.get_waypoint(location, project_to_road=True)
        vehicle_road_id = closest_waypoint.road_id
        vehicle_lane_id = closest_waypoint.lane_id
        return {
            "id": vehicle_state["cosim_data"]["id"],
            "class": vehicle_state["cosim_data"]["class"],
            "length": self.get_vehicle_length(vehicle_state),
            "width": self.get_vehicle_width(vehicle_state),
            "time": vehicle_state["cosim_data"]["time"],
            "s": self.offset_s_by_blueprint_for_co_sim(closest_waypoint.s, vehicle_actor.type_id),
            "t": (vehicle_lane_id * 3.5) + 1.75, # Hack kludge to estimate
            "velocity": float(vehicle_actor.get_velocity().length()),
            "lane_id": vehicle_lane_id,
            "road_id": vehicle_road_id
        }
    def rebuild_co_sim_hero_state(self):
        return self.rebuild_co_sim_vehicle_state(self.hero_state)

    def rebuild_co_sim_visible_state(self, hero_rebuilt=None, visible_rebuilt=None):
        cosim_visible_states = {}
        road_str = str(self.coupler.hero_road)
        for lane_count in range(1, self.mapping[road_str]["lanes"] + 1):
            lane_id = lane_count * -1
            cosim_visible_states[lane_id] = {}
        for cosim_id in self.visible_states:
            rebuilt_state = self.rebuild_co_sim_vehicle_state(self.visible_states[cosim_id])
            if (rebuilt_state is not None):
                cosim_visible_states[rebuilt_state["lane_id"]][rebuilt_state["id"]] = rebuilt_state
        if (visible_rebuilt is not None):
            for lane in visible_rebuilt:
                for cosim_id in visible_rebuilt[lane]:
                    in_rebuilt = False
                    for lane2 in cosim_visible_states:
                        for cosim_id_rebuilt in cosim_visible_states[lane2]:
                            in_rebuilt = in_rebuilt or (cosim_id_rebuilt == cosim_id)
                    if (not in_rebuilt):
                        self.coupler.credit_lost_vehicle(int(visible_rebuilt[lane][cosim_id]["lane_id"]), self._spawn_side_for(hero_rebuilt, visible_rebuilt[lane][cosim_id]))
                        self.visible_states.pop(cosim_id)

        return cosim_visible_states

    def spawn_and_despawn_visible_vehicles_from_co_sim(self):
        cosim_visible_vehicles = self.coupler.get_visible_data()
        self.reset_visible_states()
        for lane in cosim_visible_vehicles:
            for id in cosim_visible_vehicles[lane]:
                self.spawn_visible_vehicle(cosim_visible_vehicles[lane][id])

    def update_visible_vehicles_from_co_sim(self):
        current_visible_state_ids = list(self.visible_states.keys())
        cosim_visible_vehicles = self.coupler.get_visible_data()
        cosim_visible_vehicles_ids = self.coupler.get_visible_ids_flat()
        for id in current_visible_state_ids:
            if id not in cosim_visible_vehicles_ids:
                self.despawn_visible_vehicle(self.visible_states[id])
        for lane in cosim_visible_vehicles:
            for id in cosim_visible_vehicles[lane]:
                if id not in self.visible_states: #New vehicle!
                    self.spawn_visible_vehicle(cosim_visible_vehicles[lane][id])

    def get_carla_actor_id(self, cosim_id):
        if (cosim_id in self.visible_states):
            return self.visible_states[cosim_id]["carla_actor_id"]
        elif (self.hero_state["cosim_data"]["id"] == cosim_id):
            return self.hero_state["carla_actor_id"]
        return None
    
    def get_internal_vehicle_state(self, cosim_id):
        if (cosim_id in self.visible_states):
            return self.visible_states[cosim_id]
        elif (self.hero_state["cosim_data"]["id"] == cosim_id):
            return self.hero_state
        return None

    def update_hero_and_visible_vehicles_from_tm(self, hero_rebuilt, visible_rebuilt):
        vehicle_actor = self.world.get_actors([self.hero_state["carla_actor_id"]])[0]
        self.tm.set_desired_speed(vehicle_actor, self.hero_vehicle_speed)
        #self.tm.set_desired_speed(vehicle_actor, 70.0)
        cosim_lead_vehicles = {}
        cosim_nominal_vehicle_actor_ids = []
        cosim_tm_controlled_vehicles = []
        cosim_pushed_vehicles = []
        road_str = str(self.coupler.hero_road)
        min_s, max_s = hero_rebuilt["s"] - self.coupler.visible_window, hero_rebuilt["s"] + self.coupler.visible_window
        middle_s = (max_s + min_s) / 2.0
        lead_vehicle_region_start = max_s - self.coupler.transition_region_size
        for lane_count in range(1, self.mapping[road_str]["lanes"] + 1):
            lane_id = lane_count * -1
            #potential_cosim_vehicles = [self.visible_states[cosim_id] for cosim_id in self.visible_states if self.visible_states[cosim_id]["cosim_data"]["lane_id"] == lane_id]
            potential_cosim_vehicles = [visible_rebuilt[lane_id][cosim_id] for cosim_id in visible_rebuilt[lane_id]]
            potential_cosim_vehicles.append(hero_rebuilt)
            potential_cosim_vehicles_sorted = sorted(potential_cosim_vehicles, key=lambda entry: entry["s"], reverse=True)
            cosim_lead_vehicles[lane_id] = []
            for entry in potential_cosim_vehicles_sorted:
                if (entry["id"] != hero_rebuilt["id"]) \
                and (entry["s"] > lead_vehicle_region_start):
                    if (len(cosim_lead_vehicles[lane_id]) == 0):
                        cosim_lead_vehicles[lane_id].append(entry)
                    elif (cosim_lead_vehicles[lane_id][0]["s"] < entry["s"]):
                        cosim_lead_vehicles[lane_id][0] = entry
                if (self.get_internal_vehicle_state(entry["id"])["just_spawned_cooldown"] == 0):
                    cosim_tm_controlled_vehicles.append([entry, self.get_carla_actor_id(entry["id"])])
                else:
                    cosim_pushed_vehicles.append([entry, self.get_carla_actor_id(entry["id"])])
            for entry in potential_cosim_vehicles_sorted:
                if (self.get_carla_actor_id(entry["id"]) not in [self.get_carla_actor_id(cosim_lead_vehicle["id"]) for cosim_lead_vehicle in cosim_lead_vehicles[lane_id]]):
                    cosim_nominal_vehicle_actor_ids.append(self.get_carla_actor_id(entry["id"]))

        for entry, actor_id in cosim_tm_controlled_vehicles:
            self.world.get_actors([actor_id])[0].disable_constant_velocity()
        batch = []
        for entry, actor_id in cosim_tm_controlled_vehicles:
            entry_internal = self.get_internal_vehicle_state(entry["id"])
            if (not entry_internal["autopilot_enabled"]):
                batch.append(carla.command.SetAutopilot(actor_id, True, self.tm_port))
                entry_internal["autopilot_enabled"] = True
        self.client.apply_batch_sync(batch, False)
        for entry, actor_id in cosim_tm_controlled_vehicles:
            entry_internal = self.get_internal_vehicle_state(entry["id"])
            if (entry_internal["autopilot_enabled"]):
                self.tm.auto_lane_change(self.world.get_actors([actor_id])[0], True)
        batch = []
        for entry, actor_id in cosim_pushed_vehicles:
            batch.append(carla.command.ApplyTargetVelocity(actor_id, carla.Vector3D(*self.get_internal_vehicle_state(entry["id"])["cosim_data"]["temporary_velocity_push"])))
        self.client.apply_batch_sync(batch, False)

        for actor_id in cosim_nominal_vehicle_actor_ids:
            cosim_actor = self.world.get_actors([actor_id])[0]
            self.tm.set_desired_speed(cosim_actor, self.hero_vehicle_speed)
        for lane_count in range(1, self.mapping[road_str]["lanes"] + 1):
            lane_id = lane_count * -1
            cosim_lane_lead_vehicles = cosim_lead_vehicles[lane_id]
            cosim_ghost_lead_state = self.coupler.get_lowest_ahead_ghost_vehicle(lane_id)
            cosim_ghost_lead_velocity = cosim_ghost_lead_state["velocity"] * self.mps_to_kph if cosim_ghost_lead_state is not None else self.coupler.get_ahead_lane_velocity_macro(lane_id, self.hero_vehicle_speed / self.mps_to_kph) * self.mps_to_kph
            for cosim_lane_lead_vehicle in cosim_lane_lead_vehicles:
                cosim_actor = self.world.get_actors([self.get_carla_actor_id(cosim_lane_lead_vehicle["id"])])[0]
                self.tm.set_desired_speed(cosim_actor, cosim_ghost_lead_velocity)

    def update_co_sim(self, hero_rebuilt_prev = None, visible_rebuilt_prev = None):
        hero_rebuilt = self.rebuild_co_sim_hero_state()
        visible_rebuilt = self.rebuild_co_sim_visible_state(hero_rebuilt_prev, visible_rebuilt_prev)
        return (hero_rebuilt, visible_rebuilt)

    def update_co_sim_snapshot_data(self):
        vehicles = []
        hero_state = self.coupler.get_hero_data()
        visible_state = self.coupler.get_visible_data()
        ghost_state = self.coupler.get_ghost_data()
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

    def fetch_data(self):
        self.update_co_sim_snapshot_data()
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
                #print("skipping!")
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
        #print("Velocities ", numpy.mean(velocities), self.current_timestamp)
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

    def mark_heroes_and_visibles_in_previous_step_as_spawned(self):
        self.hero_state["just_spawned_cooldown"] = self.hero_state["just_spawned_cooldown"] - 1 if self.hero_state["just_spawned_cooldown"] > 0 else 0
        for cosim_id in self.visible_states:
            self.visible_states[cosim_id]["just_spawned_cooldown"] = self.visible_states[cosim_id]["just_spawned_cooldown"] - 1 if self.visible_states[cosim_id]["just_spawned_cooldown"] > 0 else 0

    def push_eligible_vehicles(self):
        if (self.hero_state["just_spawned_cooldown"] == 0) and (not self.hero_state["pushed"]):
            self.push_vehicle(self.hero_state, self.hero_state["cosim_data"]["velocity"])
            self.hero_state["pushed"] = True
            self.tm.auto_lane_change(self.world.get_actors([self.hero_state["carla_actor_id"]])[0], False)
        for cosim_id in self.visible_states:
            if (self.visible_states[cosim_id]["just_spawned_cooldown"] == 0) and (not self.visible_states[cosim_id]["pushed"]):
                self.push_vehicle(self.visible_states[cosim_id], self.visible_states[cosim_id]["cosim_data"]["velocity"])
                self.visible_states[cosim_id]["pushed"] = True

    def initialize_simulation(self):
        vehicle_list = self.world.get_actors().filter('vehicle.*')
        self.client.apply_batch_sync([carla.command.DestroyActor(x.id) for x in vehicle_list], True)

        camera_list = self.world.get_actors().filter('sensor.camera.*')
        self.client.apply_batch_sync([carla.command.DestroyActor(x.id) for x in camera_list], True)
        
        #print(f'Successfully destroyed {len(vehicle_list)} vehicles.')
        self.spawn_hero_vehicle("vehicle.audi.a2") # Lets keep us at it an audi for now ;)!
        self.spawn_and_despawn_visible_vehicles_from_co_sim()

    def configure_topdown_camera(self, mpp, cover_fwd_back, cover_lateral, height):
        img_w = round(cover_lateral  / mpp)   # horizontal extent
        img_h = round(cover_fwd_back / mpp)   # vertical extent (front + behind)
        fov   = math.degrees(2 * math.atan(img_w * mpp / (2 * height)))
        return img_w, img_h, fov

    def carla_to_bgr(self, image):
        """carla.Image (BGRA) -> H x W x 3 uint8 BGR array for OpenCV."""
        arr = numpy.frombuffer(image.raw_data, dtype=numpy.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        return arr[:, :, :3]
    
    def carla_to_rgb(self, image):
        """carla.Image (BGRA) -> H x W x 3 uint8 BGR array for OpenCV."""
        arr = numpy.frombuffer(image.raw_data, dtype=numpy.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        arr = arr[:, :, :3]
        arr = arr[:, :, ::-1]
        return arr
    
    def run_simulation_over(self, t=1.0, eps=1e-2):
        starting_timestamp = self.current_timestamp
        hero_rebuilt, visible_rebuilt = self.update_co_sim()
        while (self.current_timestamp < (starting_timestamp + t - eps)):
            self.update_hero_and_visible_vehicles_from_tm(hero_rebuilt, visible_rebuilt)
            self.world.tick()
            for video_name, video_queue in self.hero_camera_queues.items():
                if (not video_queue.empty()):
                    new_hero_image = video_queue.get()
                    self.hero_camera_writers[video_name].append_data(self.carla_to_rgb(new_hero_image))
            self.fetch_data()
            self.mark_heroes_and_visibles_in_previous_step_as_spawned()
            self.push_eligible_vehicles()
            hero_rebuilt, visible_rebuilt = self.update_co_sim(hero_rebuilt, visible_rebuilt)
            self.current_timestamp += self.tick_step
        print(self.current_timestamp)
        #print(hero_rebuilt, visible_rebuilt)
        return hero_rebuilt, visible_rebuilt
    
    def update_simulation(self):
        self.update_visible_vehicles_from_co_sim()

    def destroy_simulation(self):
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
            #print(f"Despawned {len(all_cars)} cars!")

            for video_name, writer in self.hero_camera_writers.items():
                writer.close()
                self.hero_cameras[video_name].stop()
        except:
            pass