# I24-Motion CARLA Traffic Simulation Map Helper Tools
This repository allows you to simulate in CARLA real world traffic data from I-24 Motion, as per the arXiv paper https://arxiv.org/abs/2511.23236. It supports simulation of approximately a 1 mile block of I-24, from the 60.6 mile marker to the 61.6 mile marker. This consits of two roads - one eastbound (road 1) heading out of Nashville, and one Westbound (road 2) leading into Nashville. Both have 4 lanes. Currently this is setup as a demo to replicate the experiments illustrated in the arXiv paper, but the framework is flexible and can be readily applied to other trajectory data in the I-24 dataset.

We use CARLA 0.9.16 for this work.

## Link your account with Epic Games for Unreal Engine access
To have access to the Unreal Engine source (necessary for the next download step to be successful), link your github account
with Epic Games as per this guide here: [https://www.unrealengine.com/en-US/ue-on-github](https://www.unrealengine.com/en-US/ue-on-github).

## Download
Run:

        git clone https://github.com/AlexOSAdventurer/i24motion_to_carla_helpers.git

It comes with the CARLA branch and Unreal Engine source linked as submodules.

Then do:

        cd i24motion_to_carla_helpers
        git submodule init
        git submodule update

To download the CARLA and Unreal Engine branches.

## Installation and Use

This overall stack and framework is used within the compiled CARLA docker image - Utils/Docker/run.sh. 

### I-24 Motion Dataset Download
Go to https://i24motion.org/ and setup an account - this is necessary to ensure you agree to how this data is used. Then, go to https://i24motion.org/access_data and download 11-30-2022. Unzip 11-30-2022 to the i24motion_to_carla/i24motion_to_carla/11-30-2022 path. 

### Launch CARLA docker image
To continue the rest of this workflow, build and launch the CARLA docker container - it already has the needed dependencies installed:

        cd carla-0.9.16/
        Util/Docker/run.sh --dev

This will launch the docker image shell. CARLA and the i24motion_to_carla subfolder (i24motion_to_carla/i24motion_to_carla) are already mounted within the container (read/write) for you as 
1. /workspaces/carla
2. /workspaces/i24motion_to_carla

#### CARLA Map download
The I24 map is a map called "FinalMapFinal" and is stored remotely. We download it once we're in the docker container to simplify discussion of dependencies and workflow.
To download and import the map, run:

        cd /workspaces/carla
        ./I24_CARLA_download.sh
        ./I24_CARLA_import.sh

The map is now in the Unreal Engine's "/Game/Carla/Maps/FinalMapFinal" path, like the other canonical maps.

#### Process the I-24 Motion data into a parquet dataset
The I-24 motion data is post-processed into a parquet dataset to enable real-time lookup of trajectories with respect to time and space bounding boxes (as in, a bounding box on a time-space traffic diagram), even when the dataset is large. The cost is that it can take several minutes of initial preprocessing before you can use it - the reward is that you only run it once, then never again. 

Run in the CARLA docker container:

        cd /workspaces/i24motion_to_carla/scripts
        python3.8 process_i24_motion_data.py

This will result in a "road_data" folder being created inside of the scripts folder, with each parquet database corresponding to a unique road-lane pairing.

#### Run simulation
In one shell within the container, launch CARLA with `make launch`, open the "/Game/Carla/Maps/FinalMapFinal/FinalMapFinal.umap" map, and launch the simulation to expose the server.
Then, in the other shell, do:

        cd /workspaces/i24motion_to_carla/scripts
        python3.8 i24_motion_carla.py

This will generate the result CSVs and Bird's Eye View videos from the paper.
Then, execute:

        python3.8 generate_result_figures.py

This will generate the corresponding graphs.
