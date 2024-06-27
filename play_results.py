import argparse
import pdb
import joblib
import numpy as np
import torch
import time
import copy
from lib.models import build_body_model
from lib.utils.transforms import matrix_to_euler_angles
from pydrake.all import (
    StartMeshcat,
    Meshcat,
    RigidTransform,
    Rgba,
    Cylinder,
    AngleAxis,
    MeshcatVisualizer,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    RollPitchYaw,
    GeometrySet,
    CollisionFilterDeclaration,
    Solve,
    InverseKinematics,
    RotationMatrix,
    )
import re
import xml.etree.ElementTree as ET
from pydrake.visualization._model_visualizer import (
    ModelVisualizer as _ModelVisualizer,
)
import pandas as pd
from datetime import datetime

def add_punyo(
        plant, scene_graph=None,
        env_origin=RigidTransform()):
    parser = Parser(plant)

    model_file = "punyo.urdf"

    robot, = parser.AddModels(model_file)
    plant.WeldFrames(
        plant.world_frame(), plant.GetFrameByName("torso", robot),
        env_origin.multiply(RigidTransform(
            RollPitchYaw(0, 0, 0),
            np.array([0, 0, 0])))
    )

    if scene_graph is not None:
        # Collision filters.
        filter_manager = scene_graph.collision_filter_manager()
        # TODO(josebarreiros): verify these collisions.
        body_pairs = [
            ["floatie_2.0a_on_link_2_r", "j2s7s300_link_base_r"],
            ["floatie_2.0a_on_link_2_l", "j2s7s300_link_base_l"],

            ["floatie_2.0a_on_link_2_r", "j2s7s300_link_1_r"],
            ["floatie_2.0a_on_link_2_l", "j2s7s300_link_1_l"],

            ["floatie_2.0b_on_link_2_r", "j2s7s300_link_3_r"],
            ["floatie_2.0b_on_link_2_l", "j2s7s300_link_3_l"],

            ["floatie_2.0b_on_link_2_r", "floatie_2.0b_on_link_4_r"],
            ["floatie_2.0b_on_link_2_l", "floatie_2.0b_on_link_4_l"],

            ["floatie_2.0b_on_link_2_r", "j2s7s300_link_4_r"],
            ["floatie_2.0b_on_link_2_l", "j2s7s300_link_4_l"],

            ["floatie_2.0a_on_link_2_r", "torso"],
            ["floatie_2.0a_on_link_2_l", "torso"],
            ["floatie_2.0b_on_link_2_r", "torso"],
            ["floatie_2.0b_on_link_2_l", "torso"],

            ["floatie_2.0a_on_link_3_r", "j2s7s300_link_4_r"],
            ["floatie_2.0a_on_link_3_l", "j2s7s300_link_4_l"],

            ["floatie_2.0b_on_link_3_r", "j2s7s300_link_4_r"],
            ["floatie_2.0b_on_link_3_l", "j2s7s300_link_4_l"],

            ["floatie_2.0a_on_link_4_r", "j2s7s300_link_3_r"],
            ["floatie_2.0a_on_link_4_l", "j2s7s300_link_3_l"],

            ["floatie_2.0b_on_link_4_r", "j2s7s300_link_3_r"],
            ["floatie_2.0b_on_link_4_l", "j2s7s300_link_3_l"],

            ["floatie_2.0a_on_link_4_r", "floatie_2.0a_on_link_3_r"],
            ["floatie_2.0a_on_link_4_r", "floatie_2.0b_on_link_3_r"],
            ["floatie_2.0a_on_link_4_l", "floatie_2.0a_on_link_3_l"],
            ["floatie_2.0a_on_link_4_l", "floatie_2.0b_on_link_3_l"],

            ["floatie_2.0b_on_link_4_r", "floatie_2.0a_on_link_3_r"],
            ["floatie_2.0b_on_link_4_r", "floatie_2.0b_on_link_3_r"],
            ["floatie_2.0b_on_link_4_l", "floatie_2.0a_on_link_3_l"],
            ["floatie_2.0b_on_link_4_l", "floatie_2.0b_on_link_3_l"],

            ["floatie_1.0_on_link_6_r", "j2s7s300_link_4_r"],
            ["floatie_1.0_on_link_6_l", "j2s7s300_link_4_l"],
            ["floatie_1.0_on_link_6_r", "j2s7s300_link_5_r"],
            ["floatie_1.0_on_link_6_l", "j2s7s300_link_5_l"],
            ["floatie_1.0_on_link_5_r", "j2s7s300_link_6_r"],
            ["floatie_1.0_on_link_5_l", "j2s7s300_link_6_l"],
            ["floatie_1.0_on_link_6_r", "floatie_1.0_on_link_5_r"],
            ["floatie_1.0_on_link_6_l", "floatie_1.0_on_link_5_l"],
        ]

        for pair in body_pairs:
            parent = plant.GetBodyByName(pair[0])
            child = plant.GetBodyByName(pair[1])
            set = GeometrySet(
                plant.GetCollisionGeometriesForBody(parent) +
                plant.GetCollisionGeometriesForBody(child))
            filter_manager.Apply(
                declaration=CollisionFilterDeclaration().ExcludeWithin(
                    set))
    return robot


def get_smpl_joint_names():
    return [
        'hips',            # 0
        'leftUpLeg',       # 1
        'rightUpLeg',      # 2
        'spine',           # 3
        'leftLeg',         # 4
        'rightLeg',        # 5
        'spine1',          # 6
        'leftFoot',        # 7
        'rightFoot',       # 8
        'spine2',          # 9
        'leftToeBase',     # 10
        'rightToeBase',    # 11
        'neck',            # 12
        'leftShoulder',    # 13
        'rightShoulder',   # 14
        'head',            # 15
        'leftArm',         # 16
        'rightArm',        # 17
        'leftForeArm',     # 18
        'rightForeArm',    # 19
        'leftHand',        # 20
        'rightHand',       # 21
        'leftHandIndex1',  # 22
        'rightHandIndex1', # 23
    ]

def get_smpl_skeleton():
    return np.array(
        [
            [ 0, 1 ],
            [ 0, 2 ],
            [ 0, 3 ],
            [ 1, 4 ],
            [ 2, 5 ],
            [ 3, 6 ],
            [ 4, 7 ],
            [ 5, 8 ],
            [ 6, 9 ],
            [ 7, 10],
            [ 8, 11],
            [ 9, 12],
            [ 9, 13],
            [ 9, 14],
            [12, 15],
            [13, 16],
            [14, 17],
            [16, 18],
            [17, 19],
            [18, 20],
            [19, 21],
            [20, 22],
            [21, 23],
        ]
    )


def add_triad(
    vis: Meshcat,
    name: str,
    prefix: str,
    length=1.0,
    radius=0.04,
    opacity=1.0,
    Xt=RigidTransform(),
    axes_rgb=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
):
    """
    Initializes coordinate axes of a frame T. The x-axis is drawn red,
    y-axis green and z-axis blue. The axes point in +x, +y and +z directions,
    respectively.
    Args:
        vis: a meshcat.Visualizer object.
        name: (string) the name of the triad in meshcat.
        prefix: (string) name of the node in the meshcat tree to which this
            triad is added.
        length: the length of each axis in meters.
        radius: the radius of each axis in meters.
        opacity: the opacity of the coordinate axes, between 0 and 1.
    """
    delta_xyz = np.array(
        [[length / 2, 0, 0], [0, length / 2, 0], [0, 0, length / 2]]
    )

    axes_name = ["x", "y", "z"]
    axes_color = [
        Rgba(axes_rgb[0][0], axes_rgb[0][1], axes_rgb[0][2], opacity),
        Rgba(axes_rgb[1][0], axes_rgb[1][1], axes_rgb[1][2], opacity),
        Rgba(axes_rgb[2][0], axes_rgb[2][1], axes_rgb[2][2], opacity),
    ]
    rotation_axes = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]

    for i in range(3):
        path = f"{prefix}/{name}/{axes_name[i]}"
        vis.SetObject(path, Cylinder(radius, length), axes_color[i])
        X = RigidTransform(
            AngleAxis(np.pi / 2, rotation_axes[i]), delta_xyz[i]
        )
        vis.SetTransform(path, Xt @ X)

def matrix_poses_to_euler(matrix_poses):
    matrix_poses = torch.from_numpy(matrix_poses)
    euler_poses=[]
    for pose in matrix_poses:
        euler_poses.append(matrix_to_euler_angles(pose, 'XYZ').numpy())
    return euler_poses


class Human:
    def __init__(self, meshcat):
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0)
        parser = Parser(plant)
        human = parser.AddModels(args.model_file)[0]
        plant.WeldFrames(
            plant.world_frame(), plant.GetFrameByName("body_j0", human),
            RigidTransform(RollPitchYaw(-np.pi/2,0,np.pi/2),
                            np.array([0, 0, -0.25]))
        )
        plant.Finalize()
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
        self.diagram = builder.Build()
        self.plant = plant
        self.diagram_context = self.diagram.CreateDefaultContext()
        self.plant_context = plant.GetMyContextFromRoot(self.diagram_context)

    def get_joint_pose(self, joint_idx):
        frame = self.plant.GetFrameByName(f"body_j{joint_idx}")
        return frame.CalcPoseInWorld(self.plant_context)

class Punyo:
    def __init__(self, meshcat):
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0)

        self.env_origin = RigidTransform(RollPitchYaw([0,0,0]), [0,2,0])
        add_punyo(plant, scene_graph=None, env_origin= self.env_origin)
        plant.Finalize()
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
        self.diagram = builder.Build()
        self.plant = plant
        self.diagram_context = self.diagram.CreateDefaultContext()
        self.plant_context = plant.GetMyContextFromRoot(self.diagram_context)
        self.Na = plant.num_actuators()

        self.origin_frame = self.plant.GetFrameByName(f"torso")
        self.world_frame = self.plant.world_frame()

        # R, L
        self.end_effector_frames = [
            self.plant.GetFrameByName(f"paw_link_r"),
            self.plant.GetFrameByName(f"paw_link_l"),
        ]
        self.elbow_frames = [
            self.plant.GetFrameByName(f"j2s7s300_link_4_r"),
            self.plant.GetFrameByName(f"j2s7s300_link_4_l"),
        ]

        self.obs_type = ["state", "robot_dof", "robot_vel", "end_effectors_rpyxyz", "ee_pose", "manipuland_rpyxyz"]

        # Dictionary helper {"observation_type": (start_idx, end_idx)}.
        self.obs_view_helper = {}
        self.obs_size = 0
        if "state" in self.obs_type:
            # Adds the measured positions and velocities
            # to the observation.
            start_idx = self.obs_size
            self.obs_size += self.Na * 2
            self.obs_view_helper["state"] = (start_idx, self.obs_size)

        if "robot_dof" in self.obs_type:
            # Adds the measured positions and velocities
            # to the observation.
            start_idx = self.obs_size
            self.obs_size += self.Na
            self.obs_view_helper["robot_dof"] = (start_idx, self.obs_size)

        if "robot_vel" in self.obs_type:
            # Adds the measured positions and velocities
            # to the observation.
            start_idx = self.obs_size
            self.obs_size += self.Na
            self.obs_view_helper["robot_vel"] = (start_idx, self.obs_size)

        if "end_effectors_rpyxyz" in self.obs_type:
            # Adds the pose (xyz_l, quat_l, xyz_r, quat_r) of
            # the end effectors
            # in the world frame.
            start_idx = self.obs_size
            self.obs_size += 12
            self.obs_view_helper["end_effectors_rpyxyz"] = (
                start_idx,
                self.obs_size,
            )

        if "ee_pose" in self.obs_type:
            # Adds the pose (xyz_l, quat_l, xyz_r, quat_r) of
            # the end effectors
            # in the world frame.
            start_idx = self.obs_size
            self.obs_size += 14
            self.obs_view_helper["ee_pose"] = (
                start_idx,
                self.obs_size,
            )

        if "manipuland_rpyxyz" in self.obs_type:
            # Adds the pose (xyz_l, quat_l, xyz_r, quat_r) of
            # the end effectors
            # in the world frame.
            start_idx = self.obs_size
            self.obs_size += 7
            self.obs_view_helper["manipuland_rpyxyz"] = (
                start_idx,
                self.obs_size,
            )
        # breakpoint()

    def retarget(self, h_end_effector_poses, h_elbow_poses, q_current):
        # breakpoint()
        ik = InverseKinematics(
            self.plant,
            self.plant_context,
        )

        prog = ik.get_mutable_prog()
        q = ik.q()
        TOL = 0.2
        # breakpoint()
        F = 1.
        for i, pose in enumerate(h_end_effector_poses):
            # cartesian constraint
            # ik.AddPositionConstraint(
            #     frameB=self.end_effector_frames[i],
            #     p_BQ=[0, 0, 0],
            #     frameAbar=self.origin_frame,
            #     X_AbarA=pose,
            #     p_AQ_lower=-TOL * np.ones(3),
            #     p_AQ_upper=TOL * np.ones(3),
            # )
            ik.AddPositionCost(
            frameA=self.end_effector_frames[i],
            p_AP=[0, 0, 0],
            frameB=self.world_frame,
            # frameB=self.origin_frame,
            p_BQ=pose.translation()*F + self.env_origin.translation(),
            C=10 * np.eye(3),
            )

        for i, pose in enumerate(h_elbow_poses):
        #     # cartesian constraint
        #     # ik.AddPositionConstraint(
        #     #     frameB=self.elbow_frames[i],
        #     #     p_BQ=[0, 0, 0],
        #     #     frameAbar=self.origin_frame,
        #     #     X_AbarA=pose,
        #     #     p_AQ_lower=-TOL * np.ones(3),
        #     #     p_AQ_upper=TOL * np.ones(3),
        #     # )
            ik.AddPositionCost(
            frameA=self.elbow_frames[i],
            p_AP=[0, 0, 0],
            frameB=self.world_frame,
            p_BQ=pose.translation()*F + self.env_origin.translation(),
            C=10 * np.eye(3),
            )
            # breakpoint()
            ik.AddOrientationCost(
            frameAbar=self.elbow_frames[i],
            R_AbarA=RotationMatrix().multiply(RollPitchYaw([np.pi/2,0,0]).ToRotationMatrix()),
            frameBbar=self.world_frame,
            R_BbarB=pose.rotation(),
            c=10,
            )

        prog.SetInitialGuess(q, q_current)
        result = Solve(ik.prog())

        # if self.debug:
        # Print result details.
        print(f"\tSolver type: {result.get_solver_id().name()}")
        # print(f"\tSolver took {time.time() - t_start} s")
        print(f"\tSuccess: {result.is_success()}")
        print(f"\tOptimal cost: {result.get_optimal_cost()}")
        print(f"Exit condition: {result.get_solver_details().info}")

        if result.is_success():
            q_new = result.GetSolution(q)
        else:
            print("Solver did not converged.")
            q_new = copy.copy(q_current)

        return q_new

    def obs_to_list(self, obs):
        obs_list = []
        for key in self.obs_view_helper.keys():
            (start_idx, end_idx) = self.obs_view_helper[key]
            obs_list.append(np.array(obs[start_idx:end_idx]))
        return obs_list

    def get_obs(self, state):
        manipuland_pose = [0,0,0, 0.3, 0, 0.2]
        ee_pose = []
        ee_rpy_xyz_pose = []
        for ee_frame in self.end_effector_frames[::-1]:
            # this should be L, R
            X =ee_frame.CalcPoseInWorld(self.plant_context)
            # breakpoint()
            ee_pose+= (X.translation() - self.env_origin.translation()).tolist()
            ee_pose+= X.rotation().ToQuaternion().wxyz().tolist()
            # breakpoint()
            ee_rpy_xyz_pose+=  X.rotation().ToRollPitchYaw().vector().tolist()
            ee_rpy_xyz_pose+=  (X.translation() - self.env_origin.translation()).tolist()
        observations = np.array([])

        if "state" in self.obs_type:
            observations = np.concatenate(
                (observations, state))

        if "robot_dof" in self.obs_type:
            observations = np.concatenate(
                (observations, state[:14]))

        if "robot_vel" in self.obs_type:
            observations = np.concatenate(
                (observations, state[14:]))

        if "ee_pose" in self.obs_type:
            observations = np.concatenate(
                (observations, ee_pose))

        if "end_effectors_rpyxyz" in self.obs_type:
            observations = np.concatenate(
                (observations, ee_rpy_xyz_pose))

        if "manipuland_rpyxyz" in self.obs_type:
            # breakpoint()
            observations = np.concatenate(
                (observations, manipuland_pose))

        return observations

if __name__ == '__main__':
    np.set_printoptions(precision=3)
    date = datetime.now().strftime("%Y%m%d_%H_%M")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_path",
        help="path to the results directory.",
        required=True,
        )
    parser.add_argument(
        "--model_file",
        help="path to sdf.",
        required=True,
        )
    parser.add_argument(
        "--pkl_output_path",
        help="path to the logs directory.",
        )
    args = parser.parse_args()

    tracking_results = joblib.load(args.results_path + f"/tracking_results.pth")
    wham_results = joblib.load(args.results_path + f"/wham_results.pth")

    smpl = build_body_model('cpu')
    init_output = smpl.get_output(
            global_orient=tracking_results[0]['init_global_orient'],
            body_pose=tracking_results[0]['init_body_pose'],
            betas=tracking_results[0]['init_betas'],
            pose2rot=False,
            return_full_pose=False
        )

    meshcat = StartMeshcat()

    human = Human(meshcat)
    punyo = Punyo(meshcat)

    q0 = [  1., np.pi/2, 4.71, np.pi, np.pi, np.pi, 4.171,
            2.14, np.pi/2, 4.71, np.pi, np.pi, np.pi, 1.047
            ]
    punyo.plant.SetPositions(punyo.plant_context, q0)
    punyo.diagram.ForcedPublish(punyo.diagram_context)

    human.diagram.ForcedPublish(human.diagram_context)
    Np = human.plant.num_positions()
    Ns = human.plant.num_multibody_states()
    print(f"num_positions: {Np}, num_states: {Ns}")
    joint_names_to_positions = human.plant.GetPositionNames(False,False)
    joint_ids_to_positions = [int(name.split("_")[1]) for name in joint_names_to_positions]
    joint_ids_to_positions = list(dict.fromkeys(joint_ids_to_positions)) # remove rpy

    dt = 0.2

    # Variables for data dump.
    episode_actions = []
    episode_observations = []
    episode_rewards = []
    config = {}
    config["gym_timestep"] = dt
    config["manipuland_type"] = "None"
    config["teleop_mode"] = "wham"
    # DataFrame columns.
    y = ([("policy_action", "action")] +
         [("observations", x) for x in punyo.obs_type] +
         [("rewards", "reward")] +
         [("config", "config")])
    col_list = pd.MultiIndex.from_tuples(y)


    elapsed_time= 0
    input("enter to play")
    # breakpoint()
    for poses_body in wham_results[0]["poses_body"]:

        elapsed_time+=dt
        human.diagram_context.SetTime(elapsed_time)

        euler_poses = matrix_poses_to_euler(poses_body)
        euler_poses[2] = np.zeros(3)
        # breakpoint()
        positions = [euler_poses[joint_id-1] for joint_id in joint_ids_to_positions]
        positions = np.array(positions).flatten()
        human.plant.SetPositions(human.plant_context, positions)
        human.diagram.ForcedPublish(human.diagram_context)

        # R, L
        end_effector_poses = [human.get_joint_pose(23), human.get_joint_pose(22)]
        elbow_poses = [human.get_joint_pose(19), human.get_joint_pose(18)]
        q_current = punyo.plant.GetPositions(punyo.plant_context)
        punyo_positions = punyo.retarget(end_effector_poses, elbow_poses, q_current)
        punyo.plant.SetPositions(punyo.plant_context, punyo_positions)

        obs = punyo.get_obs(np.concatenate((punyo_positions, np.zeros(14))))
        # breakpoint()
        episode_observations.append(obs)
        episode_rewards.append(0)
        episode_actions.append(punyo_positions)

        human.diagram.ForcedPublish(human.diagram_context)
        punyo.diagram.ForcedPublish(punyo.diagram_context)
        # input("enter")
        time.sleep(dt)

    # breakpoint()
    # Append to dataframe.
    x = [(f"t{t}") for t in range(len(episode_actions))]
    row_list = pd.MultiIndex.from_tuples(x)
    obs_list = [
        punyo.obs_to_list(ep_obs)
        for ep_obs in episode_observations
    ]
    obs_list_transpose = [list(i) for i in zip(*obs_list)]
    data = ([episode_actions] +
            obs_list_transpose +
            [episode_rewards] +
            # The config dict is saved in the first row.
            [[config]+[{}]*len(episode_actions)])
    data_transpose = [list(i) for i in zip(*data)]
    df_data = pd.DataFrame(
        data_transpose, index=row_list, columns=col_list
    )
    # breakpoint()
    # Save pickle file.
    if args.pkl_output_path:
        pkl_output_path = args.pkl_output_path
    else:
        pkl_output_path= args.results_path
    data_log_path_pkl = pkl_output_path+f"/wham_{date}.pkl"
    df_data.to_pickle(data_log_path_pkl)
    print(f"Episode saved in {data_log_path_pkl}.")







