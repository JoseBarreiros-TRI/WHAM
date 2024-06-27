import argparse
import pdb
import joblib
import numpy as np
from lib.models import build_body_model

from pydrake.geometry import StartMeshcat, Meshcat

from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw

from pydrake.geometry import Rgba, Cylinder

from pydrake.common.eigen_geometry import AngleAxis

import re
import xml.etree.ElementTree as ET
from pydrake.visualization._model_visualizer import (
    ModelVisualizer as _ModelVisualizer,
)
import torch


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


def add_joint_visual_sdf_body(model_element, name, pose, relative_to=None):

    link_element = ET.SubElement(model_element, "link")
    link_element.set("name", name)

    pose_element = ET.SubElement(link_element, "pose")
    if relative_to is not None:
        pose_element.set("relative_to", relative_to)
    pose_element.text = pose

    visual_element = ET.SubElement(link_element, "visual")
    visual_element.set("name", "visual")
    geometry_element = ET.Element("geometry")
    geo_element = ET.SubElement(geometry_element, "sphere")
    dim_element = ET.SubElement(geo_element, "radius")
    dim_element.text = "0.02"
    visual_element.append(geometry_element)

def add_sdf_limb_body(model_element, name, pose, length, visual_pose, relative_to=None):
    name = name+"_limb"
    link_element = ET.SubElement(model_element, "link")
    link_element.set("name", name)

    pose_element = ET.SubElement(link_element, "pose")
    if relative_to is not None:
        pose_element.set("relative_to", relative_to)
    pose_element.text = pose

    visual_element = ET.SubElement(link_element, "visual")
    visual_element.set("name", "visual")
    pose_element = ET.SubElement(visual_element, "pose")
    pose_element.text = visual_pose
    geometry_element = ET.Element("geometry")
    geo_element = ET.SubElement(geometry_element, "capsule")
    dim_element = ET.SubElement(geo_element, "radius")
    dim_element.text = "0.02"
    dim_element = ET.SubElement(geo_element, "length")
    dim_element.text = length
    visual_element.append(geometry_element)

    material_element = ET.Element("material")
    diffuse_element = ET.SubElement(material_element, "diffuse")
    diffuse_element.text = "0.12 0.45 0.7 0.5"
    visual_element.append(material_element)
    return name


def add_fixed_sdf_joint(model_element, name, body_parent_name, body_child_name):
    joint_element = ET.SubElement(model_element, "joint")
    joint_element.set("name", name+"_fixed")
    joint_element.set("type", "fixed")

    parent_element = ET.SubElement(joint_element, "parent")
    parent_element.text = body_parent_name

    child_element = ET.SubElement(joint_element, "child")
    child_element.text = body_child_name


def add_revolute_sdf_joint(model_element, name, pose,
                body_parent_name, body_child_name, type, axis):
    joint_element = ET.SubElement(model_element, "joint")
    joint_element.set("name", name)
    joint_element.set("type", type)

    pose_element = ET.SubElement(joint_element, "pose")
    pose_element.text = pose

    parent_element = ET.SubElement(joint_element, "parent")
    parent_element.text = body_parent_name

    child_element = ET.SubElement(joint_element, "child")
    child_element.text = body_child_name

    axis_element = ET.SubElement(joint_element, "axis")
    xyz_element = ET.SubElement(axis_element, "xyz")
    xyz_element.text = axis

def add_glue_sdf_link(model_element, name, relative_to):
    link_element = ET.SubElement(model_element, "link")
    link_element.set("name", name)

    pose_element = ET.SubElement(link_element, "pose")
    pose_element.set("relative_to", relative_to)
    pose_element.text = "0 0 0 0 0 0"


def add_sdf_joint(model_element, name, pose,
                body_parent_name, body_child_name, type, axis="1 0 0"):

    if type == "revolute":
        add_revolute_sdf_joint(model_element, name, pose,
                body_parent_name, body_child_name, type, axis)
    elif type =="ball":

        glue_p_name = name+"glue_p"
        add_glue_sdf_link(model_element, glue_p_name, body_child_name)
        # R
        add_revolute_sdf_joint(model_element, name+"_r", pose,
                body_parent_name, glue_p_name, "revolute", "1 0 0")

        glue_y_name = name+"glue_y"
        add_glue_sdf_link(model_element, glue_y_name, glue_p_name)
        # P
        add_revolute_sdf_joint(model_element, name+"_p", pose,
                glue_p_name, glue_y_name, "revolute", "0 -1 0")

        # Y
        add_revolute_sdf_joint(model_element, name+"_y", pose,
                glue_y_name, body_child_name, "revolute", "0 0 -1")


class Joint:
    def __init__(self, name, id):
        self.name = name
        self.id = id
        self.children = []
        self.parent = None
        self.pose = None

    def __str__(self):
        return f"Joint {self.id}, {self.name} \n children: {self.children} parent: {self.parent}"

    def __repr__(self):
        children_ids = [child_joint.id for child_joint in self.children]
        if self.parent is not None:
            parent_id = self.parent.id
        else:
            parent_id = "is_root"

        return f"Joint {self.id}, {self.name} \n children: {children_ids} parent: {parent_id}"

    def add_children(self, joint):
        self.children.append(joint)

    def set_parent(self, joint):
        # breakpoint()
        if self.parent is None:
            self.parent = joint
        elif self.parent == joint:
            print("skipping, parent is already set to the same joint")
        else:
            assert False, "this joint has a parent already"

    def set_pose(self, pose):
        # breakpoint()
        if self.pose is None:
            self.pose = pose
        else:
            assert False, "this joint has a pose already"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_path",
        help="path to the results directory.",
        )
    parser.add_argument(
        "--sdf_output_path",
        help="path to the logs directory.",
        )
    parser.add_argument(
        "--show_frames",
        action="store_true",
        )
    args = parser.parse_args()

    tracking_results = joblib.load(args.results_path + f"/tracking_results.pth")
    wham_results = joblib.load(args.results_path + f"/wham_results.pth")

    smpl = build_body_model('cpu')
    # breakpoint()
    init_output = smpl.get_output(
            # global_orient=torch.eye(3,3).reshape(1,1,3,3),
            global_orient=tracking_results[0]["init_global_orient"],
            # body_pose=tracking_results[0]["init_body_pose"],
            body_pose=torch.eye(3,3).repeat(23,1,1).reshape(1,23,3,3),
            betas=tracking_results[0]['init_betas'],
            pose2rot=False,
            return_full_pose=False
        )

    joints_dict = {}
    # breakpoint()
    # joint_poses = init_output.original_joints[0]
    joint_poses = torch.matmul(smpl.J_regressor.unsqueeze(0),init_output.vertices)[0]
    # breakpoint()
    F = 1.7

    for j_idx, j_name in enumerate(get_smpl_joint_names()):
        joints_dict[j_idx] = Joint(j_name, j_idx)
        pose = joint_poses[j_idx].numpy() * F
        # pose[2] = 0
        joints_dict[j_idx].set_pose(pose)

    # build tree
    tree_by_ids = get_smpl_skeleton()
    for edge in tree_by_ids:
        p_idx = edge[0]
        c_idx = edge[1]
        parent = joints_dict[p_idx]
        child = joints_dict[c_idx]
        parent.add_children(child)
        if c_idx != 0:
            child.set_parent(parent)

    # build SDF
    sdf_element = ET.Element("sdf")
    sdf_element.set("version", "1.7")
    sdf_element.append(
        ET.Comment(
            """This model was generated code."""  # noqa
        )
    )
    model_element = ET.Element("model")
    model_element.set("name", "smpl")
    sdf_element.append(model_element)

    global_pose_element = ET.SubElement(model_element, "pose")
    global_pose_element.text = f"0 0 0 {-np.pi/2} 0 {np.pi/2}"

    # for simplicity create bodies at the joints.
    # add base body
    base_joint = joints_dict[0]
    base_joint_pose = base_joint.pose
    # base_joint_pose -= base_joint_pose
    pose_str = re.sub(' +', ' ', str(base_joint_pose)[1:-2]) + " 0 0 0"
    name = f"body_j{base_joint.id}"
    add_joint_visual_sdf_body(model_element, name, pose_str)

    for joint_idx in joints_dict.keys():
        joint = joints_dict[joint_idx]
        for child_joint in joint.children:
            child_pose = child_joint.pose #- base_joint_pose
            child_pose_str = re.sub(' +', ' ', str(child_pose)[1:-2]) + " 0 0 0"
            name = f"body_j{child_joint.id}"
            add_joint_visual_sdf_body(model_element, name, child_pose_str)


            parent_pose = joint.pose #- base_joint_pose
            parent_name = f"body_j{joint.id}"
            M = child_pose - parent_pose
            length = np.linalg.norm(M)
            uM = M/length
            mid_pose = (length/2)*(uM) + parent_pose

            a = np.array([0,0,1])
            b = uM
            v = np.cross(a,b)
            s = np.linalg.norm(v)
            c = a.dot(b)
            v_x = np.array([
                [0, -v[2], v[2]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            # t = ((1-c)/s**2)
            t = 1 /(1+c)
            R = np.eye(3) + v_x + t * np.matmul(v_x,v_x)


            # f = np.array([1,1,1])
            # f = f/np.linalg.norm(f)
            # up = uM
            # r = np.cross(f,up)
            # r = r/np.linalg.norm(r)
            # R=np.array([
            #     [r[0], r[1], r[2]],
            #     [up[0], up[1], up[2]],
            #     [uM[0], uM[1], uM[2]],]
            # )
            rpy = RotationMatrix(R).ToRollPitchYaw().vector()

            if joint_idx in [14,17,19,21,23]:
                rpy[1] -= np.pi/4
            if joint_idx in [13,16,18,20,22]:
                rpy[1] += np.pi/4
            # R = np.outer(uM+mid_pose, p)
            # R = np.matmul((parent_pose-mid_pose).reshape(3,1),p.reshape(1,3))

            # r = - np.pi/2 + np.arctan(M[2]/M[1])
            # p = 0 #-np.arctan(M[0]/M[2])
            # y = np.arctan(M[0]/M[2])
            # rpy = [r, p, y]
            # v_pose = "0 0 0 " + rpy_str

            v_pose = "0 0 0 0 0 0"
            rpy_str = f" {rpy[0]} {rpy[1]} {rpy[2]}"

            mid_pose_str = re.sub(' +', ' ', str(mid_pose)[1:-2]) + rpy_str
            length_str = str(length)
            limb_name = add_sdf_limb_body(model_element, name, mid_pose_str, length_str, v_pose)
            add_fixed_sdf_joint(model_element, name, parent_name,limb_name)

        # add terminal body
        # if joint.children == []:
        #     pose = joint.pose + 0.005 #- base_joint_pose
        #     pose_str = re.sub(' +', ' ', str(pose)[1:-2]) + " 0 0 0"
        #     name = f"terminal_body_j{joint.id}"
        #     add_joint_visual_sdf_body(model_element, name, pose_str)

    # add joints
    for joint_idx in joints_dict.keys():
        joint = joints_dict[joint_idx]
        body_parent_name = f"body_j{joint.id}"
        for child_joint in joint.children:
            body_child_name = f"body_j{child_joint.id}"
            pose_str = re.sub(' +', ' ', str(child_joint.pose)[1:-2])
            name = f"joint_{child_joint.id}"
            # pose_str = f"0 0 0 {-np.pi/2} 0 {np.pi/2}"
            pose_str = f"0 0 0 0 0 0"
            add_sdf_joint(
                model_element, name, pose_str,
                body_parent_name, body_child_name, "ball")

    ET.indent(sdf_element)
    model_file = args.sdf_output_path+"/smpl.sdf"
    ET.ElementTree(sdf_element).write(
        model_file,
        xml_declaration=True,
        encoding="utf-8",
    )

    # visualize sdf
    visualizer = _ModelVisualizer(
        visualize_frames=args.show_frames,
        )
    visualizer.AddModels(model_file)
    visualizer.Finalize()
    visualizer.Run()


    # meshcat = StartMeshcat()
    # for i in range(25):
    #     joint_xyz = init_output.original_joints[0][i]
    #     X = RigidTransform()
    #     X.set_translation(joint_xyz)
    #     add_triad(
    #         meshcat,
    #         f"j{i}",
    #         "joint",
    #         length=0.1,
    #         radius=0.01,
    #         opacity=0.04 + 0.02*i,
    #         Xt=X,
    #     )












