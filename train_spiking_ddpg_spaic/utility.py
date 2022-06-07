import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import math
import random
import numpy as np
from shapely.geometry import Point, Polygon


def gen_rand_list_env1(size, robot_goal_diff=1.0):
    """
    Generate Random Training Lists for Environment 1
    :param size: size of list
    :param robot_goal_diff: distance between init pose and goal
    :return: poly_list, raw_poly_list, goal_list, robot_init_pose_list
    """
    robot_init_pose = [8, 8]
    env_range = ((4, 12), (4, 12))
    poly_raw_list = [[(2, 14), (14, 14), (14, 2), (2, 2)]]
    goal_raw_list = [[(4, 12), (12, 12), (12, 4), (4, 4)]]
    '''
    Generate poly_list and goal_list
    '''
    poly_list = gen_polygon_exterior_list(poly_raw_list)
    goal_poly_list = gen_polygon_exterior_list(goal_raw_list)
    goal_list = gen_goal_position_list(goal_poly_list, env_size=env_range)
    '''
    Generate Random Goal List and Robot Init Pose List
    '''
    rand_goal_list = []
    rand_robot_init_pose_list = []
    for num in range(size):
        goal = random.choice(goal_list)
        distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        while distance < robot_goal_diff:
            goal = random.choice(goal_list)
            distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        pose = [robot_init_pose[0], robot_init_pose[1], random.random() * 2 * math.pi]
        rand_goal_list.append(goal)
        rand_robot_init_pose_list.append(pose)
    return poly_list, poly_raw_list, rand_goal_list, rand_robot_init_pose_list

def gen_rand_list_env1_maze(size, robot_goal_diff=1.0):
    """
    Generate Random Training Lists for Environment 1
    :param size: size of list
    :param robot_goal_diff: distance between init pose and goal
    :return: poly_list, raw_poly_list, goal_list, robot_init_pose_list
    """
    robot_init_pose = [3, 3]
    env_range = ((2, 14), (2, 14))
    poly_raw_list = [[(2, 14), (14, 14), (14, 2), (2, 2)],
                     [(3.75, 2.0), (4.25, 2.0), (4.25, 6.25), (3.75, 6.25)],
                     [(3.75, 7.75), (4.25, 7.75), (4.25, 12.25), (3.75, 12.25)],
                     [(5.75, 3.75), (6.25, 3.75), (6.25, 6.25), (5.75, 6.25)],
                     [(5.75, 9.75), (6.25, 9.75), (6.25, 14.0), (5.75, 14.0)],
                     [(7.75, 3.75), (8.25, 3.75), (8.25, 6.25), (7.75, 6.25)],
                     [(7.75, 9.75), (8.25, 9.75), (8.25, 12.25), (7.75, 12.25)],
                     [(9.75, 4.25), (10.25, 4.25), (10.25, 10.25), (9.75, 10.25)],
                     [(8.25, 3.75), (12.25, 3.75), (12.25, 4.25), (8.25, 4.25)],
                     [(11.75, 4.25), (12.25, 4.25), (12.25, 8.25), (11.75, 8.25)],
                     [(11.75, 11.75), (12.25, 11.75), (12.25, 14.0), (11.75, 14.0)],
                     [(6.25, 5.75), (7.75, 5.75), (7.75, 6.25), (6.25, 6.25)],
                     [(4.25, 7.75), (9.75, 7.75), (9.75, 8.25), (4.25, 8.25)],
                     [(10.25, 9.75), (14.0, 9.75), (14.0, 10.25), (10.25, 10.25)],
                     [(8.25, 11.75), (11.75, 11.75), (11.75, 12.25), (8.25, 12.25)],

                     ]
    goal_raw_list = [[(2, 14), (14, 14), (14, 2), (2, 2)],
                     [(3.75, 2.0), (4.25, 2.0), (4.25, 6.25), (3.75, 6.25)],
                     [(3.75, 7.75), (4.25, 7.75), (4.25, 12.25), (3.75, 12.25)],
                     [(5.75, 3.75), (6.25, 3.75), (6.25, 6.25), (5.75, 6.25)],
                     [(5.75, 9.75), (6.25, 9.75), (6.25, 14.0), (5.75, 14.0)],
                     [(7.75, 3.75), (8.25, 3.75), (8.25, 6.25), (7.75, 6.25)],
                     [(7.75, 9.75), (8.25, 9.75), (8.25, 12.25), (7.75, 12.25)],
                     [(9.75, 4.25), (10.25, 4.25), (10.25, 10.25), (9.75, 10.25)],
                     [(8.25, 3.75), (12.25, 3.75), (12.25, 4.25), (8.25, 4.25)],
                     [(11.75, 4.25), (12.25, 4.25), (12.25, 8.25), (11.75, 8.25)],
                     [(11.75, 11.75), (12.25, 11.75), (12.25, 14.0), (11.75, 14.0)],
                     [(6.25, 5.75), (7.75, 5.75), (7.75, 6.25), (6.25, 6.25)],
                     [(4.25, 7.75), (9.75, 7.75), (9.75, 8.25), (4.25, 8.25)],
                     [(10.25, 9.75), (14.0, 9.75), (14.0, 10.25), (10.25, 10.25)],
                     [(8.25, 11.75), (11.75, 11.75), (11.75, 12.25), (8.25, 12.25)],
                     [(2, 2), (8, 2), (8, 14), (2, 14)],
                     [(8, 2), (14, 2), (14, 4), (8, 4)],
                     [(8, 6), (14, 6), (14, 12), (8, 12)],
                     [(12, 4), (14, 4), (14, 8), (12, 8)],
                     [(8, 12), (10, 12), (10, 14), (8, 14)],
                     ]

    '''
    Generate poly_list and goal_list
    '''
    poly_list = gen_polygon_exterior_list(poly_raw_list)
    goal_poly_list = gen_polygon_exterior_list(goal_raw_list)
    goal_list = gen_goal_position_list(goal_poly_list, env_size=env_range)
    '''
    Generate Random Goal List and Robot Init Pose List
    '''
    rand_goal_list = []
    rand_robot_init_pose_list = []
    for num in range(size):
        goal = random.choice(goal_list)
        distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        while distance < robot_goal_diff:
            goal = random.choice(goal_list)
            distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        pose = [robot_init_pose[0], robot_init_pose[1], random.random() * 2 * math.pi]
        rand_goal_list.append(goal)
        rand_robot_init_pose_list.append(pose)
    return poly_list, poly_raw_list, rand_goal_list, rand_robot_init_pose_list

def gen_rand_list_env_grid(size, robot_goal_diff=1.0,hw=2.5,hh=2.5,seed=11):
    """
    Generate Random Training Lists for Environment 1
    :param size: size of list
    :param robot_goal_diff: distance between init pose and goal
    :return: poly_list, raw_poly_list, goal_list, robot_init_pose_list
    """
    random.seed(seed)
    robot_init_pose = [0, 0]
    env_range = ((-hw, hw), (-hh, hh))
    poly_raw_list = [[(-hw, -hh), (-hw, hh), (hw, hh), (hw, -hh)],
                     ]
    goal_raw_list = [[(-hw, -hh), (-hw, hh), (hw, hh), (hw, -hh)],
                     ]

    '''
    Generate poly_list and goal_list
    '''
    poly_list = gen_polygon_exterior_list(poly_raw_list)
    goal_poly_list = gen_polygon_exterior_list(goal_raw_list)
    goal_list = gen_goal_position_list(goal_poly_list, env_size=env_range)
    '''
    Generate Random Goal List and Robot Init Pose List
    '''
    rand_goal_list = []
    rand_robot_init_pose_list = []
    for num in range(size):
        # goal = random.choice(goal_list)
        goal = [2.0,1.0]
        robot_init_pose = random.choice(goal_list)
        distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        while distance < robot_goal_diff:
            robot_init_pose = random.choice(goal_list)
            distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        pose = [robot_init_pose[0], robot_init_pose[1], random.random() * 2 * math.pi]
        rand_goal_list.append(goal)
        rand_robot_init_pose_list.append(pose)
    return poly_list, poly_raw_list, rand_goal_list, rand_robot_init_pose_list


def gen_rand_list_env2(size, robot_goal_diff=3.0):
    """
    Generate Random Training Lists for Environment 2
    :param size: size of list
    :param robot_goal_diff: distance between init pose and goal
    :return: poly_list, raw_poly_list, goal_list, robot_init_pose_list
    """
    robot_init_pose_range = ((5.75, 10.25), (-11, -5))
    robot_init_pose_list = []
    init_x, init_y = np.mgrid[robot_init_pose_range[0][0]:robot_init_pose_range[0][1]:0.1,
                     robot_init_pose_range[1][0]:robot_init_pose_range[1][1]:0.1]
    for x in range(init_x.shape[0]):
        for y in range(init_y.shape[1]):
            robot_init_pose_list.append([init_x[x, y], init_y[x, y]])
    env_range = ((2, 14), (-14, -2))
    poly_raw_list = [[(2, -14), (14, -14), (14, -2), (2, -2)],
                     [(5, -4.5), (5, -4), (11, -4), (11, -4.5)],
                     [(5, -11.5), (5, -12), (11, -12), (11, -11.5)],
                     [(5.25, -6), (4.75, -6), (4.75, -10), (5.25, -10)],
                     [(10.75, -6), (11.25, -6), (11.25, -10), (10.75, -10)]]
    goal_raw_list = [[(2, -14), (14, -14), (14, -2), (2, -2)],
                     [(10.75, -11.5), (10.75, -4), (5.25, -4), (5.25, -11.5)],
                     [(5, -4.5), (5, -4), (11, -4), (11, -4.5)],
                     [(5, -11.5), (5, -12), (11, -12), (11, -11.5)],
                     [(5.25, -6), (4.75, -6), (4.75, -10), (5.25, -10)],
                     [(10.75, -6), (11.25, -6), (11.25, -10), (10.75, -10)],
                     [(6, -4.5), (6, -2), (10, -2), (10, -4.5)],
                     [(6, -11.5), (6, -14), (10, -14), (10, -11.5)],
                     [(5.25, -7), (2, -7), (2, -9), (5.25, -9)],
                     [(10.75, -7), (14, -7), (14, -9), (10.75, -9)]]
    poly_list = gen_polygon_exterior_list(poly_raw_list)
    goal_poly_list = gen_polygon_exterior_list(goal_raw_list)
    goal_list = gen_goal_position_list(goal_poly_list, env_size=env_range)
    '''
    Generate Random Goal List and Robot Init Pose List
    '''
    rand_goal_list = []
    rand_robot_init_pose_list = []
    for num in range(size):
        goal = random.choice(goal_list)
        robot_init_pose = random.choice(robot_init_pose_list)
        distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        while distance < robot_goal_diff:
            robot_init_pose = random.choice(robot_init_pose_list)
            distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        pose = [robot_init_pose[0], robot_init_pose[1], random.random() * 2 * math.pi]
        rand_goal_list.append(goal)
        rand_robot_init_pose_list.append(pose)
    return poly_list, poly_raw_list, rand_goal_list, rand_robot_init_pose_list

def gen_rand_list_env3(size, robot_goal_diff=1.0):
    """
    Generate Random Training Lists for Environment 3
    :param size: size of list
    :param robot_goal_diff: distance between init pose and goal
    :return: poly_list, raw_poly_list, goal_list, robot_init_pose_list
    """
    robot_init_pose_range = ((-9.75, -6.25), (-9, -7))
    robot_init_pose_list = []
    init_x, init_y = np.mgrid[robot_init_pose_range[0][0]:robot_init_pose_range[0][1]:0.1,
                     robot_init_pose_range[1][0]:robot_init_pose_range[1][1]:0.1]
    for x in range(init_x.shape[0]):
        for y in range(init_y.shape[1]):
            robot_init_pose_list.append([init_x[x, y], init_y[x, y]])
    env_range = ((-14, -2), (-14, -2))
    poly_raw_list = [[(-2, -14), (-14, -14), (-14, -2), (-2, -2)],
                     [(-6.5, -6.5), (-6.5, -6), (-9.5, -6), (-9.5, -6.5)],
                     [(-6.5, -9.5), (-6.5, -10), (-9.5, -10), (-9.5, -9.5)],
                     [(-5.25, -6.5), (-4.75, -6.5), (-4.75, -9.5), (-5.25, -9.5)],
                     [(-10.75, -6.5), (-11.25, -6.5), (-11.25, -9.5), (-10.75, -9.5)]]
    goal_raw_list = [[(-2, -14), (-14, -14), (-14, -2), (-2, -2)],
                     [(-5.25, -9.5), (-5.25, -6.5), (-10.75, -6.5), (-10.75, -9.5)],
                     [(-9.5, -6.5), (-9.5, -2), (-14, -2), (-14, -6.5)],
                     [(-6.5, -6.5), (-2, -6.5), (-2, -2), (-6.5, -2)],
                     [(-6.5, -9.5), (-6.5, -14), (-2, -14), (-2, -9.5)],
                     [(-9.5, -9.5), (-14, -9.5), (-14, -14), (-9.5, -14)],
                     [(-6.5, -6.5), (-6.5, -6), (-9.5, -6), (-9.5, -6.5)],
                     [(-6.5, -9.5), (-6.5, -10), (-9.5, -10), (-9.5, -9.5)],
                     [(-5.25, -6.5), (-4.75, -6.5), (-4.75, -9.5), (-5.25, -9.5)],
                     [(-10.75, -6.5), (-11.25, -6.5), (-11.25, -9.5), (-10.75, -9.5)]]
    poly_list = gen_polygon_exterior_list(poly_raw_list)
    goal_poly_list = gen_polygon_exterior_list(goal_raw_list)
    goal_list = gen_goal_position_list(goal_poly_list, env_size=env_range)
    '''
    Generate Random Goal List and Robot Init Pose List
    '''
    rand_goal_list = []
    rand_robot_init_pose_list = []
    for num in range(size):
        goal = random.choice(goal_list)
        robot_init_pose = random.choice(robot_init_pose_list)
        distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        while distance < robot_goal_diff:
            robot_init_pose = random.choice(robot_init_pose_list)
            distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        pose = [robot_init_pose[0], robot_init_pose[1], random.random() * 2 * math.pi]
        rand_goal_list.append(goal)
        rand_robot_init_pose_list.append(pose)
    return poly_list, poly_raw_list, rand_goal_list, rand_robot_init_pose_list

def gen_rand_list_env3_exp(size, robot_goal_diff=1.0):
    """
    Generate Random Training Lists for Environment 3
    :param size: size of list
    :param robot_goal_diff: distance between init pose and goal
    :return: poly_list, raw_poly_list, goal_list, robot_init_pose_list
    """
    robot_init_pose_range = ((-9.75, -6.25), (-9.75, -6.25))
    robot_init_pose_list = []
    init_x, init_y = np.mgrid[robot_init_pose_range[0][0]:robot_init_pose_range[0][1]:0.1,
                     robot_init_pose_range[1][0]:robot_init_pose_range[1][1]:0.1]
    for x in range(init_x.shape[0]):
        for y in range(init_y.shape[1]):
            robot_init_pose_list.append([init_x[x, y], init_y[x, y]])
    env_range = ((-14, -2), (-14, -2))
    poly_raw_list = [[(-2, -14), (-14, -14), (-14, -2), (-2, -2)],
                     [(-6.75, -3.75), (-7.25, -3.75), (-7.25, -5.5), (-6.75, -5.5)],
                     [(-7.25, -3.75), (-12.25, -3.75), (-12.25, -4.25), (-7.25, -4.25)],
                     [(-12.25, -4.25), (-12.25, -8.75), (-11.75, -8.75), (-11.75, -4.25)],
                     [(-12.25, -8.75), (-12.25, -9.25), (-10.5, -9.25), (-10.5, -8.75)],
                     [(-3.75, -6.75), (-5.5, -6.75), (-5.5, -7.25), (-3.75, -7.25)],
                     [(-3.75, -7.25), (-4.25, -7.25), (-4.25, -12.25), (-3.75, -12.25)],
                     [(-4.25, -11.75), (-8.75, -11.75), (-8.75, -12.25), (-4.25, -12.25)],
                     [(-8.75, -12.25), (-8.75, -10.5), (-9.25, -10.5), (-9.25, -12.25)]]
    goal_raw_list = [[(-2, -14), (-14, -14), (-14, -2), (-2, -2)],
                     [(-6.75, -3.75), (-7.25, -3.75), (-7.25, -5.5), (-6.75, -5.5)],
                     [(-7.25, -3.75), (-12.25, -3.75), (-12.25, -4.25), (-7.25, -4.25)],
                     [(-12.25, -4.25), (-12.25, -8.75), (-11.75, -8.75), (-11.75, -4.25)],
                     [(-12.25, -8.75), (-12.25, -9.25), (-10.5, -9.25), (-10.5, -8.75)],
                     [(-3.75, -6.75), (-5.5, -6.75), (-5.5, -7.25), (-3.75, -7.25)],
                     [(-3.75, -7.25), (-4.25, -7.25), (-4.25, -12.25), (-3.75, -12.25)],
                     [(-4.25, -11.75), (-8.75, -11.75), (-8.75, -12.25), (-4.25, -12.25)],
                     [(-8.75, -12.25), (-8.75, -10.5), (-9.25, -10.5), (-9.25, -12.25)],
                     [(-3.75, -3.75), (-3.75, -12.25), (-12.25, -12.25), (-12.25, -3.75)],
                     [(-2, -6.75), (-6.75, -6.75), (-6.75, -2), (-2, -2)],
                     [(-14, -9.25), (-9.25, -9.25), (-9.25, -14), (-14, -14)],
                     ]
    poly_list = gen_polygon_exterior_list(poly_raw_list)
    goal_poly_list = gen_polygon_exterior_list(goal_raw_list)
    goal_list = gen_goal_position_list(goal_poly_list, env_size=env_range)
    '''
    Generate Random Goal List and Robot Init Pose List
    '''
    rand_goal_list = []
    rand_robot_init_pose_list = []
    for num in range(size):
        goal = random.choice(goal_list)
        robot_init_pose = random.choice(robot_init_pose_list)
        distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        while distance < robot_goal_diff:
            robot_init_pose = random.choice(robot_init_pose_list)
            distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        pose = [robot_init_pose[0], robot_init_pose[1], random.random() * 2 * math.pi]
        rand_goal_list.append(goal)
        rand_robot_init_pose_list.append(pose)
    return poly_list, poly_raw_list, rand_goal_list, rand_robot_init_pose_list

def gen_rand_list_env4(size, robot_goal_diff=5.0):
    """
    Generate Random Training Lists for Environment 4
    :param size: size of list
    :param robot_goal_diff: distance between init pose and goal
    :return: poly_list, raw_poly_list, goal_list, robot_init_pose_list
    """
    env_range = ((-14, -2), (2, 14))
    poly_raw_list = [[(-2, 14), (-14, 14), (-14, 2), (-2, 2)],
                     [(-6, 9), (-6, 9.5), (-9, 9.5), (-9, 9)],
                     [(-6, 7), (-6, 6.5), (-10, 6.5), (-10, 7)],
                     [(-4, 11.5), (-4, 12), (-8, 12), (-8, 11.5)],
                     [(-11.5, 9), (-11.5, 12), (-12, 12), (-12, 9)],
                     [(-3.75, 4), (-3.75, 7), (-4.25, 7), (-4.25, 4)],
                     [(-8, 4), (-8, 4.5), (-12, 4.5), (-12, 4)]]
    goal_raw_list = [[(-2, 14), (-14, 14), (-14, 2), (-2, 2)],
                     [(-6, 9), (-6, 9.5), (-9, 9.5), (-9, 9)],
                     [(-6, 7), (-6, 6.5), (-10, 6.5), (-10, 7)],
                     [(-4, 11.5), (-4, 12), (-8, 12), (-8, 11.5)],
                     [(-11.5, 9), (-11.5, 12), (-12, 12), (-12, 9)],
                     [(-3.75, 4), (-3.75, 7), (-4.25, 7), (-4.25, 4)],
                     [(-8, 4), (-8, 4.5), (-12, 4.5), (-12, 4)]]
    poly_list = gen_polygon_exterior_list(poly_raw_list)
    goal_poly_list = gen_polygon_exterior_list(goal_raw_list)
    goal_list = gen_goal_position_list(goal_poly_list, env_size=env_range)
    '''
    Generate Random Goal List and Robot Init Pose List
    '''
    rand_goal_list = []
    rand_robot_init_pose_list = []
    for num in range(size):
        goal = random.choice(goal_list)
        robot_init_pose = random.choice(goal_list)
        distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        while distance < robot_goal_diff:
            robot_init_pose = random.choice(goal_list)
            distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        pose = [robot_init_pose[0], robot_init_pose[1], random.random() * 2 * math.pi]
        rand_goal_list.append(goal)
        rand_robot_init_pose_list.append(pose)
    return poly_list, poly_raw_list, rand_goal_list, rand_robot_init_pose_list

def gen_rand_list_env4_exp(size, robot_goal_diff=5.0):
    """
    Generate Random Training Lists for Environment 4
    :param size: size of list
    :param robot_goal_diff: distance between init pose and goal
    :return: poly_list, raw_poly_list, goal_list, robot_init_pose_list
    """
    env_range = ((-14, -2), (2, 14))
    poly_raw_list = [[(-2, 14), (-14, 14), (-14, 2), (-2, 2)],
                     [(-7, 3.5), (-7.2, 3.5), (-7.2, 6.8), (-7, 7)],
                     [(-3.5, 7), (-7, 7), (-6.8, 7.2), (-3.5, 7.2)],
                     [(-7, 3.5), (-7, 4), (-6, 4), (-6, 3.5)],
                     [(-3.5, 7), (-4, 7), (-4, 6), (-3.5, 6)],
                     [(-8.8, 3.5), (-9, 3.5), (-9, 7), (-8.8, 6.8)],
                     [(-9, 7), (-12.5, 7), (-12.5, 7.2), (-9.2, 7.2)],
                     [(-9, 3.5), (-10, 3.5), (-10, 4), (-9, 4)],
                     [(-12.5, 7), (-12.5, 6), (-12, 6), (-12, 7)],
                     [(-10.25, 5.25), (-10.25, 5.75), (-10.75, 5.75), (-10.75, 5.25)],
                     [(-7, 9), (-7.2, 9.2), (-7.2, 12.5), (-7, 12.5)],
                     [(-3.5, 8.8), (-6.8, 8.8), (-7, 9), (-3.5, 9)],
                     [(-2, 12.3), (-7, 12.3), (-7, 12.5), (-2, 12.5)],
                     [(-8.8, 9.2), (-9, 9), (-9, 12.5), (-8.8, 12.5)],
                     [(-9.2, 8.8), (-12.5, 8.8), (-12.5, 9), (-9, 9)],
                     [(-9, 12.3), (-14, 12.3), (-14, 12.5), (-9, 12.5)],
                     [(-11, 10.4), (-14, 10.4), (-14, 10.6), (-11, 10.6)],
                     ]
    goal_raw_list = [[(-2, 14), (-14, 14), (-14, 2), (-2, 2)],
                     [(-7, 3.5), (-7.2, 3.5), (-7.2, 6.8), (-7, 7)],
                     [(-3.5, 7), (-7, 7), (-6.8, 7.2), (-3.5, 7.2)],
                     [(-7, 3.5), (-7, 4), (-6, 4), (-6, 3.5)],
                     [(-3.5, 7), (-4, 7), (-4, 6), (-3.5, 6)],
                     [(-8.8, 3.5), (-9, 3.5), (-9, 7), (-8.8, 6.8)],
                     [(-9, 7), (-12.5, 7), (-12.5, 7.2), (-9.2, 7.2)],
                     [(-9, 3.5), (-10, 3.5), (-10, 4), (-9, 4)],
                     [(-12.5, 7), (-12.5, 6), (-12, 6), (-12, 7)],
                     [(-10.25, 5.25), (-10.25, 5.75), (-10.75, 5.75), (-10.75, 5.25)],
                     [(-7, 9), (-7.2, 9.2), (-7.2, 12.5), (-7, 12.5)],
                     [(-3.5, 8.8), (-6.8, 8.8), (-7, 9), (-3.5, 9)],
                     [(-2, 12.3), (-7, 12.3), (-7, 12.5), (-2, 12.5)],
                     [(-8.8, 9.2), (-9, 9), (-9, 12.5), (-8.8, 12.5)],
                     [(-9.2, 8.8), (-12.5, 8.8), (-12.5, 9), (-9, 9)],
                     [(-9, 12.3), (-14, 12.3), (-14, 12.5), (-9, 12.5)],
                     [(-11, 10.4), (-14, 10.4), (-14, 10.6), (-11, 10.6)]
                     ]
    poly_list = gen_polygon_exterior_list(poly_raw_list)
    goal_poly_list = gen_polygon_exterior_list(goal_raw_list)
    goal_list = gen_goal_position_list(goal_poly_list, env_size=env_range)
    '''
    Generate Random Goal List and Robot Init Pose List
    '''
    rand_goal_list = []
    rand_robot_init_pose_list = []
    for num in range(size):
        goal = random.choice(goal_list)
        robot_init_pose = random.choice(goal_list)
        distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        while distance < robot_goal_diff:
            robot_init_pose = random.choice(goal_list)
            distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
        pose = [robot_init_pose[0], robot_init_pose[1], random.random() * 2 * math.pi]
        rand_goal_list.append(goal)
        rand_robot_init_pose_list.append(pose)
    return poly_list, poly_raw_list, rand_goal_list, rand_robot_init_pose_list

def gen_goal_position_list(poly_list, env_size=((-6, 6), (-6, 6)), obs_near_th=0.5, sample_step=0.1):
    """
    Generate list of goal positions
    :param poly_list: list of obstacle polygon
    :param env_size: size of the environment
    :param obs_near_th: Threshold for near an obstacle
    :param sample_step: sample step for goal generation
    :return: goal position list
    """
    goal_pos_list = []
    x_pos, y_pos = np.mgrid[env_size[0][0]:env_size[0][1]:sample_step, env_size[1][0]:env_size[1][1]:sample_step]
    for x in range(x_pos.shape[0]):
        for y in range(x_pos.shape[1]):
            tmp_pos = [x_pos[x, y], y_pos[x, y]]
            tmp_point = Point(tmp_pos[0], tmp_pos[1])
            near_obstacle = False
            for poly in poly_list:
                tmp_dis = tmp_point.distance(poly)
                if tmp_dis < obs_near_th:
                    near_obstacle = True
            if near_obstacle is False:
                goal_pos_list.append(tmp_pos)
    return goal_pos_list


def gen_polygon_exterior_list(poly_point_list):
    """
    Generate list of obstacle in the environment as polygon exterior list
    :param poly_point_list: list of points of polygon (with first always be the out wall)
    :return: polygon exterior list
    """
    poly_list = []
    for i, points in enumerate(poly_point_list, 0):
        tmp_poly = Polygon(points)
        if i > 0:
            poly_list.append(tmp_poly)
        else:
            poly_list.append(tmp_poly.exterior)
    return poly_list


def network_2_robot_action_decoder(action, l_max, l_min, a_max, a_min):
    """
    Decode action from network output to action for the robot
    :param action: action from network output
    :param l_max: linear speed max
    :param l_min: linear speed min
    :param a_max: angular speed max
    :param a_min: angular speed min
    :return: robot_action
    """
    linear = action[0] * (l_max - l_min) + l_min
    angular = (action[1] + 1) * a_max + a_min
    return [linear, angular]


def wheeled_network_2_robot_action_decoder(action, wheel_max, wheel_min, diff=0.25):
    """
    Decode wheeled action from network to linear and angular speed for the robot
    :param action: action for wheel spd
    :param wheel_max: max wheel spd
    :param wheel_min: min wheel spd
    :param diff: diff of wheel for decoding angular spd
    :return: robot_action
    """
    l_spd = action[0] * (wheel_max - wheel_min) + wheel_min
    r_spd = action[1] * (wheel_max - wheel_min) + wheel_min
    linear = (l_spd + r_spd) / 2
    angular = (r_spd - l_spd) / diff
    return [linear, angular]


def robot_2_goal_dis_dir(robot_pose, goal_pos):
    """
    Compuate Relative Distance and Direction between robot and goal
    :param robot_pose: robot pose
    :param goal_pos: goal position
    :return: distance, direction
    """
    delta_x = goal_pos[0] - robot_pose[0]
    delta_y = goal_pos[1] - robot_pose[1]
    distance = math.sqrt(delta_x ** 2 + delta_y ** 2)
    ego_direction = math.atan2(delta_y, delta_x)
    robot_direction = robot_pose[2]
    while robot_direction < 0:
        robot_direction += 2 * math.pi
    while robot_direction > 2 * math.pi:
        robot_direction -= 2 * math.pi
    while ego_direction < 0:
        ego_direction += 2 * math.pi
    while ego_direction > 2 * math.pi:
        ego_direction -= 2 * math.pi
    pos_dir = abs(ego_direction - robot_direction)
    neg_dir = 2 * math.pi - abs(ego_direction - robot_direction)
    if pos_dir <= neg_dir:
        direction = math.copysign(pos_dir, ego_direction - robot_direction)
    else:
        direction = math.copysign(neg_dir, -(ego_direction - robot_direction))
    return distance, direction


def ddpg_state_2_spike_value_state(state, spike_state_num,
                                   goal_dir_range=math.pi, linear_spd_range=0.5, angular_spd_range=2.0):
    """
    Transform DDPG state to Spike Value State for SNN
    :param state: single ddpg state
    :param spike_state_num: number of spike states
    :param goal_dir_range: range of goal dir
    :param linear_spd_range: range of linear spd
    :param angular_spd_range: range of angular spd
    :return: spike_state_value
    """
    spike_state_value = [0 for _ in range(spike_state_num)]
    if state[0] > 0:
        spike_state_value[0] = state[0] / goal_dir_range
        spike_state_value[1] = 0
    else:
        spike_state_value[0] = 0
        spike_state_value[1] = abs(state[0]) / goal_dir_range
    spike_state_value[2] = state[1]
    spike_state_value[3] = state[2] / linear_spd_range
    if state[3] > 0:
        spike_state_value[4] = state[3] / angular_spd_range
        spike_state_value[5] = 0
    else:
        spike_state_value[4] = 0
        spike_state_value[5] = abs(state[3]) / angular_spd_range
    spike_state_value[6:] = state[4:]
    return spike_state_value


def ddpg_state_rescale(state, state_num,
                       goal_dir_range=math.pi, linear_spd_range=0.5, angular_spd_range=2.0):
    """
    Rescale DDPG states to make them same scale with SNN input
    :param state: single ddpg state
    :param state_num: number of states
    :param goal_dir_range: range of goal dir
    :param linear_spd_range: range of linear spd
    :param angular_spd_range: range of angular spd
    :return: rescale_state_value
    """
    rescale_state_value = [0 for _ in range(state_num)]
    if state[0] > 0:
        rescale_state_value[0] = state[0] / goal_dir_range
    else:
        rescale_state_value[0] = -abs(state[0]) / goal_dir_range
    rescale_state_value[1] = state[1]
    rescale_state_value[2] = state[2] / linear_spd_range
    if state[3] > 0:
        rescale_state_value[3] = state[3] / angular_spd_range
    else:
        rescale_state_value[3] = -abs(state[3]) / angular_spd_range
    rescale_state_value[4:] = state[4:]
    return rescale_state_value

def ddpg_GC_state_rescale(state, state_num):
    """
    Rescale DDPG states to make them same scale with SNN input
    :param state: single ddpg state
    :param state_num: number of states
    :param goal_dir_range: range of goal dir
    :param linear_spd_range: range of linear spd
    :param angular_spd_range: range of angular spd
    :return: rescale_state_value
    """
    rescale_state_value = [0 for _ in range(state_num)]

    for i in range(state_num):
        if i <256:
            rescale_state_value[i]=state[0][0][i]
        elif i >=256 and i<512:
            rescale_state_value[i]=state[1][0][i-256]
        elif i >=512 and i<515:
            rescale_state_value[i]=state[2][i-512]
        elif i >=515 and i<771:
            rescale_state_value[i]=state[3][0][i-515]
        elif i >=771 and i<783:
            rescale_state_value[i]=state[4][0][i-771]

    # rescale_state_value[0:256]=state[0][0:256]
    # rescale_state_value[256:512] = state[1][0:256]
    #
    # rescale_state_value[512] = state[2][0]
    # rescale_state_value[513] = state[2][1]
    # rescale_state_value[514] = state[2][2]

    # rescale_state_value[515:771] = state[3][0:256]
    # rescale_state_value[771:783] = state[4][0:12]

    return rescale_state_value



if __name__ == '__main__':
    import cv2

    b = np.random.randint(250, 255, (300, 300), dtype=np.uint8)
    g = np.random.randint(250, 255, (300, 300), dtype=np.uint8)
    r = np.random.randint(250, 255, (300, 300), dtype=np.uint8)
    img = cv2.merge([b, g, r])

    poly_list, poly_raw_list, rand_goal_list, rand_robot_init_pose_list = gen_rand_list_env1_maze(10)
    # poly_list, poly_raw_list, rand_goal_list, rand_robot_init_pose_list = gen_rand_list_env3_exp(300)
    # poly_list, poly_raw_list, rand_goal_list, rand_robot_init_pose_list = gen_rand_list_env4_exp(400)

    for goal in rand_goal_list:
        goal=(int(abs(goal[0]) * 20),int(abs(goal[1]) * 20))
        cv2.circle(img,goal,1,(0,255,0),2)

    for pose in rand_robot_init_pose_list:
        rbt=(int(abs(pose[0]) * 20),int(abs(pose[1]) * 20))
        cv2.circle(img,rbt,1,(0,128,255),2)

    goal_raw_list = [
        # [(-2, 14), (-14, 14), (-14, 2), (-2, 2)],
        # [(-6, 9), (-6, 9.5), (-9, 9.5), (-9, 9)],
        # [(-6, 7), (-6, 6.5), (-10, 6.5), (-10, 7)],
        # [(-4, 11.5), (-4, 12), (-8, 12), (-8, 11.5)],
        # [(-11.5, 9), (-11.5, 12), (-12, 12), (-12, 9)],
        # [(-3.75, 4), (-3.75, 7), (-4.25, 7), (-4.25, 4)],
        # [(-8, 4), (-8, 4.5), (-12, 4.5), (-12, 4)]
        # [(-3.75, -3.75), (-3.75, -12.25), (-12.25, -12.25), (-12.25, -3.75)]
                     ]

    pcnt=0
    for p in poly_raw_list:
        color = (255, 0, 0)
        if pcnt==1 or pcnt==5:
            color = (255, 128, 0)
        pcnt+=1

        thickness = 2

        minx= 300
        miny=300
        maxx=0
        maxy=0
        for t in p:
            x = int(abs(t[0]) * 20)
            y = int(abs(t[1]) * 20)

            if x<minx:
                minx=x
            if x>maxx:
                maxx=x

            if y<miny:
                miny=y
            if y>maxy:
                maxy=y

        start_point=(minx,miny)
        end_point = (maxx,maxy)

        img = cv2.rectangle(img, start_point, end_point, color, thickness)

    for p in goal_raw_list:
        color = (0, 0, 255)
        thickness = 1

        minx = 300
        miny = 300
        maxx = 0
        maxy = 0
        for t in p:
            x = int(abs(t[0]) * 20)
            y = int(abs(t[1]) * 20)

            if x < minx:
                minx = x
            if x > maxx:
                maxx = x

            if y < miny:
                miny = y
            if y > maxy:
                maxy = y

        start_point = (minx, miny)
        end_point = (maxx, maxy)

        img = cv2.rectangle(img, start_point, end_point, color, thickness)
        # break



    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.destroyWindow('test')


