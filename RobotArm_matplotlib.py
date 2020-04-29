"""
Created on Thu Apr 23 12:43:29 2020

@author: Emily Brady
"""
from numpy import cos, sin, sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

#creates three joint arm
class ThreeLinkArm():
    def __init__(self, joint_angles=[0, 0, 0]):
        self.shoulder = np.array([0, 0])
        self.link_lengths = [1, 1, 1]
        self.update_joints(joint_angles)
        self.phi = np.pi/4

    def update_joints(self, joint_angles):
        self.joint_angles = joint_angles
        self.forward_kinematics()

    def forward_kinematics(self):
        theta0 = self.joint_angles[0]
        theta1 = self.joint_angles[1]
        theta2 = self.joint_angles[2]
        l0 = self.link_lengths[0]
        l1 = self.link_lengths[1]
        l2 = self.link_lengths[2]
        self.elbow = self.shoulder + np.array([l0*cos(theta0), l0*sin(theta0)])
        self.wrist = self.elbow + np.array([l1*cos(theta0 + theta1), l1*sin(theta0 + theta1)])
        self.finger = self.wrist + np.array([l2*cos(theta0 + theta1 + theta2), l2*sin(theta0 + theta1 + theta2)])

    def inverse_kinematics(self, xf, yf, phi, **kwargs):
        self.phi = phi
        self.xp = xf - self.link_lengths[2]*np.cos(self.phi)
        self.yp = yf - self.link_lengths[2]*np.sin(self.phi)
        sigma = 1
        method = 1
        for key, value in kwargs.items():
            if key == 'sigma':
                sigma = value
            elif key == 'method':
                method = value

        gamma = np.arctan2(-self.yp/sqrt(self.xp**2 + self.yp**2), self.xp/sqrt(self.xp**2 + self.yp**2))

        if method == 1:
            theta0 = gamma + sigma*np.arccos(-(self.xp**2 + self.yp**2 + self.link_lengths[0]**2 -
                                    self.link_lengths[1]**2) / (2*self.link_lengths[0]*sqrt(self.xp**2 + self.yp**2)))
        elif method == 2:
            theta0 = gamma + sigma*np.arccos((self.xp**2 + self.yp**2 + self.link_lengths[0]**2 -
                                    self.link_lengths[1]**2) / (2*self.link_lengths[0]*sqrt(self.xp**2 + self.yp**2)))

        theta1 = np.arctan2((self.yp - self.link_lengths[0]*sin(theta0))/self.link_lengths[1],
                            (self.xp - self.link_lengths[0]*cos(theta0))/self.link_lengths[1]) - theta0

        theta2 = self.phi - theta0 - theta1

        return self.update_joints([theta0, theta1, theta2])

    def plot(self):
        plt.plot([self.shoulder[0], self.elbow[0],],
                 [self.shoulder[1], self.elbow[1]],
                 'r-')
        plt.plot([self.elbow[0], self.wrist[0]],
                 [self.elbow[1], self.wrist[1]],
                 'r-')
        plt.plot([self.wrist[0], self.finger[0]],
                 [self.wrist[1], self.finger[1]],
                 'r-')

        plt.plot(self.shoulder[0], self.shoulder[1], 'ko')
        plt.plot(self.elbow[0], self.elbow[1], 'ko')
        plt.plot(self.wrist[0], self.wrist[1], 'ko')
        plt.plot(self.finger[0], self.finger[1], 'ko')


'''functions to help draw the angles did not modify these in any way'''
def transform_points(points, theta, origin):
    T = np.array([[cos(theta), -sin(theta), origin[0]],
                  [sin(theta), cos(theta), origin[1]],
                  [0, 0, 1]])
    return np.matmul(T, np.array(points))

def draw_angle(angle, offset=0, origin=[0, 0], r=0.5, n_points=100):
        x_start = r*cos(angle)
        x_end = r
        dx = (x_end - x_start)/(n_points-1)
        coords = [[0 for _ in range(n_points)] for _ in range(3)]
        x = x_start
        for i in range(n_points-1):
            y = sqrt(r**2 - x**2)
            coords[0][i] = x
            coords[1][i] = y
            coords[2][i] = 1
            x += dx
        coords[0][-1] = r
        coords[2][-1] = 1
        coords = transform_points(coords, offset, origin)
        plt.plot(coords[0], coords[1], 'k-')

class insertObject:
    def __init__(self, xpos, ypos):
        self.x0 = xpos
        self.y0 = ypos

    def moveObj(self, xnew, ynew):
        self.x0 = xnew
        self.y0 = ynew

    def plotObj(self):
        plt.plot(self.x0, self.y0, 'b*', lw=2.0)
        plt.annotate('Object', (self.x0+.1, self.y0))
class drawPlatforms:
    def __init__(self, x, y, w=0.2):
        self.x = x
        self.ytop = y
        self.xbottom = x
        self.ybottom = 0
        self.width = w

    def plotObj(self):
        # pat.Rectangle((self.xbottom-self.width/2,self.ybottom),self.width,self.ytop)
        plt.plot([self.x - self.width/2, self.x - self.width/2],[self.ytop, self.ybottom], 'b')
        plt.plot([self.x + self.width/2, self.x + self.width/2],[self.ytop, self.ybottom], 'b')
        plt.plot([self.x + self.width/2, self.x - self.width/2],[self.ytop, self.ytop], 'b')
        plt.plot([self.x + self.width/2, self.x - self.width/2],[self.ybottom, self.ybottom], 'b')

''' Function to create the animation '''
def update(i):
    global objPos, made2Object
    #start with cleared figure!
    plt.cla()
    ax.set_xlim(0,3)
    ax.set_ylim(0,3)
    label = 'timestep {0}, {1} ms, Configuration {2}'.format(i, (i*dt), (n+m))

    #insert object here and can now update the objects final position
    plat = drawPlatforms(objPos[0],objPos[1])
    obj = insertObject(xpos=objPos[0],ypos=objPos[1])
    # Rounding position on finger and object to two places after the decimal
    FingerXTol = round(arm.finger[0], 2)
    FingerYTol =round(arm.finger[1], 2)
    objXTol = round(objPos[0], 2)
    objYTol = round(objPos[1], 2)
    # print('Finger X pos: ' + str(FingerXTol))

    # Update the arm to the new orientation with angle changes
    max_dtheta = np.multiply(dt*10**-3, w)  # maximum angle joint can move within a timestep

    if i == 0:
        dtheta = np.subtract(Angles2Object[n], initial_angles)
        #insert object here and can now update the objects final position
        obj = insertObject(xpos=objPos[0],ypos=objPos[1])
        made2Object = 0

    # made it to the object to pick it up!
    elif FingerXTol == objXTol and FingerYTol == objYTol and made2Object == 0:
        made2Object += 1
        dtheta = np.subtract(Angles2Goal[m], arm.joint_angles)
        print('made it to object!')

    # already "grabbed object"
    elif made2Object > 0:
        dtheta = np.subtract(Angles2Goal[m], arm.joint_angles)

    # enroute to grab object after 1st frame
    else:
        dtheta = np.subtract(Angles2Object[n], arm.joint_angles)

    # find direction of joint rotations
    dir = []
    for a in dtheta:
        if a > 0: dir.append(1)
        elif a == 0: dir.append(0)
        elif a < 0: dir.append(-1)

    new_theta = []

    for a in range(len(dtheta)):
        if i > 0:
            if max_dtheta[a] <= abs(dtheta[a]):
                new_theta.append(arm.joint_angles[a] + max_dtheta[a]*dir[a])
            elif max_dtheta[a] > abs(dtheta[a]):
                new_theta.append(arm.joint_angles[a] + dtheta[a])
        else:
            if max_dtheta[a] <= abs(dtheta[a]):
                new_theta.append(initial_angles[a] + max_dtheta[a]*dir[a])
            elif max_dtheta[a] > abs(dtheta[a]):
                new_theta.append(initial_angles[a] + dtheta[a])
    i += 1
    arm.update_joints(new_theta)
    ax.set_xlabel(label)
    print('Made2Object: ' + str(made2Object))
    if made2Object > 0 :
        obj.moveObj(arm.finger[0], arm.finger[1])
        print('Trying to move object')
    return arm.plot(), obj.plotObj(), plat.plotObj()


def objectCoord():
    """ Get object coordinates """
    objX = input('Input Object X Position: ')
    objY = input('Input Object Y Position: ')
    return float(objX), float(objY)

def goalCoord():
    """ Get object coordinates """
    goalX = input('Input Goal X Position: ')
    goalY = input('Input Goal Y Position: ')
    return float(goalX), float(goalY)

def conditional(link3, objPosition, phi):
    """ This function determines if phi can be solved for with given object position.
    Returns TRUE if so. """
    xpos, ypos = objPosition
    constraint = xpos * cos(phi) + ypos * sin(phi)
    requirement = (2 - (link3**2 + xpos**2 + ypos**2))/(-2*link3)
    if constraint >= requirement: return True
    elif constraint < requirement: return False


'''create arm, object, and end goal '''
arm = ThreeLinkArm()
(objPos) = objectCoord()
obj = insertObject(xpos=objPos[0], ypos=objPos[1])
(goal) = goalCoord()
made2Object = 0

initial_angles = [0.5, 1, 1]  # initial joint positions [rad]
w = np.array([0.5, 1, 1.5])  # angular velocity of joints [rad/s]
phi = 0; phi0 = 0

# Determine if object can be reached with the arm's specified geometry
while not conditional(arm.link_lengths[2], objPos, phi):
    phi += 0.001
    if phi >= np.pi*2:
        print('ERROR. Object can not be reached. \nPlease enter another location:\n')
        (objPos) = objectCoord()
        phi = 0
    phi0 = phi

tol = 1e-3  # tolerance btw arm's finger and object

# Initial orientation
arm.update_joints(initial_angles)
fig, ax = plt.subplots()  # initialize plot
arm.plot()  # plot the first orientation

obj.plotObj()

# this loop calculates the final angles needed to reach object for each possible IK method
Angles2Object = []
cnt = 0  # keep track of how many times loop failed or executed
arm.inverse_kinematics(objPos[0], objPos[1], phi)
parameters = {'sigma': 1, 'method': 1}  # dictionary to specify how to solve inverse kinematics (IK)
solutions = []  # save solution parameters [sigma, method, phi]
while cnt < 4:
    phi += 0.001
    if conditional(arm.link_lengths[2], objPos, phi): arm.inverse_kinematics(objPos[0], objPos[1], phi, **parameters)
    if phi >= np.pi * 2 and cnt > 3:
        print('Error. Solution can not be reached within the specified tolerance. Failed: %d' % cnt)
        break
    if phi >= np.pi*2:
        print('Error. Solution can not be reached within the specified tolerance. Failed: %d' % cnt)
        cnt += 1
        phi = 0
        if cnt % 2 == 1: parameters['sigma'] *= - 1
        elif parameters['method'] == 1: parameters['method'] = 2
    if np.linalg.norm(arm.finger - np.array(objPos)) < tol:
        print('SOLVED!')
        Angles2Object.append(arm.joint_angles.copy())
        solutions.append([parameters['sigma'], parameters['method'], arm.phi*360/2/np.pi])
        phi += 0.001
        arm.inverse_kinematics(objPos[0], objPos[1], phi, **parameters)



# Create animation of arm moving to final location
dt = 100


'''Goal maneuver calculation below'''
# Determine if goal can be reached with the arm's specified geometry
phi = 0; phi0 = 0
while not conditional(arm.link_lengths[2], goal, phi):
    phi += 0.001
    if phi >= np.pi*2:
        print('ERROR. Goal can not be reached. \nPlease enter another location:\n')
        (goal) = goalCoord()
        phi = 0
    phi0 = phi

# Loop to find end orientations for goal
print('\n==========\nStarting Goal...\n')
Angles2Goal = []
cnt = 0  # keep track of how many times loop failed or executed
goal_parameters = {'sigma': 1, 'method': 1}  # dictionary to specify how to solve inverse kinematics (IK)
goal_solutions = []  # save solution parameters [sigma, method, phi]

arm.inverse_kinematics(goal[0], goal[1], phi)
while cnt < 4:
    phi += 0.001
    if conditional(arm.link_lengths[2], goal, phi): arm.inverse_kinematics(goal[0], goal[1], phi, **goal_parameters)
    if phi >= np.pi * 2 and cnt > 3:
        print('Error. Solution can not be reached within the specified tolerance. Failed: %d' % cnt)
        break
    if phi >= np.pi*2:
        print('Error. Solution can not be reached within the specified tolerance. Failed: %d' % cnt)
        cnt += 1
        phi = 0
        if cnt % 2 == 1: goal_parameters['sigma'] *= - 1
        elif goal_parameters['method'] == 1: goal_parameters['method'] = 2
    if np.linalg.norm(arm.finger - np.array(goal)) < tol:
        print('SOLVED!')
        Angles2Goal.append(arm.joint_angles.copy())
        goal_solutions.append([goal_parameters['sigma'], goal_parameters['method'], arm.phi*360/2/np.pi])
        phi += 0.001
        arm.inverse_kinematics(goal[0], goal[1], phi, **goal_parameters)


# runs animation for each optimization
steps = []; steps2 = []; totalsteps = []
for n in range(len(Angles2Object)):
    steps.append(np.divide((np.subtract(Angles2Object[n], initial_angles)),w*dt*10**-3))
    for m in range(len(Angles2Goal)):
        steps2.append(np.divide((np.subtract(Angles2Goal[m], Angles2Object[n])), w*dt*10**-3))
        totalsteps.append(max(abs(steps[n])) + max(abs(steps2[m])))
        anim = FuncAnimation(fig, update, frames=np.arange(0, math.ceil(totalsteps[-1])), interval=dt)

        # save a gif of the animation for each solution
        anim.save('optimization {}.gif'.format(len(totalsteps)), dpi=80, writer='imagemagick')

# Print results
print('\n=============\nResults:\n')
print('The Fastest Solution Time: %.2f s' % min(np.array(totalsteps)/10))
print('Configuration: %d' % totalsteps.index(min(totalsteps)))
# print('To Object Solution Paramters: \nphi = %.2f degrees\nsigma = %d' % (solutions[totalsteps.index(min(totalsteps))][2],
#                                                                 solutions[totalsteps.index(min(totalsteps))][0]))
# print('To Goal Solution Paramters: \nphi = %.2f degrees\nsigma = %d' % (solutions[totalsteps.index(min(totalsteps))][2],
#                                                                         solutions[totalsteps.index(min(totalsteps))][
#                                                                             0]))



# ''' Arm Schematic Plot Below'''
# arm = ThreeLinkArm()
# initial_angles = [0.5, 1, 1]  # initial joint positions [rad]
# [theta0, theta1, theta2] = initial_angles
# arm.update_joints(initial_angles)  # test configuraton
# arm.plot()  # test plot
#
# def label_diagram():
#     plt.plot([0, 0.5], [0, 0], 'k--')
#     plt.plot([arm.elbow[0], arm.elbow[0]+0.5*cos(theta0)],
#              [arm.elbow[1], arm.elbow[1]+0.5*sin(theta0)],
#              'k--')
#
#     # [arm.wrist[1], arm.wrist[1]+0.5*sin(theta2)]
#
#     draw_angle(theta0, r=0.25)
#     draw_angle(theta1, offset=theta0, origin=[arm.elbow[0], arm.elbow[1]], r=0.25)
#     draw_angle(theta2, offset=theta1+theta0, origin=[arm.wrist[0], arm.wrist[1]], r=0.25)
#
#     plt.annotate("$l_0$", xy=(0.5, 0.4), size=15, color="r")
#     plt.annotate("$l_1$", xy=(0.8, 1), size=15, color="r")
#     plt.annotate("$l_2$", xy=(0.7, 1.75), size=15, color="r")
#
#     plt.annotate(r"$\theta_0$", xy=(0.35, 0.05), size=15)
#     plt.annotate(r"$\theta_1$", xy=(1, 0.8), size=15)
#     plt.annotate(r"$\theta_2$", xy=(1, 1.75), size=15)
#
# label_diagram()
#
# plt.annotate("Shoulder", xy=(arm.shoulder[0], arm.shoulder[1]), xytext=(0.15, 0.5),
#     arrowprops=dict(facecolor='black', shrink=0.05))
# plt.annotate("Elbow", xy=(arm.elbow[0], arm.elbow[1]), xytext=(1.25, 0.25),
#     arrowprops=dict(facecolor='black', shrink=0.05))
# plt.annotate("Wrist", xy=(arm.wrist[0], arm.wrist[1]), xytext=(1.5, 1.5),
#     arrowprops=dict(facecolor='black', shrink=0.05))
# plt.annotate("Finger", xy=(arm.finger[0], arm.finger[1]), xytext=(0, 1.5),
#     arrowprops=dict(facecolor='black', shrink=0.05))
#
#
# plt.axis("equal")
#
# plt.show()