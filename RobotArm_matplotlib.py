"""
Created on Thu Apr 23 12:43:29 2020

@author: Emily Brady
"""
from numpy import cos, sin, sqrt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from matplotlib.animation import FuncAnimation
import math
import time

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

    def inverse_kinematics(self, xf, yf, phi):
        self.phi = phi
        self.xp = xf - self.link_lengths[2]*np.cos(self.phi)
        self.yp = yf - self.link_lengths[2]*np.sin(self.phi)

        gamma = np.arctan2(-self.yp/sqrt(self.xp**2 + self.yp**2), self.xp/sqrt(self.xp**2 + self.yp**2))
        sigma = 1
        
        arccosTerm = -(self.xp**2 + self.yp**2 + self.link_lengths[0]**2 - self.link_lengths[1]**2)/(2*self.link_lengths[0]*sqrt(self.xp**2 + self.yp**2))
                                         
        #take care of possible unsolvable problems 
        if arccosTerm < -1 or arccosTerm > 1:
            return

        theta0 = gamma + sigma*np.arccos(-(self.xp**2 + self.yp**2 + self.link_lengths[0]**2 - self.link_lengths[1]**2)
                                         / (2*self.link_lengths[0]*sqrt(self.xp**2 + self.yp**2)))

        theta1 = np.arctan2((self.yp - self.link_lengths[0]*sin(theta0))/self.link_lengths[1],
                            (self.xp - self.link_lengths[0]*cos(theta0))/self.link_lengths[1]) - theta0

        theta2 = self.phi - theta0 - theta1

        return self.update_joints([theta0, theta1, theta2])
    
    
    def findBestPhi (self, listPhi, objX, objY, startAngles):
        bestPhi = 0
        minSteps = 0
        
#        print('MinSteps type ' + str(type(minSteps)))

        for phi in listPhi_object:
            self.inverse_kinematics(objX, objY, phi)
            Angles2Object = self.joint_angles
            MaxSteps = max(abs(np.divide(np.subtract(Angles2Object,startAngles),w*dt*10**-3)))
            
#            print('MaxSteps: ' +str(MaxSteps) + ' steps type: ' + str(type(MaxSteps)))

            if minSteps == 0:
                minSteps = MaxSteps
                bestPhi = phi
            elif MaxSteps < minSteps:
                minSteps = MaxSteps
                bestPhi = phi
        return (bestPhi)
        
        
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

    def axplot(self):
        def plot(self):
            ax.plot([self.shoulder[0], self.elbow[0], ],
                     [self.shoulder[1], self.elbow[1]],
                     'r-')
            ax.plot([self.elbow[0], self.wrist[0]],
                     [self.elbow[1], self.wrist[1]],
                     'r-')
            ax.plot([self.wrist[0], self.finger[0]],
                     [self.wrist[1], self.finger[1]],
                     'r-')

            ax.plot(self.shoulder[0], self.shoulder[1], 'ko')
            ax.plot(self.elbow[0], self.elbow[1], 'ko')
            ax.plot(self.wrist[0], self.wrist[1], 'ko')
            ax.plot(self.finger[0], self.finger[1], 'ko')

'''functions to help draw the angles
did not modify these in any way'''
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
    #start with cleared figure!
    plt.cla()
    ax.set_xlim(-3,3)
    ax.set_ylim(0,3)
    label = 'timestep {0}, {1} ms'.format(i,(i*dt))
    
    global made2Object
    #insert object here and can now update the objects final position
    objPos = (1.5, 1.5)
    plat = drawPlatforms(objPos[0],objPos[1])
    obj = insertObject(xpos=objPos[0],ypos=objPos[1])
    # Rounding position on finger and object to two places after the decimal
    FingerXTol = round(arm.finger[0], 2)
    FingerYTol =round(arm.finger[1], 2)
    objXTol = round(objPos[0], 2)
    objYTol = round(objPos[1], 2)
 
    #print('Finger X pos: ' + str(FingerXTol))
    
    

    # Update the arm to the new orientation with angle changes
    max_dtheta = np.multiply(dt*10**-3,w)  # maximum angle joint can move within a timestep
    if i == 0:
        dtheta = np.subtract(Angles2Obj, initial_angles)
        #insert object here and can now update the objects final position
        objPos = (1.5, 1.5)
        obj = insertObject(xpos=objPos[0],ypos=objPos[1])

    # made it to the object to pick it up!
    elif FingerXTol == objXTol and FingerYTol == objYTol:
        made2Object += 1
        dtheta = np.subtract(Angles2Goal, arm.joint_angles)
    #already "grabbed object"
    elif made2Object > 0:
        dtheta = np.subtract(Angles2Goal, arm.joint_angles)
        print('made it to object!')
    else:
        dtheta = np.subtract(Angles2Obj, arm.joint_angles)
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

'''create arm, object, and end goal '''
arm = ThreeLinkArm()
objPos = (1.5, 1.5)
obj = insertObject(xpos=objPos[0],ypos=objPos[1])
goal = (0, 1.4)
made2Object = 0
        
initial_angles = [0.5, 1, 1]  # initial joint positions [rad]
w = np.array([0.5, 1, 1.5])  # angular velocity of joints [rad/s]
phi = 0  # end effector orientation (must be solved for)
tol = 1e-3  # tolerance btw arm's finger and object

#Initial orientation
arm.update_joints(initial_angles)
fig, ax = plt.subplots()  # initialize plot
# fig.set_tight_layout(True)
arm.plot()  # plot the first orientation

#obj.plotObj()

#this calculates the final angles needed to reach object
listPhi_object = []
arm.inverse_kinematics(objPos[0], objPos[1], phi)

#checking phi from 0 to 2pi radians
while phi < 6.2832:
    while np.linalg.norm(arm.finger - np.array(objPos)) >= tol:
        phi += 0.001
        arm.inverse_kinematics(objPos[0], objPos[1], phi)
    if np.linalg.norm(arm.finger - np.array(objPos)) < tol:
        listPhi_object.append(phi)
        print('Added new phi value Obj: ' + str(phi))
        phi += 0.001
        arm.inverse_kinematics(objPos[0], objPos[1], phi)
    
print('ListPhi Object: ' + str(listPhi_object))

# Create animation of arm moving to final location
dt = 100
'''
Angles2Object = arm.joint_angles.copy()
print('Angles2Object ' +str(Angles2Object))
'''

'''might need this in the update function'''
#solve for end position to move object to!
listPhi_goal = []
arm.inverse_kinematics(goal[0], goal[1], phi)
#checking phi from 0 to 2pi radians
while phi < 6.2832:
    while np.linalg.norm(arm.finger - np.array(goal)) >= tol:
        phi += 0.001
        arm.inverse_kinematics(goal[0], goal[1], phi)
    listPhi_goal.append(phi)
    print('Added new phi value Goal: ' + str(phi))
    phi += 0.001

'''
Angles2Goal = arm.joint_angles.copy()
print('Angles 2 goal' + str(Angles2Goal))
'''
#determin phi's with least amount of steps -> least amount of time
bestPhi_obj = arm.findBestPhi(listPhi_object, objPos[0], objPos[1], initial_angles)
arm.inverse_kinematics(objPos[0], objPos[1], bestPhi_obj)
Angles2Obj = arm.joint_angles.copy()

bestPhi_goal = arm.findBestPhi(listPhi_goal, goal[0], goal[1], Angles2Obj)
arm.inverse_kinematics(goal[0], goal[1], bestPhi_goal)
Angles2Goal = arm.joint_angles.copy()


# for phi in listPhi_object:
#     arm.inverse_kinematics(objPos[0], objPos[1], phi)
#     Angles2Object = arm.joint_angles.copy()
#     steps = np.divide(np.subtract(Angles2Object,initial_angles),w*dt*10**-3)
#     if minSteps == 0:
#         minSteps = steps
#         bestPhi_obj = phi
#     elif steps < minSteps:
#         minSteps = steps
#         bestPhi_obj = phi


# bestPhi_goal = 0
# minStepsGoal = 0
# for phi in listPhi_goal:
#     arm.inverse_kinematics(objPos[0], objPos[1], phi)
#     Angles2Goal = arm.joint_angles.copy()
#     steps = np.divide(np.subtract(Angles2Goal,initial_angles),w*dt*10**-3)
#     if minStepsGoal == 0:
#         minStepsGoal = steps
#         bestPhi_goal = phi
#     elif steps < minStepsGoal:
#         minStepsGoal = steps
#         bestPhi_goal = phi


steps = np.divide((np.subtract(Angles2Obj,initial_angles)),w*dt*10**-3)
steps2 = np.divide(np.subtract(Angles2Goal, Angles2Obj), w*dt*10**-3)
print('Number of steps: ' + str(steps))
print('Number of steps2: ' + str(steps2))
totalsteps = max(abs(steps)) + max(abs(steps2))
anim = FuncAnimation(fig, update, frames=np.arange(0, math.ceil(totalsteps)), interval=dt, repeat_delay = 300)

# save a gif of the animation using the writing package from magick
anim.save('arm_test1.gif', dpi=80, writer='imagemagick')

# arm.update_joints([theta0, theta1, theta2])  # test configuraton
# arm.plot()  # test plot

# ''' Test Annotation '''
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


# plt.axis("equal")

# plt.show()
