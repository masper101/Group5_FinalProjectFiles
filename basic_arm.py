"""
@author: Emily Brady, Matt Asper, Connor Gunsbury
"""
from numpy import cos, sin, sqrt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from matplotlib.animation import FuncAnimation
import math
import time

print("Enter the object starting and target locations below. Range 0-3")
while True:
    objx = eval(input("Starting x position: "))
    if objx >= 0 and objx <= 3:
        break
    else:
        print("Position out of range. Please enter a value between 0 and 3.")
while True:
    objy = eval(input("Starting y position: "))
    if objy >= 0 and objy <= 3:
        break
    else:
        print("Position out of range. Please enter a value between 0 and 3.")
while True:
    targx = eval(input("Target x position: "))
    if targx >= 0 and targx <= 3:
        break
    else:
        print("Position out of range. Please enter a value between 0 and 3.")
while True:
    targy = eval(input("Target y position: "))
    if targy >= 0 and targy <= 3:
        break
    else:
        print("Position out of range. Please enter a value between 0 and 3.")
gif = input("Please enter the filename for the saved GIF: ")

if gif[-4:] == ".gif":
    gifname = gif
else:
    gifname = gif + ".gif"


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

        theta0 = gamma + sigma*np.arccos(-(self.xp**2 + self.yp**2 + self.link_lengths[0]**2 - self.link_lengths[1]**2)
                                         / (2*self.link_lengths[0]*sqrt(self.xp**2 + self.yp**2)))

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
        
    def plotObj(self,color='b'):
        rect = pat.Rectangle((self.x - self.width/2,0),self.width,self.ytop,facecolor=color)
        ax.add_patch(rect)
        
''' Function to create the animation '''
def update(i):
    global totalsteps
    if i > math.ceil(totalsteps):
        return
    #start with cleared figure!
    plt.cla()
    ax.set_xlim(-.5,3)
    ax.set_ylim(0,3)
    label = 'timestep {0}, {1} ms'.format(i,(i*dt))
    
    global made2Object
    #insert object here and can now update the objects final position
    objPos = (objx, objy)
    plat = drawPlatforms(objPos[0],objPos[1])
    plattarg = drawPlatforms(targx,targy)
    obj = insertObject(xpos=objPos[0],ypos=objPos[1])
    # Rounding position on finger and object to two places after the decimal
    FingerXTol = round(arm.finger[0], 2)
    FingerYTol =round(arm.finger[1], 2)
    objXTol = round(objPos[0], 2)
    objYTol = round(objPos[1], 2)
    # print('Finger X pos: ' + str(FingerXTol))
    
    

    # Update the arm to the new orientation with angle changes
    max_dtheta = np.multiply(dt*10**-3,w)  # maximum angle joint can move within a timestep
    if i == 0:
        dtheta = np.subtract(Angles2Object, initial_angles)
        #insert object here and can now update the objects final position
        objPos = (objx, objy)
        obj = insertObject(xpos=objPos[0],ypos=objPos[1])

    # made it to the object to pick it up!
    elif FingerXTol == objXTol and FingerYTol == objYTol:
        made2Object += 1
        dtheta = np.subtract(Angles2Goal, arm.joint_angles)
    #already "grabbed object"
    elif made2Object > 0:
        dtheta = np.subtract(Angles2Goal, arm.joint_angles)
        # print('made it to object!')
    else:
        dtheta = np.subtract(Angles2Object, arm.joint_angles)
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
    # print('Made2Object: ' + str(made2Object))
    if made2Object > 0 :
        obj.moveObj(arm.finger[0], arm.finger[1])
        # print('Trying to move object')
    return arm.plot(), obj.plotObj(), plat.plotObj(), plattarg.plotObj(color='g')

'''create arm, object, and end goal '''
arm = ThreeLinkArm()
objPos = (objx, objy)
obj = insertObject(xpos=objPos[0],ypos=objPos[1])
goal = (targx, targy)
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

obj.plotObj()
#this calculates the final angles needed to reach object
arm.inverse_kinematics(objPos[0], objPos[1], phi)
while np.linalg.norm(arm.finger - np.array(objPos)) >= tol:
    phi += 0.001
    arm.inverse_kinematics(objPos[0], objPos[1], phi)

# Create animation of arm moving to final location
dt = 100
Angles2Object = arm.joint_angles.copy()
# print('Angles2Object ' +str(Angles2Object))

'''might need this in the update function'''
#solve for end position to move object to!
arm.inverse_kinematics(goal[0], goal[1], phi)
while np.linalg.norm(arm.finger - np.array(goal)) >= tol:
    phi += 0.001
    arm.inverse_kinematics(goal[0], goal[1], phi)

Angles2Goal = arm.joint_angles.copy()
# print('Angles 2 goal' + str(Angles2Goal))
steps = np.divide((np.subtract(Angles2Object,initial_angles)),w*dt*10**-3)
steps2 = np.divide(+ np.subtract(Angles2Goal, Angles2Object), w*dt*10**-3)
# print('Number of steps: ' + str(steps))
# print('Number of steps2: ' + str(steps2))
totalsteps = max(abs(steps)) + max(abs(steps2))
anim = FuncAnimation(fig, update, frames=np.arange(0, math.ceil(totalsteps)+30), interval=dt)

# save a gif of the animation using the writing package from magick
anim.save(gifname, dpi=80, writer='imagemagick')
