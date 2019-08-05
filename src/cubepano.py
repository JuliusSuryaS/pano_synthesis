import sys
from PIL import Image
from math import sin, cos, tan, pi


def cot(angle: float):
    return 1/tan(angle)

def rpojection(theta: float, phi: float):
    if theta < 0.615:
        return projectTop(theta, phi)
    elif theta > 2.527:
        return projectBottom(theta, phi)
    elif phi <= pi/4 or phi > 7*pi/4:
        return projectLeft(theta, phi)
    elif phi > pi/4 or phi <= 3*pi/4:
        return projectFront(theta, phi)
    elif phi > 3*pi/4 or phi <= 5*pi/4:
        return projectRight(theta, phi)
    elif phi > 5*pi/4 or phi <= 7*pi/4:
        return projectBack(theta, phi)

def projectLeft(theta, phi):
    x = 1
    y = tan(phi)
    z = cot(theta)/cos(phi)
    if z < -1:
        return projectBottom(theta, phi)
    if z > 1:
        return projectTop(theta, phi)
    return ('Left', x, y, z)

def projectFront(theta, phi):
    x = tan(phi-pi/2)
    y = 1
    z = cot(theta)/cos(phi-pi/2)
    if z < -1:
        return projectBottom(theta, phi)
    if z > 1:
        return projectTop(theta, phi)
    return ('Front', x, y, z)

def projectRight(theta, phi):
    x = -1
    y = tan(phi)
    z = -cot(theta)/cos(phi)
    if z < -1:
        return projectBottom(theta, phi)
    if z > 1:
        return projectTop(theta, phi)
    return ('Right', x, y, z)

def projectBack(theta, phi):
    x = tan(phi-3*pi/2)
    y = -1
    z = cot(theta)/cos(phi-3*pi/2)
    if z < -1:
        return projectBottom(theta, phi)
    if z > 1:
        return projectTop(theta, phi)
    return ('Back', x, y, z)

def projectTop(theta, phi):
    a = 1/cos(theta)
    x = tan(theta) * cos(phi)
    y = tan(theta) * sin(phi)
    z = 1
    return ('Top', x, y, z)

def projectBottom(theta, phi):
    a = -1/cos(theta)
    x = -tan(theta) * cos(phi)
    y = -tan(theta) * sin(phi)
    z = -1
    return ('Bottom', x, y, z)

