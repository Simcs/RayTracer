#!/usr/bin/env python3
# -*- coding: utf-8 -*
# sample_python aims to allow seamless integration with lua.
# see examples below

import os
import sys
import pdb  # use pdb.set_trace() for debugging
import code # or use code.interact(local=dict(globals(), **locals()))  for debugging.
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from abc import * 

class Color:
    def __init__(self, R, G, B):
        self.color = np.array([R, G, B]).astype(np.float)

    # Gamma corrects this color.
    # @param gamma the gamma value to use (2.2 is generally used).
    def gammaCorrect(self, gamma):
        inverseGamma = 1.0 / gamma
        self.color=np.power(self.color, inverseGamma)

    def toUINT8(self):
        return (np.clip(self.color, 0, 1) * 255).astype(np.uint8)

def normalize(x):
    return x / np.linalg.norm(x)

# abstract class for shader 'Lambertian' and 'Phong'
class Shader(metaclass=ABCMeta):
    def __init__(self, name, dColor):
        self.name = name
        self.dColor = dColor

class Lambertian(Shader):
    def __str__(self):
        return 'name : ' + str(self.name) + ' dColor : ' + str(self.dColor)

class Phong(Shader):
    def __init__(self, name, dColor, sColor, exp):
        self.name = name
        self.dColor = dColor
        self.sColor = sColor
        self.exp = exp

    def __str__(self):
        return 'name : ' + str(self.name) + ' dColor : ' + str(self.dColor) + ' sColor : ' \
            + str(self.sColor) + ' exp : ' + str(self.exp)

class Light:
    def __init__(self, position, intensity):
        self.position = position
        self.intensity = intensity

    def __str__(self):
        return 'position : ' + str(self.position) + ' intensity : ' + str(self.intensity)

class Camera:
    def __init__(self, viewPoint, viewDir, projNormal, projDistance, viewUp, viewWidth, viewHeight):
        self.viewPoint = viewPoint
        self.viewDir = viewDir
        self.projNormal = projNormal
        self.projDistance = projDistance
        self.viewUp = viewUp
        self.viewWidth = viewWidth
        self.viewHeight = viewHeight

    def __str__(self):
        return 'viewPoint : ' + str(self.viewPoint) + ' viewDir : ' + str(self.viewDir) + ' distance : ' + str(self.projDistance)

# abstract class for surface 'Sphere' and 'Box'
class Surface(metaclass=ABCMeta):
    def __init__(self, shader):
        self.shader = shader

    @abstractmethod
    def normal(self, m):
        pass

    @abstractmethod
    def light(self, eye, ray, t, scene, light):
        pass

    @abstractmethod
    def isBlocked(self, m, l, scene):
        pass
    
    @abstractmethod
    def intersect(self, eye, ray):
        pass

class Sphere(Surface):
    def __init__(self, shader, center, radius):
        self.shader = shader
        self.center = center
        self.radius = radius

    def __str__(self):
        return 'shader : { ' + str(self.shader) + ' } center : '+ str(self.center) + ' radius : ' + str(self.radius)

    def normal(self, m):
        return normalize(m - self.center)

    def light(self, eye, ray, t, scene, light):
        # intersection point
        m = eye + ray * t
        # vector to light
        l = normalize(light.position - m)
        # vector to eye
        v = (-1) * ray

        # check if shadowed
        if self.isBlocked(m, l, scene):
            return np.zeros(3).astype(float)
        
        color = np.zeros(3).astype(float)

        # Lambertian shading
        n = self.normal(m)
        color += light.intensity * self.shader.dColor * max(np.dot(n, l), 0)

        # Phong shading
        if type(self.shader) == Phong:
            phong = np.dot(n, normalize(v + l))
            color += light.intensity * self.shader.sColor * max(np.power(np.clip(phong, 0, 1), self.shader.exp), 0)

        return color
    
    # returns true if ray blocked by any surface
    def isBlocked(self, m, l, scene):
        shadow_bias = 0.0001
        n = self.normal(m)
        for s in scene:
            if s.intersect(m + n * shadow_bias, l) != np.inf:
                return True
        return False
    
    # use discriminant in quadratic function
    def intersect(self, eye, ray):
        p = eye - self.center
        a = np.dot(ray, ray)
        b = 2 * np.dot(ray, p)
        c = np.dot(p, p) - (self.radius ** 2)
        disc = b ** 2 - 4 * a * c
        if disc > 0:
            delta = np.sqrt(disc)
            tmax = ((-1) * b + delta) / (2.0 * a)
            tmin = ((-1) * b - delta) / (2.0 * a)
            # returns minimum positive t value if exists
            if tmin >= 0:
                return tmin
            elif tmax >= 0:
                return tmax
        # if both t values are negative
        return np.inf

class Box(Surface):
    def __init__(self, shader, minPt, maxPt):
        self.shader = shader
        self.minPt = minPt
        self.maxPt = maxPt
    
    def __str__(self):
        return 'shader : { ' + str(self.shader) + ' } minPt : ' + str(self.minPt) + ' maxPt : ' + str(self.maxPt)

    # returns list of possible normal vectors
    # if intersection point m is on the edge or vertex of the box, returns every possible normal vectors 
    def normal(self, m):
        box_bias = .0001
        x, y, z = m[0], m[1], m[2]
        normal = []
        if x < self.minPt[0] + box_bias and x > self.minPt[0] - box_bias:
            normal.append(np.array([-1, 0, 0]).astype(float))
        elif x < self.maxPt[0] + box_bias and x > self.maxPt[0] - box_bias:
            normal.append(np.array([1, 0, 0]).astype(float))
        if y < self.minPt[1] + box_bias and y > self.minPt[1] - box_bias:
            normal.append(np.array([0, -1, 0]).astype(float))
        elif y < self.maxPt[1] + box_bias and y > self.maxPt[1] - box_bias:
            normal.append(np.array([0, 1, 0]).astype(float))
        if z < self.minPt[2] + box_bias and z > self.minPt[2] - box_bias:
            normal.append(np.array([0, 0, -1]).astype(float))
        elif z < self.maxPt[2] + box_bias and z > self.maxPt[2] - box_bias:
            normal.append(np.array([0, 0, 1]).astype(float))
        return normal
    
    def light(self, eye, ray, t, scene, light):
        # intersection point
        m = eye + ray * t
        # vector to light
        l = normalize(light.position - m)
        # vector to eye
        v = (-1) * ray

        # check if shadowed
        if self.isBlocked(m, l, scene):
            return np.zeros(3).astype(float)
        
        color = np.zeros(3).astype(float)

        # Lambertian shading
        normals = self.normal(m)
        avg_dot = 0
        for n in normals:
            avg_dot += max(np.dot(n, l), 0)
        avg_dot /= len(normals)
        color += light.intensity * self.shader.dColor * avg_dot

        # Phong shading
        if type(self.shader) == Phong:
            avg_dot = 0
            for n in normals:
                avg_dot += np.clip(np.dot(n, normalize(v + l)), 0, 1)
            avg_dot /= len(normals)
            color += light.intensity * self.shader.sColor * np.power(avg_dot, self.shader.exp)

        return color
    
    def isBlocked(self, m, l, scene):
        shadow_bias = 0.0001
        normals = self.normal(m)
        for s in scene:
            # returns true if blocked by every possible normal vectors
            blocked = True
            for n in normals:
                if s.intersect(m + n * shadow_bias, l) == np.inf:
                    blocked = False
            if blocked:
                return True
        return False
    
    def intersect(self, O, D):
        invdir = 1 / D
        if invdir[0] >= 0:  
            tmin = (self.minPt[0] - O[0]) * invdir[0]
            tmax = (self.maxPt[0] - O[0]) * invdir[0]
        else:
            tmin = (self.maxPt[0] - O[0]) * invdir[0]
            tmax = (self.minPt[0] - O[0]) * invdir[0]
        
        if invdir[1] >= 0:  
            tymin = (self.minPt[1] - O[1]) * invdir[1]
            tymax = (self.maxPt[1] - O[1]) * invdir[1]
        else:
            tymin = (self.maxPt[1] - O[1]) * invdir[1]
            tymax = (self.minPt[1] - O[1]) * invdir[1]

        if (tmin > tymax) or (tymin > tmax):
            return np.inf
        tmin = max(tmin, tymin)
        tmax = min(tmax, tymax)

        if invdir[2] >= 0:  
            tzmin = (self.minPt[2] - O[2]) * invdir[2]
            tzmax = (self.maxPt[2] - O[2]) * invdir[2]
        else:
            tzmin = (self.maxPt[2] - O[2]) * invdir[2]
            tzmax = (self.minPt[2] - O[2]) * invdir[2]

        if (tmin > tzmax) or (tzmin > tmax):
            return np.inf
        tmin = max(tmin, tzmin)
        tmax = min(tmax, tzmax)

        if tmin >= 0:
            return tmin
        elif tmin < 0 and tmax >= 0:
            return tmax
        return np.inf        

def parseShader(root):
    shaders = []
    for s in root.findall('shader'):
        t = s.get('type')
        name = s.get('name')
        dColor = np.array(s.findtext('diffuseColor').split()).astype(np.float)
        if t == 'Lambertian':
            shaders.append(Lambertian(name, dColor))
        elif t == 'Phong':
            sColor = np.array(s.findtext('specularColor').split()).astype(np.float)
            exp = int(s.findtext('exponent'))
            shaders.append(Phong(name, dColor, sColor, exp))
    return shaders

def parseSurface(root, shaders):
    surfaces = []
    for s in root.findall('surface'):
        t = s.get('type')
        ref = s.find('shader').get('ref')
        for sh in shaders:
            if ref == sh.name:
                shader = sh
                break

        if t == 'Sphere':
            center = np.array(s.findtext('center').split()).astype(np.float)
            radius = float(s.findtext('radius'))
            surfaces.append(Sphere(shader, center, radius))
        elif t == 'Box':
            minPt = np.array(s.findtext('minPt').split()).astype(np.float)
            maxPt = np.array(s.findtext('maxPt').split()).astype(np.float)
            surfaces.append(Box(shader, minPt, maxPt))
    return surfaces

def parseLight(root):
    lights = []
    for s in root.findall('light'):
        position = np.array(s.findtext('position').split()).astype(np.float)
        intensity = np.array(s.findtext('intensity').split()).astype(np.float)
        lights.append(Light(position, intensity))
    return lights

def parseCamera(root):
    c = root.find('camera')
    viewPoint = np.array(c.findtext('viewPoint').split()).astype(np.float)
    viewDir = np.array(c.findtext('viewDir').split()).astype(np.float)
    projNormal = np.array(c.findtext('projNormal').split()).astype(np.float) # projNormal = -1 * viewDir
    projDistance = float(c.findtext('projDistance')) if c.findtext('projDistance') is not None else 1.
    viewUp = np.array(c.findtext('viewUp').split()).astype(np.float)
    viewWidth = float(c.findtext('viewWidth'))
    viewHeight = float(c.findtext('viewHeight'))
    return Camera(viewPoint, viewDir, projNormal, projDistance, viewUp, viewWidth, viewHeight)

def trace_ray(eye, ray, scene, lights):
    # find nearest intersection point with ray
    t_min = np.inf
    for i, s in enumerate(scene):
        tmp = s.intersect(eye, ray)
        if tmp < t_min:
            t_min, idx = tmp, i
            
    # no surface intersects with ray
    if t_min == np.inf:
        return np.zeros(3).astype(float)

    color = np.zeros(3).astype(float)
    s = scene[idx]
    for light in lights:
        color += s.light(eye, ray, t_min, scene, light)
    return color


def main():
    tree = ET.parse(sys.argv[1])
    root = tree.getroot()

    # Create an empty image
    imgSize = np.array(root.findtext('image').split()).astype(np.int)
    channels = 3
    img = np.zeros((imgSize[1], imgSize[0], channels), dtype=np.uint8)
    img[:,:] = 0
    
    # parse elements from xml file
    shaders = parseShader(root)
    scene = parseSurface(root, shaders)
    lights = parseLight(root)
    camera = parseCamera(root)

    # create camera space
    u = normalize(np.cross(camera.viewDir, camera.viewUp))
    v = normalize(np.cross(u, camera.viewDir))
    w = np.cross(u, v)
    eye = camera.viewPoint
    dist = camera.projDistance

    ratio = camera.viewWidth / camera.viewHeight
    S = (-1., -1. / ratio, 1., 1. / ratio)

    for i, x in enumerate(np.linspace(S[0], S[2], imgSize[0])):
        for j, y in enumerate(np.linspace(S[1], S[3], imgSize[1])):
            # pixel-to-image mapping
            x1 = x + (S[2] - S[0]) / (imgSize[0] * 2)
            y1 = y + (S[3] - S[1]) / (imgSize[1] * 2)

            ray = normalize(x1 * u + y1 * v - dist * w)
            color = trace_ray(eye, ray, scene, lights)

            # correct gamma using skeleton Color class
            c = Color(color[0], color[1], color[2])
            c.gammaCorrect(2.2)
            img[imgSize[1] - j - 1][i] = c.toUINT8()

    rawimg = Image.fromarray(img, 'RGB')
    rawimg.save(sys.argv[1] + '.png')
    
if __name__=="__main__":
    main()