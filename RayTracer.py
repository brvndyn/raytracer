import sys
import re
import numpy as np
import math

class Sphere:
    def __init__(self, name, posX, posY, posZ, sclX, sclY, sclZ, r, g, b, Ka, Kd, Ks, Kr, n):
        self.name = name
        self.posX = posX
        self.posY = posY
        self.posZ = posZ
        self.pos = np.array([posX,posY,posZ])
        self.sclX = sclX
        self.sclY = sclY
        self.sclZ = sclZ
        self.r = r
        self.g = g
        self.b = b
        self.colour = np.array([r,g,b])
        self.Ka = Ka
        self.Kd = Kd
        self.Ks = Ks
        self.Kr = Kr
        self.n = n

        # scaling matrix and its inverse
        self.S = [[self.sclX,0,0,0],[0,self.sclY,0,0],[0,0,self.sclZ,0],[0,0,0,1]]
        self.inverseS = np.linalg.inv(self.S)

        # Translation matrix and inverse
        self.T = np.array([[1,0,0,self.posX],[0,1,0,self.posY],[0,0,1,self.posZ],[0,0,0,1]])
        self.inverseT = np.linalg.inv(self.T)

        self.transformM = np.matmul(self.T,self.S)


        self.transposeM = np.transpose(self.transformM)
        self.inverseM = np.linalg.inv(self.transformM)
        self.invTransposeM = np.matmul(self.transposeM, self.inverseM)
        # Sphere always positioned at origin for intersection calculations,
        # using inverse transform matrix to move it
        self.center = np.array([0,0,0])

        # Because the ray origin is always the eye,
        # it is more efficient to calculate the transformed
        # origin once per sphere
        origin = [0,0,0, 1]
        origin = np.matmul(self.inverseM, origin)
        self.transformedOrigin = origin[0:3]

    def intersects(self, ray):

        # Apply the inverse transform matrices to our ray direction
        direction = [ray.direction[0], ray.direction[1], ray.direction[2], 0]
        direction = np.matmul(self.inverseM, direction)
        direction = direction[0:3]
        direction = normalize(direction)


        # Quadratic equation
        L = self.transformedOrigin - self.center
        b = 2.0*direction.dot(L)
        c = L.dot(L) - 1
        d = b*b - 4*c


        if d >= 0:
            t = (-b - math.sqrt(d)) / 2.0
            return t
        return -1




class Light:
    def __init__(self, name, posX, posY, posZ, lr, lg, lb):
        self.name = name
        self.posX = posX
        self.posY = posY
        self.posZ = posZ
        self.pos = np.array([posX,posY,posZ])
        self.lr = lr
        self.lg = lg
        self.lb = lb
        self.intensity = np.array([lr,lg,lb])

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = normalize(direction)

    def pointAt(self,t):
        return self.origin + t*self.direction




def normalize(vector):
    mag = math.sqrt(vector.dot(vector))
    vector[0] = vector[0]/mag
    vector[1] = vector[1]/mag
    vector[2] = vector[2]/mag
    return(vector)


global sphereArray
global lightArray
sphereArray = []
lightArray = []

def parse():
    file_name = sys.argv[1]


    for line in open(file_name,'r'):
        if re.match('NEAR', line):
            global near
            near = int(re.sub('NEAR', '', line).strip())

        elif re.match('LEFT', line):
            global left
            left = int(re.sub('LEFT', '', line).strip())

        elif re.match('RIGHT', line):
            global right
            right = int(re.sub('RIGHT', '', line).strip())

        elif re.match('BOTTOM', line):
            global bottom
            bottom = int(re.sub('BOTTOM', '', line).strip())

        elif re.match('TOP', line):
            global top
            top = int(re.sub('TOP', '', line).strip())

        elif re.match('RES', line):
            global width
            global height
            res = re.sub('RES', '', line)
            res = res.split()
            res = list(map(int, res))
            width = res[0]
            height = res[1]

        elif re.match('SPHERE', line):
            sphereData = re.sub('SPHERE', '', line)
            sphereData = sphereData.split()

            sphereName = sphereData[0]
            sphereData[0] = 0
            sphereData = list(map(float, sphereData))

            sphere = Sphere(sphereName, sphereData[1], sphereData[2], sphereData[3], sphereData[4],
            sphereData[5], sphereData[6], sphereData[7], sphereData[8], sphereData[9], sphereData[10],
            sphereData[11], sphereData[12], sphereData[13], sphereData[14])


            sphereArray.append(sphere)

        elif re.match('LIGHT', line):
            lightData = re.sub('LIGHT', '', line)
            lightData = lightData.split()
            lightName = lightData[0]
            lightData[0] = 0
            lightData = list(map(float, lightData))

            light = Light(lightName, lightData[1], lightData[2], lightData[3], lightData[4],
            lightData[5], lightData[6])

            lightArray.append(light)


        elif re.match('BACK', line):
            global backR
            global backG
            global backB
            back = re.sub('BACK', '', line)
            back = back.split()
            back = list(map(float, back))
            backR = back[0]
            backG = back[1]
            backB = back[2]

        elif re.match('AMBIENT', line):
            global ambientLr
            global ambientLg
            global ambientLb
            ambient =  re.sub('AMBIENT', '', line)
            ambient = ambient.split()
            ambient = list(map(float, ambient))
            ambientLr = ambient[0]
            ambientLg = ambient[1]
            ambientLb = ambient[2]

        elif re.match('OUTPUT', line):
            global output
            output = re.sub('OUTPUT', '', line).strip()



def colourAt(p, sphere):
    ambient = sphere.colour*(sphere.Ka*np.array([ambientLr,ambientLg,ambientLb]))
    colour = ambient



    N = (p - sphere.pos)
    N = [N[0],N[1],N[2],0] # homogeneous


    N = np.matmul(sphere.inverseM, N)
    N = np.matmul(sphere.invTransposeM ,N)

    N = N[0:3]
    # N = normalize(N) this breaks everything idk why

    eyeRay = np.array([0,0,-1])
    if eyeRay.dot(N) > 0:
        return colour

    V = normalize(-sphere.pos)
    for light in lightArray:
        L = normalize(light.pos - p)
        R = 2*(N.dot(L))*N-L
        R = normalize(R)
        n = sphere.n

        #Diffuse
        colour = colour + (sphere.Kd*light.intensity*(N.dot(L))*sphere.colour)

        #Specular
        colour = colour + (sphere.Ks*light.intensity*((R.dot(V))**n))

    if colour[0] > 1:
        colour[0] = 1
    if colour[1] > 1:
        colour[1] = 1
    if colour[2] > 1:
        colour[2] = 1


    return colour

def trace(ray):
    colour = np.array([backR,backG,backB])
    sphereHit = None
    distMin = None

    for sphere in sphereArray:
        t = sphere.intersects(ray)

        # Check the interesection distance against the near plane and other hit objects
        if t > near and (sphereHit is None or t < distMin):
            distMin = t
            sphereHit = sphere
            p = ray.pointAt(t)
            colour = colourAt(p, sphere)

    return colour




def progressBar(current, total, barLength = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))

    print('Tracing rays... [%s%s] %d %%' % (arrow, spaces, percent), end='\r')


def write(f, colour):
    f.write("{} {} {} ".format(round(colour[0]*255),round(colour[1]*255),round(colour[2]*255)))
    f.write('\n')


def main():
    if(len(sys.argv) != 2):
        print("argument error")
        quit()

    parse()

    eye = np.array([0,0,0])
    N = np.array([0,0,near])
    horizontal = np.array([right-left,0,0])
    vertical = np.array([0,top-bottom,0])

    lowerLeftCorner = eye - N - horizontal/2 - vertical/2

    f = open(output, "w")
    f.write("P3 "+str(width)+" "+str(height)+"\n")
    f.write("255\n")


    for j in range(height, 0, -1):
        progressBar(height-j+1,height)
        for i in range(width):
            u = i/width
            v = j/height

            ray = Ray(eye, lowerLeftCorner + u*horizontal + v*vertical - eye)
            colour = trace(ray)

            write(f,colour)


    print('\nDone!')
    f.close()

if __name__ == '__main__':
    main()
