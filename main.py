# python -m pygbag --PYBUILD 3.12 --ume_block 0 --template noctx.tmpl .

# /// script
# dependencies = [
#  "numpy",
#  "pygame",
#  "struct",
#  "zengl",
#  "marshmallow",
#  "opencv-python"
# ]
# ///

import numpy as np
import asyncio
import pygame as pygame
import struct
import zengl
#import cv2
from PIL import Image
import sys

#if sys.platform == 

HEIGHT, WIDTH = 1080, 1920
halfHeight, halfWidth = HEIGHT*0.5, WIDTH*0.5

pygame.init()
pygame.display.set_mode((WIDTH, HEIGHT), flags=pygame.OPENGL|pygame.DOUBLEBUF)
clock = pygame.time.Clock()
ctx = zengl.context()
size = pygame.display.get_window_size()
image = ctx.image(size, 'rgba8unorm', samples= 4)
depth = ctx.image(size, 'depth24plus', samples= 4)
output = ctx.image(size, 'rgba8unorm')

#####################################################################################

input_map = {'right': pygame.K_d,
             'left': pygame.K_a,
             'forwards': pygame.K_w,
             'backwards': pygame.K_s,
             'jump': pygame.K_SPACE,
             'sprint': pygame.K_LSHIFT}

#####################################################################################

CONTINUE = 0
NEW_GAME = 1
OPEN_MENU = 2
EXIT = 3

ENTITY_TYPE = {"player": 0,
               "skybox": 1,
               "cube": 2,
               "Camilla's_tent": 3,
               "drone_factory": 4,
               "floors" : 5,
               "item_factory": 6,
               "item_shop": 7,
               "pool": 8,
               "street": 9,
               "upygamerade_smith": 10,
               "utilities": 11,
               "vedal's_house": 12,
               "walls": 13,
               "world_center": 14,
               "bounding_box" : 15
               }

#####################################################################################

#pyrr functions that i copied cuz import pyrr causes long load times in browsers

def create_perspective_projection_from_bounds(left,right,bottom,top,near,far,dtype=None):
    A = (right + left) / (right - left)
    B = (top + bottom) / (top - bottom)
    C = -(far + near) / (far - near)
    D = -2. * far * near / (far - near)
    E = 2. * near / (right - left)
    F = 2. * near / (top - bottom)

    return np.array(((E,  0., 0., 0.),
                     (0., F,  0., 0.),
                     (A,  B,  C, -1.),
                     (0., 0., D,  0.))
    )
def normalize(vec):
    
    return (vec.T  / np.sqrt(np.sum(vec**2,axis=-1))).T
def create_from_eulers(eulers, dtype=None):
    dtype = dtype or eulers.dtype

    roll, pitch, yaw = eulers

    sP = np.sin(pitch)
    cP = np.cos(pitch)
    sR = np.sin(roll)
    cR = np.cos(roll)
    sY = np.sin(yaw)
    cY = np.cos(yaw)

    return np.array(
        [
            # m1
            [
                cY * cP,
                -cY * sP * cR + sY * sR,
                cY * sP * sR + sY * cR,
            ],
            # m2
            [
                sP,
                cP * cR,
                -cP * sR,
            ],
            # m3
            [
                -sY * cP,
                sY * sP * cR + cY * sR,
                -sY * sP * sR + cY * cR,
            ]
        ],
        dtype=dtype
    )
def create_from_quaternion(quat, dtype=None):
    dtype = dtype

    qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]

    sqw = qw**2
    sqx = qx**2
    sqy = qy**2
    sqz = qz**2
    qxy = qx * qy
    qzw = qz * qw
    qxz = qx * qz
    qyw = qy * qw
    qyz = qy * qz
    qxw = qx * qw

    invs = 1 / (sqx + sqy + sqz + sqw)
    m00 = ( sqx - sqy - sqz + sqw) * invs
    m11 = (-sqx + sqy - sqz + sqw) * invs
    m22 = (-sqx - sqy + sqz + sqw) * invs
    m10 = 2.0 * (qxy + qzw) * invs
    m01 = 2.0 * (qxy - qzw) * invs
    m20 = 2.0 * (qxz - qyw) * invs
    m02 = 2.0 * (qxz + qyw) * invs
    m21 = 2.0 * (qyz + qxw) * invs
    m12 = 2.0 * (qyz - qxw) * invs

    return np.array([
        [m00, m01, m02, 0],
        [m10, m11, m12, 0],
        [m20, m21, m22, 0],
        [0,   0,   0,   1]
    ], dtype=dtype)
def create_from_translation(vec, dtype=None):
    
    dtype = dtype
    mat = np.identity(4, dtype=dtype)
    mat[3, 0:3] = vec[:3]
    return mat
def create_from_scale(scale, dtype=None):
    m = np.diagflat([scale[0], scale[1], scale[2], 1.0])
    if dtype:
        m = m.astype(dtype)
    return m
def ray_intersect_aabb(ray, aabb):
    
    direction = ray[1]
    dir_fraction = np.empty(3, dtype = ray.dtype)
    dir_fraction[direction == 0.0] = np.inf
    dir_fraction[direction != 0.0] = np.divide(1.0, direction[direction != 0.0])

    t1 = (aabb[0,0] - ray[0,0]) * dir_fraction[ 0 ]
    t2 = (aabb[1,0] - ray[0,0]) * dir_fraction[ 0 ]
    t3 = (aabb[0,1] - ray[0,1]) * dir_fraction[ 1 ]
    t4 = (aabb[1,1] - ray[0,1]) * dir_fraction[ 1 ]
    t5 = (aabb[0,2] - ray[0,2]) * dir_fraction[ 2 ]
    t6 = (aabb[1,2] - ray[0,2]) * dir_fraction[ 2 ]


    tmin = max(min(t1, t2), min(t3, t4), min(t5, t6))
    tmax = min(max(t1, t2), max(t3, t4), max(t5, t6))

    # if tmax < 0, ray (line) is intersecting AABB
    # but the whole AABB is behind the ray start
    if tmax < 0:
        return None

    # if tmin > tmax, ray doesn't intersect AABB
    if tmin > tmax:
        return None

    # t is the distance from the ray point
    # to intersection

    t = min(x for x in [tmin, tmax] if x >= 0)
    point = ray[0] + (ray[1] * t)
    return point

projection = create_perspective_projection_from_bounds(-0.1, 0.1, -0.1*HEIGHT/WIDTH, 0.1*HEIGHT/WIDTH, 0.1, 2000)

def shader3D(vertexBuffer, normBuffer, texBuffer, texture):
    
    return ctx.pipeline(
        vertex_shader="""
            #version 300 es
            precision highp float;
            
            layout(location = 0) in vec3 vpos;
            layout(location = 1) in vec2 vtex;
            
            uniform mat4 projection;
            uniform mat4 view;
            uniform mat4 model;
            
            out vec2 TexCoords;
            
            void main()
            {
                gl_Position = projection * view * model * vec4(vpos, 1.0);
                TexCoords = vtex;
            }
        """,
        fragment_shader="""
            #version 300 es
            precision highp float;
            
            in vec2 TexCoords;
            uniform sampler2D material;
            
            layout (location = 0) out vec4 color;
            
            void main()
            {
                color = texture(material, TexCoords);
                color = pow(color, vec4(0.45));
            }
        """,
        
        uniforms={'projection': projection.flatten(), 'view': np.identity(4).flatten(), 'model': np.identity(4).flatten()},
        
        blend={'enable': True, 'src_color': 'src_alpha', 'dst_color': 'one_minus_src_alpha'},
        layout=[{'name': 'material', 'binding': 1}],
        resources=[{'type': 'sampler', 'binding': 1, 'image': texture, 'wrap_x': 'clamp_to_edge', 'wrap_y': 'clamp_to_edge', 'min_filter': 'nearest', 'mag_filter': 'nearest'}],
        
        vertex_buffers= [*zengl.bind(ctx.buffer(vertexBuffer), "3f", 0), *zengl.bind(ctx.buffer(texBuffer), "2f", 1)],
        vertex_count= len(vertexBuffer),
        cull_face= "back",
        topology= "triangles",
        framebuffer= [image, depth]
    )
def shaderBoundingBox():
    
    return ctx.pipeline(
        vertex_shader="""
            #version 300 es
            precision highp float;
            
            uniform vec3 boundingBox[2];
            uniform mat4 projection;
            uniform mat4 view;
            
            void main() {
                gl_Position = projection * view * vec4(boundingBox[gl_VertexID/4].x, boundingBox[(gl_VertexID/2) % 2].y, boundingBox[gl_VertexID % 2].z, 1.0);
            }
        """,
        fragment_shader="""
            #version 300 es
            precision highp float;
            
            layout (location = 0) out vec4 out_color;

            void main()
            {
                out_color = vec4(1,0,0,0.75);
            }
        """,
        
        uniforms={'projection': projection.flatten(), 'view': np.identity(4).flatten(), 'boundingBox': [[0, 660, -4], [1, 661, -5]]},
        
        vertex_count=24,
        index_buffer= ctx.buffer(np.array([0,1,2,1,2,3, 2,3,6,3,6,8, 6,7,4,7,4,5, 4,5,0,5,0,1]), index= True),
        #cull_face= "back",
        topology= "triangles",
        framebuffer= [image, depth]
    )

#####################################################################################

class entity:
    
    def __init__(self, position, size, eulers= [0,0,0]):
        
        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)
        self.size = size

class pointLight(entity):

    def __init__(self, position, eulers, color, strength):

        super().__init__(position, eulers, 0)
        self.color = color
        self.strength = strength

class player(entity): 
    
    def __init__(self, position, eulers, camEulers, camZoom):
        
        super().__init__(position, 0, eulers)
        self.camera = camera(self.position, camEulers, camZoom)
    
    def update(self):
        
        cosX, sinX = self.camera.update(self.position)

        self.forwards = np.array((cosX, 0, -sinX))
        self.right = np.array((-sinX, 0, -cosX))
    
    def angle(self, frameTime, dPos):
        
        angle = np.arctan2(dPos[1], -dPos[0])
        
        angle += self.camera.eulers[0] + self.eulers[2]
        angle %= 2*np.pi
        if angle > np.pi:
            self.eulers[2] += (2*np.pi-angle)* frameTime * 0.01
        else:
            self.eulers[2] -= angle* frameTime * 0.01
        self.eulers[2] %= 2*np.pi
    
    def move(self, movement):

        self.position += movement

class camera(entity):
    
    def __init__(self, position, eulers, camZoom):
        
        super().__init__(position, 0, eulers)
        self.zoom = camZoom
        self.update(position)
        
    def update(self, pos):
        
        angleX = self.eulers[0]
        angleY = self.eulers[1]
        
        cosX = np.cos(angleX)
        sinX = np.sin(angleX)
        cosY = np.cos(angleY)
        sinY = np.sin(angleY)
        
        self.forwards = np.array((cosX*cosY, sinY, -sinX*cosY))
        self.playerForwards = (cosX, 0, -sinX)
        self.right = (sinX, 0, cosX)
        self.up = np.array((-cosX*sinY, cosY, sinX*sinY))
        
        self.center = pos
        self.position = self.center - self.zoom * self.forwards
        
        self.makeFrustum()
        
        return (cosX, sinX)

    def getViewTransform(self):
        
        return np.array(((self.right[0], self.up[0], -self.forwards[0], 0),
                         (self.right[1], self.up[1], -self.forwards[1], 0),
                         (self.right[2], self.up[2], -self.forwards[2], 0),
                         (-np.dot(self.right, self.position), -np.dot(self.up, self.position), np.dot(self.forwards,self.position), 1.0)), dtype=np.float32)
    
    def getYawMat(self):
        
        return np.array(((self.right[0], 0, -self.playerForwards[0], 0),
                         (self.right[1], 1, -self.playerForwards[1], 0),
                         (self.right[2], 0, -self.playerForwards[2], 0),
                         (-np.dot(self.right, self.position), -np.dot((0,1,0), self.position), np.dot(self.playerForwards,self.position), 1.0)), dtype=np.float32)
    
    def spin(self, dEulers):

        self.eulers += dEulers

        self.eulers[0] %= 2*np.pi
        self.eulers[1] = min(1.5, max(-1.5, self.eulers[1]))

    def makeFrustum(self):

        self.frustumParts = [normalize(self.forwards + self.right), normalize(self.forwards - self.right), normalize(self.forwards + self.up * 16/9), normalize(self.forwards - self.up * 16/9)]
        self.frustum = [np.append(normal, np.sum(normal * self.position)) for normal in self.frustumParts]

class scene:
    
    def __init__(self, sceneNr, playerPos, playerEul, camEul, camZoom):
        
        self.player = player(playerPos, playerEul, camEul, camZoom)
        self.jumpTime = 0
        self.height = 0
        self.heightmap = material("gfx/map8.png")
        
        if sceneNr == 0:
            
            self.lights = [pointLight([0, 1000, 0], [0, 0, 0], [255,255,255], 500)]
            
            self.entities = {
                ENTITY_TYPE["player"]:            [self.player,                    gltfMesh("models/vedal987/vedal987.gltf",                    [material("models/vedal987/vedal987.png")])],
                ENTITY_TYPE["Camilla's_tent"]:    [entity([-8,663.55,40],20),      gltfMesh("models/V-nexus/Camilla's_tent/Camillas_tent.gltf", [material("models/V-nexus/Camilla's_tent/Camillas_tent.png"),
                                                                                                                                                 material("models/V-nexus/Camilla's_tent/Camillas_tent.png")])],
                ENTITY_TYPE["drone_factory"]:     [entity([43,660.35,7.775],25),   gltfMesh("models/V-nexus/drone_factory/drone_factory.gltf",  [material("models/V-nexus/drone_factory/drone_factory.png"),
                                                                                                                                                 material("models/V-nexus/drone_factory/drone_factory.png"),
                                                                                                                                                 material("models/V-nexus/drone_factory/drone_factory.png"),
                                                                                                                                                 material("models/V-nexus/drone_factory/drone_factory.png")])],
                ENTITY_TYPE["floors"]:            [entity([0,660.5,0],60),         gltfMesh("models/V-nexus/floors/floors.gltf",                [material("models/V-nexus/floors/grass_field2.png"),
                                                                                                                                                 material("models/V-nexus/floors/grass_field2.png"),
                                                                                                                                                 material("models/V-nexus/floors/grass_field2.png"),
                                                                                                                                                 material("models/V-nexus/floors/grass_field2.png"),
                                                                                                                                                 material("models/V-nexus/floors/item_factory.png"),
                                                                                                                                                 material("models/V-nexus/floors/upgrade_smith.png"),
                                                                                                                                                 material("models/V-nexus/floors/vedals_house.png"),
                                                                                                                                                 material("models/V-nexus/floors/water_pump.png"),
                                                                                                                                                 material("models/V-nexus/floors/power_generator.png")])],
                ENTITY_TYPE["item_factory"]:      [entity([46,664.31,-33.75],20),  gltfMesh("models/V-nexus/item_factory/item_factory.gltf",    [material("models/V-nexus/item_factory/item_factory.png")])],
                ENTITY_TYPE["item_shop"]:         [entity([42,658.1,46],10),       gltfMesh("models/V-nexus/item_shop/item_shop.gltf",          [material("models/V-nexus/item_shop/item_shop.png")])],
                ENTITY_TYPE["street"]:            [entity([6,658,0],70),           gltfMesh("models/V-nexus/street/street.gltf",                [material("models/V-nexus/street/street.png")])],
                ENTITY_TYPE["upygamerade_smith"]: [entity([41.9,661.2,-10.05],14), gltfMesh("models/V-nexus/upgrade_smith/upgrade_smith.gltf",  [material("models/V-nexus/upgrade_smith/upgrade_smith.png"),
                                                                                                                                                 material("models/V-nexus/upgrade_smith/piston.png"),
                                                                                                                                                 material("models/V-nexus/upgrade_smith/piston_base.png"),
                                                                                                                                                 material("models/V-nexus/upgrade_smith/gear.png"),
                                                                                                                                                 material("models/V-nexus/upgrade_smith/piston.png"),
                                                                                                                                                 material("models/V-nexus/upgrade_smith/piston_base.png"),
                                                                                                                                                 material("models/V-nexus/upgrade_smith/piston.png"),
                                                                                                                                                 material("models/V-nexus/upgrade_smith/piston_base.png"),
                                                                                                                                                 material("models/V-nexus/upgrade_smith/piston.png"),
                                                                                                                                                 material("models/V-nexus/upgrade_smith/piston_base.png"),
                                                                                                                                                 material("models/V-nexus/upgrade_smith/gear.png"),
                                                                                                                                                 material("models/V-nexus/upgrade_smith/gear.png")])],
                ENTITY_TYPE["utilities"]:         [entity([7.75,658.6,-37],20),    gltfMesh("models/V-nexus/utilities/utilities.gltf",          [material("models/V-nexus/utilities/water_pump.png"),
                                                                                                                                                 material("models/V-nexus/utilities/water_pump.png"),
                                                                                                                                                 material("models/V-nexus/utilities/water_pump.png"),
                                                                                                                                                 material("models/V-nexus/utilities/water_pump.png"),
                                                                                                                                                 material("models/V-nexus/utilities/power_generator.png"),
                                                                                                                                                 material("models/V-nexus/utilities/power_generator.png")])],
                ENTITY_TYPE["walls"]:             [entity([0,660.5,0],70),         gltfMesh("models/V-nexus/walls/walls.gltf",                  [material("models/V-nexus/walls/walls.png"),
                                                                                                                                                 material("models/V-nexus/walls/walls.png"),
                                                                                                                                                 material("models/V-nexus/walls/walls.png"),
                                                                                                                                                 material("models/V-nexus/walls/walls.png"),
                                                                                                                                                 material("models/V-nexus/walls/walls.png"),
                                                                                                                                                 material("models/V-nexus/walls/walls.png"),
                                                                                                                                                 material("models/V-nexus/walls/walls.png"),
                                                                                                                                                 material("models/V-nexus/walls/walls.png"),])],
                ENTITY_TYPE["world_center"]:      [entity([2,709.1,4],70),         gltfMesh("models/V-nexus/world_center/world_center.gltf",    [material("models/V-nexus/world_center/world_center_building.png"),
                                                                                                                                                 material("models/V-nexus/world_center/beacon.png")])],
                ENTITY_TYPE["vedal's_house"]:     [entity([-36.15,663.5,26],40),   gltfMesh("models/V-nexus/vedal's_house/vedals_house.gltf",   [material("models/V-nexus/vedal's_house/vedals_house.png"),
                                                                                                                                                 material("models/V-nexus/vedal's_house/vedals_house.png"),
                                                                                                                                                 material("models/V-nexus/vedal's_house/vedals_house.png"),
                                                                                                                                                 material("models/V-nexus/vedal's_house/vedals_house.png"),
                                                                                                                                                 material("models/V-nexus/vedal's_house/vedals_house.png"),
                                                                                                                                                 material("models/V-nexus/vedal's_house/vedals_house.png"),
                                                                                                                                                 material("models/V-nexus/vedal's_house/vedals_house.png"),
                                                                                                                                                 material("models/V-nexus/vedal's_house/vedals_house.png")])],
                ENTITY_TYPE["bounding_box"]:      [entity([0,0,0],999),            boundingBoxMesh(                                              )]
                }
        
        self.entityGrid = [[[] for j in range(500)] for i in range(500)]
        for entity_type, obj in self.entities.items():
            
            #skip non-collision objects
            if entity_type in [ENTITY_TYPE["player"], ENTITY_TYPE["bounding_box"]]: continue
            
            meshBoundingBoxes = obj[1].boundingBox + obj[0].position
            for meshBoundingBox in meshBoundingBoxes:
                
                meshmin, meshmax = [list(map(int, vec3[::2])) for vec3 in meshBoundingBox/4 + 250]
                [self.entityGrid[x][y].append(obj) for x in range(meshmin[0], meshmax[0]+1) for y in range(meshmin[1], meshmax[1]+1) if obj not in self.entityGrid[x][y]]
    
    def jump(self, jump):
        
        if not self.jumpTime:
            self.jumpTime = jump
            self.jumpStartHeight = self.height
        
        t = jump - self.jumpTime
        jumpheight = 5*t - 4.9*(t**2)
        
        if jumpheight < (self.height - self.jumpStartHeight):
            self.player.position[1] = self.height
            self.jumpTime = 0
            return False
        else:
            self.player.position[1] = self.jumpStartHeight + jumpheight
            return True
    
    def movePlayer(self, dPos, sprint, frametime):
        
        movement = normalize((dPos[0]*self.player.right + dPos[1]*self.player.forwards)) * (1 + sprint)
        movement, collisionHeight = self.checkCollision(movement, self.player.position)
        
        self.player.angle(frametime, dPos)
        self.player.move(movement * 0.01 * frametime)
        
        pos = self.player.position[0:3:2] * 5/2 + 2500
        pos = [int(i) for i in pos]
        
        mapHeight = [self.heightmap.pixels.getpixel((pos[1] + x, pos[0] + y))[1]/32 for x, y in [(0, -1), (-1, 0), (0, 0), (1, 0), (0, 1)]]
        angle = [np.arctan(mapHeight[x] - mapHeight[y]) for x, y in [(0, 2), (2, 4), (2, 1), (3, 2)]]
        
        roll = (angle[0] + angle[1]) * 0.5
        pitch = (angle[2] + angle[3]) * 0.5
        
        self.player.eulers[0] = pitch
        self.player.eulers[1] = roll
        
        self.height = max(mapHeight[2] * 31.25/64, collisionHeight) - 0.1
        
        if not self.jumpTime:
            self.player.position[1] = self.height
    
    def checkCollision(self, movement, pos):
        
        meshBoundingBoxList, collisionPosList, movementList, distanceList, heightList = [], [], [], [], []
        collisionHeight = 0
        
        cell = pos[::2]/4 + 250
        grid = self.entityGrid[int(cell[0])][int(cell[1])]
        for obj in grid:
            
            meshBoundingBoxes = obj[1].boundingBox + obj[0].position
            for meshBoundingBox in meshBoundingBoxes:
                
                localBoundingBox = meshBoundingBox - pos
                
                if all(localBoundingBox[0][2*i] < 0 < localBoundingBox[1][2*i] for i in range(2)) and localBoundingBox[1][1] < 1:
                    heightList.append(localBoundingBox[1][1])
                
                if localBoundingBox[1][1] < 0.4: continue
                
                moveRay = np.array(((0,0,0), movement), dtype=np.float32)
                collisionPos = ray_intersect_aabb(moveRay, localBoundingBox)
                if collisionPos is not None:
                    
                    distance = np.linalg.norm(collisionPos)
                    if distance < 1:
                        
                        normal = self.getNormal(collisionPos, localBoundingBox)
                        leftoverMovement = movement - collisionPos
                        leftoverMovement -= (normal @ leftoverMovement) * normal
                        
                        distanceList.append(distance)
                        collisionPosList.append(collisionPos)
                        movementList.append(leftoverMovement)
                        meshBoundingBoxList.append(meshBoundingBox)
        
        if distanceList:
            index = distanceList.index(min(distanceList))
            movement = collisionPosList[index]
            movement += self.checkCollision2(movementList[index], meshBoundingBoxList-pos+movement)
            self.entities[ENTITY_TYPE["bounding_box"]][1].updateBoundingBox(meshBoundingBoxList[index])
            
        if heightList:
            collisionHeight = pos[1] + max(heightList)
        
        return movement, collisionHeight
    
    def checkCollision2(self, movement, meshBoundingBoxList):
        
        collisionPosList, distanceList = [], []
        
        for meshBoundingBox in meshBoundingBoxList:
            
            moveRay = np.array(((0,0,0), movement), dtype=np.float32)
            collisionPos = ray_intersect_aabb(moveRay, meshBoundingBox)
            if collisionPos is not None:
                
                distanceList.append(np.linalg.norm(collisionPos))
                collisionPosList.append(collisionPos)
        
        if distanceList:
            index = distanceList.index(min(distanceList))
            movement = collisionPosList[index]
        
        return movement
    
    def getNormal(self, collisionPos, boundingBox):
        
        distances = [abs(collisionPos[0] - boundingBox[0][0]),
                     abs(collisionPos[0] - boundingBox[1][0]),
                     abs(collisionPos[2] - boundingBox[0][2]),
                     abs(collisionPos[2] - boundingBox[1][2])]
        
        index = distances.index(min(distances))
        
        if index == 0: return np.array((1,0,0))
        if index == 1: return np.array((-1,0,0))
        if index == 2: return np.array((0,0,1))
        if index == 3: return np.array((0,0,-1))

    def render(self):
        
        ctx.new_frame()
        image.clear()
        depth.clear()
        
        view = self.player.camera.getViewTransform()
        frustum = self.player.camera.frustum
        
        for entity in self.entities.values():
            obj, mesh = entity
            
            if all([vec4[0:3] @ obj.position - vec4[3] > -entity[0].size for vec4 in frustum]):
                
                transformMat = np.identity(4)
                transformMat[0:3,0:3] = create_from_eulers(entity[0].eulers)
                transformMat[3,0:3] = obj.position
                
                #if mesh.hasJoints:
                    #mesh.pose += 1
                    #mesh.setUniform()
                
                mesh.draw(view, transformMat)
        
        image.blit(output)
        output.blit()
        ctx.end_frame()
        
        pygame.display.flip()

class game:
    
    __slots__ = ("window", "renderer", "scene", "sceneNr", "last_time", "window_time", "frametime", "keys", "scroll", "jump")

    def __init__(self):
        
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        
        self.jump = 0
        saveName = "savefile.txt"
        try:
            data = np.loadtxt(saveName, converters=float, dtype=np.float32)
            self.sceneNr = data[0]
            playerPos = data[1:4]
            playerEul = data[4:7]
            camEul = data[7:10]
            camZoom = data[10]
        except:
            print("no savefile found or error while reading data")
            self.sceneNr = 0
            playerPos = [0,0,0]
            playerEul = [0,0,0]
            camEul = [0,0,0]
            camZoom = 3
        
        self.scene = scene(self.sceneNr, playerPos, playerEul, camEul, camZoom)
        self.set_up_timer()
        self.gameLoop()

    def gameLoop(self):
        
        result = CONTINUE
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                result = EXIT
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    result = OPEN_MENU
            elif event.type == pygame.MOUSEWHEEL:
                self.scene.player.camera.zoom -= event.y
        
        self.calculate_framerate()
        self.handle_keys()
        self.handle_mouse()
        
        self.scene.player.update()
        self.scene.render()
        
        return result

    def handle_keys(self):

        dPos = 0
        keys = pygame.key.get_pressed()

        #this method makes it so holding multible keys doesnt prioritize the first one in the row
        if keys[input_map["forwards"]]:
            dPos += np.array([0,1])
        if keys[input_map["left"]]:
            dPos += np.array([1,0])
        if keys[input_map["backwards"]]:
            dPos -= np.array([0,1])
        if keys[input_map["right"]]:
            dPos -= np.array([1,0])
        if keys[input_map["jump"]]:
            self.jump = True
        
        #this could have been an if statement but this is cooler
        sprint = keys[input_map["sprint"]]
        
        #the jump code is an ungodly mess, dont touch it if not needed
        if self.jump:
            self.jump = self.scene.jump(self.last_time)

        if np.any(dPos):
            self.scene.movePlayer(dPos, sprint, self.frametime)

    def handle_mouse(self):
        
        (x,y) = pygame.mouse.get_rel()
        dEulers = 0.001 * -x * np.array([1,0,0])
        dEulers -= 0.001 * y * np.array([0,1,0])
        self.scene.player.camera.spin(dEulers)
    
    def set_up_timer(self):

        self.last_time = pygame.time.get_ticks()/1000
        self.window_time = 0
        self.frametime = 0
    
    def calculate_framerate(self):

        clock.tick()
        framerate = clock.get_fps()
        if framerate != 0:
            self.frametime = 1000/framerate
        
        self.last_time = pygame.time.get_ticks()/1000
        if self.last_time - self.window_time > 1:
            pygame.display.set_caption(f"Running at {int(framerate)} fps.")
            self.window_time = self.last_time
    
    def quit(self):
        
        saveName = "savefile.txt"
        np.savetxt(saveName, [self.sceneNr, *self.scene.player.position, *self.scene.player.eulers, *self.scene.player.camera.eulers, self.scene.player.camera.zoom], fmt='%f')

class menu:
    
    def __init__(self):
        
        pygame.mouse.set_visible(True)
        pygame.event.set_grab(False)
        
        self.set_up_timer()
        self.createObjects()
        self.gameLoop()
    
    def set_up_timer(self):

        self.last_time = pygame.time.get_ticks()/1000
        self.frametime = 0
    
    def createObjects(self):
        
        baseTexture = material("gfx/button.png", 0)
        hoverTexture = material("gfx/hoverButton.png", 0)
        
        self.buttons = []
        
        newGameButton = button((0, 0.3), (0.6, 0.4), baseTexture, hoverTexture)
        newGameButton.click = newGameClick
        self.buttons.append(newGameButton)
        
        quitButton = button((0, -0.3), (0.6, 0.4), baseTexture, hoverTexture)
        quitButton.click = quitClick
        self.buttons.append(quitButton)
    
    def gameLoop(self):
        
        result = CONTINUE
        click = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                result = EXIT
            if event.type == pygame.MOUSEBUTTONDOWN:
                click = True
        
        result = self.handleMouse(click)
        
        pygame.display.flip()
        
        for button in self.buttons:
            button.draw()
        
        self.calculate_framerate()
        
        return result
    
    def handleMouse(self, click):
        (x,y) = pygame.mouse.get_pos()
        x -= WIDTH * 0.5
        x /= WIDTH * 0.5
        y -= HEIGHT * 0.5
        y /= -HEIGHT * 0.5
        
        for button in self.buttons:
            result = button.handleMouse((x,y), click)
            if result != CONTINUE:
                return result
        return CONTINUE
    
    def calculate_framerate(self):

        clock.tick()
        framerate = clock.get_fps()
        if framerate != 0:
            self.frametime = 1000/framerate
        
        time = pygame.time.get_ticks()/1000
        if time - self.last_time > 1:
            pygame.display.set_caption(f"Running at {int(framerate)} fps.")
            self.last_time = time
    
    def quit(self):
        for button in self.buttons:
            button.destroy()

#####################################################################################

def newGameClick():
    return NEW_GAME

def quitClick():
    return EXIT

class button:
    
    def __init__(self, pos, size, baseTexture, hoverTexture, shader):
        
        self.click = None
        self.pos = pos
        self.size = size
        self.baseTexture, self.hoverTexture = baseTexture, hoverTexture
        self.shader = shader
        self.halfWidth, self.halfHeight = size[0]/2, size[1]/2
        
        self.vertices = (
            pos[0] - self.halfWidth, pos[1] + self.halfHeight, 0, 1,
            pos[0] - self.halfWidth, pos[1] - self.halfHeight, 0, 0,
            pos[0] + self.halfWidth, pos[1] - self.halfHeight, 1, 0,
            
            pos[0] - self.halfWidth, pos[1] + self.halfHeight, 0, 1,
            pos[0] + self.halfWidth, pos[1] - self.halfHeight, 1, 0,
            pos[0] + self.halfWidth, pos[1] + self.halfHeight, 1, 1
        )
        self.vertices = np.array(self.vertices, dtype=np.float32)
        
        #self.vao = gl.glGenVertexArrays(1)
        #gl.glBindVertexArray(self.vao)
        #self.vbo = gl.glGenBuffers(1)
        #gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        #gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, gl.GL_STATIC_DRAW)
        
        #gl.glEnableVertexAttribArray(0)
        #gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 16, gl.ctypes.c_void_p(0))
        
        #gl.glEnableVertexAttribArray(1)
        #gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 16, gl.ctypes.c_void_p(8))
    
    def inside(self, pos):
        for i in (0, 1):
            if pos[i] < (self.pos[i] - self.size[i]*0.5) or pos[i] > (self.pos[i] + self.size[i]*0.5):
                return False
        return True
    
    def handleMouse(self, pos, click):
        
        if self.inside(pos):
            self.texture = self.hoverTexture
            if click:
                return self.click()
        else:
            self.texture = self.baseTexture
        return CONTINUE

class material:
    
    def __init__(self, filepath):
        
        self.pixels = Image.open(filepath)
        self.img = ctx.image(self.pixels.size, 'rgba8unorm', self.pixels.tobytes())

class gltfMesh:

    def __init__(self, filename, textures):
        
        #create the np files with this
        #import precomputeGLTF
        #precomputeGLTF.loadGLTF(filename)
        
        hasNormals, hasTextures, hasJoints, listLenght = np.loadtxt(f"{filename}Data", dtype=np.int32)
        self.boundingBox = np.loadtxt(f"{filename}BoundingBox", dtype=np.float32)
        self.boundingBox = [[self.boundingBox[2*i], self.boundingBox[2*i + 1]] for i in range(listLenght)]
        
        vertexDataList = [np.loadtxt(f"{filename}VertexDataList{i}", dtype=np.float32) for i in range(listLenght)]
        if hasNormals:
            normalDataList = [np.loadtxt(f"{filename}NormalDataList{i}", dtype=np.float32) for i in range(listLenght)]
        if hasTextures:
            texCoordDataList = [np.loadtxt(f"{filename}TexCoordDataList{i}", dtype=np.float32) for i in range(listLenght)]
        if hasJoints:
            jointDataList = [np.loadtxt(f"{filename}JointDataList{i}", dtype=np.int32) for i in range(listLenght)]
            weightDataList = [np.loadtxt(f"{filename}WeightDataList{i}", dtype=np.float32) for i in range(listLenght)]
        indexDataList = [np.loadtxt(f"{filename}IndexDataList{i}", dtype=np.int32) for i in range(listLenght)]
        
        #index buffer fix
        vertexDataList = [np.array([vertexDataList[i][3*j:3*j+3] for j in indexDataList[i]], dtype=np.float32) for i in range(listLenght)]
        if hasNormals:
            normalDataList = [np.array([normalDataList[i][3*j:3*j+3] for j in indexDataList[i]], dtype=np.float32) for i in range(listLenght)]
        if hasTextures:
            texCoordDataList = [np.array([texCoordDataList[i][2*j:2*j+2] for j in indexDataList[i]], dtype=np.float32) for i in range(listLenght)]
        if hasJoints:
            jointDataList = [np.array([jointDataList[i][4*j:4*j+4] for j in indexDataList[i]], dtype=np.int32) for i in range(listLenght)]
            weightDataList = [np.array([weightDataList[i][4*j:4*j+4] for j in indexDataList[i]], dtype=np.float32) for i in range(listLenght)]
            
        #if self.hasJoints:
            #this shader doesnt work (yet)
            #self.shaders = [shader3D_animated(ctx.buffer(np.array(vertexDataList[i], dtype=np.float32)), vertexDataList[i].nbytes, ctx.buffer(np.array(normalDataList[i], dtype=np.float32)), ctx.buffer(np.array(texCoordDataList[i], dtype=np.float32)), ctx.buffer(np.array(indexDataList[i], dtype=np.float32)), textures[i].img, filename + str(i)) for i in range(self.listLenght)]
        
        #else:
        self.shaders = [shader3D(vertexDataList[i], normalDataList[i], texCoordDataList[i], textures[i].img) for i in range(listLenght)]
    
    def setUniform(self):
        
        for shader in self.shaders:
            animation = np.array(self.finalMatrices[0][round(self.pose//8)%self.timeData])
            shader.uniforms['animation'][:] = [struct.pack('4f4f4f4f', *anim.flatten()) for anim in animation]
    
    def draw(self, view, model):
        
        for shader in self.shaders:
            shader.uniforms['view'][:] = struct.pack('4f4f4f4f', *view.flatten())
            shader.uniforms['model'][:] = struct.pack('4f4f4f4f', *model.flatten())
            shader.render()

class boundingBoxMesh:
    
    def __init__(self):
        
        self.hasJoints = 0
        vertexData = np.array([0,0,1, 1,0,1, 0,1,1, 1,1,1, 1,0,1, 0,0,1, 1,0,0, 0,0,0, 1,1,1, 1,0,1, 1,1,0, 1,0,0, 0,1,1, 1,1,1, 0,1,0, 1,1,0, 0,0,1, 0,1,1, 0,0,0, 0,1,0, 0,0,0, 0,1,0, 1,0,0, 1,1,0], dtype= np.float32)
        indexData = np.array([0,1,2, 3,2,1, 4,5,6, 7,6,5, 8,9,10, 11,10,9, 12,13,14, 15,14,13, 16,17,18, 19,18,17, 20,21,22, 23,22,21], dtype= np.int32)
        normalData = np.array([0,0,1, 0,0,1, 0,0,1, 0,0,1, 0,-1,0, 0,-1,0, 0,-1,0, 0,-1,0, 1,0,0, 1,0,0, 1,0,0, 1,0,0, 0,1,0, 0,1,0, 0,1,0, 0,1,0, -1,0,0, -1,0,0, -1,0,0, -1,0,0, 0,0,-1, 0,0,-1, 0,0,-1, 0,0,-1], dtype= np.float32)
        
        self.shader = shaderBoundingBox()
    
    def updateBoundingBox(self, boundingBox):
        
        #self.shader.uniforms['boundingBox'][:] = struct.pack('3f3f', *boundingBox.flatten())
        
        xlist, ylist, zlist = [(boundingBox[0][i], boundingBox[1][i]) for i in range(3)]
        vertices = [(x, y, z) for x in xlist for y in ylist for z in zlist]
        vertexData = np.array([vertices[i] for i in (4,5,6,7,5,4,1,0,7,5,3,1,6,7,2,3,4,6,0,2,0,2,1,3)], dtype= np.float32)
        print(len(vertexData))
        
        self.shader.uniforms['boundinBox'][:] = struct.pack('3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f3f', *vertexData.flatten())
        
    def draw(self, view, model):
        
        self.shader.uniforms['view'][:] = struct.pack('4f4f4f4f', *view.flatten())
        self.shader.render()

#####################################################################################

async def main():
    myApp = game()
    result = CONTINUE
    while result == CONTINUE:
        result = myApp.gameLoop()
        if result == NEW_GAME:
            myApp.quit()
            myApp = game()
            result = CONTINUE
        elif result == OPEN_MENU:
            myApp.quit()
            myApp = menu()
            result = CONTINUE
        await asyncio.sleep(0)
    myApp.quit()

asyncio.run(main())