import glfw
import glfw.GLFW as GLFW_CONSTANTS
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import ctypes
import pyrr
import cv2
from pygltflib import GLTF2
import struct
import cProfile
import sys

np.set_printoptions(threshold=sys.maxsize)

#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

input_map = {'right': GLFW_CONSTANTS.GLFW_KEY_D,
             'left': GLFW_CONSTANTS.GLFW_KEY_A,
             'forwards': GLFW_CONSTANTS.GLFW_KEY_W,
             'backwards': GLFW_CONSTANTS.GLFW_KEY_S,
             'jump': GLFW_CONSTANTS.GLFW_KEY_SPACE,
             'sprint': GLFW_CONSTANTS.GLFW_KEY_LEFT_SHIFT}

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
               "upgrade_smith": 10,
               "utilities": 11,
               "vedal's_house": 12,
               "walls": 13,
               "world_center": 14,
               "bounding_box" : 15
               }
UNIFORM_TYPE = {"LIGHT_COLOR": 1,
                "LIGHT_POS": 2,
                "LIGHT_STRENGTH": 3}
ELEMENT_SIZES = {'SCALAR': 1,
                 'VEC2': 2,
                 'VEC3': 3,
                 'VEC4': 4,
                 'MAT2': 4,
                 'MAT3': 9, 
                 'MAT4': 16}
STRUCT_TYPE = {GL_BYTE: 'b',
               GL_UNSIGNED_BYTE: 'B',
               GL_SHORT: 'h',
               GL_UNSIGNED_SHORT:'H',
               GL_UNSIGNED_INT:'I',
               GL_FLOAT:'f'}
VALUE_SIZE = {'b': 1,
              'h': 2,
              'i': 4,
              'f': 4}

#####################################################################################

HEIGHT, WIDTH = 720, 1280
halfHeight, halfWidth = HEIGHT*0.5, WIDTH*0.5

#####################################################################################

def setUpGlfw():
    
    global window

    glfw.init()
    glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MINOR, 1)
    glfw.window_hint(GLFW_CONSTANTS.GLFW_OPENGL_PROFILE, GLFW_CONSTANTS.GLFW_OPENGL_CORE_PROFILE)
    glfw.window_hint(GLFW_CONSTANTS.GLFW_OPENGL_FORWARD_COMPAT, GLFW_CONSTANTS.GLFW_TRUE)
    glfw.window_hint(GLFW_CONSTANTS.GLFW_SAMPLES, 4)
    glfw.window_hint(GLFW_CONSTANTS.GLFW_DOUBLEBUFFER, GL_TRUE)
    window = glfw.create_window(WIDTH, HEIGHT, "Title", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(0)

def createShader(filepaths):

    with open(filepaths[0], 'r') as f:
        vertexSrc = f.readlines()
    
    with open(filepaths[1], 'r') as f:
        fragmentSrc = f.readlines()
    
    if len(filepaths) == 2:
        
        shader = compileProgram(compileShader(vertexSrc, GL_VERTEX_SHADER),
                                compileShader(fragmentSrc, GL_FRAGMENT_SHADER))
    else:
        with open(filepaths[2], 'r') as f:
            tesControlSrc = f.readlines()
        
        with open(filepaths[3], 'r') as f:
            tesEvalSrc = f.readlines()
    
        shader = compileProgram(compileShader(vertexSrc, GL_VERTEX_SHADER),
                                compileShader(fragmentSrc, GL_FRAGMENT_SHADER),
                                compileShader(tesControlSrc, GL_TESS_CONTROL_SHADER),
                                compileShader(tesEvalSrc, GL_TESS_EVALUATION_SHADER))
    
    return shader

#####################################################################################

class entity:

    def __init__(self, position, eulers, size):
        
        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)
        self.size = size

class billBoard:
    
    def __init__(self, w, h):
        pass
        
    def destroy():
        pass

class pointLight(entity):

    def __init__(self, position, eulers, color, strength):

        super().__init__(position, eulers, 0)
        self.color = color
        self.strength = strength

class player(entity): 
    
    def __init__(self, position, eulers, camEulers, camZoom):
        
        super().__init__(position, eulers, 0)
        self.camera = camera(self.position, camEulers, camZoom)
    
    def update(self, fps):
        
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
        
        super().__init__(position, eulers, 0)
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

        self.frustumParts = [pyrr.vector.normalize(self.forwards + self.right), pyrr.vector.normalize(self.forwards - self.right), pyrr.vector.normalize(self.forwards + self.up * 16/9), pyrr.vector.normalize(self.forwards - self.up * 16/9)]
        self.frustum = [np.append(normal, np.sum(normal * self.position)) for normal in self.frustumParts]

class scene:
    
    def __init__(self, sceneNr, playerPos, playerEul, camEul, camZoom):
        
        self.set_up_opengl()
        
        self.shaders = {
            "2D" : createShader(("shaders/shader_2D/vertex_2D.glsl", "shaders/shader_2D/fragment_2D.glsl")),
            "3D" : createShader(("shaders/shader_3D/vertex_3D.glsl","shaders/shader_3D/fragment_3D.glsl")),
            "3D_animated" : createShader(("shaders/shader_3D_animated/vertex_3D_animated.glsl","shaders/shader_3D_animated/fragment_3D_animated.glsl")),
            "skybox" : createShader(("shaders/shader_skybox/vertex_skybox.glsl","shaders/shader_skybox/fragment_skybox.glsl")),
            "terrain" : createShader(("shaders/shader_terrain/vertex_terrain.glsl","shaders/shader_terrain/fragment_terrain.glsl","shaders/shader_terrain/tesControl_terrain.glsl","shaders/shader_terrain/tesEval_terrain.glsl")),
            "grass" : createShader(("shaders/shader_grass/vertex_grass.glsl","shaders/shader_grass/fragment_grass.glsl")),
        }
        
        glUseProgram(self.shaders["skybox"])
        self.skybox = (gltfMesh("models/box 3D model/Box.gltf"), [[cubeMap(["gfx/skybox/skybox_right.png","gfx/skybox/skybox_left.png","gfx/skybox/skybox_top.png","gfx/skybox/skybox_bottom.png","gfx/skybox/skybox_front.png","gfx/skybox/skybox_back.png"], 0)]])
        
        glUseProgram(self.shaders["terrain"])
        self.terrain = (gltfMesh("models/circle2K.gltf"), [[material("gfx/map8.png", 0), material("gfx/sand-v1.png", 1), material("gfx/grass.png", 2)]])
        for i in range(3):
            glUniform1i(13+i, i)
        
        glUseProgram(self.shaders["grass"])
        self.fern = (gltfMesh("models/grass.gltf"), [[material("gfx/map8.png", 0), material("gfx/grass2D.png", 1)]])
        for i in range(2):
            glUniform1i(12+i, i)
        
        #glUseProgram(self.shaders["2D"])
        #self.userInterface = (gltfMesh("models/button.gltf"), [[material("gfx/button.png")]])
        
        self.player = player(playerPos, playerEul, camEul, camZoom)
        self.jumpTime = 0
        self.height = 0

        if sceneNr == 0:
            
            self.lights = [pointLight([0, 1000, 0], [0, 0, 0], [255,255,255], 500)]
            
            self.entities = {
                ENTITY_TYPE["player"]:               [self.player                                       , [gltfMesh("models/vedal987.gltf",                             self.shaders), [[material("models/vedal987.png", 0)]]]],
                ENTITY_TYPE["Camilla's_tent"]:       [entity([-8,       663.55,  40   ], [0, 0, 0], 20 ), [gltfMesh("models/V-nexus/Camilla's_tent/Camillas_tent.gltf", self.shaders), [[material("models/V-nexus/Camilla's_tent/Camillas_tent.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/Camilla's_tent/Camillas_tent.png", 0)]]]],
                ENTITY_TYPE["drone_factory"]:        [entity([43,       660.35,  7.775], [0, 0, 0], 25 ), [gltfMesh("models/V-nexus/drone_factory/drone_factory.gltf",  self.shaders), [[material("models/V-nexus/drone_factory/drone_factory.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/drone_factory/drone_factory.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/drone_factory/drone_factory.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/drone_factory/drone_factory.png", 0)]]]],
                ENTITY_TYPE["floors"]:               [entity([0,        660.5,   0    ], [0, 0, 0], 60 ), [gltfMesh("models/V-nexus/floors/floors.gltf",                self.shaders), [[material("models/V-nexus/floors/grass_field2.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/floors/grass_field2.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/floors/grass_field2.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/floors/grass_field2.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/floors/item_factory.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/floors/upgrade_smith.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/floors/vedals_house.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/floors/water_pump.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/floors/power_generator.png", 0)]]]],
                ENTITY_TYPE["item_factory"]:         [entity([46,       664.31, -33.75], [0, 0, 0], 20 ), [gltfMesh("models/V-nexus/item_factory/item_factory.gltf",    self.shaders), [[material("models/V-nexus/item_factory/item_factory.png", 0)]]]],
                ENTITY_TYPE["item_shop"]:            [entity([42,       658.1,   46   ], [0, 0, 0], 10 ), [gltfMesh("models/V-nexus/item_shop/item_shop.gltf",          self.shaders), [[material("models/V-nexus/item_shop/item_shop.png", 0)]]]],
                ENTITY_TYPE["street"]:               [entity([6,        658,     0    ], [0, 0, 0], 70 ), [gltfMesh("models/V-nexus/street/street.gltf",                self.shaders), [[material("models/V-nexus/street/street.png", 0)]]]],
                ENTITY_TYPE["upgrade_smith"]:        [entity([41.9,     661.2,  -10.05], [0, 0, 0], 14 ), [gltfMesh("models/V-nexus/upgrade_smith/upgrade_smith.gltf",  self.shaders), [[material("models/V-nexus/upgrade_smith/upgrade_smith.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/upgrade_smith/piston.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/upgrade_smith/piston_base.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/upgrade_smith/gear.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/upgrade_smith/piston.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/upgrade_smith/piston_base.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/upgrade_smith/piston.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/upgrade_smith/piston_base.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/upgrade_smith/piston.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/upgrade_smith/piston_base.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/upgrade_smith/gear.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/upgrade_smith/gear.png", 0)]]]],
                ENTITY_TYPE["utilities"]:            [entity([7.75,     658.6,  -37   ], [0, 0, 0], 20 ), [gltfMesh("models/V-nexus/utilities/utilities.gltf",          self.shaders), [[material("models/V-nexus/utilities/water_pump.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/utilities/water_pump.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/utilities/water_pump.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/utilities/water_pump.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/utilities/power_generator.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/utilities/power_generator.png", 0)]]]],
                ENTITY_TYPE["walls"]:                [entity([0,        660.5,   0    ], [0, 0, 0], 70 ), [gltfMesh("models/V-nexus/walls/walls.gltf",                  self.shaders), [[material("models/V-nexus/walls/walls.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/walls/walls.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/walls/walls.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/walls/walls.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/walls/walls.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/walls/walls.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/walls/walls.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/walls/walls.png", 0)],]]],
                ENTITY_TYPE["world_center"]:         [entity([2,        709.1,   4    ], [0, 0, 0], 70 ), [gltfMesh("models/V-nexus/world_center/world_center.gltf",    self.shaders), [[material("models/V-nexus/world_center/world_center_building.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/world_center/beacon.png", 0)]]]],
                
                ENTITY_TYPE["vedal's_house"]:        [entity([-36.15,   663.5,   26   ], [0, 0, 0], 40 ), [gltfMesh("models/V-nexus/vedal's_house/vedals_house.gltf",  self.shaders),  [[material("models/V-nexus/vedal's_house/vedals_house.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/vedal's_house/vedals_house.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/vedal's_house/vedals_house.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/vedal's_house/vedals_house.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/vedal's_house/vedals_house.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/vedal's_house/vedals_house.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/vedal's_house/vedals_house.png", 0)],
                                                                                                                                                                        [material("models/V-nexus/vedal's_house/vedals_house.png", 0)]]]],
                ENTITY_TYPE["bounding_box"]:         [entity([0,        0,       0    ], [0, 0, 0], 999), [boundingBoxMesh(),                                                          [[material("gfx/redA.png", 0)]]]]
                }
        
        self.entityGrid = [[[] for j in range(500)] for i in range(500)]
        for entity_type, obj in self.entities.items():
            
            #skip non-collision objects
            if entity_type in [ENTITY_TYPE["player"], ENTITY_TYPE["bounding_box"]]: continue
            
            meshBoundingBoxes = obj[1][0].boundingBox + obj[0].position
            for meshBoundingBox in meshBoundingBoxes:
                
                meshmin, meshmax = [list(map(int, vec3[::2])) for vec3 in meshBoundingBox/4 + 250]
                [self.entityGrid[x][y].append([entity_type, obj]) for x in range(meshmin[0], meshmax[0]+1) for y in range(meshmin[1], meshmax[1]+1) if [entity_type, obj] not in self.entityGrid[x][y]]
        
        lightsNr = len(self.lights)
        self.set_onetime_uniforms()
        self.get_uniform_locations(lightsNr)
    
    def set_up_opengl(self):

        glClearColor(0.0, 0.0, 0.0, 1)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthFunc(GL_LEQUAL)
        
        glPatchParameteri(GL_PATCH_VERTICES, 3)
        OpenGL.ERROR_CHECKING = False
    
    def set_onetime_uniforms(self) :
        
        projection_transform = pyrr.matrix44.create_perspective_projection_from_bounds(-0.1, 0.1, -0.1*HEIGHT/WIDTH, 0.1*HEIGHT/WIDTH, 0.1, 2000)

        glUseProgram(self.shaders["3D"])
        glUniformMatrix4fv(3, 1, GL_FALSE, projection_transform)
        
        glUseProgram(self.shaders["3D_animated"])
        glUniformMatrix4fv(5, 1, GL_FALSE, projection_transform)
        
        glUseProgram(self.shaders["skybox"])
        glUniformMatrix4fv(1, 1, GL_FALSE, projection_transform)

        glUseProgram(self.shaders["terrain"])
        glUniformMatrix4fv(1, 1, GL_FALSE, projection_transform)
        
        glUseProgram(self.shaders["grass"])
        glUniformMatrix4fv(3, 1, GL_FALSE, projection_transform)
    
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
    
    def movePlayer(self, dPos, frametime):
        
        movement = pyrr.vector.normalize((dPos[0]*self.player.right + dPos[1]*self.player.forwards))
        movement, collisionHeight = self.checkCollision(movement, self.player.position)
        
        self.player.angle(frametime, dPos)
        self.player.move(movement * 0.01 * frametime)
        
        pos = self.player.position[0:3:2] * 5/2 + 2500
        pos = [int(i) for i in pos]
        
        heightmap = self.terrain[1][0][0]
        mapHeight = [heightmap.img[pos[1] + x, pos[0] + y][1]/32 for x, y in [(0, -1), (-1, 0), (0, 0), (1, 0), (0, 1)]]
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
            
            meshBoundingBoxes = obj[1][1][0].boundingBox + obj[1][0].position
            for meshBoundingBox in meshBoundingBoxes:
                
                localBoundingBox = meshBoundingBox - pos
                
                if all(localBoundingBox[0][2*i] < 0 < localBoundingBox[1][2*i] for i in range(2)) and localBoundingBox[1][1] < 1:
                    heightList.append(localBoundingBox[1][1])
                
                if localBoundingBox[1][1] < 0.4: continue
                
                moveRay = np.array(((0,0,0), movement), dtype=np.float32)
                collisionPos = pyrr.geometric_tests.ray_intersect_aabb(moveRay, localBoundingBox)
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
            self.entities[ENTITY_TYPE["bounding_box"]][1][0].updateBoundingBox(meshBoundingBoxList[index])
            
        if heightList:
            collisionHeight = pos[1] + max(heightList)
        
        return movement, collisionHeight
    
    def checkCollision2(self, movement, meshBoundingBoxList):
        
        collisionPosList, distanceList = [], []
        
        for meshBoundingBox in meshBoundingBoxList:
            
            moveRay = np.array(((0,0,0), movement), dtype=np.float32)
            collisionPos = pyrr.geometric_tests.ray_intersect_aabb(moveRay, meshBoundingBox)
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

    def update(self, fps, time):
        
        self.player.update(fps)
        self.render(self.entities, self.lights, time)
    
    def get_uniform_locations(self, lightsNr):

        self.light_locations = [
        {
            UNIFORM_TYPE["LIGHT_COLOR"]: [glGetUniformLocation(self.shaders["3D"], f"Lights[{i}].color") for i in range(lightsNr)],
            UNIFORM_TYPE["LIGHT_POS"]: [glGetUniformLocation(self.shaders["3D"], f"Lights[{i}].position") for i in range(lightsNr)],
            UNIFORM_TYPE["LIGHT_STRENGTH"]: [glGetUniformLocation(self.shaders["3D"], f"Lights[{i}].strength") for i in range(lightsNr)]
        },
        {
            UNIFORM_TYPE["LIGHT_COLOR"]: [glGetUniformLocation(self.shaders["3D_animated"], f"Lights[{i}].color") for i in range(lightsNr)],
            UNIFORM_TYPE["LIGHT_POS"]: [glGetUniformLocation(self.shaders["3D_animated"], f"Lights[{i}].position") for i in range(lightsNr)],
            UNIFORM_TYPE["LIGHT_STRENGTH"]: [glGetUniformLocation(self.shaders["3D_animated"], f"Lights[{i}].strength") for i in range(lightsNr)]
        }
        ]

    def render(self, entities, lights, times):

        glfw.swap_buffers(window)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_CULL_FACE)
        
        camera = entities[ENTITY_TYPE["player"]][0].camera
        viewTransform = camera.getViewTransform()
        frustum = camera.frustum
        
        #draw GUI
        #glUseProgram(self.shaders["2D"])
        #GUI.draw()
        
        #draw normal entities
        glUseProgram(self.shaders["3D"])
        glUniformMatrix4fv(7, 1, GL_FALSE, viewTransform)
        self.setPointlights(lights, 0)
        glUniform3fv(15, 1, camera.position)
        
        glUseProgram(self.shaders["3D_animated"])
        glUniformMatrix4fv(9, 1, GL_FALSE, viewTransform)
        self.setPointlights(lights, 1)
        glUniform3fv(17, 1, camera.position)

        #draw 3D objects
        for entity in entities.values():

            mesh, materials = entity[1]
            
            pos = entity[0].position
            rad = entity[0].size
            
            if all([vec4[0:3] @ pos - vec4[3] > -rad for vec4 in frustum]):
                
                transformMat = np.identity(4)
                transformMat[0:3,0:3] = pyrr.matrix33.create_from_eulers(entity[0].eulers)
                transformMat[3,0:3] = pos
                
                if mesh.hasJoints:
                    glUseProgram(self.shaders["3D_animated"])
                    mesh.pose += 1
                    mesh.setUniform()
                    glUniformMatrix4fv(13, 1, GL_FALSE, transformMat)
                    
                else:
                    glUseProgram(self.shaders["3D"])
                    glUniformMatrix4fv(11, 1, GL_FALSE, transformMat)
                
                mesh.draw(materials)
        
        #draw terrain
        glUseProgram(self.shaders["terrain"])
        glUniformMatrix4fv(5, 1, GL_FALSE, viewTransform)
        yawMat = camera.getYawMat()
        glUniformMatrix4fv(9, 1, GL_FALSE, yawMat)
        self.terrain[0].drawPatches(self.terrain[1])
        
        #for seeing: skybox, both sides of grass
        glDisable(GL_CULL_FACE)
        
        #draw skybox
        glUseProgram(self.shaders["skybox"])
        glUniformMatrix4fv(5, 1, GL_FALSE, viewTransform)
        self.skybox[0].draw(self.skybox[1])
        
        #needs to be last for transparency
        
        #draw grass
        #glUseProgram(self.shaders["grass"])
        #glUniformMatrix4fv(7, 1, GL_FALSE, viewTransform)
        #glUniform1fv(1, 1, times)
        #glUniform3fv(11, 1, camera.position)
        #self.fern[0].drawInstanced(self.fern[1], 1000000)

        glFlush()
    
    def setPointlights(self, lights, shaderNr):
        
        for i in range(len(lights)):

            light = lights[i]

            glUniform3fv(self.light_locations[shaderNr][UNIFORM_TYPE["LIGHT_POS"]][i], 1, light.position)
            glUniform3fv(self.light_locations[shaderNr][UNIFORM_TYPE["LIGHT_COLOR"]][i],   1, light.color)
            glUniform1f(self.light_locations[shaderNr][UNIFORM_TYPE["LIGHT_STRENGTH"]][i],  light.strength)
    
    def destroy(self):
        
        for entity in self.entities.values():
            mesh, materialsList = entity[1]
            mesh.destroy()
            for materials in materialsList:
                for material in materials:
                    material.destroy()
        
        self.skybox[0].destroy()
        for material in self.skybox[1][0]:
            material.destroy()
        
        self.terrain[0].destroy()
        for material in self.terrain[1][0]:
            material.destroy()
        
        self.fern[0].destroy()
        for material in self.fern[1][0]:
            material.destroy()
        
        for shader in self.shaders.values():
            glDeleteProgram(shader)

class game:
    
    __slots__ = ("window", "renderer", "scene", "sceneNr", "last_time", "window_update", "frametime", "keys", "scroll", "jump")

    def __init__(self):
        
        self.jump = 0
        saveName = "savefile.txt"
        
        try:
            with open(saveName) as savefile:
                data = [[float(i) for i in line.strip().replace("[", "").replace("]", "").split(" ") if i] for line in savefile]
                self.sceneNr = int(data[0][0])
                playerPos = data[1]
                playerEul = data[2]
                camEul = data[3]
                camZoom = data[4][0]
        
        except:
            print("no savefile found or error while reading data")
            self.sceneNr = 0
            playerPos = [0,0,0]
            playerEul = [0,0,0]
            camEul = [0,0,0]
            camZoom = 3
        
        self.scene = scene(self.sceneNr, playerPos, playerEul, camEul, camZoom)
        self.set_up_input_systems()
        self.set_up_timer()
        self.gameLoop()

    def set_up_input_systems(self):

        glfw.set_input_mode(window, GLFW_CONSTANTS.GLFW_CURSOR, GLFW_CONSTANTS.GLFW_CURSOR_HIDDEN)

        self.keys = {}
        self.scroll = 0
        glfw.set_key_callback(window, self.key_callback)
        glfw.set_scroll_callback(window, self.scroll_callback)

    def key_callback(self, window, key, scancode, action, mods):

        state = False
        match action:
            case GLFW_CONSTANTS.GLFW_PRESS:
                state = True
            case GLFW_CONSTANTS.GLFW_RELEASE:
                state = False
            case _:
                return

        self.keys[key] = state

    def scroll_callback(self, window, x_offset, y_offset):
        
        self.scroll = -y_offset

    def gameLoop(self):
        
        result = CONTINUE
        
        self.calculate_framerate()
        glfw.poll_events()
        self.handle_keys()
        self.handle_mouse()
        
        self.scene.update(self.frametime, self.last_time)
        
        if glfw.window_should_close(window):
            result = EXIT
        elif self.keys.get(GLFW_CONSTANTS.GLFW_KEY_ESCAPE, False):
            result = OPEN_MENU
        
        return result

    def handle_keys(self):

        dPos = 0

        if self.keys.get(input_map["forwards"], False):
            dPos += np.array([0,1])
        if self.keys.get(input_map["left"], False):
            dPos += np.array([1,0])
        if self.keys.get(input_map["backwards"], False):
            dPos -= np.array([0,1])
        if self.keys.get(input_map["right"], False):
            dPos -= np.array([1,0])
        if self.keys.get(input_map["jump"], False):
            self.jump = True
        
        if self.jump:
            self.jump = self.scene.jump(self.last_time)

        if np.any(dPos):
            self.scene.movePlayer(dPos, self.frametime)

    def handle_mouse(self):

        (x,y) = glfw.get_cursor_pos(window)
        dEulers = 0.001 * (halfWidth - x) * np.array([1,0,0])
        dEulers += 0.001 * (halfHeight - y) * np.array([0,1,0])
        self.scene.player.camera.spin(dEulers)
        glfw.set_cursor_pos(window, halfWidth, halfHeight)
        
        self.scene.player.camera.zoom += self.scroll/5
        self.scroll = 0
    
    def set_up_timer(self):

        self.last_time = glfw.get_time()
        self.frametime = 0
        self.window_update = 0
    
    def calculate_framerate(self):

        current_time = glfw.get_time()
        delta = current_time - self.last_time
        framerate = int(1/delta)
        self.frametime = 1000 * delta
        self.last_time = current_time
        if (current_time - self.window_update) >= 1:
            glfw.set_window_title(window, f"Running at {framerate} fps.")
            self.window_update = current_time
    
    def quit(self):
        
        #saveName = f"savefile_{time.asctime().replace(" ", "_").replace(":", ".")}.txt"
        saveName = "savefile.txt"
        
        with open(saveName, "w") as savefile:
            
            savefile.write(f"{self.sceneNr}\n")
            savefile.write(f"{self.scene.player.position}\n{self.scene.player.eulers}\n")
            savefile.write(f"{self.scene.player.camera.eulers}\n{self.scene.player.camera.zoom}\n")
            
        self.scene.destroy()

class menu:
    
    def __init__(self):
        
        glClearColor(0.0, 0.0, 0.0, 1)
        glDisable(GL_DEPTH_TEST)
        self.set_up_input_systems()
        self.set_up_timer()
        self.createObjects()
        self.gameLoop()
        
    def set_up_input_systems(self):

        glfw.set_input_mode(window, GLFW_CONSTANTS.GLFW_CURSOR, GLFW_CONSTANTS.GLFW_CURSOR_NORMAL)
        
        self.keys = {}
        glfw.set_key_callback(window, self.key_callback)
    
    def set_up_timer(self):

        self.last_time = glfw.get_time()
        self.current_time = 0
        self.frames_rendered = 0
        self.frametime = 0.0
    
    def key_callback(self, window, key, scancode, action, mods):

        state = False
        match action:
            case GLFW_CONSTANTS.GLFW_PRESS:
                state = True
            case GLFW_CONSTANTS.GLFW_RELEASE:
                state = False
            case _:
                return

        self.keys[key] = state
    
    def createObjects(self):
        
        self.shader = createShader(("shaders/shader_2D/vertex_2D.glsl", "shaders/shader_2D/fragment_2D.glsl"))
        baseTexture = material("gfx/button.png", 0)
        hoverTexture = material("gfx/hoverButton.png", 0)
        
        self.buttons = []
        
        newGameButton = button((0, 0.3), (0.6, 0.4), baseTexture, hoverTexture, self.shader)
        newGameButton.click = newGameClick
        self.buttons.append(newGameButton)
        
        quitButton = button((0, -0.3), (0.6, 0.4), baseTexture, hoverTexture, self.shader)
        quitButton.click = quitClick
        self.buttons.append(quitButton)
    
    def gameLoop(self):
        
        result = CONTINUE
        
        result = self.handleMouse()
        glfw.poll_events()
        
        glfw.swap_buffers(window)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glClear(GL_COLOR_BUFFER_BIT)
        glDisable(GL_DEPTH_TEST)
        
        for button in self.buttons:
            button.draw()
        glFlush()
        self.calculate_framerate()
        
        if glfw.window_should_close(window):
            result = EXIT
        
        return result
    
    def handleMouse(self):
        (x,y) = glfw.get_cursor_pos(window)
        x -= halfWidth
        x /= halfWidth
        y -= halfHeight
        y /= -halfHeight
        
        for button in self.buttons:
            result = button.handleMouse((x,y))
            if result != CONTINUE:
                return result
        return CONTINUE
    
    def calculate_framerate(self):

        self.current_time = glfw.get_time()
        delta = self.current_time - self.last_time
        if (delta >= 1):
            framerate = max(1,int(self.frames_rendered/delta))
            glfw.set_window_title(window, f"Running at {framerate} fps.")
            self.last_time = self.current_time
            self.frames_rendered = -1
            self.frametime = float(1000.0 / max(1,framerate))
        self.frames_rendered += 1
    
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
        
        glUseProgram(self.shader)
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
    
    def inside(self, pos):
        for i in (0, 1):
            if pos[i] < (self.pos[i] - self.size[i]*0.5) or pos[i] > (self.pos[i] + self.size[i]*0.5):
                return False
        return True
    
    def handleMouse(self, pos):
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        memoryHandle = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY)
        ctypes.memmove(ctypes.c_void_p(memoryHandle), ctypes.c_void_p(self.vertices.ctypes.data), self.vertices.nbytes)
        glUnmapBuffer(GL_ARRAY_BUFFER)
        
        if self.inside(pos):
            self.texture = self.hoverTexture
            if glfw.get_mouse_button(window, GLFW_CONSTANTS.GLFW_MOUSE_BUTTON_1):
                return self.click()
        else:
            self.texture = self.baseTexture
        return CONTINUE
    
    def draw(self):
        
        glUseProgram(self.shader)
        glBindVertexArray(self.vao)
        self.texture.use()
        glDrawArrays(GL_TRIANGLES, 0, 6)
    
    def destroy(self):
        glDeleteBuffers(1, (self.vbo,))
        glDeleteVertexArrays(1, (self.vao,))

class material:
    
    def __init__(self, filepath, textureUnit):
        
        self.textureUnit = textureUnit
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        self.img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        image_height, image_width, channels = self.img.shape
        
        if img.dtype == np.uint16:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_SHORT, self.img)
        else:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.img)
        
        glGenerateMipmap(GL_TEXTURE_2D)
    
    def use(self):
        
        glActiveTexture(GL_TEXTURE0 + self.textureUnit)
        glBindTexture(GL_TEXTURE_2D, self.texture)
    
    def destroy(self):
        
        glDeleteTextures(1, (self.texture,))

class cubeMap:
    
    def __init__(self, files, textureUnit):
        
        self.textureUnit = textureUnit
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.texture)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        for i in range(6):
            img = cv2.imread(files[i], cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            image_height, image_width, channels = img.shape
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img)
        glGenerateMipmap(GL_TEXTURE_CUBE_MAP)
    
    def use(self):
        
        glActiveTexture(GL_TEXTURE0 + self.textureUnit)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.texture)
    
    def destroy(self):
        
        glDeleteTextures(1, (self.texture,))

class gltfMesh:

    __slots__ = ("vao", "vertexBuffer", "indexBuffer", "texCoordBuffer", "normalBuffer", "jointBuffer", "weightBuffer", "gltf", "hasNormals", "hasTextures", "hasJoints", "indexCountList", "listLenght", "transformMatrices", "finalMatrices", "pose", "timeData", "boundingBox")

    def __init__(self, filename, shaders=None):
        
        self.boundingBox, vertexDataList, normalDataList, texCoordDataList, jointDataList, weightDataList, indexDataList, self.indexCountList, nodeHierarchy = [], [], [], [], [], [], [], [], []
        self.hasNormals, self.hasTextures, self.hasJoints, self.pose = 0, 0, 0, 0
        
        if shaders is not None:
            glUseProgram(shaders["3D"])
        
        self.gltf = GLTF2().load(filename)
        
        scene = self.gltf.scenes[self.gltf.scene]
        
        animations = self.gltf.animations
        parentList = self.createParentlist()
        
        for node in scene.nodes:
            nodeHierarchy.append(self.createNodeHierarchy(node))
        
        for nodes in nodeHierarchy:
            for nodeNr in nodes:

                node = self.gltf.nodes[nodeNr]
                
                if node.mesh == None: continue
                mesh = self.gltf.meshes[node.mesh]
                
                if mesh.primitives[0].attributes.NORMAL:     self.hasNormals = 1
                if mesh.primitives[0].attributes.TEXCOORD_0: self.hasTextures = 1
                if node.skin is not None:
                    
                    if shaders is not None:
                        glUseProgram(shaders["3D_animated"])
                    self.hasJoints = 1
                    skin = self.gltf.skins[node.skin]
                    
                    self.timeData = self.readAccesor(self.gltf.accessors[animations[0].samplers[animations[0].channels[0].sampler].input])[1]
                    inverseBindData = self.readAccesor(self.gltf.accessors[skin.inverseBindMatrices])[0]
                    inverseBindData = [inverseBindData[i:i + 16].reshape(4,4) for i in range(0, len(inverseBindData), 16)]
                    
                    transformMatrix = [[np.identity(4) for pose in range(self.timeData)] for i in range(3)]
                    self.finalMatrices, self.transformMatrices = [[[[np.identity(4) for node in range(len(self.gltf.nodes))] for pose in range(self.timeData)] for animation in range(len(animations))] for i in range(2)]
                    
                    for animation in range(len(animations)):
                        
                        self.createTransformMatrices(self.timeData, animations, animation, transformMatrix)
                        self.createAnimation(skin, animation, parentList, inverseBindData)
                    
                    glUniformMatrix4fv(18, len(skin.joints), GL_FALSE, np.array(self.finalMatrices[0][0]))
                
                for primitive in mesh.primitives:

                    vertexAccessor = self.gltf.accessors[primitive.attributes.POSITION]
                    indexAccessor = self.gltf.accessors[primitive.indices]
                    if self.hasNormals:
                        normalAccessor = self.gltf.accessors[primitive.attributes.NORMAL]
                    if self.hasTextures:
                        texCoordAccessor = self.gltf.accessors[primitive.attributes.TEXCOORD_0]
                    if self.hasJoints:
                        jointAccesor = self.gltf.accessors[primitive.attributes.JOINTS_0]
                        weightAccesor = self.gltf.accessors[primitive.attributes.WEIGHTS_0]
                    
                    self.createBoundingBox(vertexAccessor.min, vertexAccessor.max)
                    
                    vertexData, vertexCount = self.readAccesor(vertexAccessor)
                    indexData, indexCount = self.readAccesor(indexAccessor)
                    if self.hasNormals:
                        normalData, normalCount = self.readAccesor(normalAccessor)
                    if self.hasTextures:
                        texCoordData, texCoordCount = self.readAccesor(texCoordAccessor)
                    if self.hasJoints:
                        jointData, jointCount = self.readAccesor(jointAccesor)
                        weightData, weightCount = self.readAccesor(weightAccesor)
                    
                    vertexDataList.append(vertexData)
                    self.indexCountList.append(indexCount)
                    indexDataList.append(indexData)
                    if self.hasNormals:
                        normalDataList.append(normalData)
                    if self.hasTextures:
                        texCoordDataList.append(texCoordData)
                    if self.hasJoints:
                        jointDataList.append(jointData)
                        weightDataList.append(weightData)
        
        self.listLenght = len(self.indexCountList)
        self.createBuffers()
        
        for i in range(self.listLenght):

            glBindVertexArray(self.vao[i])
            glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer[i])
            glBufferData(GL_ARRAY_BUFFER, vertexDataList[i].nbytes, vertexDataList[i], GL_STATIC_DRAW)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
            
            if self.hasNormals:
                glBindBuffer(GL_ARRAY_BUFFER, self.normalBuffer[i])
                glBufferData(GL_ARRAY_BUFFER, normalDataList[i].nbytes, normalDataList[i], GL_STATIC_DRAW)
                glEnableVertexAttribArray(1)
                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
                
            if self.hasTextures:
                glBindBuffer(GL_ARRAY_BUFFER, self.texCoordBuffer[i])
                glBufferData(GL_ARRAY_BUFFER, texCoordDataList[i].nbytes, texCoordDataList[i], GL_STATIC_DRAW)
                glEnableVertexAttribArray(2)
                glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
                
            if self.hasJoints:
                glBindBuffer(GL_ARRAY_BUFFER, self.jointBuffer[i])
                glBufferData(GL_ARRAY_BUFFER, jointDataList[i].nbytes, jointDataList[i], GL_STATIC_DRAW)
                glEnableVertexAttribArray(3)
                glVertexAttribIPointer(3, 4, GL_UNSIGNED_BYTE, 4, ctypes.c_void_p(0))
                
                glBindBuffer(GL_ARRAY_BUFFER, self.weightBuffer[i])
                glBufferData(GL_ARRAY_BUFFER, weightDataList[i].nbytes, weightDataList[i], GL_STATIC_DRAW)
                glEnableVertexAttribArray(4)
                glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
                
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.indexBuffer[i])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexDataList[i].nbytes, indexDataList[i], GL_STATIC_DRAW)
            
    def createNodeHierarchy(self, node):
        
        result = [node]
        children = self.gltf.nodes[node].children
        if children:
            for child in children:
                result.extend(self.createNodeHierarchy(child))
            
        return result    
    
    def createParentlist(self):
        
        nodes = self.gltf.nodes
        parentList = [None for i in range(len(nodes))]
        for node in range(len(nodes)):
            if nodes[node].children is not None:
                for i in nodes[node].children:
                    parentList[i] = node
        return parentList

    def createTransformMatrices(self, timeData, animations, animation, transformMatrix):
        
        for i in range(len(animations[animation].channels)):
            target = animations[animation].channels[i].target
            sampler = animations[animation].channels[i].sampler
            samplerOutput = animations[animation].samplers[sampler].output
            samplerData = self.readAccesor(self.gltf.accessors[samplerOutput])[0]
            
            if target.path == "translation":
                for pose in range(timeData):
                    transformMatrix[0][pose] = pyrr.matrix44.create_from_translation((samplerData[3*pose], samplerData[3*pose+1], samplerData[3*pose+2]))
            
            elif target.path == "rotation":
                for pose in range(timeData):
                    transformMatrix[1][pose] = pyrr.matrix44.create_from_quaternion((-samplerData[4*pose], -samplerData[4*pose+1], -samplerData[4*pose+2], samplerData[4*pose+3]))

            elif target.path == "scale":
                for pose in range(timeData):
                    transformMatrix[2][pose] = pyrr.matrix44.create_from_scale((samplerData[3*pose], samplerData[3*pose+1], samplerData[3*pose+2]))
            
            for pose in range(timeData):
                self.transformMatrices[animation][pose][target.node] = transformMatrix[2][pose] @ transformMatrix[1][pose] @ transformMatrix[0][pose]
        
    def createAnimation(self, skin, animation, parentList, inverseBindData):
        
        for joint in skin.joints:
            for pose in range(self.timeData):
                self.transformMatrices[animation][pose][joint] = self.transformMatrices[animation][pose][joint] @ self.transformMatrices[animation][pose][parentList[joint]]
                self.finalMatrices[animation][pose][skin.joints.index(joint)] = inverseBindData[skin.joints.index(joint)] @ self.transformMatrices[animation][pose][joint]
        
    def createBuffers(self):
        
        self.vao = glGenVertexArrays(self.listLenght)
        self.vertexBuffer, self.indexBuffer = glGenBuffers(self.listLenght), glGenBuffers(self.listLenght)
        if self.hasNormals:
            self.normalBuffer = glGenBuffers(self.listLenght)
        if self.hasTextures:
            self.texCoordBuffer = glGenBuffers(self.listLenght)
        if self.hasJoints:
            self.jointBuffer = glGenBuffers(self.listLenght)
            self.weightBuffer = glGenBuffers(self.listLenght)
        
        if self.listLenght == 1:
            
            self.vao, self.vertexBuffer, self.indexBuffer = [self.vao], [self.vertexBuffer], [self.indexBuffer]
            if self.hasNormals:
                self.normalBuffer = [self.normalBuffer]
            if self.hasTextures:
                self.texCoordBuffer = [self.texCoordBuffer]
            if self.hasJoints:
                self.jointBuffer = [self.jointBuffer]
                self.weightBuffer = [self.weightBuffer]

    def createBoundingBox(self, min, max):
        min -= np.array((0.7,0.1,0.7))
        max += np.array((0.7,0.1,0.7))
        self.boundingBox.append(pyrr.aabb.create_from_bounds(min, max))
    
    def readAccesor(self, accessor):
        
        bufferView = self.gltf.bufferViews[accessor.bufferView]
        buffer = self.gltf.buffers[bufferView.buffer]
        data = self.gltf.get_data_from_buffer_uri(buffer.uri)
        
        count = accessor.count * ELEMENT_SIZES[accessor.type]
        struct_type = STRUCT_TYPE[accessor.componentType]
        value_size = VALUE_SIZE[struct_type.lower()]
        data = struct.unpack(f'<{count}{struct_type}', data[bufferView.byteOffset + accessor.byteOffset:bufferView.byteOffset + accessor.byteOffset + count * value_size])
        data = np.array(data, dtype=struct_type)
        
        return data, count
    
    def setUniform(self):
        animation = np.array(self.finalMatrices[0][round(self.pose//8)%self.timeData])
        glUniformMatrix4fv(18, len(self.gltf.skins[0].joints), GL_FALSE, animation)
    
    def draw(self, textures):
        
        for i in range(self.listLenght):
            
            for texture in textures[i]:
                texture.use()
            
            glBindVertexArray(self.vao[i])
            glDrawElements(GL_TRIANGLES, self.indexCountList[i], GL_UNSIGNED_SHORT, ctypes.c_void_p(0))
            
    def drawPatches(self, textures):
        
        for i in range(self.listLenght):
            
            for texture in textures[i]:
                texture.use()
            
            glBindVertexArray(self.vao[i])
            glDrawElements(GL_PATCHES, self.indexCountList[i], GL_UNSIGNED_SHORT, ctypes.c_void_p(0))
            
    def drawInstanced(self, textures, instances):
        
        for i in range(self.listLenght):
            
            for texture in textures[i]:
                texture.use()
            
            glBindVertexArray(self.vao[i])
            glDrawElementsInstanced(GL_TRIANGLES, self.indexCountList[i], GL_UNSIGNED_SHORT, ctypes.c_void_p(0), instances)
    
    def destroy(self):

        glDeleteVertexArrays(self.listLenght, self.vao)
        glDeleteBuffers(self.listLenght, self.indexBuffer)
        glDeleteBuffers(self.listLenght, self.vertexBuffer)
        if self.hasNormals:
            glDeleteBuffers(self.listLenght, self.normalBuffer)
        if self.hasTextures:
            glDeleteBuffers(self.listLenght, self.texCoordBuffer)
        if self.hasJoints:
            glDeleteBuffers(self.listLenght, self.jointBuffer)
            glDeleteBuffers(self.listLenght, self.weightBuffer)            

class boundingBoxMesh:
    
    def __init__(self):
        
        self.hasJoints = 0
        vertexData = np.array([0,0,1, 1,0,1, 0,1,1, 1,1,1, 1,0,1, 0,0,1, 1,0,0, 0,0,0, 1,1,1, 1,0,1, 1,1,0, 1,0,0, 0,1,1, 1,1,1, 0,1,0, 1,1,0, 0,0,1, 0,1,1, 0,0,0, 0,1,0, 0,0,0, 0,1,0, 1,0,0, 1,1,0], dtype= np.float32)
        indexData = np.array([0,1,2, 3,2,1, 4,5,6, 7,6,5, 8,9,10, 11,10,9, 12,13,14, 15,14,13, 16,17,18, 19,18,17, 20,21,22, 23,22,21], dtype= np.uint16)
        normalData = np.array([0,0,1, 0,0,1, 0,0,1, 0,0,1, 0,-1,0, 0,-1,0, 0,-1,0, 0,-1,0, 1,0,0, 1,0,0, 1,0,0, 1,0,0, 0,1,0, 0,1,0, 0,1,0, 0,1,0, -1,0,0, -1,0,0, -1,0,0, -1,0,0, 0,0,-1, 0,0,-1, 0,0,-1, 0,0,-1], dtype= np.float32)
        
        self.vao = glGenVertexArrays(1)
        self.vertexBuffer, self.indexBuffer, self.normalBuffer = glGenBuffers(1), glGenBuffers(1), glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer)
        glBufferData(GL_ARRAY_BUFFER, vertexData.nbytes, vertexData, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        
        glBindBuffer(GL_ARRAY_BUFFER, self.normalBuffer)
        glBufferData(GL_ARRAY_BUFFER, normalData.nbytes, normalData, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
            
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.indexBuffer)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexData.nbytes, indexData, GL_STATIC_DRAW)
    
    def updateBoundingBox(self, boundingBox):
        
        xlist, ylist, zlist = [(boundingBox[0][i], boundingBox[1][i]) for i in range(3)]
        vertices = [(x, y, z) for x in xlist for y in ylist for z in zlist]
        vertexData = np.array([vertices[i] for i in (4,5,6,7,5,4,1,0,7,5,3,1,6,7,2,3,4,6,0,2,0,2,1,3)], dtype= np.float32)
        
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer)
        glBufferData(GL_ARRAY_BUFFER, vertexData.nbytes, vertexData, GL_STATIC_DRAW)
        
    def draw(self, textures):
        
        glDisable(GL_CULL_FACE)
        for texture in textures[0]:
            texture.use()
        
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, ctypes.c_void_p(0))
    
    def destroy(self):
        
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.indexBuffer])
        glDeleteBuffers(1, [self.vertexBuffer])

#####################################################################################

def startProgram():
    setUpGlfw()
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
    myApp.quit()

#cProfile.run('startProgram()')
#startProgram()