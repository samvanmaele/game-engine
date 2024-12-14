# python -m pygbag --PYBUILD 3.12 --ume_block 0 --template noctx.tmpl .

import numpy as np
import asyncio
import pygame as pygame
import struct
import zengl
import pyrr
import sys

HEIGHT, WIDTH = 720, 1280

pygame.init()
pygame.display.set_mode((WIDTH, HEIGHT), flags=pygame.OPENGL)

ctx = zengl.context()
size = pygame.display.get_window_size()
image = ctx.image(size, 'rgba8unorm', texture=False)
depth = ctx.image(size, 'depth24plus', texture=False)
output = ctx.image(size, 'rgba8unorm')

clock = pygame.time.Clock()

#####################################################################################

input_map = {'right': pygame.K_d,
             'left': pygame.K_a,
             'forwards': pygame.K_w,
             'backwards': pygame.K_s,
             'jump': pygame.K_SPACE,
             'sprint': pygame.K_LSHIFT}

#####################################################################################

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

STRUCT_TYPE = {5120: 'b',
               5121: 'B',
               5122: 'h',
               5123:'H',
               5125:'I',
               5126:'f'}

VALUE_SIZE = {'b': 1,
              'h': 2,
              'i': 4,
              'f': 4}

#####################################################################################

#projection = pyrr.matrix44.create_perspective_projection_from_bounds(-0.1, 0.1, -0.1*HEIGHT/WIDTH, 0.1*HEIGHT/WIDTH, 0.1, 2000)

class shader3D:

    def __init__(self, vertexBuffer, vertexCount, normBuffer, texBuffer, indexBuffer, texture):
    
        view = pyrr.matrix44.create_identity()
        model = pyrr.matrix44.create_identity()
        
        self.pipeline = ctx.pipeline(
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
                    TexCoords = vtex;
                    gl_Position = projection * view * model * vec4(vpos, 1.0);
                }
            """,
            fragment_shader="""
                #version 300 es
                precision highp float;

                in vec2 TexCoords;
                uniform sampler2D material;

                layout (location = 0) out vec4 out_color;

                void main()
                {
                    out_color = texture(material, TexCoords);
                    out_color = pow(out_color, vec4(0.45));
                }
            """,
            
            uniforms= {'projection': projection.flatten(), 'view': view.flatten(), 'model': model.flatten()},
            
            blend= {'enable': True, 'src_color': 'src_alpha', 'dst_color': 'one_minus_src_alpha'},
            
            layout= [{'name': 'material', 'binding': 0}],
            
            resources= [{'type': 'sampler', 'binding': 0, 'image': texture, 'wrap_x': 'clamp_to_edge', 'wrap_y': 'clamp_to_edge', 'min_filter': 'nearest', 'mag_filter': 'nearest'}],
            
            framebuffer= [image, depth],
            vertex_buffers= [*zengl.bind(ctx.buffer(vertexBuffer), "3f", 0),
                            *zengl.bind(ctx.buffer(texBuffer), "2f", 1)],
            index_buffer= ctx.buffer(indexBuffer, index= True),
            vertex_count= vertexCount,
            cull_face= "back",
            topology= "triangles"
        )

#####################################################################################

class entity:

    def __init__(self, position, size, eulers= [0,0,0]):
        
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
        
        super().__init__(position, 0, eulers)
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

        self.frustumParts = [pyrr.vector.normalize(self.forwards + self.right), pyrr.vector.normalize(self.forwards - self.right), pyrr.vector.normalize(self.forwards + self.up * 16/9), pyrr.vector.normalize(self.forwards - self.up * 16/9)]
        self.frustum = [np.append(normal, np.sum(normal * self.position)) for normal in self.frustumParts]

class scene:
    
    def __init__(self, sceneNr, playerPos, playerEul, camEul, camZoom):
        
        self.player = player(playerPos, playerEul, camEul, camZoom)
        self.jumpTime = 0
        self.height = 0
        
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
                ENTITY_TYPE["bounding_box"]:      [entity([0,0,0],999),            boundingBoxMesh(                                              material("gfx/redA.png"))]
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
    
    def movePlayer(self, dPos, frametime):
        
        movement = pyrr.vector.normalize((dPos[0]*self.player.right + dPos[1]*self.player.forwards))
        movement, collisionHeight = self.checkCollision(movement, self.player.position)
        
        self.player.angle(frametime, dPos)
        self.player.move(movement * 0.01 * frametime)
        """
        pos = self.player.position[0:3:2] * 5/2 + 2500
        pos = [int(i) for i in pos]
        
        heightmap = material("gfx/map8.png")
        mapHeight = [heightmap.img[pos[1] + x, pos[0] + y][1]/32 for x, y in [(0, -1), (-1, 0), (0, 0), (1, 0), (0, 1)]]
        angle = [np.arctan(mapHeight[x] - mapHeight[y]) for x, y in [(0, 2), (2, 4), (2, 1), (3, 2)]]
        
        roll = (angle[0] + angle[1]) * 0.5
        pitch = (angle[2] + angle[3]) * 0.5
        
        self.player.eulers[0] = pitch
        self.player.eulers[1] = roll
        
        self.height = max(mapHeight[2] * 31.25/64, collisionHeight) - 0.1
        """
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

    def update(self, fps):
        
        self.player.update(fps)
        self.render()

    def render(self):
        ctx.new_frame()
        image.clear()
        depth.clear()
        
        camera = self.entities[ENTITY_TYPE["player"]][0].camera
        view = camera.getViewTransform()
        frustum = camera.frustum
        
        for entity in self.entities.values():
            obj, mesh = entity
            
            if all([vec4[0:3] @ obj.position - vec4[3] > -entity[0].size for vec4 in frustum]):
                
                transformMat = np.identity(4)
                transformMat[0:3,0:3] = pyrr.matrix33.create_from_eulers(entity[0].eulers)
                transformMat[3,0:3] = obj.position
                
                #if mesh.hasJoints:
                    #mesh.pose += 1
                    #mesh.setUniform()
                
                mesh.draw(view, transformMat)
        
        pipeline.render()
        
        image.blit(output)
        output.blit()
        ctx.end_frame()
        
        pygame.display.flip()

class game:
    
    __slots__ = ("window", "renderer", "scene", "sceneNr", "last_time", "window_time", "frametime", "keys", "scroll", "jump")

    def __init__(self):
        
        #pygame.mouse.set_visible(False)
        #pygame.event.set_grab(True)
        
        self.jump = 0
        self.scroll = 0
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
        self.set_up_timer()
        self.gameLoop()

    def gameLoop(self):
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                    
            elif event.type == pygame.MOUSEWHEEL:
                self.scroll = event.y
        
        self.calculate_framerate()
        self.handle_keys()
        self.handle_mouse()
        
        self.scene.update(self.frametime)

    def handle_keys(self):

        dPos = 0
        keys = pygame.key.get_pressed()

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
        
        if self.jump:
            self.jump = self.scene.jump(self.last_time)

        if np.any(dPos):
            self.scene.movePlayer(dPos, self.frametime)

    def handle_mouse(self):

        (x,y) = pygame.mouse.get_rel()
        dEulers = 0.001 * -x * np.array([1,0,0])
        dEulers += 0.001 * -y * np.array([0,1,0])
        self.scene.player.camera.spin(dEulers)
        
        self.scene.player.camera.zoom -= self.scroll/5
        self.scroll = 0
    
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
        
        #saveName = f"savefile_{time.asctime().replace(" ", "_").replace(":", ".")}.txt"
        saveName = "savefile.txt"
        
        with open(saveName, "w") as savefile:
            
            savefile.write(f"{self.sceneNr}\n")
            savefile.write(f"{self.scene.player.position}\n{self.scene.player.eulers}\n")
            savefile.write(f"{self.scene.player.camera.eulers}\n{self.scene.player.camera.zoom}\n")

#####################################################################################

class material:
    
    def __init__(self, filepath):
        
        img = pygame.image.load(filepath)
        img = pygame.transform.flip(img, False, True)
        pixels = pygame.image.tobytes(img, 'RGBA', True)
        self.img = ctx.image(img.get_size(), 'rgba8unorm', pixels)

class gltfMesh:

    def __init__(self, filename, textures):
        
        hasNormals, hasTextures, hasJoints, listLenght = np.loadtxt(filename + "Data", converters=float, dtype=np.int32)
        self.boundingBox = np.loadtxt(filename + "BoundingBox", converters=float)
        self.boundingBox = [[self.boundingBox[2*i], self.boundingBox[2*i + 1]] for i in range(listLenght)]
        
        vertexDataList = [np.loadtxt(filename + "VertexDataList" + str(i), converters=float, dtype=np.float32) for i in range(listLenght)]
        if hasNormals:
            normalDataList = [np.loadtxt(filename + "VormalDataList" + str(i), converters=float, dtype=np.float32) for i in range(listLenght)]
        if hasTextures:
            texCoordDataList = [np.loadtxt(filename + "TexCoordDataList" + str(i), converters=float, dtype=np.float32) for i in range(listLenght)]
        if hasJoints:
            jointDataList = [np.loadtxt(filename + "JointDataList" + str(i), converters=float, dtype=np.float32) for i in range(listLenght)]
            weightDataList = [np.loadtxt(filename + "WeightDataList" + str(i), converters=float, dtype=np.float32) for i in range(listLenght)]
        indexDataList = [np.loadtxt(filename + "TndexDataList" + str(i), converters=float, dtype=np.int32) for i in range(listLenght)]
        
        #if self.hasJoints:
            #this shader doesnt work (yet)
            #self.shaders = [shader3D_animated(ctx.buffer(np.array(vertexDataList[i], dtype=np.float32)), vertexDataList[i].nbytes, ctx.buffer(np.array(normalDataList[i], dtype=np.float32)), ctx.buffer(np.array(texCoordDataList[i], dtype=np.float32)), ctx.buffer(np.array(indexDataList[i], dtype=np.float32)), textures[i].img, filename + str(i)) for i in range(self.listLenght)]
        
        #else:
        self.shaders = [shader3D(vertexDataList[0], vertexDataList[0].nbytes, normalDataList[0], texCoordDataList[0], indexDataList[0], textures[0].img) for i in range(listLenght)]
        
    def draw(self, view, transformMat):
        
        for shader in self.shaders:
            #shader.pipeline.uniforms['view'][:] = struct.pack('4f4f4f4f', *view.flatten())
            #shader.pipeline.uniforms['model'][:] = struct.pack('4f4f4f4f', *transformMat.flatten())
            #shader.pipeline.render()
            pipeline.render()

class boundingBoxMesh:
    
    def __init__(self, texture):
        
        self.hasJoints = 0
        vertexData = np.array([0,0,1, 1,0,1, 0,1,1, 1,1,1, 1,0,1, 0,0,1, 1,0,0, 0,0,0, 1,1,1, 1,0,1, 1,1,0, 1,0,0, 0,1,1, 1,1,1, 0,1,0, 1,1,0, 0,0,1, 0,1,1, 0,0,0, 0,1,0, 0,0,0, 0,1,0, 1,0,0, 1,1,0], dtype= np.float32)
        indexData = np.array([0,1,2, 3,2,1, 4,5,6, 7,6,5, 8,9,10, 11,10,9, 12,13,14, 15,14,13, 16,17,18, 19,18,17, 20,21,22, 23,22,21], dtype= np.int32)
        normalData = np.array([0,0,1, 0,0,1, 0,0,1, 0,0,1, 0,-1,0, 0,-1,0, 0,-1,0, 0,-1,0, 1,0,0, 1,0,0, 1,0,0, 1,0,0, 0,1,0, 0,1,0, 0,1,0, 0,1,0, -1,0,0, -1,0,0, -1,0,0, -1,0,0, 0,0,-1, 0,0,-1, 0,0,-1, 0,0,-1], dtype= np.float32)
        texCoordData = np.array([0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0], dtype= np.float32)
        
        self.shader = shader3D(vertexData, vertexData.nbytes, normalData, texCoordData, indexData, texture.img)
    
    def updateBoundingBox(self, boundingBox):
        
        xlist, ylist, zlist = [(boundingBox[0][i], boundingBox[1][i]) for i in range(3)]
        vertices = [(x, y, z) for x in xlist for y in ylist for z in zlist]
        vertexData = np.array([vertices[i] for i in (4,5,6,7,5,4,1,0,7,5,3,1,6,7,2,3,4,6,0,2,0,2,1,3)], dtype= np.float32)
        
    def draw(self, view, transformMat):
        
        #self.shader.pipeline.uniforms['view'][:] = struct.pack('4f4f4f4f', *view.flatten())
        #self.shader.pipeline.uniforms['model'][:] = struct.pack('4f4f4f4f', *transformMat.flatten())
        #self.shader.pipeline.render()
        pipeline.render()

#####################################################################################

pipeline = ctx.pipeline(
    vertex_shader='''
        #version 300 es
        precision highp float;

        vec2 vertices[3] = vec2[](
            vec2(0.0, 0.8),
            vec2(-0.866, -0.7),
            vec2(0.866, -0.7)
        );

        vec3 colors[3] = vec3[](
            vec3(1.0, 0.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 0.0, 1.0)
        );

        out vec3 v_color;

        void main() {
            gl_Position = vec4(vertices[gl_VertexID], 0.0, 1.0);
            v_color = colors[gl_VertexID];
        }
    ''',
    fragment_shader='''
        #version 300 es
        precision highp float;

        in vec3 v_color;

        layout (location = 0) out vec4 out_color;

        void main() {
            out_color = vec4(v_color, 1.0);
            out_color.rgb = pow(out_color.rgb, vec3(1.0 / 2.2));
        }
    ''',
    framebuffer=[image],
    topology='triangles',
    vertex_count=3,
)

async def main():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        ctx.new_frame()
        image.clear()
        pipeline.render()
        image.blit()
        ctx.end_frame()
        
        pygame.display.flip()
        await asyncio.sleep(0)

asyncio.run(main())