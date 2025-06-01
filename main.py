# python -m pygbag --PYBUILD 3.12 --ume_block 0 --no_opt --git --template noctx.tmpl .
# py-spy record -o profile.svg -- python main.py

# /// script
# dependencies = [
#  "numpy",
#  "pygame",
#  "zengl",
#  "marshmallow",
#  "PIL",
#  "PyOpenGL"
# ]
# ///

from PIL import Image
import numpy as np
import asyncio
import pygame
import zengl
import sys
import platform
import time

HEIGHT, WIDTH = 1080, 1920

pygame.init()

#pygame.mixer.init()
#pygame.mixer.music.set_volume(0.2)
#audio1 = pygame.mixer.music.load("sfx/NeuroSama-Goddess.ogg")
#pygame.mixer.music.play(-1)

pygame.display.init()

if sys.platform == "win32":
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, 1)

screen = pygame.display.set_mode((WIDTH, HEIGHT), flags=pygame.OPENGL|pygame.DOUBLEBUF)
clock = pygame.time.Clock()
ctx = zengl.context()

size = pygame.display.get_window_size()
image = ctx.image(size, 'rgba8unorm', samples= 4)
depth = ctx.image(size, 'depth24plus', samples= 4)
lightdepth = ctx.image((4096, 4096), 'rgba32float')
output = ctx.image(size, 'rgba8unorm')

fingers = {}

#####################################################################################

input_map = {'right': pygame.K_d,
             'left': pygame.K_a,
             'forwards': pygame.K_w,
             'backwards': pygame.K_s,
             'jump': pygame.K_SPACE,
             'sprint': pygame.K_LSHIFT,
             'escape': pygame.K_ESCAPE}

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
               "world_center": 14
               }

#####################################################################################

#pyrr functions that i copied cuz import pyrr causes long load times in browsers

def create_perspective_projection_from_bounds(left, right, bottom, top, near, far, dtype= np.float32):
    C = -(far + near) / (far - near)
    D = -2. * far * near / (far - near)
    E = 2. * near / (right - left)
    F = 2. * near / (top - bottom)

    return np.array(((E,  0., 0., 0.),
                     (0., F,  0., 0.),
                     (0., 0., C, -1.),
                     (0., 0., D,  0.)),
                     dtype= dtype)
def create_orthogonal_projection(left, right, bottom, top, near, far, dtype= np.float32):

    rml = right - left
    tmb = top - bottom
    fmn = far - near

    A = 2. / rml
    B = 2. / tmb
    C = -2. / fmn
    Tx = -(right + left) / rml
    Ty = -(top + bottom) / tmb
    Tz = -(far + near) / fmn

    return np.array((( A, 0., 0., 0.),
                     (0.,  B, 0., 0.),
                     (0., 0.,  C, 0.),
                     (Tx, Ty, Tz, 1.),),
                     dtype=dtype)
def normalize(vec):
    
    return (vec.T  / np.sqrt(np.sum(vec**2,axis=-1))).T
def create_from_eulers(eulers):

    sP = np.sin(eulers[1])
    cP = np.cos(eulers[1])
    sR = np.sin(eulers[0])
    cR = np.cos(eulers[0])
    sY = np.sin(eulers[2])
    cY = np.cos(eulers[2])

    return np.array([[cY * cP, -cY * sP * cR + sY * sR, cY * sP * sR + sY * cR],
                     [sP, cP * cR, -cP * sR],
                     [-sY * cP, sY * sP * cR + cY * sR, -sY * sP * sR + cY * cR,]])
def create_from_quaternion(quat, dtype= np.float32):
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

    return np.array([[m00, m01, m02, 0],
                     [m10, m11, m12, 0],
                     [m20, m21, m22, 0],
                     [0,   0,   0,   1]],
                     dtype=dtype)
def create_from_translation(vec, dtype= np.float32):
    
    mat = np.identity(4, dtype=dtype)
    mat[3, 0:3] = vec[:3]
    return mat
def create_from_scale(scale, dtype= np.float32):
    m = np.diagflat([*scale, 1.0])
    if dtype:
        m = m.astype(dtype)
    return m
def ray_intersect_aabb(ray, aabb):
    
    direction = ray
    dir_fraction = np.empty(3, dtype = ray.dtype)
    dir_fraction[direction == 0.0] = np.inf
    dir_fraction[direction != 0.0] = np.divide(1.0, direction[direction != 0.0])

    t1 = (aabb[0,0]) * dir_fraction[ 0 ]
    t2 = (aabb[1,0]) * dir_fraction[ 0 ]
    t3 = (aabb[0,1]) * dir_fraction[ 1 ]
    t4 = (aabb[1,1]) * dir_fraction[ 1 ]
    t5 = (aabb[0,2]) * dir_fraction[ 2 ]
    t6 = (aabb[1,2]) * dir_fraction[ 2 ]


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
    point = (ray * t)
    return point
def get_view(forwards, up, right, position):

    return np.array(((right[0], up[0], -forwards[0], 0),
                     (right[1], up[1], -forwards[1], 0),
                     (right[2], up[2], -forwards[2], 0),
                     (-np.dot(right, position), -np.dot(up, position), np.dot(forwards, position), 1.0)), dtype=np.float32)

#####################################################################################

nearplane, farplane = 0.1, 1000
depthlayers = [nearplane, farplane/50, farplane/10, farplane/5, farplane]
projection = create_perspective_projection_from_bounds(-0.1, 0.1, -0.1*HEIGHT/WIDTH, 0.1*HEIGHT/WIDTH, nearplane, farplane)

def shader2D(vertexBuffer, texBuffer, texture):
    
    return ctx.pipeline(
        vertex_shader="""
            #version 300 es
            precision highp float;
            
            layout(location = 0) in vec2 vpos;
            layout(location = 1) in vec2 vtex;
            
            out vec2 TexCoords;
            
            void main()
            {
                TexCoords = vtex;
                gl_Position = vec4(vpos, 0, 1);
            }
        """,
        fragment_shader="""
            #version 300 es
            precision highp float;
            
            in vec2 TexCoords;

            uniform sampler2D material;
            
            layout(location = 0) out vec4 color;
            
            void main()
            {
                color = texture(material, TexCoords);
                color = pow(color, vec4(0.45));
            }
        """,
        
        blend={'enable': True, 'src_color': 'src_alpha', 'dst_color': 'one_minus_src_alpha'},
        layout=[{'name': 'material', 'binding': 0}],
        resources=[{'type': 'sampler', 'binding': 0, 'image': texture, 'wrap_x': 'clamp_to_edge', 'wrap_y': 'clamp_to_edge', 'min_filter': 'nearest', 'mag_filter': 'nearest'}],
        
        vertex_buffers= [*zengl.bind(ctx.buffer(vertexBuffer), "2f", 0),
                         *zengl.bind(ctx.buffer(texBuffer), "2f", 1)],
        
        vertex_count= len(vertexBuffer),
        cull_face= "back",
        topology= "triangles",
        framebuffer= [image, depth]
    )
def shader2Danitex(vertexBuffer, texBuffer, texture, frameAmount):
    
    texBuffer /= [frameAmount, 1]
    
    return ctx.pipeline(
        vertex_shader="""
            #version 300 es
            precision highp float;
            
            layout(location = 0) in vec2 vpos;
            layout(location = 1) in vec2 vtex;
            uniform float ofset;
            
            out vec2 TexCoords;
            
            void main()
            {
                TexCoords = vtex + vec2(ofset, 0);
                gl_Position = vec4(vpos, 0, 1);
            }
        """,
        fragment_shader="""
            #version 300 es
            precision highp float;
            
            in vec2 TexCoords;
            uniform sampler2D material;
            
            layout(location = 0) out vec4 color;
            
            void main()
            {
                color = texture(material, TexCoords);
                color = pow(color, vec4(0.45));
            }
        """,
        
        uniforms={'ofset': 0},
        
        blend={'enable': True, 'src_color': 'src_alpha', 'dst_color': 'one_minus_src_alpha'},
        layout=[{'name': 'material', 'binding': 0}],
        resources=[{'type': 'sampler', 'binding': 0, 'image': texture, 'wrap_x': 'clamp_to_edge', 'wrap_y': 'clamp_to_edge', 'min_filter': 'nearest', 'mag_filter': 'nearest'}],
        
        vertex_buffers= [*zengl.bind(ctx.buffer(vertexBuffer), "2f", 0),
                         *zengl.bind(ctx.buffer(texBuffer), "2f", 1)],
        
        vertex_count= len(vertexBuffer),
        cull_face= "back",
        topology= "triangles",
        framebuffer= [image, depth]
    )

LIGHTING = ctx.buffer(size= 64)
VIEW = ctx.buffer(size= 64)

# 0 = no shadows
# 1 = single shadow drawcall per frame, no moving shadows
# 2 = shadow drawcall per entity, moving shadows

if hasattr(platform, "window") and platform.window.mobile_check():
    DYNAMIC_SHADOWS = 0
else:
    DYNAMIC_SHADOWS = 2

if DYNAMIC_SHADOWS:
    #depthshaders for shadowmapping
    LSM = ctx.buffer(size= 256)
    def shaderDepth(vertexBuffer):

        return ctx.pipeline(
            vertex_shader="""
                #version 300 es
                precision highp float;
                
                layout(location = 0) in vec3 vpos;
                
                layout (std140) uniform GLSM
                {
                    mat4 LSM[4];
                };

                uniform mat4 model;

                flat out int instanceID;

                void main()
                {
                    gl_Position = LSM[gl_InstanceID] * model * vec4(vpos, 1.0);
                    instanceID = gl_InstanceID;
                }
            """,
            fragment_shader="""
                #version 300 es
                precision highp float;

                flat in int instanceID;

                layout (location = 0) out vec4 colour;

                void main()
                {
                    vec4 tempcol = vec4(0.0);
                    tempcol[instanceID] = 1.0 - gl_FragCoord.z;
                    colour = tempcol;
                }
            """,
            
            uniforms={'model': [np.identity(4).flatten()]},

            blend={'enable': True, 'src_color': 'src_alpha', 'dst_color': 'one_minus_src_alpha', 'op_color': 'max', 'op_alpha' : 'max'},
            layout=[{'name': 'GLSM', 'binding': 0}],
            resources=[{'type': 'uniform_buffer', 'binding': 0, 'buffer': LSM}],

            vertex_buffers= zengl.bind(ctx.buffer(vertexBuffer), "3f", 0),
            
            vertex_count= len(vertexBuffer),
            cull_face= "front",
            instance_count= 4,
            topology= "triangles",
            framebuffer= [lightdepth]
        )
    def shaderTerrainDepth(vertexBuffer, depthmap):

        return ctx.pipeline(
            vertex_shader="""
                #version 300 es
                precision highp float;
                
                layout(location = 0) in vec3 vpos;

                layout (std140) uniform GLSM
                {
                    mat4 LSM[4];
                };
                
                uniform mat4 model;
                uniform sampler2D heightmap;
                uniform vec2 ofset;

                flat out int instanceID;

                void main()
                {
                    vec2 TexCoords = (vpos.xz + ofset)/2000.0 + 0.5;
                    vec2 rawHeight = texture(heightmap, TexCoords).bg;
                    float height = (rawHeight.x + rawHeight.y / 256.0) * 977.5;

                    vec3 pos = vpos + vec3(ofset.x, height, ofset.y);
                    gl_Position = LSM[gl_InstanceID] * model * vec4(pos, 1);
                    instanceID = gl_InstanceID;
                }
            """,
            fragment_shader="""
                #version 300 es
                precision highp float;

                flat in int instanceID;

                layout (location = 0) out vec4 colour;

                void main()
                {
                    vec4 tempcol = vec4(0.0);
                    tempcol[instanceID] = 1.0 - gl_FragCoord.z;
                    colour = tempcol;
                }
            """,
            
            uniforms={'model': [np.identity(4).flatten()],
                    'ofset': [0, 0]},

            blend={'enable': True, 'src_color': 'src_alpha', 'dst_color': 'one_minus_src_alpha', 'op_color': 'max', 'op_alpha' : 'max'},
            layout=[{'name': 'GLSM', 'binding': 0},
                    {'name': 'heightmap', 'binding': 1}],
            resources=[{'type': 'uniform_buffer', 'binding': 0, 'buffer': LSM},
                    {'type': 'sampler', 'binding': 1, 'image': depthmap, 'wrap_x': 'clamp_to_edge', 'wrap_y': 'clamp_to_edge', 'min_filter': 'nearest', 'mag_filter': 'nearest'}],

            vertex_buffers= zengl.bind(ctx.buffer(vertexBuffer), "3f", 0),
            
            vertex_count= len(vertexBuffer),
            cull_face= "front",
            instance_count= 4,
            topology= "triangles",
            framebuffer= [lightdepth]
        )

    #shadowmapping
    def shader3D(vertexBuffer, normBuffer, texBuffer, texture):
        
        return ctx.pipeline(
            vertex_shader="""
                #version 300 es
                precision highp float;
                
                layout(location = 0) in vec3 vpos;
                layout(location = 1) in vec3 vnorm;
                layout(location = 2) in vec2 vtex;

                layout (std140) uniform GLSM
                {
                    mat4 LSM[4];
                };

                layout (std140) uniform VIEW
                {
                    mat4 view;
                };
                
                uniform mat4 projection;
                uniform mat4 model;
                
                out vec2 TexCoords;
                out vec3 fragPos;
                out vec3 fragNorm;
                out vec4 lightSpace[4];
                out float viewPosZ;
                
                void main()
                {
                    vec4 vertPos = model * vec4(vpos, 1.0);
                    
                    TexCoords = vtex;
                    fragPos = vertPos.xyz;
                    fragNorm = (model * vec4(vnorm, 0)).xyz;

                    for (int i = 0; i < 4; i++) {
                        lightSpace[i] = LSM[i] * vertPos;
                    }

                    vec4 viewPos = view * vertPos;
                    viewPosZ = viewPos.z;
                    gl_Position = projection * viewPos;
                }
            """,
            fragment_shader="""
                #version 300 es
                precision highp float;
                
                in vec2 TexCoords;
                in vec3 fragPos;
                in vec3 fragNorm;
                in vec4 lightSpace[4];
                in float viewPosZ;

                layout (std140) uniform lighting {
                    vec3 camPos;
                    vec3 lightposition[1];
                    vec3 lightcolor[1];
                    float lightstrength[1];
                };

                uniform sampler2D material;
                uniform highp sampler2D lightdepth;
                uniform float cascadeClip[3];
                
                layout (location = 0) out vec4 color;
                
                float ShadowCalculation()
                {
                    int cascadeIndex = 3;
                    for (int i = 0 ; i < 3 ; i++)
                    {
                        if (-viewPosZ <= cascadeClip[i])
                        {
                            cascadeIndex = i;
                            break;
                        }
                    }
                    
                    vec4 space = lightSpace[cascadeIndex];
                    vec3 projCoords = space.xyz / space.w;
                    projCoords = projCoords * 0.5 + 0.5;

                    int shadow = 0;
                    vec2 texelSize = 1.0 / vec2(textureSize(lightdepth, 0));
                    for (int x = -1; x <= 1; ++x)
                    {
                        for (int y = -1; y <= 1; ++y)
                        {
                            vec2 local = projCoords.xy + vec2(x, y) * texelSize;
                            shadow += any(lessThan(vec2(0.5), abs(local - 0.5))) ? 1 : 0;

                            float pcfDepth = 1.0 - texture(lightdepth, local)[cascadeIndex];
                            shadow += (pcfDepth + 0.00005) > projCoords.z ? 1 : 0;
                        }
                    }

                    return min(float(shadow)/9.0, 1.0);
                }

                vec3 calcPointlight(int i)
                {
                    vec3 baseTexture = texture(material, TexCoords).rgb;
                    vec3 result = vec3(0);
                    
                    vec3 relLightPos = lightposition[i] - fragPos;
                    float distance = length(relLightPos);
                    relLightPos = normalize(relLightPos);
                    
                    vec3 relCamPos = normalize(camPos - fragPos);
                    vec3 halfVec = normalize(relLightPos + relCamPos);

                    vec3 lightval = lightcolor[i] * lightstrength[i];
                    float distsquared = distance * distance;
                    float dotfrag = dot(fragNorm, relLightPos);

                    result += lightval * max(0.0, dotfrag) / distsquared * baseTexture; //diffuse
                    result += lightval * pow(max(0.0, dot(fragNorm, halfVec)), 32.0) / distsquared; //specular

                    float shadow = ShadowCalculation();
                    return result * shadow;
                }
                
                void main()
                {
                    vec4 baseTex = texture(material, TexCoords);
                    vec3 temp = 0.2 * baseTex.rgb; //ambient
                    temp += calcPointlight(0);

                    color = pow(vec4(temp, baseTex.a), vec4(0.45));
                }
            """,
            
            uniforms={'projection': projection.flatten(),
                    'model': np.identity(4).flatten(),
                    'cascadeClip': depthlayers[1:4]},
            
            blend={'enable': True, 'src_color': 'src_alpha', 'dst_color': 'one_minus_src_alpha'},
            layout=[{'name': 'GLSM', 'binding': 0},
                    {'name': 'VIEW', 'binding': 1},
                    {'name': 'lighting', 'binding': 2},
                    {'name': 'material', 'binding': 3},
                    {'name': 'lightdepth', 'binding': 4}],
            resources=[{'type': 'uniform_buffer', 'binding': 0, 'buffer': LSM},
                    {'type': 'uniform_buffer', 'binding': 1, 'buffer': VIEW},
                    {'type': 'uniform_buffer', 'binding': 2, 'buffer': LIGHTING},
                    {'type': 'sampler', 'binding': 3, 'image': texture, 'wrap_x': 'clamp_to_edge', 'wrap_y': 'clamp_to_edge', 'min_filter': 'nearest', 'mag_filter': 'nearest'},
                    {'type': 'sampler', 'binding': 4, 'image': lightdepth, 'wrap_x': 'clamp_to_edge', 'wrap_y': 'clamp_to_edge', 'min_filter': 'nearest', 'mag_filter': 'nearest'}],
            
            vertex_buffers= [*zengl.bind(ctx.buffer(vertexBuffer), "3f", 0),
                            *zengl.bind(ctx.buffer(normBuffer), "3f", 1),
                            *zengl.bind(ctx.buffer(texBuffer), "2f", 2)],
            
            vertex_count= len(vertexBuffer),
            cull_face= "back",
            topology= "triangles",
            framebuffer= [image, depth]
        )
    def shader3Danimated(vertexBuffer, normBuffer, texBuffer, jointDataList, weightDataList, nrJoints, texture):
        
        return ctx.pipeline(
            vertex_shader="""
                #version 300 es
                precision highp float;
                
                layout(location = 0) in vec3 vpos;
                layout(location = 1) in vec3 vnorm;
                layout(location = 2) in vec2 vtex;
                layout(location = 3) in ivec4 vboneIds; 
                layout(location = 4) in vec4 vweights;

                layout (std140) uniform GLSM
                {
                    mat4 LSM[4];
                };

                layout (std140) uniform VIEW
                {
                    mat4 view;
                };
                
                uniform mat4 projection;
                uniform mat4 model;
                uniform mat4 animation[50];
                
                out vec2 TexCoords;
                out vec3 fragPos;
                out vec3 fragNorm;
                out vec4 lightSpace[4];
                out float viewPosZ;
                
                vec4 applyBone(vec4 p)
                {
                    vec4 result = vec4(0.0);
                    for(int i = 0; i < 4; ++i)
                    {
                        if(vboneIds[i] >= 100) 
                        {
                            result = p;
                            break;
                        }
                        result += vweights[i] * (animation[vboneIds[i]] * p);
                    }
                    return result;
                }
                
                void main()
                {
                    
                    vec4 position = applyBone(vec4(vpos, 1.0));
                    vec4 normal = normalize(applyBone(vec4(vnorm, 0.0)));
                    
                    vec4 vertPos = model * position;
                    
                    TexCoords = vtex;
                    fragPos = vertPos.xyz;
                    fragNorm = (view * normal).xyz;

                    for (int i = 0; i < 4; i++) {
                        lightSpace[i] = LSM[i] * vertPos;
                    }

                    vec4 viewPos = view * vertPos;
                    viewPosZ = viewPos.z;
                    gl_Position = projection * viewPos;
                }
            """,
            fragment_shader="""
                #version 300 es
                precision highp float;
                
                in vec2 TexCoords;
                in vec3 fragPos;
                in vec3 fragNorm;
                in vec4 lightSpace[4];
                in float viewPosZ;

                layout (std140) uniform lighting {
                    vec3 camPos;
                    vec3 lightposition[1];
                    vec3 lightcolor[1];
                    float lightstrength[1];
                };

                uniform sampler2D material;
                uniform highp sampler2D lightdepth;
                uniform float cascadeClip[3];
                
                layout (location = 0) out vec4 color;
                
                float ShadowCalculation()
                {
                    int cascadeIndex = 3;
                    for (int i = 0 ; i < 3 ; i++)
                    {
                        if (-viewPosZ <= cascadeClip[i])
                        {
                            cascadeIndex = i;
                            break;
                        }
                    }
                    
                    vec4 space = lightSpace[cascadeIndex];
                    vec3 projCoords = space.xyz / space.w;
                    projCoords = projCoords * 0.5 + 0.5;

                    int shadow = 0;
                    vec2 texelSize = 1.0 / vec2(textureSize(lightdepth, 0));
                    for (int x = -1; x <= 1; ++x)
                    {
                        for (int y = -1; y <= 1; ++y)
                        {
                            vec2 local = projCoords.xy + vec2(x, y) * texelSize;
                            shadow += any(lessThan(vec2(0.5), abs(local - 0.5))) ? 1 : 0;

                            float pcfDepth = 1.0 - texture(lightdepth, local)[cascadeIndex]; 
                            shadow += (pcfDepth + 0.00005) > projCoords.z ? 1 : 0;
                        }    
                    }

                    return min(float(shadow)/9.0, 1.0);
                }

                vec3 calcPointlight(int i)
                {
                    vec3 baseTexture = texture(material, TexCoords).rgb;
                    vec3 result = vec3(0);
                    
                    vec3 relLightPos = lightposition[i] - fragPos;
                    float distance = length(relLightPos);
                    relLightPos = normalize(relLightPos);
                    
                    vec3 relCamPos = normalize(camPos - fragPos);
                    vec3 halfVec = normalize(relLightPos + relCamPos);

                    vec3 lightval = lightcolor[i] * lightstrength[i];
                    float distsquared = distance * distance;
                    float dotfrag = dot(fragNorm, relLightPos);

                    result += lightval * max(0.0, dotfrag) / distsquared * baseTexture; //diffuse
                    result += lightval * pow(max(0.0, dot(fragNorm, halfVec)), 32.0) / distsquared; //specular

                    float shadow = ShadowCalculation();
                    return result * shadow;
                }
                
                void main()
                {
                    vec4 baseTex = texture(material, TexCoords);
                    vec3 temp = 0.2 * baseTex.rgb; //ambient
                    temp += calcPointlight(0);

                    color = pow(vec4(temp, baseTex.a), vec4(0.45));
                }
            """,
            
            uniforms={'projection': projection.flatten(),
                    'model': np.identity(4).flatten(),
                    'cascadeClip': depthlayers[1:4],
                    'animation': [np.identity(4) for i in range(nrJoints)]},
            
            blend={'enable': True, 'src_color': 'src_alpha', 'dst_color': 'one_minus_src_alpha'},
            layout=[{'name': 'GLSM', 'binding': 0},
                    {'name': 'VIEW', 'binding': 1},
                    {'name': 'lighting', 'binding': 2},
                    {'name': 'material', 'binding': 3},
                    {'name': 'lightdepth', 'binding': 4}],
            resources=[{'type': 'uniform_buffer', 'binding': 0, 'buffer': LSM},
                    {'type': 'uniform_buffer', 'binding': 1, 'buffer': VIEW},
                    {'type': 'uniform_buffer', 'binding': 2, 'buffer': LIGHTING},
                    {'type': 'sampler', 'binding': 3, 'image': texture, 'wrap_x': 'clamp_to_edge', 'wrap_y': 'clamp_to_edge', 'min_filter': 'nearest', 'mag_filter': 'nearest'},
                    {'type': 'sampler', 'binding': 4, 'image': lightdepth, 'wrap_x': 'clamp_to_edge', 'wrap_y': 'clamp_to_edge', 'min_filter': 'nearest', 'mag_filter': 'nearest'}],
            
            vertex_buffers= [*zengl.bind(ctx.buffer(vertexBuffer), "3f", 0),
                            *zengl.bind(ctx.buffer(normBuffer), "3f", 1),
                            *zengl.bind(ctx.buffer(texBuffer), "2f", 2),
                            *zengl.bind(ctx.buffer(jointDataList), "4i", 3),
                            *zengl.bind(ctx.buffer(weightDataList), "4f", 4)],
            
            vertex_count= len(vertexBuffer),
            cull_face= "back",
            topology= "triangles",
            framebuffer= [image, depth]
        )
    def shaderTerrain(vertexBuffer, normmap, depthmap, texture):

        return ctx.pipeline(
            vertex_shader="""
                #version 300 es
                precision highp float;
                
                layout(location = 0) in vec3 vpos;

                layout (std140) uniform GLSM
                {
                    mat4 LSM[4];
                };

                layout (std140) uniform VIEW
                {
                    mat4 view;
                };

                uniform mat4 projection;
                uniform sampler2D normmap;
                uniform sampler2D heightmap;
                uniform vec2 ofset;

                out vec2 TexCoords;
                out vec3 fragPos;
                out vec3 fragNorm;
                out vec4 lightSpace[4];
                out float viewPosZ;

                void main()
                {
                    TexCoords = (vpos.xz + ofset)/2000.0 + 0.5;
                    vec2 rawHeight = texture(heightmap, TexCoords).bg;
                    float height = (rawHeight.x + rawHeight.y / 256.0) * 977.5;

                    fragPos = vpos + vec3(ofset.x, height, ofset.y);
                    fragNorm = normalize(texture(normmap, TexCoords).rbg * 2.0 - 1.0);
                    fragNorm.b = -fragNorm.b;

                    for (int i = 0; i < 4; i++) {
                        lightSpace[i] = LSM[i] * vec4(fragPos, 1);
                    }

                    vec4 viewPos = view * vec4(fragPos, 1);
                    viewPosZ = viewPos.z;
                    gl_Position = projection * viewPos;
                }
            """,
            fragment_shader="""
                #version 300 es
                precision highp float;
                
                in vec2 TexCoords;
                in vec3 fragPos;
                in vec3 fragNorm;
                in vec4 lightSpace[4];
                in float viewPosZ;

                layout (std140) uniform lighting {
                    vec3 camPos;
                    vec3 lightposition[1];
                    vec3 lightcolor[1];
                    float lightstrength[1];
                };

                uniform sampler2D material;
                uniform highp sampler2D lightdepth;
                uniform float cascadeClip[3];
                
                layout (location = 0) out vec4 color;
                
                float ShadowCalculation()
                {
                    int cascadeIndex = 3;
                    for (int i = 0 ; i < 3 ; i++)
                    {
                        if (-viewPosZ <= cascadeClip[i])
                        {
                            cascadeIndex = i;
                            break;
                        }
                    }
                    
                    vec4 space = lightSpace[cascadeIndex];
                    vec3 projCoords = space.xyz / space.w;
                    projCoords = projCoords * 0.5 + 0.5;

                    int shadow = 0;
                    vec2 texelSize = 1.0 / vec2(textureSize(lightdepth, 0));
                    for (int x = -1; x <= 1; ++x)
                    {
                        for (int y = -1; y <= 1; ++y)
                        {
                            vec2 local = projCoords.xy + vec2(x, y) * texelSize;
                            shadow += any(lessThan(vec2(0.5), abs(local - 0.5))) ? 1 : 0;

                            float pcfDepth = 1.0 - texture(lightdepth, local)[cascadeIndex]; 
                            shadow += (pcfDepth + 0.00005) > projCoords.z ? 1 : 0;
                        }
                    }

                    return min(float(shadow)/9.0, 1.0);
                }

                vec3 calcPointlight(int i)
                {
                    vec3 baseTexture = texture(material, TexCoords).rgb;
                    vec3 result = vec3(0);
                    
                    vec3 relLightPos = lightposition[i] - fragPos;
                    float distance = length(relLightPos);
                    relLightPos = normalize(relLightPos);
                    
                    vec3 relCamPos = normalize(camPos - fragPos);
                    vec3 halfVec = normalize(relLightPos + relCamPos);

                    vec3 lightval = lightcolor[i] * lightstrength[i];
                    float distsquared = distance * distance;
                    float dotfrag = dot(fragNorm, relLightPos);

                    result += lightval * max(0.0, dotfrag) / distsquared * baseTexture; //diffuse
                    result += lightval * pow(max(0.0, dot(fragNorm, halfVec)), 32.0) / distsquared; //specular

                    float shadow = ShadowCalculation();
                    return result * shadow;
                }
                
                void main()
                {
                    vec4 baseTex = texture(material, TexCoords);
                    vec3 temp = 0.2 * baseTex.rgb; //ambient
                    temp += calcPointlight(0);

                    color = pow(vec4(temp, baseTex.a), vec4(0.45));
                }
            """,
            
            uniforms={'projection': projection.flatten(),
                    'ofset': [0, 0],
                    'cascadeClip': depthlayers[1:4]},
            
            blend={'enable': True, 'src_color': 'src_alpha', 'dst_color': 'one_minus_src_alpha'},
            layout=[{'name': 'GLSM', 'binding': 0},
                    {'name': 'VIEW', 'binding': 1},
                    {'name': 'lighting', 'binding': 2},
                    {'name': 'normmap', 'binding': 3},
                    {'name': 'heightmap', 'binding': 4},
                    {'name': 'material', 'binding': 5},
                    {'name': 'lightdepth', 'binding': 6}],
            resources=[{'type': 'uniform_buffer', 'binding': 0, 'buffer': LSM},
                       {'type': 'uniform_buffer', 'binding': 1, 'buffer': VIEW},
                       {'type': 'uniform_buffer', 'binding': 2, 'buffer': LIGHTING},
                       {'type': 'sampler', 'binding': 3, 'image': normmap, 'wrap_x': 'clamp_to_edge', 'wrap_y': 'clamp_to_edge', 'min_filter': 'nearest', 'mag_filter': 'nearest'},
                       {'type': 'sampler', 'binding': 4, 'image': depthmap, 'wrap_x': 'clamp_to_edge', 'wrap_y': 'clamp_to_edge', 'min_filter': 'nearest', 'mag_filter': 'nearest'},
                       {'type': 'sampler', 'binding': 5, 'image': texture, 'wrap_x': 'repeat', 'wrap_y': 'repeat', 'min_filter': 'nearest', 'mag_filter': 'nearest'},
                       {'type': 'sampler', 'binding': 6, 'image': lightdepth, 'wrap_x': 'clamp_to_edge', 'wrap_y': 'clamp_to_edge', 'min_filter': 'nearest', 'mag_filter': 'nearest'}],
            
            vertex_buffers= zengl.bind(ctx.buffer(vertexBuffer), "3f", 0),
            
            vertex_count= len(vertexBuffer),
            cull_face= "back",
            topology= "triangles",
            framebuffer= [image, depth]
            )
else:
    #no shadowmapping
    def shader3D(vertexBuffer, normBuffer, texBuffer, texture):
        
        return ctx.pipeline(
            vertex_shader="""
                #version 300 es
                precision highp float;
                
                layout(location = 0) in vec3 vpos;
                layout(location = 1) in vec3 vnorm;
                layout(location = 2) in vec2 vtex;

                layout (std140) uniform VIEW
                {
                    mat4 view;
                };
                
                uniform mat4 projection;
                uniform mat4 model;
                
                out vec2 TexCoords;
                out vec3 fragPos;
                out vec3 fragNorm;
                
                void main()
                {
                    vec4 vertPos = model * vec4(vpos, 1.0);
                    
                    TexCoords = vtex;
                    fragPos = vertPos.xyz;
                    fragNorm = (model * vec4(vnorm, 0)).xyz;

                    vec4 viewPos = view * vertPos;
                    gl_Position = projection * viewPos;
                }
            """,
            fragment_shader="""
                #version 300 es
                precision highp float;
                
                in vec2 TexCoords;
                in vec3 fragPos;
                in vec3 fragNorm;

                layout (std140) uniform lighting {
                    vec3 camPos;
                    vec3 lightposition[1];
                    vec3 lightcolor[1];
                    float lightstrength[1];
                };

                uniform sampler2D material;
                
                layout (location = 0) out vec4 color;

                vec3 calcPointlight(int i)
                {
                    vec3 baseTexture = texture(material, TexCoords).rgb;
                    vec3 result = vec3(0);
                    
                    vec3 relLightPos = lightposition[i] - fragPos;
                    float distance = length(relLightPos);
                    relLightPos = normalize(relLightPos);
                    
                    vec3 relCamPos = normalize(camPos - fragPos);
                    vec3 halfVec = normalize(relLightPos + relCamPos);

                    vec3 lightval = lightcolor[i] * lightstrength[i];
                    float distsquared = distance * distance;
                    float dotfrag = dot(fragNorm, relLightPos);

                    result += lightval * max(0.0, dotfrag) / distsquared * baseTexture; //diffuse
                    result += lightval * pow(max(0.0, dot(fragNorm, halfVec)), 32.0) / distsquared; //specular

                    return result;
                }
                
                void main()
                {
                    vec4 baseTex = texture(material, TexCoords);
                    vec3 temp = 0.2 * baseTex.rgb; //ambient
                    temp += calcPointlight(0);

                    color = pow(vec4(temp, baseTex.a), vec4(0.45));
                }
            """,
            
            uniforms={'projection': projection.flatten(),
                    'model': np.identity(4).flatten()},
            
            blend={'enable': True, 'src_color': 'src_alpha', 'dst_color': 'one_minus_src_alpha'},
            layout=[{'name': 'VIEW', 'binding': 0},
                    {'name': 'lighting', 'binding': 1},
                    {'name': 'material', 'binding': 2}],
            resources=[{'type': 'uniform_buffer', 'binding': 0, 'buffer': VIEW},
                       {'type': 'uniform_buffer', 'binding': 1, 'buffer': LIGHTING},
                       {'type': 'sampler', 'binding': 2, 'image': texture, 'wrap_x': 'clamp_to_edge', 'wrap_y': 'clamp_to_edge', 'min_filter': 'nearest', 'mag_filter': 'nearest'}],
            
            vertex_buffers= [*zengl.bind(ctx.buffer(vertexBuffer), "3f", 0),
                            *zengl.bind(ctx.buffer(normBuffer), "3f", 1),
                            *zengl.bind(ctx.buffer(texBuffer), "2f", 2)],
            
            vertex_count= len(vertexBuffer),
            cull_face= "back",
            topology= "triangles",
            framebuffer= [image, depth]
        )
    def shader3Danimated(vertexBuffer, normBuffer, texBuffer, jointDataList, weightDataList, nrJoints, texture):
        
        return ctx.pipeline(
            vertex_shader="""
                #version 300 es
                precision highp float;
                
                layout(location = 0) in vec3 vpos;
                layout(location = 1) in vec3 vnorm;
                layout(location = 2) in vec2 vtex;
                layout(location = 3) in ivec4 vboneIds; 
                layout(location = 4) in vec4 vweights;

                layout (std140) uniform VIEW
                {
                    mat4 view;
                };
                
                uniform mat4 projection;
                uniform mat4 model;
                uniform mat4 animation[50];
                
                out vec2 TexCoords;
                out vec3 fragPos;
                out vec3 fragNorm;
                
                vec4 applyBone(vec4 p)
                {
                    vec4 result = vec4(0.0);
                    for(int i = 0; i < 4; ++i)
                    {
                        if(vboneIds[i] >= 100) 
                        {
                            result = p;
                            break;
                        }
                        result += vweights[i] * (animation[vboneIds[i]] * p);
                    }
                    return result;
                }
                
                void main()
                {
                    
                    vec4 position = applyBone(vec4(vpos, 1.0));
                    vec4 normal = normalize(applyBone(vec4(vnorm, 0.0)));
                    
                    vec4 vertPos = model * position;
                    
                    TexCoords = vtex;
                    fragPos = vertPos.xyz;
                    fragNorm = (view * normal).xyz;

                    vec4 viewPos = view * vertPos;
                    gl_Position = projection * viewPos;
                }
            """,
            fragment_shader="""
                #version 300 es
                precision highp float;
                
                in vec2 TexCoords;
                in vec3 fragPos;
                in vec3 fragNorm;

                layout (std140) uniform lighting {
                    vec3 camPos;
                    vec3 lightposition[1];
                    vec3 lightcolor[1];
                    float lightstrength[1];
                };

                uniform sampler2D material;
                
                layout (location = 0) out vec4 color;

                vec3 calcPointlight(int i)
                {
                    vec3 baseTexture = texture(material, TexCoords).rgb;
                    vec3 result = vec3(0);
                    
                    vec3 relLightPos = lightposition[i] - fragPos;
                    float distance = length(relLightPos);
                    relLightPos = normalize(relLightPos);
                    
                    vec3 relCamPos = normalize(camPos - fragPos);
                    vec3 halfVec = normalize(relLightPos + relCamPos);

                    vec3 lightval = lightcolor[i] * lightstrength[i];
                    float distsquared = distance * distance;
                    float dotfrag = dot(fragNorm, relLightPos);

                    result += lightval * max(0.0, dotfrag) / distsquared * baseTexture; //diffuse
                    result += lightval * pow(max(0.0, dot(fragNorm, halfVec)), 32.0) / distsquared; //specular

                    return result;
                }
                
                void main()
                {
                    vec4 baseTex = texture(material, TexCoords);
                    vec3 temp = 0.2 * baseTex.rgb; //ambient
                    temp += calcPointlight(0);

                    color = pow(vec4(temp, baseTex.a), vec4(0.45));
                }
            """,
            
            uniforms={'projection': projection.flatten(),
                    'model': np.identity(4).flatten(),
                    'animation': [np.identity(4) for i in range(nrJoints)]},
            
            blend={'enable': True, 'src_color': 'src_alpha', 'dst_color': 'one_minus_src_alpha'},
            layout=[{'name': 'VIEW', 'binding': 0},
                    {'name': 'lighting', 'binding': 1},
                    {'name': 'material', 'binding': 2}],
            resources=[{'type': 'uniform_buffer', 'binding': 0, 'buffer': VIEW},
                       {'type': 'uniform_buffer', 'binding': 1, 'buffer': LIGHTING},
                       {'type': 'sampler', 'binding': 2, 'image': texture, 'wrap_x': 'clamp_to_edge', 'wrap_y': 'clamp_to_edge', 'min_filter': 'nearest', 'mag_filter': 'nearest'}],
            
            vertex_buffers= [*zengl.bind(ctx.buffer(vertexBuffer), "3f", 0),
                            *zengl.bind(ctx.buffer(normBuffer), "3f", 1),
                            *zengl.bind(ctx.buffer(texBuffer), "2f", 2),
                            *zengl.bind(ctx.buffer(jointDataList), "4i", 3),
                            *zengl.bind(ctx.buffer(weightDataList), "4f", 4)],
            
            vertex_count= len(vertexBuffer),
            cull_face= "back",
            topology= "triangles",
            framebuffer= [image, depth]
        )
    def shaderTerrain(vertexBuffer, normmap, depthmap, texture):

        return ctx.pipeline(
            vertex_shader="""
                #version 300 es
                precision highp float;
                
                layout(location = 0) in vec3 vpos;

                layout (std140) uniform VIEW
                {
                    mat4 view;
                };

                uniform mat4 projection;
                uniform sampler2D normmap;
                uniform sampler2D heightmap;
                uniform vec2 ofset;

                out vec2 TexCoords;
                out vec3 fragPos;
                out vec3 fragNorm;

                void main()
                {
                    TexCoords = (vpos.xz + ofset)/2000.0 + 0.5;
                    vec2 rawHeight = texture(heightmap, TexCoords).bg;
                    float height = (rawHeight.x + rawHeight.y / 256.0) * 977.5;

                    fragPos = vpos + vec3(ofset.x, height, ofset.y);
                    fragNorm = normalize(texture(normmap, TexCoords).rbg * 2.0 - 1.0);
                    fragNorm.b = -fragNorm.b;

                    vec4 viewPos = view * vec4(fragPos, 1);
                    gl_Position = projection * viewPos;
                }
            """,
            fragment_shader="""
                #version 300 es
                precision highp float;
                
                in vec2 TexCoords;
                in vec3 fragPos;
                in vec3 fragNorm;
                in vec4 lightSpace[4];
                in float viewPosZ;

                layout (std140) uniform lighting {
                    vec3 camPos;
                    vec3 lightposition[1];
                    vec3 lightcolor[1];
                    float lightstrength[1];
                };

                uniform sampler2D material;
                
                layout (location = 0) out vec4 color;
                
                vec3 calcPointlight(int i)
                {
                    vec3 baseTexture = texture(material, TexCoords).rgb;
                    vec3 result = vec3(0);
                    
                    vec3 relLightPos = lightposition[i] - fragPos;
                    float distance = length(relLightPos);
                    relLightPos = normalize(relLightPos);
                    
                    vec3 relCamPos = normalize(camPos - fragPos);
                    vec3 halfVec = normalize(relLightPos + relCamPos);

                    vec3 lightval = lightcolor[i] * lightstrength[i];
                    float distsquared = distance * distance;
                    float dotfrag = dot(fragNorm, relLightPos);

                    result += lightval * max(0.0, dotfrag) / distsquared * baseTexture; //diffuse
                    result += lightval * pow(max(0.0, dot(fragNorm, halfVec)), 32.0) / distsquared; //specular

                    return result;
                }
                
                void main()
                {
                    vec4 baseTex = texture(material, TexCoords);
                    vec3 temp = 0.2 * baseTex.rgb; //ambient
                    temp += calcPointlight(0);

                    color = pow(vec4(temp, baseTex.a), vec4(0.45));
                }
            """,
            
            uniforms={'projection': projection.flatten(),
                    'ofset': [0, 0]},
            
            blend={'enable': True, 'src_color': 'src_alpha', 'dst_color': 'one_minus_src_alpha'},
            layout=[{'name': 'VIEW', 'binding': 0},
                    {'name': 'lighting', 'binding': 1},
                    {'name': 'normmap', 'binding': 2},
                    {'name': 'heightmap', 'binding': 3},
                    {'name': 'material', 'binding': 4}],
            resources=[{'type': 'uniform_buffer', 'binding': 0, 'buffer': VIEW},
                       {'type': 'uniform_buffer', 'binding': 1, 'buffer': LIGHTING},
                       {'type': 'sampler', 'binding': 2, 'image': normmap, 'wrap_x': 'clamp_to_edge', 'wrap_y': 'clamp_to_edge', 'min_filter': 'nearest', 'mag_filter': 'nearest'},
                       {'type': 'sampler', 'binding': 3, 'image': depthmap, 'wrap_x': 'clamp_to_edge', 'wrap_y': 'clamp_to_edge', 'min_filter': 'nearest', 'mag_filter': 'nearest'},
                       {'type': 'sampler', 'binding': 4, 'image': texture, 'wrap_x': 'repeat', 'wrap_y': 'repeat', 'min_filter': 'nearest', 'mag_filter': 'nearest'}],
            
            vertex_buffers= zengl.bind(ctx.buffer(vertexBuffer), "3f", 0),
            
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
            
            layout (location = 0) in int vert;

            layout (std140) uniform VIEW
            {
                mat4 view;
            };
            
            uniform vec3 boundingBox[2];
            uniform mat4 projection;
            
            void main() {
                vec3 pos = vec3(boundingBox[vert%2].x, boundingBox[(vert/2)%2].y, boundingBox[vert/4].z);
                gl_Position = projection * view * vec4(pos, 1);
            }
        """,
        fragment_shader="""
            #version 300 es
            precision highp float;
            
            layout (location = 0) out vec4 out_color;

            void main()
            {
                out_color = vec4(1,0,0,0.5);
            }
        """,
        
        uniforms={'projection': projection.flatten(),
                  'boundingBox': [[0, 660, -5], [1, 661, -4]]},
        
        blend={'enable': True, 'src_color': 'src_alpha', 'dst_color': 'one_minus_src_alpha'},
        layout=[{'name': 'VIEW', 'binding': 0}],
        resources=[{'type': 'uniform_buffer', 'binding': 0, 'buffer': VIEW}],
        
        vertex_buffers= zengl.bind(ctx.buffer(np.array([0, 2, 1, 3, 1, 2, 4, 5, 6, 7, 6, 5, 0, 1, 4, 5, 4, 1, 2, 6, 3, 7, 3, 6, 0, 4, 2, 6, 2, 4, 1, 3, 5, 7, 5, 3], dtype=np.int32)), "1i", 0),

        vertex_count=36,
        cull_face= "back",
        topology= "triangles",
        framebuffer= [image, depth]
    )

#####################################################################################

class entity:
    
    def __init__(self, position, size, eulers= [0,0,0]):
        
        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)
        self.size = size

        self.transformMat = np.identity(4)
        self.transformMat[0:3,0:3] = create_from_eulers(self.eulers)
        self.transformMat[3,0:3] = position

class pointLight(entity):

    def __init__(self, position, eulers, color, strength):

        super().__init__(position, 0, eulers)
        self.color = np.array(color, dtype=np.float32)
        self.strength = strength

class player(entity): 
    
    def __init__(self, position, eulers, camEulers, camZoom):
        
        super().__init__(position, 0, eulers)
        self.camera = camera(self.position, camEulers, camZoom)
    
    def update(self):
        
        cosX, sinX = self.camera.update(self.position)

        self.forwards = np.array((cosX, 0, -sinX))
        self.right = np.array((-sinX, 0, -cosX))

        transformMat = create_from_eulers(self.eulers)
        self.transformMat = [[*transformMat[0], 0],
                             [*transformMat[1], 0],
                             [*transformMat[2], 0],
                             [*self.position,   1]]
    
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
        self.up = (-cosX*sinY, cosY, sinX*sinY)
        
        self.center = pos
        self.position = self.center - self.zoom * self.forwards
        
        self.makeFrustum()
        
        return (cosX, sinX)

    def getViewTransform(self):
        
        return get_view(self.forwards, self.up, self.right, self.position)
    
    def getYawMat(self):
        
        return get_view(self.forwards, (0, 1, 0), self.right, self.position)
    
    def spin(self, dEulers):

        self.eulers += dEulers

        self.eulers[0] %= 2*np.pi
        self.eulers[1] = min(1.5, max(-1.5, self.eulers[1]))

    def makeFrustum(self):
        
        normals = [(self.forwards + self.right)*0.70710677298, (self.forwards - self.right)*0.70710677298, (self.forwards * 16/9 + self.up)*0.4902612303, (self.forwards * 16/9 - self.up)*0.4902612303]
        self.frustum = [[normal, normal[0] * self.position[0] + normal[1] * self.position[1] + normal[2] * self.position[2]] for normal in normals]

class scene:
    
    def __init__(self, sceneNr, playerPos, playerEul, camEul, camZoom):
        
        self.player = player(playerPos, playerEul, camEul, camZoom)
        self.jumpTime = 0
        self.height = 700
        self.heightmap = material("gfx/map8RG.png")
        
        self.createEntities(sceneNr)
        
        box = [[-1,-1,-1,1], [1,-1,-1,1], [1,1,-1,1], [-1,1,-1,1], [-1,-1,1,1], [1,-1,1,1], [1,1,1,1], [-1,1,1,1]]
        self.preFrustum = np.array([box @ np.linalg.inv(create_perspective_projection_from_bounds(-depthlayers[i], depthlayers[i], -depthlayers[i], depthlayers[i], depthlayers[i], depthlayers[i+1])) for i in range(4)], dtype= np.float32)
        self.frustumdepth = np.array([depthlayers[i] + depthlayers[i+1] * 0.5 for i in range(4)], dtype= np.float32).reshape((4, 1))
        self.invView = np.identity(4, dtype= np.float32)
        self.preFrustum /= self.preFrustum[...,3,None]

        self.boundingbox = boundingBoxMesh()

        cosX = np.cos(self.light.eulers[0])
        sinX = np.sin(self.light.eulers[0])
        cosY = np.cos(self.light.eulers[1])
        sinY = np.sin(self.light.eulers[1])

        self.lightforwards = np.array((cosX*cosY, sinY, -sinX*cosY))
        self.lightright = (sinX, 0, cosX)
        self.lightup = np.array((-cosX*sinY, cosY, sinX*sinY))

        self.lightview = np.array([get_view(self.lightforwards, self.lightup, self.lightright, self.player.camera.position + self.player.camera.forwards * self.frustumdepth[i]) for i in range(4)], dtype= np.float32)

        self.setDrawFunc()
        self.createEntityGrid()
    
    def createEntities(self, sceneNr):

        if sceneNr == 0:
            
            self.light = pointLight([-2000, 2000, -2000], [-1/3*np.pi, -1/10*np.pi, 0], [218, 203, 125], 10000)

            self.terrain = gltfMesh("models/terrain/terrain.gltf", [material("gfx/map8N.png").img, self.heightmap.img, material("gfx/grass.png").img])
            self.terrain.shaders.uniforms['ofset'][:] = np.ascontiguousarray(np.round(self.player.position[0:3:2] / 5) * 5, 'f').data.cast('B')
            
            self.entities = {
                ENTITY_TYPE["player"]:            [self.player,                    gltfMesh("models/vedal987/vedal987.gltf",                    [material("models/vedal987/vedal987.png").img])],
                ENTITY_TYPE["Camilla's_tent"]:    [entity([-8,663.55,40],20),      gltfMesh("models/V-nexus/Camilla's_tent/Camillas_tent.gltf", [material("models/V-nexus/Camilla's_tent/Camillas_tent.png").img,
                                                                                                                                                 material("models/V-nexus/Camilla's_tent/Camillas_tent.png").img])],
                ENTITY_TYPE["drone_factory"]:     [entity([43,660.35,7.775],25),   gltfMesh("models/V-nexus/drone_factory/drone_factory.gltf",  [material("models/V-nexus/drone_factory/drone_factory.png").img,
                                                                                                                                                 material("models/V-nexus/drone_factory/drone_factory.png").img,
                                                                                                                                                 material("models/V-nexus/drone_factory/drone_factory.png").img,
                                                                                                                                                 material("models/V-nexus/drone_factory/drone_factory.png").img])],
                ENTITY_TYPE["floors"]:            [entity([0,660.5,0],60),         gltfMesh("models/V-nexus/floors/floors.gltf",                [material("models/V-nexus/floors/grass_field2.png").img,
                                                                                                                                                 material("models/V-nexus/floors/grass_field2.png").img,
                                                                                                                                                 material("models/V-nexus/floors/grass_field2.png").img,
                                                                                                                                                 material("models/V-nexus/floors/grass_field2.png").img,
                                                                                                                                                 material("models/V-nexus/floors/item_factory.png").img,
                                                                                                                                                 material("models/V-nexus/floors/upgrade_smith.png").img,
                                                                                                                                                 material("models/V-nexus/floors/vedals_house.png").img,
                                                                                                                                                 material("models/V-nexus/floors/water_pump.png").img,
                                                                                                                                                 material("models/V-nexus/floors/power_generator.png").img])],
                ENTITY_TYPE["item_factory"]:      [entity([46,664.31,-33.75],20),  gltfMesh("models/V-nexus/item_factory/item_factory.gltf",    [material("models/V-nexus/item_factory/item_factory.png").img])],
                ENTITY_TYPE["item_shop"]:         [entity([42,658.1,46],10),       gltfMesh("models/V-nexus/item_shop/item_shop.gltf",          [material("models/V-nexus/item_shop/item_shop.png").img])],
                ENTITY_TYPE["street"]:            [entity([6,658,0],70),           gltfMesh("models/V-nexus/street/street.gltf",                [material("models/V-nexus/street/street.png").img])],
                ENTITY_TYPE["upygamerade_smith"]: [entity([41.9,661.2,-10.05],14), gltfMesh("models/V-nexus/upgrade_smith/upgrade_smith.gltf",  [material("models/V-nexus/upgrade_smith/upgrade_smith.png").img,
                                                                                                                                                 material("models/V-nexus/upgrade_smith/piston.png").img,
                                                                                                                                                 material("models/V-nexus/upgrade_smith/piston_base.png").img,
                                                                                                                                                 material("models/V-nexus/upgrade_smith/gear.png").img,
                                                                                                                                                 material("models/V-nexus/upgrade_smith/piston.png").img,
                                                                                                                                                 material("models/V-nexus/upgrade_smith/piston_base.png").img,
                                                                                                                                                 material("models/V-nexus/upgrade_smith/piston.png").img,
                                                                                                                                                 material("models/V-nexus/upgrade_smith/piston_base.png").img,
                                                                                                                                                 material("models/V-nexus/upgrade_smith/piston.png").img,
                                                                                                                                                 material("models/V-nexus/upgrade_smith/piston_base.png").img,
                                                                                                                                                 material("models/V-nexus/upgrade_smith/gear.png").img,
                                                                                                                                                 material("models/V-nexus/upgrade_smith/gear.png").img])],
                ENTITY_TYPE["utilities"]:         [entity([7.75,658.6,-37],20),    gltfMesh("models/V-nexus/utilities/utilities.gltf",          [material("models/V-nexus/utilities/water_pump.png").img,
                                                                                                                                                 material("models/V-nexus/utilities/water_pump.png").img,
                                                                                                                                                 material("models/V-nexus/utilities/water_pump.png").img,
                                                                                                                                                 material("models/V-nexus/utilities/water_pump.png").img,
                                                                                                                                                 material("models/V-nexus/utilities/power_generator.png").img,
                                                                                                                                                 material("models/V-nexus/utilities/power_generator.png").img])],
                ENTITY_TYPE["walls"]:             [entity([0,660.5,0],70),         gltfMesh("models/V-nexus/walls/walls.gltf",                  [material("models/V-nexus/walls/walls.png").img,
                                                                                                                                                 material("models/V-nexus/walls/walls.png").img,
                                                                                                                                                 material("models/V-nexus/walls/walls.png").img,
                                                                                                                                                 material("models/V-nexus/walls/walls.png").img,
                                                                                                                                                 material("models/V-nexus/walls/walls.png").img,
                                                                                                                                                 material("models/V-nexus/walls/walls.png").img,
                                                                                                                                                 material("models/V-nexus/walls/walls.png").img,
                                                                                                                                                 material("models/V-nexus/walls/walls.png").img,])],
                ENTITY_TYPE["world_center"]:      [entity([2,709.1,4],70),         gltfMesh("models/V-nexus/world_center/world_center.gltf",    [material("models/V-nexus/world_center/world_center_building.png").img,
                                                                                                                                                 material("models/V-nexus/world_center/beacon.png").img])],
                ENTITY_TYPE["vedal's_house"]:     [entity([-36.15,663.5,26],40),   gltfMesh("models/V-nexus/vedal's_house/vedals_house.gltf",   [material("models/V-nexus/vedal's_house/vedals_house.png").img,
                                                                                                                                                 material("models/V-nexus/vedal's_house/vedals_house.png").img,
                                                                                                                                                 material("models/V-nexus/vedal's_house/vedals_house.png").img,
                                                                                                                                                 material("models/V-nexus/vedal's_house/vedals_house.png").img,
                                                                                                                                                 material("models/V-nexus/vedal's_house/vedals_house.png").img,
                                                                                                                                                 material("models/V-nexus/vedal's_house/vedals_house.png").img,
                                                                                                                                                 material("models/V-nexus/vedal's_house/vedals_house.png").img,
                                                                                                                                                 material("models/V-nexus/vedal's_house/vedals_house.png").img])],
                }

    def setDrawFunc(self):

        if DYNAMIC_SHADOWS == 2:
        
            self.drawFunc = self.drawDepthAndColor2
            self.depthshaders = [[shaderDepth(np.concatenate(entity[1].vertexDataList)), entity[0]] for entity in self.entities.values()]
            self.terraindepth = shaderTerrainDepth(np.concatenate(self.terrain.vertexDataList), self.heightmap.img)
            self.moveTerrainUniforms = [self.terrain.shaders.uniforms['ofset'], self.terraindepth.uniforms['ofset']]
        
        elif DYNAMIC_SHADOWS == 1:
            
            self.drawFunc = self.drawDepthAndColor1
            self.depthshaders = [shaderDepth(np.ascontiguousarray(np.concatenate([np.concatenate([arr + entity[0].transformMat[3,0:3] for arr in entity[1].vertexDataList]) for entity_type, entity in self.entities.items() if entity_type not in [ENTITY_TYPE['player']]]), 'f').data.cast('B'))]
            self.terraindepth = shaderTerrainDepth(np.concatenate(self.terrain.vertexDataList), self.heightmap.img)
            self.moveTerrainUniforms = [self.terrain.shaders.uniforms['ofset'], self.terraindepth.uniforms['ofset']]
        
        else:

            self.drawFunc = self.drawColor
            self.moveTerrainUniforms = [self.terrain.shaders.uniforms['ofset'][:]]

    def createEntityGrid(self):

        self.entityGrid = [[[] for j in range(500)] for i in range(500)]
        for entity_type, obj in self.entities.items():
            
            #skip non-collision objects
            if entity_type is ENTITY_TYPE["player"]: continue
            
            meshBoundingBoxes = obj[1].boundingBox + obj[0].position
            for meshBoundingBox in meshBoundingBoxes:
                
                meshmin, meshmax = [list(map(int, vec3[::2])) for vec3 in meshBoundingBox/4 + 250]
                [self.entityGrid[x][y].append(obj) for x in range(meshmin[0], meshmax[0]+1) for y in range(meshmin[1], meshmax[1]+1) if obj not in self.entityGrid[x][y]]

    def jump(self, jump):
        
        if not self.jumpTime:
            self.jumpTime = jump
            self.jumpStartHeight = self.height
        
        t = (jump - self.jumpTime)
        jumpheight = 0.005*t - 0.0000049 * (t**2)
        
        if jumpheight < (self.height - self.jumpStartHeight):
            self.player.position[1] = self.height
            self.jumpTime = 0
            return False
        else:
            self.player.position[1] = self.jumpStartHeight + jumpheight
            return True
    
    def movePlayer(self, dPos, sprint, frametime):
        
        if dPos[0] and dPos[1]: dPos *= 0.7071
        
        movement = (dPos[0]*self.player.right + dPos[1]*self.player.forwards) * (1 + sprint)
        movement, collisionHeight = self.checkCollision(movement, self.player.position)
        
        self.player.angle(frametime, dPos)
        self.player.move(movement * 0.01 * frametime)
        

        pos = self.player.position[0:3:2]

        for uniform in self.moveTerrainUniforms:
            uniform[:] = np.ascontiguousarray([np.round(pos / 5) * 5], 'f').data.cast('B')
        
        pos = [int(i * 4096/2000 + 2048) for i in pos]

        mapHeight = [self.heightmap.pixels.getpixel([pos[0] + x, pos[1] + y]) for x, y in [(0, -1), (-1, 0), (0, 0), (1, 0), (0, 1)]]
        mapHeight = [b * 8 + g * 8/256 for r, g, b, a in mapHeight]

        angle = [np.arctan(mapHeight[y] - mapHeight[x]) for x, y in [(2, 1), (3, 2), (0, 2), (2, 4)]]
        
        roll = (angle[0] + angle[1]) * 0.5
        pitch = (angle[2] + angle[3]) * 0.5

        self.player.eulers[0] = pitch
        self.player.eulers[1] = roll
        self.height = max(mapHeight[2]  * 977.5/(255*8), collisionHeight - 0.1)
        
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
                
                collisionPos = ray_intersect_aabb(movement, localBoundingBox)
                if collisionPos is not None:
                    
                    distance = np.linalg.norm(collisionPos)
                    if distance < 1:
                        
                        normal = self.getNormal(collisionPos, localBoundingBox)
                        leftoverMovement = movement - collisionPos
                        leftoverMovement -= (normal[0] * leftoverMovement[0] + normal[1] * leftoverMovement[1] + normal[2] * leftoverMovement[2]) * normal
                        
                        distanceList.append(distance)
                        collisionPosList.append(collisionPos)
                        movementList.append(leftoverMovement)
                        meshBoundingBoxList.append(meshBoundingBox)
        
        if distanceList:
            index = distanceList.index(min(distanceList))
            movement = collisionPosList[index]
            movement += self.checkCollision2(movementList[index], meshBoundingBoxList-pos+movement)
            self.boundingbox.updateBoundingBox(meshBoundingBoxList[index])
        
        if heightList:
            collisionHeight = pos[1] + max(heightList)
        
        return movement, collisionHeight
    
    def checkCollision2(self, movement, meshBoundingBoxList):
        
        collisionPosList, distanceList = [], []
        
        for meshBoundingBox in meshBoundingBoxList:
            
            collisionPos = ray_intersect_aabb(movement, meshBoundingBox)
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
        
        cam = self.player.camera
        view = cam.getViewTransform()
        frustum = cam.frustum

        VIEW.write(view)
        LIGHTING.write(np.ascontiguousarray([*cam.position, 0,
                                             *self.light.position, 0,
                                             *self.light.color, 0,
                                             self.light.strength], 'f').data.cast('B'))
        
        terrain = self.terrain

        self.drawFunc(frustum, view)
        
        terrain.shaders.render()
        self.boundingbox.draw()
        
        image.blit(output)
        output.blit()
        ctx.end_frame()
        pygame.display.flip()

    def drawDepthAndColor2(self, frustum, view):

        lightdepth.clear()
        
        lightSpaceMatrix = self.getLightSpaceMatrix(view)
        LSM.write(lightSpaceMatrix)

        for depthshader, transformMat in self.depthshaders:

            depthshader.uniforms['model'][:] = np.ascontiguousarray(transformMat.transformMat, 'f').data.cast('B')
            depthshader.render()
        
        self.terraindepth.render()

        self.drawColor(frustum, view)
    
    def drawDepthAndColor1(self, frustum, view):

        lightdepth.clear()
        
        lightSpaceMatrix = self.getLightSpaceMatrix(view)
        LSM.write(lightSpaceMatrix)

        for depthshader in self.depthshaders:
            depthshader.render()
        
        self.terraindepth.render()

        self.drawColor(frustum, view)

    def drawColor(self, frustum, view):

        for entity in self.entities.values():

            if self.insideFrustum(entity[0], frustum):

                if entity[1].hasJoints:
                    entity[1].pose += 1
                    entity[1].setUniform()

                entity[1].draw(entity[0].transformMat)

    def getLightSpaceMatrix(self, view):

        position = self.player.camera.position + self.player.camera.forwards * self.frustumdepth
        self.lightview[:, 3, 0:3] = np.stack([-position @ self.lightright, -position @ self.lightup, position @ self.lightforwards], axis=1)

        transView = view[:3,:3].T
        self.invView[:3,:3] = transView
        self.invView[3,:3] = -view[3,:3] @ transView

        viewcorners = self.preFrustum @ self.invView @ self.lightview

        mins = viewcorners.min(axis=1)[:, :3]
        maxs = viewcorners.max(axis=1)[:, :3]

        minZ = -10 * mins[:, 2]
        maxZ = -10 * maxs[:, 2]

        return self.lightview @ np.array([create_orthogonal_projection(mins[i, 0], maxs[i, 0], mins[i, 1], maxs[i, 1], maxZ[i], minZ[i]) for i in range(4)])
    
    def insideFrustum(self, ent, frustum):

        px,py,pz = ent.position
        s = ent.size

        for (nx,ny,nz), d in frustum:
            if nx*px + ny*py + nz*pz - d + s <= 0:
                return 0
        
        return 1

class game:
    
    __slots__ = ("window", "renderer", "scene", "sceneNr", "time", "last_time", "savedtime", "frametime", "savedFramerate", "savedFrames", "keys", "scroll", "jump")

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
                if event.key == input_map["escape"]:
                    result = OPEN_MENU
            elif event.type == pygame.MOUSEWHEEL:
                self.scene.player.camera.zoom -= event.y
            if event.type == pygame.FINGERDOWN:
                x = event.x
                y = event.y
                fingers[event.finger_id] = x, y
                print(x, y)
            if event.type == pygame.FINGERUP:
                fingers.pop(event.finger_id, None)
        
        self.calculate_framerate()
        self.handle_keys()
        self.handle_mouse()
        
        self.scene.player.update()
        self.scene.render()
        
        return result

    def handle_keys(self):

        dPos = np.zeros(2)
        keys = pygame.key.get_pressed()

        #this method makes it so holding multible keys doesnt prioritize the first one in the row
        if keys[input_map["forwards"]]:  dPos[1] += 1
        if keys[input_map["left"]]:      dPos[0] += 1
        if keys[input_map["backwards"]]: dPos[1] -= 1
        if keys[input_map["right"]]:     dPos[0] -= 1
        if keys[input_map["jump"]]:      self.jump = True
        sprint = keys[input_map["sprint"]]

        if fingers.values():
            dPos += fingers.values()
        
        #the jump code is an ungodly mess, dont touch it if not needed
        if self.jump: self.jump = self.scene.jump(self.time)
        
        if dPos[0] or dPos[1]: self.scene.movePlayer(dPos, sprint, self.frametime)

    def handle_mouse(self):
        
        (x,y) = pygame.mouse.get_rel()
        dEulers = 0.001 * -x * np.array([1,0,0])
        dEulers -= 0.001 * y * np.array([0,1,0])
        self.scene.player.camera.spin(dEulers)
    
    def set_up_timer(self):

        self.last_time = 0
        self.time = 0
        self.frametime = 0
        self.savedtime = 0
        self.savedFramerate = 0
        self.savedFrames = 0
    
    def calculate_framerate(self):
        
        self.time = time.perf_counter_ns() * 0.000001
        self.frametime = (self.time - self.last_time)
        framerate = 1000/self.frametime
        self.last_time = self.time

        self.savedFramerate += framerate
        self.savedFrames += 1

        if self.time - self.savedtime > 250:

            pygame.display.set_caption(f"Running at {int(self.savedFramerate/self.savedFrames)} fps.")
            self.savedtime = self.time
            self.savedFramerate = 0
            self.savedFrames = 0
    
    def quit(self):
        
        saveName = "savefile.txt"
        np.savetxt(saveName, [self.sceneNr, *self.scene.player.position, *self.scene.player.eulers, *self.scene.player.camera.eulers, self.scene.player.camera.zoom], fmt='%f')

class menu:
    
    def __init__(self):
        
        pygame.mouse.set_visible(True)
        pygame.event.set_grab(False)
        
        self.createObjects()
        self.set_up_timer()
        self.gameLoop()
    
    def createObjects(self):
        
        texture = material("gfx/button.png")
        
        self.buttons = []
        
        newGameButton = button((0, 0.3), (0.6, 0.4), texture, newGameClick)
        self.buttons.append(newGameButton)
        
        if sys.platform != "emscripten":
            quitButton = button((0, -0.3), (0.6, 0.4), texture, quitClick)
            self.buttons.append(quitButton)
    
    def gameLoop(self):
        
        result = CONTINUE
        click = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                result = EXIT
            elif event.type == pygame.MOUSEBUTTONDOWN:
                click = True
        
        self.calculate_framerate()
        result = self.handleMouse(click)
        
        ctx.new_frame()
        image.clear()
        depth.clear()
        
        for button in self.buttons:
            button.draw()
        
        image.blit()
        ctx.end_frame()
        
        pygame.display.flip()
        
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
    
    def set_up_timer(self):

        self.last_time = 0
        self.time = 0
        self.frametime = 0
        self.savedtime = 0
        self.savedFramerate = 0
        self.savedFrames = 0
    
    def calculate_framerate(self):

        self.time = time.perf_counter_ns() * 0.000001
        self.frametime = (self.time - self.last_time)
        framerate = 1000/self.frametime
        self.last_time = self.time

        self.savedFramerate += framerate
        self.savedFrames += 1

        if self.time - self.savedtime > 250:

            pygame.display.set_caption(f"Running at {int(self.savedFramerate/self.savedFrames)} fps.")
            self.savedtime = self.time
            self.savedFramerate = 0
            self.savedFrames = 0
    
    def quit(self):
        
        pass

#####################################################################################

def newGameClick():
    return NEW_GAME

def quitClick():
    return EXIT

class button:
    
    def __init__(self, pos, size, texture, function):
        
        self.click = function
        self.pos = pos
        self.size = size
        self.frameCount = 2
        
        vertices = np.array([[pos[0] - self.size[0]*0.5, pos[1] + self.size[1]*0.5],
                             [pos[0] - self.size[0]*0.5, pos[1] - self.size[1]*0.5],
                             [pos[0] + self.size[0]*0.5, pos[1] - self.size[1]*0.5],
                             
                             [pos[0] - self.size[0]*0.5, pos[1] + self.size[1]*0.5],
                             [pos[0] + self.size[0]*0.5, pos[1] - self.size[1]*0.5],
                             [pos[0] + self.size[0]*0.5, pos[1] + self.size[1]*0.5]], dtype=np.float32)
        
        texCoords = np.array([[0,1], [0,0], [1,0], [0,1], [1,0], [1,1]], dtype=np.float32)
        
        self.shader = shader2Danitex(vertices, texCoords, texture.img, self.frameCount)
    
    def draw(self):
        
        self.shader.render()
    
    def handleMouse(self, pos, click):
        
        if self.inside(pos):
            self.shader.uniforms['ofset'][:] = np.ascontiguousarray(1 / self.frameCount, 'f').data.cast('B')
            if click:
                return self.click()
        else:
            self.shader.uniforms['ofset'][:] = np.ascontiguousarray(0 / self.frameCount, 'f').data.cast('B')
        
        return CONTINUE
    
    def inside(self, pos):
        
        for i in (0, 1):
            if pos[i] < (self.pos[i] - self.size[i]*0.5) or pos[i] > (self.pos[i] + self.size[i]*0.5):
                return False
        return True

class material:
    
    def __init__(self, filepath):
        
        self.pixels = Image.open(filepath)
        self.img = ctx.image(self.pixels.size, 'rgba8unorm',  np.array(self.pixels))

class gltfMesh:

    def __init__(self, filename, textures):
        
        #create the np files with this
        #import precomputeGLTF
        #precomputeGLTF.loadGLTF(filename)
        
        hasNormals, hasTextures, self.hasJoints, listLenght = np.loadtxt(f"{filename}.Data").astype(np.int32)
        self.boundingBox = np.loadtxt(f"{filename}.BoundingBox").astype(np.float32)
        self.boundingBox = [[self.boundingBox[2*i], self.boundingBox[2*i + 1]] for i in range(listLenght)]
        
        indexDataList = [np.loadtxt(f"{filename}.IndexDataList{i}").astype(np.int32) for i in range(listLenght)]
        
        self.vertexDataList = [np.loadtxt(f"{filename}.VertexDataList{i}").astype(np.float32) for i in range(listLenght)]
        self.vertexDataList = [np.array([self.vertexDataList[i][3*j:3*j+3] for j in indexDataList[i]], dtype=np.float32) for i in range(listLenght)]
        
        if hasNormals:
            normalDataList = [np.loadtxt(f"{filename}.NormalDataList{i}").astype(np.float32) for i in range(listLenght)]
            normalDataList = [np.array([normalDataList[i][3*j:3*j+3] for j in indexDataList[i]], dtype=np.float32) for i in range(listLenght)]
        
        if hasTextures:
            texCoordDataList = [np.loadtxt(f"{filename}.TexCoordDataList{i}").astype(np.float32) for i in range(listLenght)]
            texCoordDataList = [np.array([texCoordDataList[i][2*j:2*j+2] for j in indexDataList[i]], dtype=np.float32) for i in range(listLenght)]
        
        if self.hasJoints:
            jointDataList = [np.loadtxt(f"{filename}.JointDataList{i}").astype(np.int32) for i in range(listLenght)]
            jointDataList = [np.array([jointDataList[i][4*j:4*j+4] for j in indexDataList[i]], dtype=np.int32) for i in range(listLenght)]
            weightDataList = [np.loadtxt(f"{filename}.WeightDataList{i}").astype(np.float32) for i in range(listLenght)]
            weightDataList = [np.array([weightDataList[i][4*j:4*j+4] for j in indexDataList[i]], dtype=np.float32) for i in range(listLenght)]
            
            self.pose = 0
            nrAnimations, self.timeData = np.loadtxt(f"{filename}.MatData").astype(np.int32)
            self.transformMat = [np.loadtxt(f"{filename}.Anim{i}Matrices").astype(np.float32) for i in range(nrAnimations)]
            self.nrJoints = len(self.transformMat[0]) // (16 * self.timeData)
            self.transformMat = [[[self.transformMat[anim][i+j*self.nrJoints : i+j*self.nrJoints+16] for i in range(0, 16*self.nrJoints, 16)] for j in range(0, 16*self.timeData, 16)] for anim in range(nrAnimations)]
            
        #shaders

        if filename == "models/terrain/terrain.gltf": self.shaders = shaderTerrain(self.vertexDataList[0], textures[0], textures[1], textures[2])
        elif self.hasJoints:                          self.shaders = [shader3Danimated(self.vertexDataList[i], normalDataList[i], texCoordDataList[i], jointDataList[i], weightDataList[i], self.nrJoints, textures[i]) for i in range(listLenght)]
        else:                                         self.shaders = [shader3D(self.vertexDataList[i], normalDataList[i], texCoordDataList[i], textures[i]) for i in range(listLenght)]
    
    def setUniform(self):
        
        for shader in self.shaders:
            animation = self.transformMat[0][self.pose%self.timeData]
            shader.uniforms['animation'][:] = np.ascontiguousarray(animation, 'f').data.cast('B')
    
    def draw(self, model):

        for shader in self.shaders:
            shader.uniforms['model'][:] = np.ascontiguousarray(model, 'f').data.cast('B')
            shader.render()
    
class boundingBoxMesh:
    
    def __init__(self):
        
        self.hasJoints = 0
        self.shader = shaderBoundingBox()
    
    def updateBoundingBox(self, boundingBox):
        
        self.shader.uniforms['boundingBox'][:] = np.ascontiguousarray(boundingBox, 'f').data.cast('B')
        
    def draw(self):
        
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

#import cv2
#img = cv2.imread("gfx/map8.png", cv2.IMREAD_UNCHANGED)
#pixels = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
#*size, channels = pixels.shape
#newpixels = np.array([[[r//256, r%256, 0, 255] for r, g, b, a in row] for row in pixels])
#cv2.imwrite("gfx/map8RG.png", newpixels)

asyncio.run(main())