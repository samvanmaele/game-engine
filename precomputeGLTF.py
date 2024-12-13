from pygltflib import GLTF2
import numpy as np
import struct
import pyrr

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

def createNodeHierarchy(gltf, node):
    
    result = [node]
    children = gltf.nodes[node].children
    if children:
        for child in children:
            result.extend(createNodeHierarchy(gltf, child))
        
    return result    

def createParentlist(gltf):
    
    nodes = gltf.nodes
    parentList = [None for i in range(len(nodes))]
    for node in range(len(nodes)):
        if nodes[node].children is not None:
            for i in nodes[node].children:
                parentList[i] = node
    return parentList

def createTransformMatrices(gltf, transformMatrices, timeData, animations, animation, transformMatrix):
    
    for i in range(len(animations[animation].channels)):
        target = animations[animation].channels[i].target
        sampler = animations[animation].channels[i].sampler
        samplerOutput = animations[animation].samplers[sampler].output
        samplerData = readAccesor(gltf, gltf.accessors[samplerOutput])
        
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
            transformMatrices[animation][pose][target.node] = transformMatrix[2][pose] @ transformMatrix[1][pose] @ transformMatrix[0][pose]
    
def createAnimation(transformMatrices, finalMatrices, timeData, skin, animation, parentList, inverseBindData):
    
    for joint in skin.joints:
        for pose in range(timeData):
            transformMatrices[animation][pose][joint] = transformMatrices[animation][pose][joint] @ transformMatrices[animation][pose][parentList[joint]]
            finalMatrices[animation][pose][skin.joints.index(joint)] = inverseBindData[skin.joints.index(joint)] @ transformMatrices[animation][pose][joint]

def readAccesor(gltf, accessor):
    
    bufferView = gltf.bufferViews[accessor.bufferView]
    buffer = gltf.buffers[bufferView.buffer]
    data = gltf.get_data_from_buffer_uri(buffer.uri)
    
    count = accessor.count * ELEMENT_SIZES[accessor.type]
    struct_type = STRUCT_TYPE[accessor.componentType]
    value_size = VALUE_SIZE[struct_type.lower()]
    data = struct.unpack(f'<{count}{struct_type}', data[bufferView.byteOffset + accessor.byteOffset:bufferView.byteOffset + accessor.byteOffset + count * value_size])
    data = np.array(data, dtype=struct_type)
    
    return data

def loadGLTF(filename):
    boundingBox, vertexDataList, normalDataList, texCoordDataList, jointDataList, weightDataList, indexDataList, nodeHierarchy = [], [], [], [], [], [], [], []
    hasNormals, hasTextures, hasJoints, pose = 0, 0, 0, 0

    gltf = GLTF2().load(filename)

    scene = gltf.scenes[gltf.scene]
    animations = gltf.animations
    parentList = createParentlist(gltf)

    for node in scene.nodes:
        nodeHierarchy.append(createNodeHierarchy(gltf, node))

    for nodes in nodeHierarchy:
        for nodeNr in nodes:

            node = gltf.nodes[nodeNr]
            
            if node.mesh == None: continue
            mesh = gltf.meshes[node.mesh]
            
            if mesh.primitives[0].attributes.NORMAL:     hasNormals = 1
            if mesh.primitives[0].attributes.TEXCOORD_0: hasTextures = 1
            if node.skin is not None:
                
                hasJoints = 1
                skin = gltf.skins[node.skin]
                
                timeData = len(readAccesor(gltf, gltf.accessors[animations[0].samplers[animations[0].channels[0].sampler].input]))
                inverseBindData = readAccesor(gltf, gltf.accessors[skin.inverseBindMatrices])
                inverseBindData = [inverseBindData[i:i + 16].reshape(4,4) for i in range(0, len(inverseBindData), 16)]
                
                transformMatrix = [[np.identity(4) for pose in range(timeData)] for i in range(3)]
                finalMatrices, transformMatrices = [[[[np.identity(4) for node in range(len(gltf.nodes))] for pose in range(timeData)] for animation in range(len(animations))] for i in range(2)]
                
                for animation in range(len(animations)):
                    
                    createTransformMatrices(gltf, transformMatrices, timeData, animations, animation, transformMatrix)
                    createAnimation(transformMatrices, finalMatrices, timeData, skin, animation, parentList, inverseBindData)
                
                #set uniform to finalMatrices[0][0]
            
            for primitive in mesh.primitives:
                
                vertexAccessor = gltf.accessors[primitive.attributes.POSITION]
                indexAccessor = gltf.accessors[primitive.indices]
                if hasNormals:
                    normalAccessor = gltf.accessors[primitive.attributes.NORMAL]
                if hasTextures:
                    texCoordAccessor = gltf.accessors[primitive.attributes.TEXCOORD_0]
                if hasJoints:
                    jointAccesor = gltf.accessors[primitive.attributes.JOINTS_0]
                    weightAccesor = gltf.accessors[primitive.attributes.WEIGHTS_0]
                
                boundingBox.append(vertexAccessor.min - np.array((0.7,0.1,0.7)))
                boundingBox.append(vertexAccessor.max + np.array((0.7,0.1,0.7)))
                
                vertexData = readAccesor(gltf, vertexAccessor)
                indexData = readAccesor(gltf, indexAccessor)
                if hasNormals:
                    normalData = readAccesor(gltf, normalAccessor)
                if hasTextures:
                    texCoordData = readAccesor(gltf, texCoordAccessor)
                if hasJoints:
                    jointData = readAccesor(gltf, jointAccesor)
                    weightData = readAccesor(gltf, weightAccesor)
                
                vertexDataList.append(vertexData)
                indexDataList.append(indexData)
                if hasNormals:
                    normalDataList.append(normalData)
                if hasTextures:
                    texCoordDataList.append(texCoordData)
                if hasJoints:
                    jointDataList.append(jointData)
                    weightDataList.append(weightData)
    
    listLenght = len(vertexDataList)
    
    np.savetxt(filename + "Data", [hasNormals, hasTextures, hasJoints, listLenght], fmt='%d')
    np.savetxt(filename + "BoundingBox", boundingBox, fmt='%d')
    for i in range(listLenght):
        
        np.savetxt(filename + "VertexDataList" + str(i), vertexDataList[i], fmt='%d')
        if hasNormals:
            np.savetxt(filename + "VormalDataList" + str(i), normalDataList[i], fmt='%d')
        if hasTextures:
            np.savetxt(filename + "TexCoordDataList" + str(i), texCoordDataList[i], fmt='%d')
        if hasJoints:
            np.savetxt(filename + "JointDataList" + str(i), jointDataList[i], fmt='%d')
            np.savetxt(filename + "WeightDataList" + str(i), weightDataList[i], fmt='%d')
        np.savetxt(filename + "TndexDataList" + str(i), indexDataList[i], fmt='%d')