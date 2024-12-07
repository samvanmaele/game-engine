import math
import pygame
import pygame.gfxdraw
import os
import sys

def get_path(filename):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, filename)
    else:
        return filename

pygame.init()

gamePaused = True
gameRunning = False
gameMainMenu = False
gameOptions = False
editingWindow = False
customWindow = False
customFPS = False
gameWin = False

clock = pygame.time.Clock()
frameRate = 60
FPSLimit = True

horizontalModifier = 0
verticalModifier = 0
DepthModifier = 0

walkspeedModifier = 1

key = pygame.key.get_pressed()
input_map = {'right': pygame.K_d, 'left': pygame.K_a, 'forwards': pygame.K_w, 'backwards': pygame.K_s, 'jump': pygame.K_SPACE, 'sprint': pygame.K_LSHIFT}
prevClick = False

jumpHeight = 30
jumpVelocity = jumpHeight
gravity = 1
jumping = False

moveGrassX = 0
moveGrassY = 0

HEIGHT, WIDTH = 1080, 1920
WIN = pygame.display.set_mode((WIDTH, HEIGHT))

ANGLEx = math.radians(0)
ANGLEy = math.radians(-30)

playerAngleX = 0
playerAngleDif = 0

objectListCube = []
objectListTemple = []

buttonTexture = pygame.image.load(get_path("gfx/button.png")).convert_alpha()

UIfont = pygame.font.Font(None, 36)
buttonFont = pygame.font.Font(None, 50)

textColour = (255, 255, 255)
textColourHover = (255, 255, 0)
textColourSelected = (50, 50, 50)
typeColour = (0, 0, 0)

windowSizeText = buttonFont.render("window zise", True, textColour)
frameRateText = buttonFont.render("frame rate limit", True, textColour)
controlsText = buttonFont.render("controls", True, textColour)
widthText = buttonFont.render("width", True, textColour)
heightText = buttonFont.render("height", True, textColour)
FPSText = buttonFont.render("FPS", True, textColour)
introtext = buttonFont.render("You gave out a free code and lost 150M.", True, textColour)
introtext2 = buttonFont.render("To pay of your debts you need to collect 25 Evil Neuro plushies.", True, textColour)

tutelColour = (50,230,50)
evilNeuroPlushColour = (255,200,200)

player2D = []
tutel = [[130,0,-150], #leg
          [130,0,150], #arm
          [10,30,-100], #in leg
          [40,30,100], #shoulder
          [60,80,-30], #hip deep
          [60,75,30], #elbow deep
          [80,0,-50], #hip undeep
          [80,5,50]] #elbow undeep
for playerMirror in range(len(tutel)):
    tutel += [[-tutel[playerMirror][0], tutel[playerMirror][1], tutel[playerMirror][2]]]

evilNeuroPlush = [[10,16,30], #leg
                  [22,36,10], #arm
                  [2,16,0], #in leg
                  [4,50,0], #shoulder
                  [8,26,6], #hip deep
                  [8,42,5], #elbow deep
                  [12,26,-6], #hip undeep
                  [8,42,-5]] #elbow undeep
for evilNeuroPlushMirror in range(len(tutel)):
    evilNeuroPlush += [[-evilNeuroPlush[evilNeuroPlushMirror][0], evilNeuroPlush[evilNeuroPlushMirror][1], evilNeuroPlush[evilNeuroPlushMirror][2]]]
plushCollected = [False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]

legZ = 0
legZModifier = 10
oldLegZ1 = tutel[0][2]
oldLegZ2 = tutel[8][2]
oldArmZ1 = tutel[1][2]
oldArmZ2 = tutel[9][2]

def Turn3Dx(x, y, z, angle, xmod, ymod, zmod):
    x2 = x - xmod
    y2 = y - ymod
    z2 = z - zmod
    x3 =  x2*math.cos(angle) + z2*math.sin(angle)
    z3 = -x2*math.sin(angle) + z2*math.cos(angle)
    returnTurn3D = [x3, y2, z3]
    return returnTurn3D

def Turn3Dy(x, y, z, angle):
    y2 =  y*math.cos(angle) + z*math.sin(angle)
    z2 = -y*math.sin(angle) + z*math.cos(angle)
    returnTurn3D = [x, y2, z2]
    return returnTurn3D

def Convert3Dto2D(x, y, z):
    if z <= -1000:
        x = 1001*x+WIDTH/2
        y = 1001*y+HEIGHT/2
    else:
        x = (1000*x)/(z+1000)+WIDTH/2
        y = (1000*y)/(z+1000)+HEIGHT/2
    return [(x, y), z]

def Turn3Dto2D(angleX, angleY, coordinatelist, xmod, ymod, zmod):
    returnTurn3Dto2D = coordinatelist[:]
    for listchooser1 in range(len(coordinatelist)):
        x1, y1, z1 = coordinatelist[listchooser1]
        x2, y2, z2 = Turn3Dx(x1, -y1, z1, angleX, xmod, ymod, zmod)
        x3, y3, z3 = Turn3Dy(x2, y2, z2, angleY)
        returnTurn3Dto2D[listchooser1] = Convert3Dto2D(x3, y3, z3)
    return returnTurn3Dto2D

def angleCorection(oldAngle, goalAngle):
    deltaAngle = goalAngle - oldAngle
    if deltaAngle < -math.pi:
        deltaAngle += math.pi*2
    if deltaAngle > math.pi:
        oldAngle -= 3*deltaAngle/FPS
    else:
        oldAngle += 3*deltaAngle/FPS
    return oldAngle%(math.pi*2)

colisionObjects = []

def createObjectCube(width, height, depth, x, y, z, angle):
    global objectListCube, colisionObjects
    colisionObjects += [pygame.Rect(x, z, width, depth)]
    for objectWidth in (0+x, width+x):
        for objectHeight in (0+y, height+y):
            for objectDepth in (0+z, depth+z):
                objectListCube += [Turn3Dx(objectWidth, objectHeight, objectDepth, angle, 0, 0, 0)]

for mapLines in range(50):
    mapDepth = (mapLines-10)*1000
    if mapDepth == -10000:
        mapWidth = (-10000, -9000, -8000, -7000, -6000, -5000, -4000, -3000, -2000, -1000, 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000)
    elif mapDepth == -9000:
        mapWidth = (-10000, -6000, 0, 6000, 9000)
    elif mapDepth == -8000:
        mapWidth = (-10000, -8000, -7000, -6000, -4000, -3000, -2000, -1000, 0, 2000, 3000, 4000, 6000, 7000, 9000)
    elif mapDepth == -7000: 
        mapWidth = (-10000, -8000, -6000, -4000, -2000, 4000, 7000, 9000)
    elif mapDepth == -6000:
        mapWidth = (-10000, -6000, -2000, 0, 1000, 2000, 4000, 5000, 7000, 9000)
    elif mapDepth == -5000:
        mapWidth = (-10000, -8000, -7000, -6000, -5000, -4000, 2000, 3000, 7000, 9000)
    elif mapDepth == -4000:
        mapWidth = (-10000, -7000, -4000, -2000, 0, 3000, 4000, 5000, 7000, 9000)
    elif mapDepth == -3000:
        mapWidth = (-10000, -8000, -7000, -6000, -2000, 0, 1000, 7000, 9000)
    elif mapDepth == -2000:
        mapWidth = (-10000, -8000, -4000, -3000, -2000, 1000, 2000, 3000, 5000, 6000, 7000, 9000)
    elif mapDepth == -1000:
        mapWidth = (-10000, -8000, -6000, 1000, 3000,  5000, 9000)
    elif mapDepth == 0:
        mapWidth = (-10000, -6000, -4000, -3000, -2000, 3000, 5000, 7000, 9000)
    elif mapDepth == 1000:
        mapWidth = (-10000, -8000, -6000, -2000, 1000, 3000, 4000, 5000, 7000, 9000)
    elif mapDepth == 2000:
        mapWidth = (-10000, -8000, -6000, -5000, -4000, -2000, 1000, 7000, 9000)
    elif mapDepth == 3000:
        mapWidth = (-10000, -4000, -3000, -2000, 1000, 2000, 3000, 5000, 7000, 9000)
    elif mapDepth == 4000:
        mapWidth = (-10000, -9000, -8000, -6000, -5000, -4000, -2000, 1000, 5000, 6000, 9000)
    elif mapDepth == 5000:
        mapWidth = (-10000, -2000, 4000, 8000, 9000)
    elif mapDepth == 6000:
        mapWidth = (-10000, -8000, -7000, -6000, -5000, -4000, -3000, -2000, 1000, 3000, 4000, 5000, 7000, 8000, 9000)
    elif mapDepth == 7000:
        mapWidth = (-10000, -5000, -4000, 1000, 2000, 4000, 9000)
    elif mapDepth == 8000:
        mapWidth = (-10000, -9000, -8000, -7000, -2000, 1000, 4000, 6000, 8000, 9000)
    elif mapDepth == 9000:
        mapWidth = (-10000, -5000, -4000, -3000, -2000, 1000, 3000, 4000, 6000, 7000, 9000)
    elif mapDepth == 10000:
        mapWidth = (-10000, -9000, -7000, -5000, -2000, 1000, 3000, 7000, 9000)
    elif mapDepth == 11000:
        mapWidth = (-10000, -6000, -4000, -2000, 1000, 3000, 4000, 5000, 7000, 9000)
    elif mapDepth == 12000:
        mapWidth = (-10000, -9000, -8000, -6000, 1000, 9000)
    elif mapDepth == 13000:
        mapWidth = (-10000, -4000, -2000, 1000, 2000, 3000, 4000, 6000, 7000, 9000)
    elif mapDepth == 14000:
        mapWidth = (-10000, -8000, -7000, -6000, -4000, -3000, -2000, 1000, 6000, 9000)
    elif mapDepth == 15000:
        mapWidth = (-10000, -8000, -2000, 1000, 3000, 4000, 5000, 6000, 8000, 9000)
    elif mapDepth == 16000:
        mapWidth = (-10000, -8000, -6000, -5000, -4000, -2000, 1000, 3000, 9000)
    elif mapDepth == 17000:
        mapWidth = (-10000, -8000, -7000, -6000, -4000, 1000, 2000, 3000, 5000, 6000, 7000, 8000, 9000)
    elif mapDepth == 18000:
        mapWidth = (-10000, -4000, -3000, -2000, 3000, 4000, 5000, 9000)
    elif mapDepth == 19000:
        mapWidth = (-10000, -8000, -6000, 1000, 7000, 9000)
    elif mapDepth == 20000:
        mapWidth = (-10000, -8000, -7000, -6000, -4000, -3000, -2000, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 9000)
    elif mapDepth == 21000:
        mapWidth = (-10000, -8000, -4000, 3000, 7000, 9000)
    elif mapDepth == 22000:
        mapWidth = (-10000, -8000, -6000, -5000, -4000, 3000, 4000, 5000, 9000)
    elif mapDepth == 23000:
        mapWidth = (-10000, -8000, -6000, 5000, 7000, 9000)
    elif mapDepth == 24000:
        mapWidth = (-10000, -9000, -8000, -6000, 7000, 8000, 9000)
    elif mapDepth == 25000:
        mapWidth = (-10000, 5000, 7000, 9000)
    elif mapDepth == 26000:
        mapWidth = (-10000, -8000, -7000, -6000, 5000, 9000)
    elif mapDepth == 27000:
        mapWidth = (-10000, -6000, 5000, 6000, 7000, 9000)
    elif mapDepth == 28000:
        mapWidth = (-10000, -9000, -8000, -6000, 5000, 9000)
    elif mapDepth == 29000:
        mapWidth = (-10000, -8000, 5000, 7000, 9000)
    elif mapDepth == 30000:
        mapWidth = (-10000, -8000, -6000, 7000, 9000)
    elif mapDepth == 31000:
        mapWidth = (-10000, -6000, 5000, 7000, 8000, 9000)
    elif mapDepth == 32000:
        mapWidth = (-10000, -9000, -8000, -6000, 5000, 9000)
    elif mapDepth == 33000:
        mapWidth = (-10000, -6000, 5000, 6000, 8000, 9000)
    elif mapDepth == 34000:
        mapWidth = (-10000, -8000, -7000, -6000, 9000)
    elif mapDepth == 35000:
        mapWidth = (-10000, -8000, -6000, -5000, -3000, -2000, -1000, 1000, 3000, 4000, 5000, 6000, 7000, 9000)
    elif mapDepth == 36000:
        mapWidth = (-10000, -8000, -3000, 1000, 6000, 9000)
    elif mapDepth == 37000:
        mapWidth = (-10000, -8000, -7000, -6000, -5000, -3000, -1000, 0, 1000, 3000, 4000, 5000, 6000, 8000, 9000)
    elif mapDepth == 38000:
        mapWidth = (-10000, -3000, 9000)
    elif mapDepth == 39000:
        mapWidth = (-10000, -9000, -8000, -7000, -6000, -5000, -4000, -3000, -2000, -1000, 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000)
    for mapBuilder in mapWidth:
        createObjectCube(1000, 1000, 1000, mapBuilder, 0, mapDepth, 0)

def createHarrisonTemple(width, height, depth, x, y, z, angle):
    global objectListTemple, colisionObjects
    colisionObjects += [pygame.Rect(x, z, width, depth)]
    for objectWidth in (0+x, width+x):
        for objectHeight in (0+y, height+y):
            for objectDepth in (0+z, depth+z):
                objectListTemple += [Turn3Dx(objectWidth, objectHeight, objectDepth, angle, 0, 0, 0)]

for harrisonTempleBaseBuilderX in (-4000,-3000,-2000,-1000,0,1000,2000,3000):
    for harrisonTempleBaseBuilderY in (26000,27000,28000,29000,30000,31000,32000,33000):
        createHarrisonTemple(1000, 1000, 1000, harrisonTempleBaseBuilderX, 0, harrisonTempleBaseBuilderY, 0)
for harrisonTempleBaseBuilderX in (-3000,-2000,-1000,0,1000,2000):
    for harrisonTempleBaseBuilderY in (27000,28000,29000,30000,31000,32000):
        createHarrisonTemple(1000, 1000, 1000, harrisonTempleBaseBuilderX, 1000, harrisonTempleBaseBuilderY, 0)
for harrisonTempleBaseBuilderX in (-2000,-1000,0,1000):
    for harrisonTempleBaseBuilderY in (28000,29000,30000,31000):
        createHarrisonTemple(1000, 1000, 1000, harrisonTempleBaseBuilderX, 2000, harrisonTempleBaseBuilderY, 0)
for harrisonTempleBaseBuilderX in (-1000,0):
    for harrisonTempleBaseBuilderY in (29000,30000):
        createHarrisonTemple(1000, 1000, 1000, harrisonTempleBaseBuilderX, 3000, harrisonTempleBaseBuilderY, 0)

def movement():
    global horizontalModifier, DepthModifier, ANGLEx, playerAngleX, run, moveGrassX, moveGrassY, jumping
    key = pygame.key.get_pressed()
    walkspeedModifier = 1+key[input_map['sprint']]
    if  key[input_map['forwards']] and key[input_map['right']]:
        moving = True
        moveDirection = math.pi/4
    elif key[input_map['forwards']] and key[input_map['left']]:
        moving = True
        moveDirection = 7*math.pi/4 
    elif key[input_map['backwards']] and key[input_map['right']]:
        moving = True
        moveDirection = 3*math.pi/4 
    elif key[input_map['backwards']] and key[input_map['left']]:
        moving = True
        moveDirection = 5*math.pi/4
    elif key[input_map['forwards']]:
        moving = True
        moveDirection = 0
    elif key[input_map['right']]:
        moving = True
        moveDirection = math.pi/2 
    elif key[input_map['left']]:
        moving = True
        moveDirection = 3*math.pi/2
    elif key[input_map['backwards']]:
        moving = True
        moveDirection = math.pi
    else:
        moving = False
    if moving:
        playerColisionRect1 = pygame.Rect(playerRect[0]-(1000 * walkspeedModifier * math.cos(moveDirection) * math.sin(ANGLEx) - 1000 * walkspeedModifier * math.sin(moveDirection) * math.cos(ANGLEx) )/ FPS, playerRect[1], playerRect[2], playerRect[3])
        if not playerColisionRect1.collidelistall(colisionObjects):
            horizontalModifier -= (1000 * walkspeedModifier * math.cos(moveDirection) * math.sin(ANGLEx) - 1000 * walkspeedModifier * math.sin(moveDirection) * math.cos(ANGLEx) )/ FPS
        playerColisionRect2 = pygame.Rect(playerRect[0], playerRect[1] + (1000 * walkspeedModifier * math.cos(moveDirection) * math.cos(ANGLEx) + 1000 * walkspeedModifier * math.sin(moveDirection) * math.sin(ANGLEx))/FPS, playerRect[2], playerRect[3])
        if not playerColisionRect2.collidelistall(colisionObjects):
            DepthModifier += (1000 * walkspeedModifier * math.cos(moveDirection) * math.cos(ANGLEx) + 1000 * walkspeedModifier * math.sin(moveDirection) * math.sin(ANGLEx))/FPS
        playerAngleX = angleCorection(playerAngleX, moveDirection)
        moveGrassY += 10
        moveLeg()
    if key[input_map['jump']]:
        jumping = True
    if key[pygame.K_ESCAPE]:
        run = False

def jump():
    global jumping, verticalModifier, jumpVelocity, gravity
    if jumping:
        verticalModifier -= 100*jumpVelocity/FPS
        jumpVelocity -= 100*gravity/FPS
        if verticalModifier >= 0:
            jumpVelocity = jumpHeight
            verticalModifier = 0
            jumping = False

def moveMouse():
    global ANGLEx, ANGLEy
    horizontalMouse, verticalMouse = pygame.mouse.get_rel()
    ANGLEx -= horizontalMouse/1000
    if -math.pi/2 < ANGLEy-verticalMouse/1000 < math.pi/2:
        ANGLEy -= verticalMouse/1000

def moveLeg():
    global legZModifier, legZ
    if abs(legZ) >= 50:
        legZModifier = -legZModifier
    legZ += 25*legZModifier*walkspeedModifier/FPS
    tutel[0][2] = oldLegZ1 + legZ
    tutel[8][2] = oldLegZ2 - legZ
    tutel[1][2] = oldArmZ1 - legZ
    tutel[9][2] = oldArmZ2 + legZ

def drawPlayerRibs(model, colour):
    for playerListChooser1 in (0,1):
        pygame.draw.line(WIN, colour, player2D[2+playerListChooser1][0], player2D[10+playerListChooser1][0], 5)
        for playerListChooser2 in (0,2):
            pygame.draw.line(WIN, colour, player2D[4+playerListChooser1+playerListChooser2][0], player2D[12+playerListChooser1+playerListChooser2][0], 5)
            pygame.draw.line(WIN, colour, player2D[8*playerListChooser1-playerListChooser2+6][0], player2D[8*playerListChooser1-playerListChooser2+7][0], 5)
            pygame.draw.line(WIN, colour, player2D[playerListChooser1+4*playerListChooser2+4][0], player2D[playerListChooser1+4*playerListChooser2+6][0], 5)
            for playerListChooser3 in (2,4,6):
                pygame.draw.line(WIN, colour, player2D[playerListChooser1+4*playerListChooser2][0], player2D[playerListChooser1+4*playerListChooser2+playerListChooser3][0], 5)
            for playerListChooser3 in (0,2):
                pygame.draw.line(WIN, colour, player2D[playerListChooser1+4*playerListChooser2+2][0], player2D[playerListChooser1+4*playerListChooser2+playerListChooser3+4][0], 5)
    if model == tutel:
        pygame.draw.circle(WIN, colour, ((player2D[3][0][0]+player2D[11][0][0])/2, (player2D[3][0][1]+player2D[11][0][1])/2-40), 40, 5)
    else:
        pygame.draw.circle(WIN, colour, ((player2D[3][0][0]+player2D[11][0][0])/2, (player2D[3][0][1]+player2D[11][0][1])/2-10), 10, 5)
    
def drawObjectCubeRibs():
    coordinates2D = Turn3Dto2D(ANGLEx, ANGLEy ,objectListCube, horizontalModifier, verticalModifier, DepthModifier)
    objectNr = len(coordinates2D)/8
    for multiObject in range(int(objectNr)):
        for listchooser1 in (0,1,4,5):
            if coordinates2D[listchooser1+8*multiObject][1] > -1000 or coordinates2D[listchooser1+8*multiObject+2][1] > -1000:
                pygame.draw.line(WIN, (0,0,255), coordinates2D[listchooser1+8*multiObject][0], coordinates2D[listchooser1+8*multiObject+2][0], 5)  
        for listchooser2 in (0,1,2,3):
            if coordinates2D[listchooser2*2+8*multiObject][1] > -1000 or coordinates2D[listchooser2*2+8*multiObject+1][1] > -1000:
                pygame.draw.line(WIN, (0,0,255), coordinates2D[listchooser2*2+8*multiObject][0], coordinates2D[listchooser2*2+8*multiObject+1][0], 5)
            if coordinates2D[listchooser2+8*multiObject][1] > -1000 or coordinates2D[listchooser2+8*multiObject+4][1] > -1000:
                pygame.draw.line(WIN, (0,0,255), coordinates2D[listchooser2+8*multiObject][0], coordinates2D[listchooser2+8*multiObject+4][0], 5)

def drawObjectHarrisonTemple():
    coordinates2D = Turn3Dto2D(ANGLEx, ANGLEy ,objectListTemple, horizontalModifier, verticalModifier, DepthModifier)
    objectNr = len(coordinates2D)/8
    for multiObject in range(int(objectNr)):
        for listchooser1 in (0,1,4,5):
            if coordinates2D[listchooser1+8*multiObject][1] > -1000 or coordinates2D[listchooser1+8*multiObject+2][1] > -1000:
                pygame.draw.line(WIN, (255,255,0), coordinates2D[listchooser1+8*multiObject][0], coordinates2D[listchooser1+8*multiObject+2][0], 5)  
        for listchooser2 in (0,1,2,3):
            if coordinates2D[listchooser2*2+8*multiObject][1] > -1000 or coordinates2D[listchooser2*2+8*multiObject+1][1] > -1000:
                pygame.draw.line(WIN, (255,255,0), coordinates2D[listchooser2*2+8*multiObject][0], coordinates2D[listchooser2*2+8*multiObject+1][0], 5)
            if coordinates2D[listchooser2+8*multiObject][1] > -1000 or coordinates2D[listchooser2+8*multiObject+4][1] > -1000:
                pygame.draw.line(WIN, (255,255,0), coordinates2D[listchooser2+8*multiObject][0], coordinates2D[listchooser2+8*multiObject+4][0], 5)

def displayFPS():
    global FPS, FPStext
    FPStext = UIfont.render(str(round(FPS)) + " FPS", True, (255, 255, 255))
    WIN.blit(FPStext, (10, 10))
    
def button(position, mouseposition, text, image, function, parameter1, parameter2):
    global event, prevClick, WIN, WIDTH, HEIGHT, frameRate, gamePaused, gameOptions, editingWindow, customWindow, editingFPS, customFPS, gameMainMenu, gameWin
    image = pygame.transform.smoothscale(image, (350,150))
    buttonWidth, buttonHeight = image.get_width(), image.get_height()
    
    if function == "window" and parameter1 == HEIGHT and parameter2 == WIDTH:
        buttonText = buttonFont.render(text, True, textColourSelected)
        
    elif function == "window" and parameter1 == "custom" and customWindow == True:
        buttonText = buttonFont.render(text + ": " + str(HEIGHT) + " x " + str(WIDTH), True, textColourSelected)
        
    elif function == "frameRate" and parameter1 == frameRate:
        buttonText = buttonFont.render(text, True, textColourSelected)
    
    elif function == "frameRate" and parameter1 == "custom" and customFPS == True:
        buttonText = buttonFont.render(text + ": " + str(frameRate), True, textColourSelected)
    
    elif function == "controls":
        buttonText = buttonFont.render(text + ": " + pygame.key.name(parameter1), True, textColour)
    
    else:
        buttonText = buttonFont.render(text, True, textColour)
        
    if position[0]-buttonWidth/2+20 < mouseposition[0] < position[0]+buttonWidth/2-20 and position[1]-buttonHeight/2+20 < mouseposition[1] < position[1]+buttonHeight/2-20:
        if function == "frameRate" and parameter1 == "custom" and customFPS:
            buttonText = buttonFont.render(text + ": " + str(frameRate), True, textColourHover)
        elif function == "window" and parameter1 == "custom" and customWindow == True:
            buttonText = buttonFont.render(text + ": " + str(HEIGHT) + " x " + str(WIDTH), True, textColourHover)
        elif function == "controls":
            buttonText = buttonFont.render(text + ": " + pygame.key.name(parameter1), True, textColourHover)
        else:
            buttonText = buttonFont.render(text, True, textColourHover)
        if event.type == pygame.MOUSEBUTTONDOWN and not prevClick:
            if function == "game" or function == "options":
                gamePaused, gameMainMenu = False, False
                parameter1()
            elif function == "quit":
                gameWin = False
                gamePaused = False
                gameMainMenu = False
                quit()
            elif function == "window":
                if parameter1 == "custom":
                    customWindowSize()
                    gameOptions = False
                    customWindow = False
                else:
                    HEIGHT, WIDTH = parameter1, parameter2
                    WIN = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
                customWindow = False
            elif function == "newWindowSize":
                if parameter1 == "":
                    parameter1 = WIDTH
                if parameter2 == "":
                    parameter2 = HEIGHT
                HEIGHT, WIDTH = int(parameter1), int(parameter2)
                WIN = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
                editingWindow = False
                if ((HEIGHT == 1080 and WIDTH == 1920) or (HEIGHT == 1440 and WIDTH == 2560) or (HEIGHT == 2160 and WIDTH == 3840)):
                    customWindow = False
                else:
                    customWindow = True
                options()
            elif function == "frameRate":
                if parameter1 == "custom":
                    customFrameRate()
                    gameOptions = False
                else:
                    frameRate = parameter1
                    customFPS = False
            elif function == "newFPS":
                if parameter1 == "":
                    parameter1 = frameRate
                frameRate = int(parameter1)
                editingFPS = False
                if frameRate == 30 or frameRate == 60 or frameRate == 120:
                    customFPS = False
                else:
                    customFPS = True
                options()
            elif function == "done":
                if parameter2 == "gameOptions":
                    gameOptions = False
                    paused()
            elif function == "controls":
                parameter2(text)
            elif function == "gameMainMenu":
                mainMenu()
                gameWin = False
            prevClick = True
        else:
            prevClick = False

    textWidth, textHeight = buttonText.get_width(), buttonText.get_height()
    
    WIN.blit(image, (position[0]-buttonWidth/2,position[1]-buttonHeight/2))
    WIN.blit(buttonText, (position[0]-textWidth/2,position[1]-textHeight/2))

def game():
    global WIN, FPS, key, player2D, walkspeedModifier, event, gameRunning, tutelColour, playerRect, plushCollected

    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)

    gameRunning = True
    while gameRunning:
    
        WIN.fill((0,0,0))
        
        if FPSLimit:
            clock.tick(frameRate)
        else:
            clock.tick()
        FPS = clock.get_fps()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gameRunning = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    paused()
                    gameRunning = False
        
        playerRect = [-100+horizontalModifier, -100+DepthModifier, 200, 200]
        
        movement()
        jump()
        moveMouse()
        for evilNeuroPlushHider in ((-500,-8500,0),(-6500,-6500,1),(8500,-5500,2),(3500,-5500,3),(-5500,-3500,4),
                                    (2500,-2500,5),(-2500,2500,6),(-6500,3500,7),(6500,3500,8),(-5500,7500,9),
                                    (3500,7500,10),(8500,7500,11),(-3500,10500,12),(4500,10500,13),(-4500,15500,14),
                                    (2500,16500,15),(8500,16500,16),(-6500,19500,17),(4500,21500,18),(-3500,24500,19),
                                    (-8500,29500,20),(8500,30500,21),(-6500,35500,22),(-1500,36500,23),(5500,36500,24)):
            if plushCollected[evilNeuroPlushHider[2]]:
                player2D = Turn3Dto2D(playerAngleX, ANGLEy, evilNeuroPlush, 0, 80, 20*evilNeuroPlushHider[2])
                drawPlayerRibs(evilNeuroPlush, evilNeuroPlushColour)
            else:
                player2D = Turn3Dto2D(ANGLEx, ANGLEy, evilNeuroPlush, horizontalModifier-evilNeuroPlushHider[0], verticalModifier, DepthModifier-evilNeuroPlushHider[1])
                drawPlayerRibs(evilNeuroPlush, evilNeuroPlushColour)
                plushRect = pygame.Rect(evilNeuroPlushHider[0]-20, evilNeuroPlushHider[1]-20, 40, 40)
                plushCollected[evilNeuroPlushHider[2]] = pygame.Rect.colliderect(pygame.Rect(playerRect), plushRect)
        
        drawObjectCubeRibs()
        drawObjectHarrisonTemple()
        player2D = Turn3Dto2D(playerAngleX, ANGLEy, tutel, 0, 0, 0)
        drawPlayerRibs(tutel, tutelColour)
        displayFPS()
        displayPlushcount()
    
        pygame.display.flip()
        
def mainMenu():
    global event, gameMainMenu
    
    pygame.mouse.set_visible(True)
    pygame.event.set_grab(False)
    
    WIN.fill((0,0,0))
    
    gameMainMenu = True
    while gameMainMenu:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gameMainMenu = False
                
        mousePosition = pygame.mouse.get_pos()
        
        WIN.fill((0,0,0))
                
        button((WIDTH/8, 5*HEIGHT/8), mousePosition, "start game", buttonTexture, "game", game, "gameMainMenu")
        button((WIDTH/8, 3*HEIGHT/4), mousePosition, "options", buttonTexture, "options", options, "gameMainMenu")
        button((WIDTH/8, 7*HEIGHT/8), mousePosition, "quit", buttonTexture, "quit", 0, "gameMainMenu")
        
        WIN.blit(introtext, ((WIDTH-introtext.get_width())/2, HEIGHT/8))
        WIN.blit(introtext2, ((WIDTH-introtext2.get_width())/2, HEIGHT/4))
        
        pygame.display.flip()

def paused():
    global WIN, WIDTH, HEIGHT, event, gamePaused
    
    pygame.mouse.set_visible(True)
    pygame.event.set_grab(False)
    
    gamePaused = True
    while gamePaused:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gamePaused = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    game()
                    gamePaused = False
                
        clock.tick(20)

        mousePosition = pygame.mouse.get_pos()
        
        WIN.fill((0,0,0))
        
        button((WIDTH/2, HEIGHT/3), mousePosition, "resume", buttonTexture, "game", game, "gamePaused")
        button((WIDTH/2, HEIGHT/2), mousePosition, "options", buttonTexture, "options", options, "gamePaused")
        button((WIDTH/2, 2*HEIGHT/3), mousePosition, "quit", buttonTexture, "quit", 0, "gamePaused")
        
        pygame.display.flip()

def options():
    global WIN, WIDTH, HEIGHT, event, gameOptions
    
    pygame.mouse.set_visible(True)
    pygame.event.set_grab(False)
    
    gameOptions = True
    while gameOptions:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gameOptions = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    paused()
                    gameOptions = False
        
        clock.tick(20)

        mousePosition = pygame.mouse.get_pos()
        
        WIN.fill((0,0,0))
        
        WIN.blit(windowSizeText, (WIDTH/5-windowSizeText.get_width()/2,HEIGHT/8-windowSizeText.get_height()/2))
        WIN.blit(frameRateText, (2*WIDTH/5-frameRateText.get_width()/2,HEIGHT/8-frameRateText.get_height()/2))
        WIN.blit(controlsText, (3.5*WIDTH/5-controlsText.get_width()/2,HEIGHT/8-controlsText.get_height()/2))
        
        button((3*WIDTH/5, HEIGHT/4), mousePosition, "forwards", buttonTexture, "controls", input_map['forwards'], keyRemap)
        button((3*WIDTH/5, 3*HEIGHT/8), mousePosition, "backwards", buttonTexture, "controls", input_map['backwards'], keyRemap)
        button((3*WIDTH/5, HEIGHT/2), mousePosition, "left", buttonTexture, "controls", input_map['left'], keyRemap)
        button((3*WIDTH/5, 5*HEIGHT/8), mousePosition, "right", buttonTexture, "controls", input_map['right'], keyRemap)
        button((4*WIDTH/5, HEIGHT/4), mousePosition, "jump", buttonTexture, "controls", input_map['jump'], keyRemap)
        button((4*WIDTH/5, 3*HEIGHT/8), mousePosition, "sprint", buttonTexture, "controls", input_map['sprint'], keyRemap)
        
        button((WIDTH/5, HEIGHT/4), mousePosition, "1080p", buttonTexture, "window", 1080, 1920)
        button((WIDTH/5, 3*HEIGHT/8), mousePosition, "1440p", buttonTexture, "window", 1440, 2560)
        button((WIDTH/5, HEIGHT/2), mousePosition, "4K", buttonTexture, "window", 2160, 3860)
        button((WIDTH/5, 5*HEIGHT/8), mousePosition, "custom", buttonTexture, "window", "custom", 0)
        
        button((2*WIDTH/5, HEIGHT/4), mousePosition, "30 FPS", buttonTexture, "frameRate", 30, 0)
        button((2*WIDTH/5, 3*HEIGHT/8), mousePosition, "60 FPS", buttonTexture, "frameRate", 60, 0)
        button((2*WIDTH/5, HEIGHT/2), mousePosition, "120 FPS", buttonTexture, "frameRate", 120, 0)
        button((2*WIDTH/5, 5*HEIGHT/8), mousePosition, "custom", buttonTexture, "frameRate", "custom", 0)
        
        button((WIDTH/2, 7*HEIGHT/8), mousePosition, "done", buttonTexture, "done", 0, "gameOptions")
        
        pygame.display.flip()

def customWindowSize():
    global WIN, WIDTH, HEIGHT, event, editingWindow, gameOptions
    text1 = str(HEIGHT)
    text2 = str(WIDTH)
    text1Selected = False
    text2Selected = False
    textBackgroundColour1 = (255, 255, 255)
    textBackgroundColour2 = (255, 255, 255)
    editingWindow = True
    while editingWindow:
        WIN.fill((0,0,0))
        mousePosition = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                editingWindow = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    editingWindow = False
                    options()
                
            if text1Selected:
                textBackgroundColour1 = (255, 0, 0)
                if event.type == pygame.MOUSEBUTTONDOWN and not (WIDTH/3-60 < mousePosition[0] < WIDTH/3+60 and 3*HEIGHT/8-20 < mousePosition[1] < 3*HEIGHT/8+20):
                    text1Selected = False
                    textBackgroundColour1 = (255, 255, 255)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        text1Selected = False
                        textBackgroundColour1 = (255, 255, 255)
                    elif event.key == pygame.K_BACKSPACE:
                        text1 = ""
                    elif event.unicode.isdigit() and len(text1) < 4:
                        text1 += event.unicode
            elif text2Selected:
                textBackgroundColour2 = (255, 0, 0)
                if event.type == pygame.MOUSEBUTTONDOWN and not (2*WIDTH/3-60 < mousePosition[0] < 2*WIDTH/3+60 and 3*HEIGHT/8-20 < mousePosition[1] < 3*HEIGHT/8+20):
                    text2Selected = False
                    textBackgroundColour2 = (255, 255, 255)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        text2Selected = False
                        textBackgroundColour2 = (255, 255, 255)
                    elif event.key == pygame.K_BACKSPACE:
                        text2 = ""
                    elif event.unicode.isdigit() and len(text2) < 4:
                        text2 += event.unicode
            elif WIDTH/3-60 < mousePosition[0] < WIDTH/3+60 and 3*HEIGHT/8-20 < mousePosition[1] < 3*HEIGHT/8+20:
                textBackgroundColour1 = (255, 255, 0)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    text1Selected = True
            elif 2*WIDTH/3-60 < mousePosition[0] < 2*WIDTH/3+60 and 3*HEIGHT/8-20 < mousePosition[1] < 3*HEIGHT/8+20:
                textBackgroundColour2 = (255, 255, 0)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    text2Selected = True
            else:
                textBackgroundColour1 = (255, 255, 255)
                textBackgroundColour2 = (255, 255, 255)
        
        WIN.fill(textBackgroundColour1, ((WIDTH/3-60, 3*HEIGHT/8-20, 120, 40)))
        WIN.fill(textBackgroundColour2, (2*WIDTH/3-60, 3*HEIGHT/8-20, 120, 40))
        
        customWindowText1 = buttonFont.render(text1, True, typeColour)
        customWindowText2 = buttonFont.render(text2, True, typeColour)
        
        WIN.blit(heightText, (WIDTH/3-heightText.get_width()/2, HEIGHT/4-heightText.get_height()/2))
        WIN.blit(widthText, (2*WIDTH/3-widthText.get_width()/2, HEIGHT/4-widthText.get_height()/2))
        WIN.blit(customWindowText1, (WIDTH/3-customWindowText1.get_width()/2, 3*HEIGHT/8-customWindowText1.get_height()/2))
        WIN.blit(customWindowText2, (2*WIDTH/3-customWindowText2.get_width()/2, 3*HEIGHT/8-customWindowText2.get_height()/2))
        
        button((WIDTH/2, 3*HEIGHT/4), mousePosition, "done", buttonTexture, "newWindowSize", text1, text2)
        
        pygame.display.flip()

def customFrameRate():
    global WIN, WIDTH, HEIGHT, event, editingFPS, gameOptions
    text = str(frameRate)
    textSelected = False
    textBackgroundColour = (255, 255, 255)
    editingFPS = True
    while editingFPS:
        WIN.fill((0,0,0))
        mousePosition = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                editingFPS = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    editingFPS = False
                    options()
                
            if textSelected:
                textBackgroundColour = (255, 0, 0)
                if event.type == pygame.MOUSEBUTTONDOWN and not (WIDTH/2-60 < mousePosition[0] < 3*WIDTH/8+60 and HEIGHT/2 < mousePosition[1] < 3*HEIGHT/8+44):
                    textSelected = False
                    textBackgroundColour = (255, 255, 255)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        textSelected = False
                        textBackgroundColour = (255, 255, 255)
                    elif event.key == pygame.K_BACKSPACE:
                        text = ""
                    elif event.unicode.isdigit() and len(text) < 3:
                        text += event.unicode
            elif WIDTH/2-60 < mousePosition[0] < WIDTH/2+60 and 3*HEIGHT/8 < mousePosition[1] < 3*HEIGHT/8+44:
                textBackgroundColour = (255, 255, 0)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    textSelected = True
            else:
                textBackgroundColour = (255, 255, 255)
        
        WIN.fill(textBackgroundColour, ((WIDTH/2-60, 3*HEIGHT/8, 120, 44)))
        
        customFPSText = buttonFont.render(text, True, typeColour)
        
        WIN.blit(FPSText, ((WIDTH-FPSText.get_width())/2, HEIGHT/4))
        WIN.blit(customFPSText, ((WIDTH-customFPSText.get_width())/2, 3*HEIGHT/8))
        
        button((WIDTH/2, 3*HEIGHT/4), mousePosition, "done", buttonTexture, "newFPS", text, 0)
        
        pygame.display.flip()

def keyRemap(key):
    global gameOptions
    gameKeyRemap = True
    while gameKeyRemap:
        event = pygame.event.wait()
        if event.type == pygame.QUIT:
            gameKeyRemap, gameOptions = False, False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                options()
                gameKeyRemap = False
            else:
                input_map[key] = event.key
                gameKeyRemap = False
                options()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            options()
            gameKeyRemap = False

def displayPlushcount():
    global gameRunning
    plushcount = 0
    for plushInspector in range(len(plushCollected)):
        plushcount += plushCollected[plushInspector]
    plushText = UIfont.render(str(plushcount) + "/25", True, (255, 255, 255))
    WIN.blit(plushText, (WIDTH-plushText.get_width(), 10))
    if plushcount == 25:
        winCondition()
        gameRunning = False

def winCondition():
    global gameWin
    pygame.mouse.set_visible(True)
    pygame.event.set_grab(False)
    gameWin = True
    while gameWin:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gameWin = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    mainMenu()
                    gameWin = False
        
        mousePosition = pygame.mouse.get_pos()
        
        WIN.fill((0,0,0))
        
        winText = buttonFont.render("thanks for playing!", True, (255, 255, 255))
        WIN.blit(winText, ((WIDTH-winText.get_width())/2, HEIGHT/2))
        
        pygame.display.flip()

mainMenu()