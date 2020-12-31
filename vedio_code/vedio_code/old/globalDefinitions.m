global scale;

%size of a square tile in pixels
global tileSize;

%the center pixel of a tile
global midTile; 

global maxCols; maxCols = 28;
global maxRows; maxRows = 36;

%the game map
global gameMap;
global DIR_UP; DIR_UP = 0;
global DIR_LEFT; DIR_LEFT = 1;
global DIR_DOWN; DIR_DOWN = 2;
global DIR_RIGHT; DIR_RIGHT = 3;

% global mapMargin; % margin between the map and the screen
% global mapPad; % padding between the map and its clipping

global mapWidth; 
global mapHeight; 

global gameScreenWidth; 
global gameScreenHeight; 
global gameScreenXOffset;
global gameScreenYOffset;

global screens;
global screenNumber;
global gameWindow;
global gameWindow0;
global gameWindowRect;
global gameWindowRect0;
global screenWidth;
global screenHeight;

global flipInterval;

global energizer;

global pacMan;
    
% fruits
global CHERRY; CHERRY=0;
global STRAWBERRY; STRAWBERRY=1;
global ORANGE; ORANGE=2;
global APPLE; APPLE=3;
global MELON; MELON=4;
global fruitDotLimits;  fruitDotLimits=[70 170]; 

% ghosts
global ghostNumber;
global ghostActive;
global ghosts;
global ghostEndPosition;
global BLINKY; BLINKY=1;
global PINKY; PINKY=4; % 2 ZWY
global INKY; INKY=3;
global CLYDE; CLYDE=2; % 4 ZWY
global GHOST_COLOR; GHOST_COLOR = {[1 0 0 0.8],[255 184 81 204]/255.0,[0 1 1 0.8],[255 184 255 204]/255.0,}; % switch 2 and 4 ZWY
% ghost home stuff
%two separate counter modes for releasing the ghosts from home
global MODE_PERSONAL; MODE_PERSONAL = 0;
global MODE_GLOBAL; MODE_GLOBAL = 1;
  
global framesSinceLastDot; % frames elapsed since last dot was eaten
global mode;               % personal or global dot counter mode
global ghostCounts;        % personal dot counts for each ghost
global globalCount;        % global dot count


% ghost modes
global GHOST_OUTSIDE; GHOST_OUTSIDE = 0;
global GHOST_EATEN; GHOST_EATEN = 1;
global GHOST_GOING_HOME; GHOST_GOING_HOME = 2;
global GHOST_ENTERING_HOME; GHOST_ENTERING_HOME = 3;
global GHOST_PACING_HOME; GHOST_PACING_HOME = 4;
global GHOST_LEAVING_HOME; GHOST_LEAVING_HOME = 5;

% reward
global totalReward;
% % zzw 20161220 change the way of reward delivering
% % % global pelletReward; pelletReward = 10;
% % % global energizerReward; energizerReward = 20;
% % % global ghostReward; ghostReward = 100;
% % % global fruitReward; fruitReward = [100 200 300 400 500];
global rewd
global reward_count

% keyboard
global keyDown;
global keyCode;
global upKey;    upKey = KbName('UpArrow');
global downKey;  downKey = KbName('DownArrow');
global leftKey;  leftKey = KbName('LeftArrow');
global rightKey; rightKey = KbName('RightArrow');

% Joysticks
global JSup; JSup = 1;
global JSdown; JSdown = 2;
global JSleft; JSleft = 3;
global JSright; JSright = 4;
global JSMoved JSCode;

% system
global topPriorityLevel OldPriority;

% trial interval
global StimulusOnsetTime

% online control 
global ol_info_win % online information window  -- 20170119 lzq
