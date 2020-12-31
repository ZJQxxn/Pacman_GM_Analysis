KbName('UnifyKeyNames')
maxCols = 28;
maxRows = 36;
Screen('Preference', 'SkipSyncTests', 1);
Screen('Preference', 'VisualDebugLevel', 3);
Screen('Preference', 'Verbosity', 2);
PsychDefaultSetup(2);
screens = Screen('Screens');
screenNumber = max(screens);
black = BlackIndex(screenNumber);
[gameWindow0, gameWindowRect0] = PsychImaging('OpenWindow', screenNumber, black,[0,0,900,1440]);
[screenWidth, screenHeight] = Screen('WindowSize', gameWindow0);

tileSize = ceil(1080 / 44.125/2)*2-1;
midTile = struct('x',ceil(tileSize/2),'y',ceil(tileSize/2));
scale = tileSize / 8.0;

mapWidth = maxCols*tileSize;
mapHeight = maxRows*tileSize;
gameScreenWidth = mapWidth;
gameScreenHeight = mapHeight;
gameScreenXOffset = (screenWidth - gameScreenWidth)/2;
gameScreenYOffset = (screenHeight - gameScreenHeight)/2;
vedioWidth = screenWidth+length(char(label))*20-gameScreenXOffset;
vedioHeight = gameScreenYOffset+gameScreenHeight-gameScreenYOffset;
[gameWindow, ~] = PsychImaging('OpenWindow', screenNumber, black, ...
    [gameScreenXOffset gameScreenYOffset ...
    vedioWidth+gameScreenXOffset vedioHeight+gameScreenYOffset]);
flipInterval = Screen('GetFlipInterval', gameWindow);
totalReward = 0;
ljs_map;
