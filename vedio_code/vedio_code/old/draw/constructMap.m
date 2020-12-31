function constructMap(numCols, numRows, tiles)
% initialize gameMap from an ASCII map

    globalDefinitions;

    %%2017/1/4 by hy
    
    
    %%
    gameMap.numCols = numCols;
    gameMap.numRows = numRows;
    gameMap.numTiles = numCols*numRows;
    gameMap.widthPixels = numCols*tileSize;
    gameMap.heightPixels = numRows*tileSize;

    % ascii map
    gameMap.tiles = tiles;
    gameMap.currentTiles = tiles;

    % ghost home location
    gameMap.doorTile.x = 14;
    gameMap.doorTile.y = 16;
%         gameMap.doorTile.x = 14; % DoubleFaceGhostTrial_v6
%         gameMap.doorTile.y = 15; % DoubleFaceGhostTrial_v6
    gameMap.doorPixel.x = (gameMap.doorTile.x)*tileSize; %possible bug
    gameMap.doorPixel.y = (gameMap.doorTile.y-2)*tileSize + midTile.y;
%     
    gameMap.homeTopPixel = 16*tileSize;
    gameMap.homeBottomPixel = 19*tileSize;
%     gameMap.homeTopPixel = 17*tileSize;  % DoubleFaceGhostTrial_v6
%     gameMap.homeBottomPixel = 18*tileSize; % DoubleFaceGhostTrial_v6

    % rendering settings
    gameMap.messageRow = 20;
    gameMap.pelletSize = floor(2*scale);
    gameMap.energizerSize = tileSize;

    gameMap.backColor = [0 0 0];
    gameMap.floorColor = [0.5 0.5 0.5];
    gameMap.flashWallColor = [1 1 1];
    
%     gameMap.wallStrokeColor = [33 33 255]/255.0; % from original
    gameMap.wallStrokeColor = [200 200 255]/255.0; 
    gameMap.wallFillColor = [0 0 0];
    gameMap.pelletColor = [255 184 174]/255.0;


    gameMap.name = 'Arcade';

    %gameMap.timeEaten = {};

    %gameMap.resetCurrent();
    gameMap.dotsEaten=0;
    
    %gameMap.parseDots();
    i=1;
    gameMap.numDots = 0;
    gameMap.numEnergizers = 0;
    for y=1:gameMap.numRows
        for x=1:gameMap.numCols
            if (gameMap.tiles(i) == '.')
                gameMap.numDots = gameMap.numDots + 1;
            elseif (gameMap.tiles(i) == 'o')
                gameMap.numDots = gameMap.numDots + 1;
                gameMap.numEnergizers = gameMap.numEnergizers + 1;
                if isfield(gameMap, 'energizers')
                    gameMap.energizers(end+1)=struct('x',x,'y',y);
                else
                    gameMap.energizers(1)=struct('x',x,'y',y);
                end
            elseif (gameMap.tiles(i) == 'C' || gameMap.tiles(i) == 'S' || gameMap.tiles(i) == 'O' ...
                    || gameMap.tiles(i) == 'A' || gameMap.tiles(i) == 'M')
                gameMap.numDots = gameMap.numDots + 1; 
            end
            i=i+1;
        end
    end
    
    %%%gameMap.parseTunnels();
    % the number of margin tiles outside of the map on one side of a tunnel
    % There are (2*marginTiles) tiles outside of the map per tunnel.
    marginTiles = 2;
    for y=1:gameMap.numRows
        % a map row is a tunnel if opposite ends are both walkable tiles
        if (isFloorTile(1,y) && isFloorTile(gameMap.numCols,y))
            gameMap.tunnelRows(y) = struct( ...
                'leftEntrance', getTunnelEntrance(1,y,1), ...
                'rightEntrance', getTunnelEntrance(gameMap.numCols,y,-1), ...
                'leftExit', -marginTiles, ...
                'rightExit', gameMap.numCols+marginTiles-1);
%               'leftExit', -marginTiles*tileSize, ...
%               'rightExit', (gameMap.numCols+marginTiles)*tileSize-1);
        else
            gameMap.tunnelRows(y) = struct( ...
                'leftEntrance', -1, ...
                'rightEntrance', -1, ...
                'leftExit', -1, ...
                'rightExit', -1);
        end
    end
    
    %%%gameMap.parseWalls();
    % creates a list of drawable canvas paths to render the map walls
    gameMap.paths = [];

    % a map of wall tiles that already belong to a built path
    visited = zeros(1,gameMap.numRows * (gameMap.numCols+4));

    % a map of which wall tiles that are not completely surrounded by other wall tiles
    gameMap.edges = zeros(1,gameMap.numRows * (gameMap.numCols+4));
    i=0;
    for y=1:gameMap.numRows
        for x=-1:gameMap.numCols+2
            i=i+1;
            if (getTile(x,y) == '|' && ...
                (getTile(x-1,y) ~= '|' || ...
                getTile(x+1,y) ~= '|' || ...
                getTile(x,y-1) ~= '|' || ...
                getTile(x,y+1) ~= '|' || ...
                getTile(x-1,y-1) ~= '|' || ...
                getTile(x-1,y+1) ~= '|' || ...
                getTile(x+1,y-1) ~= '|' || ...
                getTile(x+1,y+1) ~= '|')) 
                gameMap.edges(i) = 1;
            else
               gameMap.edges(i) = 0;
            end
        end
    end

   
    % iterate through all edges, making a new path after hitting an unvisited wall edge
    i=0;
    for y=1:gameMap.numRows
        for x=-1:gameMap.numCols+2
            i=i+1;
            if (gameMap.edges(i) && ~visited(i))
                visited(i) = 1;
                
                % makePath(gameMap,x,y);
                % walks along edge wall tiles starting at the given index to build a canvas path
                
                tx = x; ty = y;
                % get initial direction
                if (gameMap.edges(toIndex(tx+1,ty)))
                    dirEnum = DIR_RIGHT;
                elseif (gameMap.edges(toIndex(tx, ty+1)))
                    dirEnum = DIR_DOWN;
                else
                    ME = MException('ParseWalls:outOfRange', 'tile should not be 1x1 at %d,%d', tx, ty);
                    throw(ME);
                end
                dir = setDirFromEnum(dirEnum);
                
                % increment to next tile
                tx = tx+dir.x;
                ty = ty+dir.y;
                
                % backup initial location and direction
                init_tx = tx;
                init_ty = ty;
                init_dirEnum = dirEnum;
                
                turn = 0;
                path = {};
                %   We employ the 'right-hand rule' by keeping our right hand in contact
                %   with the wall to outline an individual wall piece.
                %
                %   Since we parse the tiles in row major order, we will always start
                %   walking along the wall at the leftmost tile of its topmost row.  We
                %   then proceed walking to the right.
                %
                %   When facing the direction of the walk at each tile, the outline will
                %   hug the left side of the tile unless there is a walkable tile to the
                %   left.  In that case, there will be a padding distance applied.
                
                pad = 0;
                while (1)
                    
                    visited(toIndex(tx,ty)) = 1;
                    
                    % determine start point
                    [point pad] = getStartPoint(tx,ty,dirEnum,pad);
                    %fprintf('(%d,%d,%d)',tx,ty,dirEnum);
                   
                    if (turn)
                        % if we're turning into gameMap tile, create a control point for the curve
                        %
                        % >---+  <- control point
                        %     |
                        %     V
                        lastPoint = path(end);
                        if (dir.x == 0)
                            point.cx = point.x;
                            point.cy = lastPoint.y;
                        else
                            point.cx = lastPoint.x;
                            point.cy = point.y;
                        end
                    end
                    
                    % update direction
                    turn = 0;
                    turnAround = 0;
                    ind = toIndex(tx+dir.y, ty-dir.x);
                    if ind~=0 && gameMap.edges(ind) % turn left
                        dirEnum = rotateLeft(dirEnum);
                        turn = 1;
                    else
                        ind = toIndex(tx+dir.x, ty+dir.y);
                        if ind~=0 && gameMap.edges(ind) % continue straight
                            1;
                        else
                            ind = toIndex(tx-dir.y, ty+dir.x);
                            if ind~=0 && gameMap.edges(ind) %turn right
                                dirEnum = rotateRight(dirEnum);
                                turn = 1;
                            else % turn around
                                dirEnum = rotateAboutFace(dirEnum);
                                turnAround = 1;
                            end
                        end
                    end
                    dir=setDirFromEnum(dirEnum);
                    
                    % commit path point
                    if ~isempty(path)
                        path(end+1)=point;
                    else
                        path=point;
                    end
                    
                    % special case for turning around (have to connect more dots manually)
                    if (turnAround)
                        [path(end+1) pad]=getStartPoint(tx-dir.x, ty-dir.y, rotateAboutFace(dirEnum), pad);
                        [path(end+1) pad]=getStartPoint(tx, ty, dirEnum, pad);
                    end
                    
                    % advance to the next wall
                    tx = tx + dir.x;
                    ty = ty + dir.y;
                    
                    % exit at full cycle
                    if (tx==init_tx && ty==init_ty && dirEnum == init_dirEnum)
                        if length(gameMap.paths)>0
                            gameMap.paths{end+1}=path;
                        else
                            gameMap.paths{1} = path;
                        end
                        break;
                    end
                end
            end
        end
    end
    
    i=1; set=0;
    for y=1:gameMap.numRows
        for x=1:gameMap.numCols
            if (gameMap.tiles(i) == '-' && gameMap.tiles(i+1) == '-')
                gameMap.ghostHouseTile.x = x;
                gameMap.ghostHouseTile.y = y;
                set=1;
                break;
            end
            i=i+1;
        end
        if set
            break;
        end
    end
    if set==0
        gameMap.ghostHouseTile.x = -1;
    end

    gameMap.walls={};
    for i=1:length(gameMap.paths)
        path = gameMap.paths{i};
        wall = [path(1).x;path(1).y];

        for j=2:length(path)
            if (path(j).cx >= 0)
                %Screen('DrawLine', window, currentx, currenty, path(j).x, path(j).y);
                wall = [wall quadraticCurvePath(wall(1,end), wall(2,end), path(j).cx, path(j).cy, path(j).x, path(j).y)];
            else
                wall = [wall [path(j).x; path(j).y]];
            end
        end
        %Screen('DrawLine', window, currentx, currenty, path(1).x, path(1).y);
        gameMap.walls{end+1} = [wall quadraticCurvePath(wall(1,end), wall(2,end), path(end).x, path(1).y, path(1).x, path(1).y)]';
    end
        
    % count dots
    gameMap.totalDots = 0;
    for y=1:gameMap.numRows
        for x=1:gameMap.numCols
            % gameMap.refreshPellet(x,y,true);
            i = x+(y-1)*gameMap.numCols;
            tile = gameMap.currentTiles(i);
            if (tile == '.')
                gameMap.totalDots = gameMap.totalDots + 1;
            end
        end
    end      
end

function gx = getTunnelEntrance(x, y, dx)
    while (~isFloorTile(x,y-1) && ~isFloorTile(x,y+1) && isFloorTile(x,y))
        x = x + dx;
    end
    gx = x;
end

%we extend the x range to suggest the continuation of the tunnels
function index = toIndex(x,y)
    global gameMap
    if (x>-2 && x<gameMap.numCols+3 && y>0 && y<=gameMap.numRows)
        index = (x+2)+(y-1)*(gameMap.numCols+4);
    else
        index = 0;
    end
end

function [point pad] = getStartPoint(tx,ty,dirEnum,pad) 
    global gameMap tileSize scale;
    dir = setDirFromEnum(dirEnum);
    ind = toIndex(tx+dir.y,ty-dir.x);
    if ind~=0 && ~(gameMap.edges(ind))
        if isFloorTile(tx+dir.y,ty-dir.x)
            pad = floor(5*scale);
        else
            pad = 0;
        end
    end

    px = -tileSize/2+pad;
    py = tileSize/2;
    a = getClockwiseAngleFromTop(dirEnum);
    c = cos(a);
    s = sin(a);
    % the first expression is the rotated point centered at origin
    % the second expression is to translate it to the tile
    point.x=(px*c - py*s) + (tx-0.5)*tileSize + 1;
    point.y=(px*s + py*c) + (ty-0.5)*tileSize;
    point.cx=-1;
    point.cy=-1;
end
