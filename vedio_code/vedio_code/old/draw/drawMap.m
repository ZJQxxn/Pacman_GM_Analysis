function drawMap

global gameWindow gameMap midTile tileSize scale;

% ghost house door
%rectColor = [1 0.7216 0.8706]; %[255 184 222] / 255.0;
if gameMap.ghostHouseTile.x>=0
    Screen('FillRect', gameWindow, [1 0.7216 0.8706], [(gameMap.ghostHouseTile.x-1)*tileSize ...
        gameMap.ghostHouseTile.y*tileSize-2*scale (gameMap.ghostHouseTile.x+1)*tileSize ...
        gameMap.ghostHouseTile.y*tileSize]);
end


fillStyle = gameMap.wallFillColor;
strokeStyle = gameMap.wallStrokeColor;

for i=1:length(gameMap.walls)
    Screen('FillPoly', gameWindow, fillStyle, gameMap.walls{i}, 0);
    Screen('FramePoly', gameWindow, strokeStyle, gameMap.walls{i});
end

[dots,sizes,colors,n]=dotsProfiling(gameMap.numRows,gameMap.numCols,gameMap.currentTiles,...
    tileSize, midTile.x, midTile.y,gameMap.pelletSize,gameMap.energizerSize,scale);

% % draw pellet tiles
% dots=zeros(2,gameMap.numRows*gameMap.numCols);
% sizes=zeros(1,gameMap.numRows*gameMap.numCols);
% colors=zeros(3,gameMap.numRows*gameMap.numCols);
% 
% % mex_drawmap(gameMap.numRows,gameMap.numCols,dots,sizes,colors,gameMap.currentTiles,...
% % tileSize,midTile.x,midTile.y,gameMap.pelletSize,gameMap.energizerSize,scale);
% n=0;
for y=1:gameMap.numRows
    for x=1:gameMap.numCols
%         % this.refreshPellet(x,y,true);
        i = x+(y-1)*gameMap.numCols;
        tile = gameMap.currentTiles(i);
%         switch tile
% %             case ' '
%             case '.'
%                 n=n+1;
%                 dots(:,n)= [(x-1)*tileSize+midTile.x; (y-1)*tileSize+midTile.y];
%                 sizes(n) = gameMap.pelletSize;
%                 colors(:,n) = gameMap.pelletColor';
%             case 'o'
%                 n=n+1;
%                 dots(:,n)= [(x-1)*tileSize+0.5*scale+midTile.x; (y-1)*tileSize+0.5*scale+midTile.y];
%                 sizes(n) = gameMap.energizerSize;
%                 colors(:,n) = gameMap.pelletColor';
%         end
         if (tile == 'C' || tile  == 'S' || tile == 'O' || tile == 'A' || tile == 'M')
             drawFruit(gameWindow, tile, (x-1)*tileSize+midTile.x, (y-1)*tileSize+midTile.y);
         end
         
    end
end

if n>0
    Screen('DrawDots', gameWindow, dots(:,1:n), sizes(1:n), colors(:,1:n), [0 0], 1);
end

%     drawgrid
end