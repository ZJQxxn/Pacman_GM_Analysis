function [num_j,rewd, currentTiles, scared] = draw_ljs_temp(f,data,eldata,file_name,rewd,currentTiles,success,name,num_j,scared,version,Endreward,rt)

global ghosts pacMan gameMap energizer;
tileSize = 25;
% GHOST_OUTSIDE = 0;
% GHOST_EATEN = 1;
GHOST_GOING_HOME = 2;
GHOST_ENTERING_HOME = 3;
% GHOST_PACING_HOME = 4;
% GHOST_LEAVING_HOME = 5;
gameWindow = 11;
ghostNumber = 2;
gameScreenWidth = 700;
gameScreenHeight = 900;

%% update map
gameMap.currentTiles = data.gameMap.currentTiles(:,f);

%% update ghosts
if ghostNumber>0
    for i=1:ghostNumber
        %% update
        ghosts(i).pixel.x = data.ghosts.pixel_x(i,f);
        ghosts(i).pixel.y = data.ghosts.pixel_y(i,f);
        ghosts(i).tile.x = floor(ghosts(i).pixel.x / tileSize)+1;
        ghosts(i).tile.y = floor(ghosts(i).pixel.y / tileSize)+1;
        ghosts(i).frames = data.ghosts.frames(i,f);
        ghosts(i).dirEnum = data.ghosts.dirEnum(i,f);
        ghosts(i).scared = data.ghosts.scared(i,f);
        ghosts(i).mode = data.ghosts.mode(i,f);
    end
end
energizer.count = data.energizer.count(f);
%% update pacman
pacMan.pixel.x = data.pacMan.pixel_x(f);
pacMan.pixel.y = data.pacMan.pixel_y(f);
pacMan.tile.x = floor(pacMan.pixel.x / tileSize)+1;
pacMan.tile.y = floor(pacMan.pixel.y / tileSize)+1;
pacMan.dirEnum = data.pacMan.dirEnum(f);
pacMan.frames = data.pacMan.frames(f);
up = data.direction.up(f);
down = data.direction.down(f);
right = data.direction.right(f);
left = data.direction.left(f);

%% update reward
[rewd.numdot,currentTiles,scared] = DotsRewd(pacMan,rewd,gameMap,currentTiles,scared);
[rewd.numghoast,scared] = eatGhosts(pacMan, ghosts, rewd, scared);
rewd.count = rewd.count + rewd.numdot + rewd.numghoast;

%% draw everything
drawMap;
% draw eyelink point
if ~isempty(eldata)
    eye_x = eldata(f,1);
    eye_y = eldata(f,2);
    Screen('DrawDots',gameWindow,[eye_x,eye_y],25,255,[],1);
end
% draw ghosts
for i=1:ghostNumber
    drawGhostSprite(gameWindow, ghosts(i).pixel.x, ghosts(i).pixel.y, ...
        mod(floor(ghosts(i).frames/8),2), ghosts(i).dirEnum, ...
        ghosts(i).scared, isEnergizerFlash, ...
        ghosts(i).mode == GHOST_GOING_HOME || ghosts(i).mode == GHOST_ENTERING_HOME, ...
        ghosts(i).color);
end
% draw player and others
drawPlayer;
drawMonkeyMove_ljs(up,right,down,left);
draw_text(gameWindow, file_name, rewd);
Screen('TextSize', gameWindow, 40);
DrawFormattedText(gameWindow, sprintf('%d',rt), 800,...
    50, [1 1 1]);
% Fix Tunnel bug
Screen('FillRect', gameWindow, [0,0,0], ...
    [gameScreenWidth+1;gameScreenHeight/2-50; ...
    gameScreenWidth+51;gameScreenHeight/2+26]);
Screen('Flip', gameWindow, 0, 0, 1);

if mod(f,2) || f == length(data.direction.up)
    imageArray = Screen('GetImage', gameWindow);
    imwrite(imageArray, sprintf('%s/%05d.jpg',name, num_j))
    num_j = num_j + 1;
end

if f == length(data.direction.up)
    if ~success
        %% Draw dead Pacman
        for number = 1:15
            drawMap;
            drawPlayer;
            for i=1:ghostNumber
                ghosts(i).frames = ghosts(i).frames + 1;
                drawGhostSprite(gameWindow, ghosts(i).pixel.x, ghosts(i).pixel.y, ...
                    mod(floor(ghosts(i).frames/8),2), ghosts(i).dirEnum, ...
                    ghosts(i).scared, isEnergizerFlash, ...
                    ghosts(i).mode == GHOST_GOING_HOME || ghosts(i).mode == GHOST_ENTERING_HOME, ...
                    ghosts(i).color);
            end
            drawMonkeyMove_ljs(up,right,down,left);
            draw_text(gameWindow, file_name, rewd);
            Screen('FillRect', gameWindow, [0,0,0], ...
                [gameScreenWidth+1;gameScreenHeight/2-50; ...
                gameScreenWidth+51;gameScreenHeight/2+26]);
            Screen('Flip', gameWindow, 0, 0, 1);
            imageArray = Screen('GetImage', gameWindow);
            imwrite(imageArray, sprintf('%s/%05d.jpg',name, num_j))
            num_j = num_j + 1;
        end
        
        for angle = 330-mod(floor(pacMan.frames/4),2)*30:-20:0
            drawMap;
            drawPacmanSprite(gameWindow,pacMan.pixel.x,pacMan.pixel.y,pacMan.dirEnum, angle, pacMan.color);
            drawMonkeyMove_ljs(up,right,down,left);
            draw_text(gameWindow, file_name, rewd);
            Screen('FillRect', gameWindow, [0,0,0], ...
                [gameScreenWidth+1;gameScreenHeight/2-50; ...
                gameScreenWidth+51;gameScreenHeight/2+26]);
            Screen('Flip', gameWindow, 0, 0, 1);
            imageArray = Screen('GetImage', gameWindow);
            imwrite(imageArray, sprintf('%s/%05d.jpg',name, num_j))
            num_j = num_j + 1;
        end
        
        for size = 1:2:tileSize*0.9
            drawMap;
            rect = [pacMan.pixel.x-size pacMan.pixel.y-size pacMan.pixel.x+size pacMan.pixel.y+size];
            if ~mod(size-1,3)
                Screen('FillOval', gameWindow, pacMan.color*0.5, rect);
            else
                Screen('FillOval', gameWindow, [0 0 0], rect);
            end
            drawMonkeyMove_ljs(up,right,down,left);
            draw_text(gameWindow, file_name, rewd);
            Screen('FillRect', gameWindow, [0,0,0], ...
                [gameScreenWidth+1;gameScreenHeight/2-50; ...
                gameScreenWidth+51;gameScreenHeight/2+26]);
            Screen('Flip', gameWindow, 0, 0, 1);
            imageArray = Screen('GetImage', gameWindow);
            imwrite(imageArray, sprintf('%s/%05d.jpg',name, num_j))
            num_j = num_j + 1;
        end
    elseif success
        if strcmp(version,'good')
            rewd.final = 20;
        elseif strcmp(version,'bad')
            rewd.final = 10;
        else
            rewd.final = 10;
        end
        final = -1;
        while final < rewd.final
            final = final + 1;
            number = 25;
            rewd_succ = final * Endreward;
            while number > 0
                drawMap;
                for i=1:ghostNumber
                    drawGhostSprite(gameWindow, ghosts(i).pixel.x, ghosts(i).pixel.y, ...
                        mod(floor(ghosts(i).frames/8),2), ghosts(i).dirEnum, ...
                        ghosts(i).scared, isEnergizerFlash, ...
                        ghosts(i).mode == GHOST_GOING_HOME || ghosts(i).mode == GHOST_ENTERING_HOME, ...
                        ghosts(i).color);
                end
                drawPlayer;
                drawMonkeyMove_ljs(up,right,down,left);
                Screen('TextSize', gameWindow, 40);
                DrawFormattedText(gameWindow, sprintf('%s',file_name), 70,...
                    50, [1 1 1]);
                Screen('TextSize', gameWindow, 30);
                DrawFormattedText(gameWindow, sprintf('rew = %2.2fs',rewd.count*1/60+rewd_succ), 750,...
                    300, [1 1 1]);
                Screen('FillRect', gameWindow, [0,0,0], ...
                    [gameScreenWidth+1;gameScreenHeight/2-50; ...
                    gameScreenWidth+51;gameScreenHeight/2+26]);
                Screen('Flip', gameWindow, 0, 0, 1);
                imageArray = Screen('GetImage', gameWindow);
                imwrite(imageArray, sprintf('%s/%05d.jpg',name, num_j))
                num_j = num_j + 1;
                number = number - 1;
            end
        end
        
    end
end

end