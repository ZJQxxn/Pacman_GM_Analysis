function num_j = draw_csv(index, empty_map, T, num_j, name, label, file_name, labelPos)
global ghosts pacMan gameMap;
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
gameMap.currentTiles = mapFromCsv(index, empty_map, T);

    function tiles = mapFromCsv(index, empty_map, T)
        ep = str2double(T.energizers{index,1});
        dp = str2double(T.beans{index,1});
        empty_map((dp(2 * (1:length(dp)/2))-1)*28 + dp(2 * (1:length(dp)/2)-1)) = '.';
        empty_map((ep(2 * (1:length(ep)/2))-1)*28 + ep(2 * (1:length(ep)/2)-1)) = 'o';
        tiles = empty_map;
    end



%% update ghosts
if ghostNumber>0
    for i=1:ghostNumber
        gp = posFromCsv(index, T, i);
        %% update
        ghosts(i).pixel.x = (gp(1)-1)*25+12;
        ghosts(i).pixel.y = (gp(2)-1)*25+12;
        ghosts(i).tile.x = gp(1);
        ghosts(i).tile.y = gp(2);
        ghosts(i).frames = 1;
        if i == 1
            gdir = string(T.ghost1_dir(index));
            gmode = T.ifscared1(index);
        elseif i == 2
            gdir = string(T.ghost2_dir(index));
            gmode = T.ifscared2(index);
        end
        switch gdir
            case 'up'
                ghosts(i).dirEnum = 0;
            case 'left'
                ghosts(i).dirEnum = 1;
            case 'down'
                ghosts(i).dirEnum = 2;
            case 'right'
                ghosts(i).dirEnum = 3;
            otherwise
                ghosts(i).dirEnum = 0;
        end
        ghosts(i).scared = 0;
        ghosts(i).mode = 0;
        switch gmode
            case 3
                ghosts(i).mode = 2;
            case 4
                ghosts(i).scared = 1;
            case 5
                ghosts(i).scared = 1;
        end
    end
end
    function gp = posFromCsv(index, T, i)
        if i == 1
            gp = str2double(T.ghost1Pos{index,1});
        elseif i == 2
            gp = str2double(T.ghost2Pos{index,1});
        else
            error('ghost number > 2')
        end
    end

%% update pacman
pp = str2double(T.pacmanPos{index,1});
pdir = string(T.pacman_dir(index));
pacMan.pixel.x = (pp(1)-1)*25+12;
pacMan.pixel.y = (pp(2)-1)*25+12;
pacMan.tile.x = pp(1);
pacMan.tile.y = pp(2);
switch pdir
    case 'up'
        pacMan.dirEnum = 0;
    case 'left'
        pacMan.dirEnum = 1;
    case 'down'
        pacMan.dirEnum = 2;
    case 'right'
        pacMan.dirEnum = 3;
    otherwise
        pacMan.dirEnum = -1;
end
pacMan.frames = 1;
up = 0;
down = 0;
right = 0;
left = 0;
switch T.handler(index)
    case 1
        up = 1;
    case 2
        down = 1;
    case 3
        left = 1;
    case 4
        right = 1;
end

%% update reward
% [rewd.numdot,currentTiles,scared] = DotsRewd(pacMan,rewd,gameMap,currentTiles,scared);
% [rewd.numghoast,scared] = eatGhosts(pacMan, ghosts, rewd, scared);
% rewd.count = rewd.count + rewd.numdot + rewd.numghoast;

%% draw everything
drawMap;
% % draw eyelink point
% eye_x = T.x(index);
% eye_y = T.y(index);
% Screen('DrawDots',gameWindow,[eye_x,eye_y],25,255,[],1);
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
drawMonkeyMove(up,right,down,left);
draw_text(gameWindow, T, index, label, file_name, labelPos);
    function draw_text(gameWindow, T, index, label, file_name, labelPos)
        label_i = label(~isnan(double(T{index,labelPos})));
        Screen('TextSize', gameWindow, 40);
        DrawFormattedText(gameWindow, sprintf('%s',file_name), 10,...
            50, [1 1 1]);
        if ~isempty(label_i)
            DrawFormattedText(gameWindow, char(join(label_i,'\n')), 800,...
                100, [1 1 1]);
        end
        DrawFormattedText(gameWindow, sprintf('%d',T.index(index)), 800,...
            50, [1 1 1]);
    end
% Fix Tunnel bug
Screen('FillRect', gameWindow, [0,0,0], ...
    [gameScreenWidth+1;gameScreenHeight/2-51; ...
    gameScreenWidth+51;gameScreenHeight/2+26]);
Screen('Flip', gameWindow, 0, 0, 1);

imageArray = Screen('GetImage', gameWindow);
for yql = 1:10
    imwrite(imageArray, sprintf('%s/%05d.jpg',name, num_j))
    num_j = num_j + 1;
end

% if f == length(data.direction.up)
%     if ~success
%         %% Draw dead Pacman
%         for number = 1:15
%             drawMap;
%             drawPlayer;
%             for i=1:ghostNumber
%                 ghosts(i).frames = ghosts(i).frames + 1;
%                 drawGhostSprite(gameWindow, ghosts(i).pixel.x, ghosts(i).pixel.y, ...
%                     mod(floor(ghosts(i).frames/8),2), ghosts(i).dirEnum, ...
%                     ghosts(i).scared, isEnergizerFlash, ...
%                     ghosts(i).mode == GHOST_GOING_HOME || ghosts(i).mode == GHOST_ENTERING_HOME, ...
%                     ghosts(i).color);
%             end
%             drawMonkeyMove_ljs(up,right,down,left);
%             draw_text(gameWindow, file_name, rewd);
%             Screen('FillRect', gameWindow, [0,0,0], ...
%                 [gameScreenWidth+1;gameScreenHeight/2-50; ...
%                 gameScreenWidth+51;gameScreenHeight/2+26]);
%             Screen('Flip', gameWindow, 0, 0, 1);
%             imageArray = Screen('GetImage', gameWindow);
%             imwrite(imageArray, sprintf('%s/%05d.jpg',name, num_j))
%             num_j = num_j + 1;
%         end
%
%         for angle = 330-mod(floor(pacMan.frames/4),2)*30:-20:0
%             drawMap;
%             drawPacmanSprite(gameWindow,pacMan.pixel.x,pacMan.pixel.y,pacMan.dirEnum, angle, pacMan.color);
%             drawMonkeyMove_ljs(up,right,down,left);
%             draw_text(gameWindow, file_name, rewd);
%             Screen('FillRect', gameWindow, [0,0,0], ...
%                 [gameScreenWidth+1;gameScreenHeight/2-50; ...
%                 gameScreenWidth+51;gameScreenHeight/2+26]);
%             Screen('Flip', gameWindow, 0, 0, 1);
%             imageArray = Screen('GetImage', gameWindow);
%             imwrite(imageArray, sprintf('%s/%05d.jpg',name, num_j))
%             num_j = num_j + 1;
%         end
%
%         for size = 1:2:tileSize*0.9
%             drawMap;
%             rect = [pacMan.pixel.x-size pacMan.pixel.y-size pacMan.pixel.x+size pacMan.pixel.y+size];
%             if ~mod(size-1,3)
%                 Screen('FillOval', gameWindow, pacMan.color*0.5, rect);
%             else
%                 Screen('FillOval', gameWindow, [0 0 0], rect);
%             end
%             drawMonkeyMove_ljs(up,right,down,left);
%             draw_text(gameWindow, file_name, rewd);
%             Screen('FillRect', gameWindow, [0,0,0], ...
%                 [gameScreenWidth+1;gameScreenHeight/2-50; ...
%                 gameScreenWidth+51;gameScreenHeight/2+26]);
%             Screen('Flip', gameWindow, 0, 0, 1);
%             imageArray = Screen('GetImage', gameWindow);
%             imwrite(imageArray, sprintf('%s/%05d.jpg',name, num_j))
%             num_j = num_j + 1;
%         end
%     elseif success
%         if strcmp(version,'good')
%             rewd.final = 20;
%         elseif strcmp(version,'bad')
%             rewd.final = 10;
%         end
%         final = -1;
%         while final < rewd.final
%             final = final + 1;
%             number = 25;
%             rewd_succ = final * Endreward;
%             while number > 0
%                 drawMap;
%                 for i=1:ghostNumber
%                     drawGhostSprite(gameWindow, ghosts(i).pixel.x, ghosts(i).pixel.y, ...
%                         mod(floor(ghosts(i).frames/8),2), ghosts(i).dirEnum, ...
%                         ghosts(i).scared, isEnergizerFlash, ...
%                         ghosts(i).mode == GHOST_GOING_HOME || ghosts(i).mode == GHOST_ENTERING_HOME, ...
%                         ghosts(i).color);
%                 end
%                 drawPlayer;
%                 drawMonkeyMove_ljs(up,right,down,left);
%                 Screen('TextSize', gameWindow, 40);
%                 DrawFormattedText(gameWindow, sprintf('%s',file_name), 70,...
%                     50, [1 1 1]);
%                 Screen('TextSize', gameWindow, 30);
%                 DrawFormattedText(gameWindow, sprintf('rew = %2.2fs',rewd.count*1/60+rewd_succ), 750,...
%                     300, [1 1 1]);
%                 Screen('FillRect', gameWindow, [0,0,0], ...
%                     [gameScreenWidth+1;gameScreenHeight/2-50; ...
%                     gameScreenWidth+51;gameScreenHeight/2+26]);
%                 Screen('Flip', gameWindow, 0, 0, 1);
%                 imageArray = Screen('GetImage', gameWindow);
%                 imwrite(imageArray, sprintf('%s/%05d.jpg',name, num_j))
%                 num_j = num_j + 1;
%                 number = number - 1;
%             end
%         end
%
%     end
% end

end