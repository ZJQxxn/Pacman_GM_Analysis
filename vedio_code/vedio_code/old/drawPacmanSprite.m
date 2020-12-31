function drawPacmanSprite(window,x,y,dirEnum,angle,color)

    global DIR_UP DIR_RIGHT DIR_DOWN DIR_LEFT scale tileSize;
    
    w = tileSize*0.9;
    
    % rotate to current heading direction
    if (dirEnum == DIR_UP) 
        startAngle = 0;
    elseif (dirEnum == DIR_RIGHT) 
        startAngle = 90;
    elseif (dirEnum == DIR_DOWN) 
        startAngle = 180;
    elseif (dirEnum == DIR_LEFT)
        startAngle = -90;
    end
    
    rect = [x-w y-w+scale x+w y+w+scale];
    %[rect startAngle+180-angle/2 angle]
    
    if dirEnum>=0
        Screen('FillArc', window, color, rect, startAngle+180-angle/2, angle);
    else
        Screen('FillOval', window, color, rect);
    end    
    
end