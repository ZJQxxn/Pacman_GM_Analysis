function drawGhostSprite(window,x,y,frame,dirEnum,scared,flash,eyes_only,color)
    
    global DIR_UP DIR_RIGHT DIR_DOWN DIR_LEFT scale;
    
    x=x-6*scale;
    y=y-6.5*scale;
    
    if (scared)
        if flash
            color = [1 1 1];
        else
            color = [33 33 255]/255.0;
        end
    end
    
    if (~eyes_only)
        % draw body
        % add top of the ghost head to the current canvas path
        coords = [0,6,1,3,2,2,3,1,4,1,5,0,8,0,9,1,10,1,11,2,12,3,13,6];
        
        % translate by half a pixel to the right
        % to try to force centering
        path = quadraticCurvePath(x+0.5*scale,y+6*scale,x+2*scale,y,x+7*scale,y);
        path = [path quadraticCurvePath(x+7*scale,y,x+12*scale,y,x+13.5*scale,y+6*scale)];
        
        % draw lines between pixel coordinates
        %             for (i=1:2:length(coords)
        %                 path = [path [x+(0.5+coords(i))*scale; y+coords(i+1)*scale]];
        %             end
        
        if (frame == 0)
            % add first ghost animation frame feet to the current canvas path
            
            % pixel coordinates for the first feet animation
            % on the original arcade ghost sprite
            coords = [13,13,11,11,9,13,8,13,8,11,5,11,5,13,4,13,2,11,0,13];
        else
            % add second ghost animation frame feet to the current canvas path
            
            
            % pixel coordinates for the second feet animation
            % on the original arcade ghost sprite
            coords = [13,12,12,13,11,13,9,11,7,13,6,13,4,11,2,13,1,13,0,12];
        end
        % translate half a pixel right and down
        % to try to force centering and proper height
        % continue previous path (assuming ghost head)
        % by drawing lines to each of the pixel coordinates
        for i=1:2:length(coords)
            path = [path [x+(0.5+coords(i))*scale; y+(0.5+coords(i+1))*scale]];
        end
        Screen('FillPoly', window, color, path');
    end
    
    % draw face
    if (scared)
        % draw scared ghost face
        if flash
            color = [1 0 0];
        else
            color = [1 1 0];
        end
        % eyes
        Screen('FillRect', window, color, [x+4*scale,y+5*scale,x+6*scale,y+7*scale]);
        Screen('FillRect', window, color, [x+8*scale,y+5*scale,x+10*scale,y+7*scale]);
        
        % mouth
        %coords = [1,10,2,9,3,9,4,10,5,10,6,9,7,9,8,10,9,10,10,9,11,9,12,10];
        coords = [1 2 3 4 5 6 7 8 9 10 11 12; ...
                  10 9 9 10 10 9 9 10 10 9 9 10];
        path=[x+(0.5+coords(1,:))*scale;y+(0.5+coords(2,:))*scale];
%         for i=3:2:length(coords)
%             path=[path [x+(0.5+coords(i))*scale;y+(0.5+coords(i+1))*scale]];
%         end
        Screen('DrawLines', window, path, scale, color); %, [0 0], 1);
    else
        % draw regular ghost eyes
        
        coords = [0,1,1,0,2,0,3,1,3,3,2,4,1,4,0,3];
        % translate eye balls to correct position
        if (dirEnum == DIR_LEFT)
            xoff=-1;
            yoff=0;
        elseif (dirEnum == DIR_RIGHT)
            xoff=1;
            yoff=0;
        elseif (dirEnum == DIR_UP)
            xoff=0;
            yoff=-1;
        elseif (dirEnum == DIR_DOWN)
            xoff=0;
            yoff=1;
        end
        
        % draw eye balls
        path=[x+(xoff+2.5+coords(1))*scale;y+(yoff+3.5+coords(2))*scale];
        for i=3:2:length(coords)
            path=[path [x+(xoff+2.5+coords(i))*scale;y+(yoff+3.5+coords(i+1))*scale]];
        end
        Screen('FillPoly', window, [1 1 1], path');
        path=[x+(xoff+8.5+coords(1))*scale;y+(yoff+3.5+coords(2))*scale];
        for i=3:2:length(coords)
            path=[path [x+(xoff+8.5+coords(i))*scale;y+(yoff+3.5+coords(i+1))*scale]];
        end
        Screen('FillPoly', window, [1 1 1], path');
        
        
        % translate pupils to correct position
        if (dirEnum == DIR_LEFT)
            xoff=5;
            yoff=1;
        elseif (dirEnum == DIR_RIGHT)
            xoff=9;
            yoff=1;
        elseif (dirEnum == DIR_UP)
            xoff=7;
            yoff=-1;
        elseif (dirEnum == DIR_DOWN)
            xoff=7;
            yoff=4;
        end
        
        % draw pupils
        Screen('FillRect', window, [0 0 1], [x+(2+xoff)*scale,y+(3+yoff)*scale,x+(4+xoff)*scale,y+(5+yoff)*scale]); % right
        Screen('FillRect', window, [0 0 1], [x+(-4+xoff)*scale,y+(3+yoff)*scale,x+(-2+xoff)*scale,y+(5+yoff)*scale]); % left
    end
    
    
    
end
