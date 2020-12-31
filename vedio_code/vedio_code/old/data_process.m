% set direction vector from a direction enum
function dirEnum = data_process(dir_x, dir_y)

    DIR_UP = 0; 
    DIR_RIGHT = 3; 
    DIR_DOWN = 2; 
    DIR_LEFT = 1;
    dirEnum = ones(2, length(dir_x)) * 8;
    for i = 1:length(dir_x)
        for j = 1:2
            
            if (dir_x(j,i) == 0 && dir_y(j,i) == -1)  
                dirEnum(j,i) = DIR_UP;
            elseif (dir_x(j,i) == 1 && dir_y(j,i) == 0)  
                dirEnum(j,i) = DIR_RIGHT; 
            elseif (dir_x(j,i) == 0 && dir_y(j,i) == 1)  
                dirEnum(j,i) = DIR_DOWN;
            elseif (dir_x(j,i) == -1 && dir_y(j,i) == 0) 
                dirEnum(j,i) = DIR_LEFT; 
            else
                dirEnum(j,i) = 4;
            end
        end
    end
end