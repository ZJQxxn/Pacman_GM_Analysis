% set direction vector from a direction enum
function dir = setDirFromEnum (dirEnum)

    global DIR_UP DIR_RIGHT DIR_DOWN DIR_LEFT ;
    
    if (dirEnum == DIR_UP)  
        dir.x = 0; dir.y =-1;
    elseif (dirEnum == DIR_RIGHT)  
        dir.x = 1; dir.y = 0; 
    elseif (dirEnum == DIR_DOWN)  
        dir.x = 0; dir.y = 1; 
    elseif (dirEnum == DIR_LEFT) 
        dir.x = -1; dir.y = 0; 
    else
        dir.x=0; dir.y=0;
    end
end