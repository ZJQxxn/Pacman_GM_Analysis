function [point pad] = getStartPoint(tx,ty,dirEnum,pad) 
    global gameMap tileSize scale;
    dir = setDirFromEnum(dirEnum);
    % if left-hand side tile is floor other than edge, pad = floor(5*scale), ljs
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
    c = cos(-dirEnum*pi/2);
    s = sin(-dirEnum*pi/2);
    % the first expression is the rotated point centered at origin
    % the second expression is to translate it to the tile
    point.x=(px*c - py*s) + (tx-0.5)*tileSize + 1;
    point.y=(px*s + py*c) + (ty-0.5)*tileSize;
    point.cx=-1;
    point.cy=-1;
end
