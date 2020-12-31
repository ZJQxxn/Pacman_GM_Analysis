function gx = getTunnelEntrance(x, y, dx)
    % (x,y)'s up and down are not FloorTile and itself is FloorTile 
    while (~isFloorTile(x,y-1) && ~isFloorTile(x,y+1) && isFloorTile(x,y))
        x = x + dx;
    end
    gx = x;
end