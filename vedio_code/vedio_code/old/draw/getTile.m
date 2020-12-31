function tile = getTile(x, y)
    global gameMap;
    
    if (x>0 && x<=gameMap.numCols && y>0 && y<=gameMap.numRows)
        tile = gameMap.currentTiles(posToIndex(x,y));
    elseif ((x<1 || x>gameMap.numCols) && (isTunnelTile(x,y-1) || isTunnelTile(x,y+1)))
        tile = '|';
    elseif (isTunnelTile(x,y))
        tile = ' ';
    else
        tile = 0;
    end
end