function tile = getTile_ljs(x, y, gameMap, currentTiles)

if (x>0 && x<=gameMap.numCols && y>0 && y<=gameMap.numRows)
    
    tile = currentTiles(x+(y-1)*gameMap.numCols);
else
    tile = 0;
end

end