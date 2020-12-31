function is = isTunnelTile(x,y) 
    global gameMap;
    
    if y<1 || y>gameMap.numRows
        is = 0;
    else
        is = gameMap.tunnelRows(y).leftEntrance~=-1 && (x < gameMap.tunnelRows(y).leftEntrance || x > gameMap.tunnelRows(y).rightEntrance);
    
    end
end
