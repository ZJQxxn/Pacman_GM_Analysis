function index = posToIndex(x,y) 
% % % % get the index of the tile
    global gameMap;
    
    if (x>0 && x<=gameMap.numCols && y>0 && y<=gameMap.numRows) 
        index = x+(y-1)*gameMap.numCols;
    end

end
