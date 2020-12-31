% tunnels version posToIndex
% we extend the x range to suggest the continuation of the tunnels
function index = toIndex(x,y)
    global gameMap
    if (x>-2 && x<gameMap.numCols+3 && y>0 && y<=gameMap.numRows)
        index = (x+2)+(y-1)*(gameMap.numCols+4);
    else
        index = 0;
    end
end