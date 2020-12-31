function [numdot,currentTiles,scared] = DotsRewd(pacMan,rewd,gameMap,currentTiles,scared)
t = getTile_ljs(pacMan.tile.x,pacMan.tile.y,gameMap,currentTiles);
if (t == '.' || t == 'o' || t == 'C' || t == 'S' || t == 'O' || t == 'A' || t == 'M')
    currentTiles(pacMan.tile.x+(pacMan.tile.y-1)*28)=' ';
    if (t=='.')
        numdot = rewd.magdot;
    elseif (t=='o')
        numdot = rewd.mageneg;
        scared(1:2) = 1;
    elseif (t == 'C')
        numdot = rewd.magcherry;
    elseif (t == 'S')
        numdot = rewd.magstrawberry;
    elseif (t == 'O')
        numdot = rewd.magorange;
    elseif (t == 'A')
        numdot = rewd.magapple;
    elseif (t == 'M')
        numdot = rewd.magmelon;
    else
        numdot = 0;
    end
else
    numdot = 0;
end

end