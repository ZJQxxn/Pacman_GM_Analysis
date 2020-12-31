function [numghoast,scared] = eatGhosts(pacMan, ghosts, rewd, scared)
reward = zeros(1,2);
numghoast = 0;
for i = 1:2
    if ghosts(i).tile.x == pacMan.tile.x && ghosts(i).tile.y == pacMan.tile.y ...
            && logical(scared(i))
        reward(i) = rewd.maggoast;
        numghoast = reward(i) + numghoast;
        scared(i) = 0;
    end
end

end