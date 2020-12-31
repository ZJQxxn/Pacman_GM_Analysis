function drawPlayer
    global gameWindow pacMan;
    
    drawPacmanSprite(gameWindow, pacMan.pixel.x, pacMan.pixel.y, pacMan.dirEnum,...
        330-mod(floor(pacMan.frames/4),2)*30, pacMan.color);
end