function fruitInit

    global fruit tileSize midTile;
    
    fruit.pixel.x = tileSize*14;
    fruit.pixel.y = tileSize*21 + midTile.y;
    fruit.duration = 5;
    fruit.framesLeft = 60*fruit.duration;
    fruit.current = 1;
    
end
