function is = isEnergizerFlash
    global energizer;
    
    i = floor((energizer.duration-energizer.count)/energizer.flashInterval);
    if (i<=2*energizer.flashes-1)
        is = mod(i,2);
    else
        is = 0;
    end
end