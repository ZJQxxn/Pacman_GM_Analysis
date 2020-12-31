function energizerInit
    global energizer;
    
    energizer.pointsDuration = 1;

    % how long to stay energized based on current level
    energizer.duration = 840;
    % how many ghost flashes happen near the end of frightened mode based on current level
    energizer.flashes = 5;

    % "The ghosts change colors every 14 game cycles when they start 'flashing'" -Jamey Pittman
    energizer.flashInterval = 14;

    energizer.count = 0;  % how long in frames energizer has been active
    energizer.active = 0; % indicates if energizer is currently active
    energizer.points = 0; % points that the last eaten ghost was worth
    energizer.pointsFramesLeft = 0; % number of frames left to display points earned from eating ghost
end
