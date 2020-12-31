function draw_text(gameWindow, file_name, rewd)

Screen('TextSize', gameWindow, 40);
DrawFormattedText(gameWindow, sprintf('%s',file_name), 70,...
    50, [1 1 1]);
Screen('TextSize', gameWindow, 30);
DrawFormattedText(gameWindow, sprintf('rew = %2.2fs',rewd.count*1/60), 750,...
    300, [1 1 1]);

end