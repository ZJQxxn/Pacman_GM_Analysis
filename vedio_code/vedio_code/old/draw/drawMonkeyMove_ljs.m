function drawMonkeyMove_ljs(up,right,down,left)

global gameWindow;

x = 815;
y = 425;
size = 25 / 2;
uparrow = [0,0;0,-3*size;-size,-3*size;size,-5*size;3*size,-3*size;2*size,-3*size;2*size,0;0,0];
rightarrow = ([cos(pi/2), -sin(pi/2);sin(pi/2), cos(pi/2)] * uparrow')';
downarrow = ([cos(pi/2), -sin(pi/2);sin(pi/2), cos(pi/2)] * rightarrow')';
leftarrow = ([cos(pi/2), -sin(pi/2);sin(pi/2), cos(pi/2)] * downarrow')';

if up
    Screen('FillPoly', gameWindow, [1,1,1], uparrow(3:5,:) + [x,y], 1);
    Screen('FillPoly', gameWindow, [1,1,1], uparrow([1,2,6,7],:) + [x,y], 1);
end

if right
    Screen('FillPoly', gameWindow, [1,1,1], rightarrow(3:5,:) + [x+2*size,y], 1);
    Screen('FillPoly', gameWindow, [1,1,1], rightarrow([1,2,6,7],:) + [x+2*size,y], 1);
end

if down
    Screen('FillPoly', gameWindow, [1,1,1], downarrow(3:5,:) + [x+2*size,y+2*size], 1);
    Screen('FillPoly', gameWindow, [1,1,1], downarrow([1,2,6,7],:) + [x+2*size,y+2*size], 1);
end

if left
    Screen('FillPoly', gameWindow, [1,1,1], leftarrow(3:5,:) + [x,y+2*size], 1);
    Screen('FillPoly', gameWindow, [1,1,1], leftarrow([1,2,6,7],:) + [x,y+2*size], 1);
end

if ~up && ~right && ~down && ~left
    Screen('FramePoly', gameWindow, [1,1,1], uparrow + [x,y], 2);
    Screen('FramePoly', gameWindow, [1,1,1], rightarrow + [x+2*size,y], 2);
    Screen('FramePoly', gameWindow, [1,1,1], downarrow + [x+2*size,y+2*size], 2);
    Screen('FramePoly', gameWindow, [1,1,1], leftarrow + [x,y+2*size], 2);
end

end