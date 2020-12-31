function drawFruit(window, fruit, x, y)
%////////////////////////////////////////////////////////////////////
%// FRUIT SPRITES

%     globalDefinitions;

    global scale;
    
    switch fruit
        case 'C'
            drawCherry(window,scale,x,y);
        case 'S'
            drawStrawberry(window,scale,x,y);
        case 'O'
            drawOrange(window,scale,x,y);
        case 'A'
            drawApple(window,scale,x,y);
        case 'M'
            drawMelon(window,scale,x,y);
    end
end
        
function drawCherry(window,scale,x,y)
  
 
    
    function cherry(x,y)
        % red fruit
        Screen('FillOval', window, [1 0 0], [x-0.5*scale y-0.5*scale x+5.5*scale y+5.5*scale]);
        Screen('FrameOval', window, [0 0 0], [x-0.5*scale y-0.5*scale x+5.5*scale y+5.5*scale]);
        % white shine
        Screen('DrawLine', window, [1 1 1], x+scale, y+3*scale, x+2*scale, y+4*scale, scale);
    end
%     t2 = tic;
    % draw both cherries
    cherry(x-6*scale,y-scale);
    cherry(x-scale,y+scale);

    % draw stems
    strokeColor = [255 153 0]/255.0;
    drawBezierCurve(window, strokeColor, ...
        x-3*scale, y, x-scale, y-2*scale, x+2*scale, y-4*scale, x+5*scale,y-5*scale, scale);
    Screen('DrawLine', window, strokeColor, x+5*scale, y-5*scale, x+5*scale, y-4*scale, scale);
    drawBezierCurve(window, strokeColor, ...
        x+5*scale, y-4*scale, x+3*scale, y-4*scale, x+scale, y, x+scale,y+2*scale, scale);
%     t2 = toc(t2);
%     display(['cherryt2 = ',num2str(t2)])
end

function drawStrawberry(window,scale,x,y)
    % red body
    path = bezierCurvePath(x-scale,y-4*scale,x-3*scale,y-4*scale,x-5*scale,y-3*scale,x-5*scale,y-scale);
    path = [path bezierCurvePath(x-5*scale,y-scale,x-5*scale,y+3*scale,x-2*scale,y+5*scale,x,y+6*scale)];
    path = [path bezierCurvePath(x,y+6*scale,x+3*scale,y+5*scale,x+5*scale,y+2*scale,x+5*scale,y)];
    path = [path bezierCurvePath(x+5*scale,y,x+5*scale,y-3*scale,x+3*scale,y-4*scale,x,y-4*scale)];
    Screen('FillPoly', window, [1 0 0], path');
        
    % white spots
    spots = [-4 -1;-3 2;-2 0;-1 4;0 2;0 0;1.5 3.7;2 -1;3 1;4 -2];
    for i=1:size(spots,1)
        Screen('FillOval', window, [1 1 1], ...
            [x+(spots(i,1)-0.5)*scale y+(spots(i,2)-0.5)*scale x+(spots(i,1)+1)*scale y+(spots(i,2)+1)*scale]);
    end
   
    % green leaf
    path=[0 -3 0 -2 -1 0 0 0 1 2 0 3;
            -4 -4 -4 -3 -3 -4 -2 -4 -3 -3 -4 -4].*scale;
    path(1,:) = path(1,:)+x;
    path(2,:) = path(2,:)+y;
    Screen('BlendFunction', window, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    Screen('DrawLines', window, path, scale, [0 1 0], [0 0], 1);
    
    % stem
    Screen('DrawLine', window, [1 1 1], x, y-4*scale, x, y-5*scale, scale);
end

function drawOrange(window,scale,x,y) 
   
    % orange body
    path = bezierCurvePath(x-2*scale,y-2*scale,x-3*scale,y-2*scale,x-5*scale,y-scale,x-5*scale,y+scale);
    path = [path bezierCurvePath(x-5*scale,y+scale,x-5*scale,y+4*scale,x-3*scale,y+6*scale,x,y+6*scale)];
    path = [path bezierCurvePath(x,y+6*scale,x+3*scale,y+6*scale,x+5*scale,y+4*scale,x+5*scale,y+scale)];
    path = [path bezierCurvePath(x+5*scale,y+scale,x+5*scale,y-scale,x+3*scale,y-2*scale,x+2*scale,y-2*scale)];
%     t2 = tic;
    Screen('FillPoly', window, [255 204 51]/255.0, path');
%     t2 = toc(t2);
%     display(['orangeS1= ',num2str(t2)])
%     % stem
%     path = quadraticCurvePath(x-scale,y-scale,x-scale,y-2*scale,x-2*scale,y-2*scale);
%     path = [path quadraticCurvePath(x-2*scale,y-2*scale,x-scale,y-2*scale,x-scale,y-4*scale)];
%     path = [path quadraticCurvePath(x-scale,y-4*scale,x-scale,y-2*scale,x,y-2*scale)];
%     path = [path quadraticCurvePath(x,y-2*scale,x-scale,y-2*scale,x-scale,y-scale)];
%     Screen('DrawLines', window, path, scale, [255 153 0]/255.0, [0 0], 1);
    
    % green leaf
    path = quadraticCurvePath(x-0.5*scale,y-4*scale,x,y-5*scale,x+scale,y-5*scale);
    path = [path bezierCurvePath(x+scale,y-5*scale,x+2*scale,y-5*scale,x+3*scale,y-4*scale,x+4*scale,y-4*scale)];
    path = [path bezierCurvePath(x+4*scale,y-4*scale,x+3*scale,y-4*scale,x+3*scale,y-3*scale,x+2*scale,y-3*scale)];
    path = [path bezierCurvePath(x+2*scale,y-3*scale,x+scale,y-3*scale,x+scale,y-4*scale,x-0.5*scale,y-4*scale)];
%     t2 = tic;
    Screen('FillPoly', window, [0 1 0], path');
%     t2 = toc(t2);
%     display(['orangeS2= ',num2str(t2)])
end

function drawApple(window,scale,x,y)
    
    % red fruit
    path = bezierCurvePath(x-2*scale,y-3*scale,x-2*scale,y-4*scale,x-3*scale,y-4*scale,x-4*scale,y-4*scale);
    path = [path bezierCurvePath(x-4*scale,y-4*scale,x-5*scale,y-4*scale,x-6*scale,y-3*scale,x-6*scale,y)];
    path = [path bezierCurvePath(x-6*scale,y,x-6*scale,y+3*scale,x-4*scale,y+6*scale,x-2.5*scale,y+6*scale)];
    path = [path quadraticCurvePath(x-2.5*scale,y+6*scale,x-scale,y+6*scale,x-scale,y+5*scale)];
    path = [path bezierCurvePath(x-scale,y+5*scale,x-scale,y+6*scale,x,y+6*scale,x+scale,y+6*scale)];
    path = [path bezierCurvePath(x+scale,y+6*scale,x+3*scale,y+6*scale,x+5*scale,y+3*scale,x+5*scale,y)];
    path = [path bezierCurvePath(x+5*scale,y,x+5*scale,y-3*scale,x+3*scale,y-4*scale,x+2*scale,y-4*scale)];
    path = [path quadraticCurvePath(x+2*scale,y-4*scale,x,y-4*scale,x,y-3*scale) [x-2*scale; y-3*scale]];
    Screen('FillPoly', window, [1 0 0], path');
    
    % stem
    drawQuadraticCurve(window,[255 153 0]/255.0,x-scale,y-3*scale,x-scale,y-5*scale,x,y-5*scale,scale);
    
    % shine
    drawQuadraticCurve(window,[1 1 1],x+2*scale,y+3*scale,x+3*scale,y+3*scale,x+3*scale,y+scale,scale);
end

function drawMelon(window,scale,x,y)
    % draw body
    Screen('FillOval', window, [123 243 49]/255.0, [x-5.5*scale y-3.5*scale x+5.5*scale y+7.5*scale]);
   
    % draw stem
    Screen('DrawLine', window, [105 180 175]/255.0, x, y-4*scale, x, y-5*scale, scale);
    drawQuadraticCurve(window,[105 180 175]/255.0,x+2*scale,y-5*scale,x-3*scale,y-5*scale,x-3*scale,y-6*scale,scale);
    

    % dark lines
%     Screen('DrawLine', window, [105 180 175]/255.0, x, y-2*scale, x-4*scale, y+2*scale, scale);
%     Screen('DrawLine', window, [105 180 175]/255.0, x-4*scale, y+2*scale, x-scale, y+5*scale, scale);
%     Screen('DrawLine', window, [105 180 175]/255.0, x-3*scale, y-scale, x-2*scale, y, scale);
%     Screen('DrawLine', window, [105 180 175]/255.0, x-2*scale, y+6*scale, x+scale, y+3*scale, scale);
%     Screen('DrawLine', window, [105 180 175]/255.0, x+scale, y+7*scale, x+3*scale, y+5*scale, scale);
%     Screen('DrawLine', window, [105 180 175]/255.0, x+3*scale, y+5*scale, x, y+2*scale, scale);
%     Screen('DrawLine', window, [105 180 175]/255.0, x, y+2*scale, x+3*scale, y-scale, scale);
%     Screen('DrawLine', window, [105 180 175]/255.0, x+2*scale, y, x+4*scale, y+2*scale, scale);
        
    % dark spots
    spots = [0,-2,-1,-1,-2,0,-3,1,-4,2,-3,3,-2,4,-1,5,-2,6,-3,-1,1,7,2,6, ...
        3,5,2,4,1,3, 0,2,1,1,2,0,3,-1,3,1,4,2];

    for i=1:2:length(spots)
        x1 = spots(i)*scale+x;
        y1 = spots(i+1)*scale+y;
        Screen('FillOval', window, [105 180 175]/255.0, [x1-0.65*scale y1-0.65*scale x1+0.65*scale y1+0.65*scale]);
    end

    % white spots
    spots = [0 -3 -2 -1 -4 1 -3 3 1 0 -1 2 -1 4 3 2 1 4];
    for i=1:2:length(spots)
        x1 = spots(i)*scale+x;
        y1 = spots(i+1)*scale+y;
        Screen('FillOval', window, [1 1 1], [x1-0.65*scale y1-0.65*scale x1+0.65*scale y1+0.65*scale]);
    end

end