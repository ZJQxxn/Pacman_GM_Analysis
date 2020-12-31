function path = bezierCurvePath(p0x,p0y,p1x,p1y,p2x,p2y,p3x,p3y)
    
    %d=sqrt((p3x-p0x)*(p3x-p0x)+(p3y-p0y)*(p3y-p0y));
    t=0:0.25:1;
    px=(1-t).*(1-t).*(1-t)*p0x+3*(1-t).*(1-t).*t*p1x+3*(1-t).*t.*t*p2x+t.*t.*t*p3x;
    py=(1-t).*(1-t).*(1-t)*p0y+3*(1-t).*(1-t).*t*p1y+3*(1-t).*t.*t*p2y+t.*t.*t*p3y;
    
    path = [px; py];
end