function path = quadraticCurvePath(p0x,p0y,p1x,p1y,p2x,p2y)

t=0:0.25:1;
px=(1-t).*(1-t)*p0x+2*(1-t).*t*p1x+t.*t*p2x;
py=(1-t).*(1-t)*p0y+2*(1-t).*t*p1y+t.*t*p2y;
path = [px; py];
