function position = IndexToPosition(index)
%INDEXTOPOSITION Summary of this function goes here
%   Detailed explanation goes here

pos_x= rem(index,28);
if pos_x ==0
    pos_x =28;
    pos_y = fix(index/28);
else
    pos_y = fix(index/28)+1;
end
position = [pos_x pos_y];
end

