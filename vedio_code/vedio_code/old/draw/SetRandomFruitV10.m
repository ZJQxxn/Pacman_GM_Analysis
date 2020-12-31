function map_with_fruit = SetRandomFruitV10( map )

fruits = ['A' ; 'O' ; 'M' ; 'C' ; 'S'];
load('map_v10_regions.mat');
fruit_region = [];
e_count = 1;
for i = 1:1008
    if map(i) == 'o'
        energizer(e_count) = i;
        e_count = e_count +1;
        if ismember(i,region1)
            fruit_region = [fruit_region ;region1];
        elseif ismember(i,region2)
            fruit_region = [fruit_region ;region2];
        elseif ismember(i,region3)
            fruit_region = [fruit_region ;region3];
        elseif ismember(i,region4)
            fruit_region = [fruit_region ;region4];
        end
            
    end
    
end
fruit_position = IndexToPosition(datasample(fruit_region,1));
e1 = IndexToPosition(energizer(1));
e2 = IndexToPosition(energizer(2));
e3 = IndexToPosition(energizer(3));

while abs(fruit_position(1) - e1(1)) + abs(fruit_position(2) - e1(2)) < 6 ...
        || abs(fruit_position(1) - e2(1)) + abs(fruit_position(2) - e2(2)) < 6 ...
        || abs(fruit_position(1) - e3(1)) + abs(fruit_position(2) - e3(2)) < 6
    fruit_position = IndexToPosition(datasample(fruit_region,1));
end
fruit = PositionToIndex(fruit_position(1),fruit_position(2));
map(fruit) = datasample(fruits,1);
map_with_fruit = map;

end

