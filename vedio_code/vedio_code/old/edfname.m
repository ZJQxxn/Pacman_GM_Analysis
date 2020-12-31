function edfFile = edfname(folder, folder_position, round_now)
inum = split(folder(folder_position),'-');
day_num = num2str(round_now(4));
if strcmp(day_num(1),'0')
    dnum = day_num(2);
else
    dnum = day_num(1:2);
end

if strcmp(round_now(3),'Patamon')
    edfFile = sprintf('p%d%s%s', month(datetime(join(round_now(4:6)','-'))), dnum,inum(5));
elseif strcmp(round_now(3),'Omega')
    edfFile = sprintf('o%d%s%s', month(datetime(join(round_now(4:6)','-'))), dnum,inum(5));
end
end