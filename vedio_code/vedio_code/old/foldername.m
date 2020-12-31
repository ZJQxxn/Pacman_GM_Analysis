function folder_name = foldername(edfFile)

if isa(edfFile, 'string')
    edfFile = char(edfFile);
end
if edfFile(1) == 'p'
    Monkey = 'Patamon';
elseif edfFile(1) == 'o'
    Monkey = 'Omega';
end

if length(edfFile) == 5
    t = datetime(2019,str2double(edfFile(2)),str2double(edfFile(3:4)));
    inum = edfFile(5);
    folder_name = sprintf('%s-%s-%s', Monkey, t, inum);
elseif length(edfFile) == 4
    t = datetime(2019,str2double(edfFile(2)),str2double(edfFile(3)));
    inum = edfFile(4);
    folder_name = sprintf('%s-%s-%s', Monkey, t, inum);
end

end