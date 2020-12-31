path = "/media/tree/HardDisk/pacman_data/data_2018/monkeys";
a = struct2table(dir(path));
b = string(a.name(6:end));
folder_name = string(zeros(length(b), 1));
for i = 1:length(b)
    name = b(i);
    Elements = split(name, '-');
    if strcmp(Elements(1), "O")
        folder_name = join(["Omega", ...
            string(datetime(2018, double(Elements(2)), double(Elements(3)))), ...
            "1"], '-');
        Omega = 1;
    elseif strcmp(Elements(1), "P")
        folder_name = join(["Patamon", ...
            string(datetime(2018, double(Elements(2)), double(Elements(3)))), ...
            "1"], '-');
        Omega = 0;
    end
    mkdir(folder_name)
    %% rename file
    file_table = struct2table(dir(fullfile(strcat(path, '/', name), "*.mat")));
    file_name = string(file_table.name); % the list of files
    temp = split(file_name, '.');
    file_name = temp(:,1);
    if Omega
        a = join(["Omega", ...
            string(datetime(2018, double(Elements(2)), double(Elements(3))))], ...
            '-');
        fileList = strcat(file_name, "-", a, ".mat");
    else
        a = join(["Patamon", ...
            string(datetime(2018, double(Elements(2)), double(Elements(3))))], ...
            '-');
        fileList = strcat(file_name, "-", a, ".mat");
    end
    for j = 1:length(file_name)
        copyfile(strcat(path, "/", name, "/", file_name(j), ".mat"),strcat(folder_name, "/"))
        movefile(strcat(folder_name,"/", file_name(j), ".mat"), ...
            strcat(folder_name,"/", fileList(j)))
    end
    
end