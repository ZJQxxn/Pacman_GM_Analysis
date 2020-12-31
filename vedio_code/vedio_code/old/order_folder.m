function [folder, folder_order] = order_folder(folder_path)

% folder_path =  uigetdir(cd,'Open Data folder');
if ~folder_path
    error('you must select a data folder')
end

folder_table = struct2table(dir(folder_path));
folder_table = folder_table(3:end,:);
folder_name = string(folder_table.name); % the list of folders
folder_name = folder_name(folder_table.isdir == 1);

for i = 1:length(folder_name)
    char_i = split(folder_name(i),'-');
    char(i,1) = strjoin(char_i(2:4),'-');
end

char = datetime(char,'InputFormat','dd-MMM-yyyy','Locale','en_US');
[~, I] = sortrows(char); % sort folders by current_round and used_trial

folder(1:length(I)) = folder_name(I);
folder_order = I;
end