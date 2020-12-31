function [file_list, file_order] = order_file(type, file_path)

% file_path =  uigetdir(cd,'Open Data folder');
% if ~file_path
%     error('you must select a data folder')
% end

type = strcat('*', type);
file_table = struct2table(dir(fullfile(file_path, type)));
file_name = string(file_table.name); % the list of files
if isempty(file_name)
    file_list = [];
    file_order = [];
    return
end
%% 
char_i = split(file_name,{'-','.'},2);
char_num = str2double(char_i(:,1:2));
char_trial =  char_num(:,1) * 1000 + char_num(:,2);
[~, I] = sortrows(char_trial); % sort files by current_round and used_trial
file_name(1:length(I)) = file_name(I);

char_time = "";
for i = 1:length(file_name)
    char_i = split(file_name(i),{'-','.'})';
    char_time(i,1) = strjoin(char_i(4:6),'-');
end
char = datetime(char_time,'InputFormat','dd-MMM-yyyy','Locale','en_US');
[~, I] = sortrows(char); % sort files by current_round and used_trial
file_list(1:length(I)) = file_name(I);
file_order = I;


end
