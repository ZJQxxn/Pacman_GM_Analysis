sca
clearvars
clear -global
clear PsychImaging
close all force
clc
warning on
KbName('UnifyKeyNames')
addpath(genpath("old"))
%% initial variables
dataName = 'out.csv'; %% draw vedio from dataName in data folder
globalDefinitions;
suc_num = 0;
Endreward = 0.16;
num_j = 1; % the number of images
time_cp = datetime('01-Mar-2019','InputFormat','dd-MMM-yyyy','Locale','en_US');
rewd.count = 0;
num = 1;
%% detect data struct
dataPath = strcat('data/',dataName);
opts = detectImportOptions(dataPath);
labelPos = find(contains(string(opts.VariableNames), 'label_'));
opts = setvartype(opts,opts.VariableNames(labelPos),'double');
%% create vedio folder
% the path of vedio folder
datapath = "vedio/";
fprintf('create %s folder\n', datapath);
mkdir(sprintf('%s', datapath));
%% create diary
cur_path = cd;
if ~exist(strcat(datapath,'myDiaryFile'), 'file')
    cd(datapath)
    diary myDiaryFile
    cd(cur_path);
else
    warning('myDiaryFile is already in %s',strcat(datapath,'myDiaryFile'))
end
%% init keyboard
key_escape = KbName('escape');
[keyDown, secs, keyCode] = KbCheck;
%% read data
T = readtable(dataPath);
% T = readtable(dataPath, opts);
% Tdata = readtable('/home/tree/Desktop/notoptimal_files_small.csv', 'ReadVariableNames',false);
% validFile = ismember(string(T.file), string(Tdata.Var1));
% T = T(validFile,:);
T.energizers = regexp(T.energizers,'\d*','Match');
T.beans = regexp(T.beans,'\d*','Match');
T.ghost1Pos = regexp(T.ghost1Pos,'\d*','Match');
T.ghost2Pos = regexp(T.ghost2Pos,'\d*','Match');
T.pacmanPos = regexp(T.pacmanPos,'\d*','Match');
T.label_planned_hunting = cellfun(@str2double, T.label_planned_hunting);
label = pickFromT(T,labelPos);
load('map.mat','empty_map');
[indexLabel,trialLabel] = indexFromLabel(T);
%% open window
init_csv;

for i = 1:length(indexLabel)
    file_name = trialLabel(i);
    fileNum = sum(ismember(trialLabel, file_name));
    round_now = split(file_name, {'-','.'});
    %% vedio folder name
    fname1 = sprintf('%s%s', datapath, join(round_now(3:6)','-'));
    fname2 = strcat(join(round_now(1:6)','-'), '-', sprintf('%d', fileNum));
    fname = sprintf('%s/%s', fname1, fname2);
    if exist(fname,'dir')
        fprintf('%s folder already exists\n', fname);
    else
        fprintf('create %s folder\n', fname);
        mkdir(fname);
    end
    %% create folder for pictures with vedio file name
    vname = strcat(join(round_now(1:6)','-'), '-', sprintf('%d', num));
    name = sprintf('%s%s', datapath, vname);
    if exist(sprintf('%s', name),'dir')
        num = num + 1;
        vname = strcat(join(round_now(1:6)','-'), '-', sprintf('%d', num));
        name = sprintf('%s%s', datapath, vname);
        fprintf('create %s folder\n', name);
        mkdir(sprintf('%s', name));
    else
        num = 1;
        vname = strcat(join(round_now(1:6)','-'), '-', sprintf('%d', num));
        name = sprintf('%s%s', datapath, vname);
        fprintf('create %s folder\n', name);
        mkdir(sprintf('%s', name));
    end
    
    %% draw
    for index = indexLabel(i,1):indexLabel(i,2)
        if file_name ~= string(T.file(index))
            error("file name has something wrong")
        end
        num_j = draw_csv(index, empty_map, T, num_j, name, label, file_name, labelPos);
        [keyDown, secs, keyCode] = KbCheck;
        if keyCode(key_escape)
            break;
        end
    end
    %% create mp4 file
%     if ~exist(sprintf('%s/%s.mp4', fname, vname),'file')
%         [status,~] = unix(sprintf('ffmpeg -r 30 -f image2 -i %s/%%5d.jpg -vf pad=%d:%d:100:0:black %s/%s.mp4\n', ...
%             name, vedioWidth, vedioHeight, fname, vname));
%         if status
%             fprintf('fail to create %s/%s.mp4\n', fname, vname)
%             fprintf('********************\n')
%         else
%             fprintf('create %s/%s.mp4 successfully\n', fname, vname)
%             fprintf('====================\n')
%         end
%     else
%         fprintf('%s/%s.mp4 is already exist\n', fname, vname)
%         fprintf('********************\n')
%     end
    num_j = 1;
end
sca
rmpath(genpath("old"))
diary off

function [index,trial] = indexFromLabel(T)
% index of notoptimal
a = T.Var1+1;
a(:,2) = conv(a,[1,-1], 'same');
a(:,3) = T.index(a(:,1));
a(:,4) = conv(T.index(a(:,1)),[1,-1], 'same');
a(a(:,2) <= 5 & a(:,2) > 1 & a(:,2) == a(:,4),[2,4]) = 1;
index_i = find(a(:,2) ~= 1 | a(:,4) ~= 1);
index(:,2) = index_i; % end point of "a index"
index(1,1) = 1;
index(2:end,1) = index_i(1:end-1)+1; % start point of "a index"
% trial
trial = string(T.file(a(:,1)));
trial = trial(index_i(:,1));
[~,LocbStart] = ismember(trial, T.file);
[~,LocbEnd] = ismember(trial, T.file, 'legacy');
index(:,3) = LocbStart;% lowest T index of every range
index(:,4) = LocbEnd; % highest T index of every range
% index include -5:5
index(:,5) = a(index(:,1)); % start point of T index
index(:,6) = a(index(:,2)); % end point of T index
lowBound = index(:,3) <= index(:,5)-5;
index(~lowBound,7) = index(~lowBound,3);
index(lowBound,7) = index(lowBound,5) - 5;
highBound = index(:,4) >= index(:,6)+5;
index(~highBound,8) = index(~highBound,4);
index(highBound,8) = index(highBound,6) + 5;
index = index(:,[7,8]);
end

function label = pickFromT(T,labelPos)
a = char(T.Properties.VariableNames(labelPos));
b = a(:,7:end);
label = "";
for index = 1:length(labelPos)
    label(index) = b(index,~isspace(b(index,:)));
end
end