clc
clear
close

% Set5 Set14 Urban100 B100
target = 'nova'; %name of testing dataset
dataDir = fullfile('../../raw/', target);

count = 0;
f_lst = [];
f_lst = [f_lst; dir(fullfile(dataDir, '*.jpg'))];
f_lst = [f_lst; dir(fullfile(dataDir, '*.bmp'))];
f_lst = [f_lst; dir(fullfile(dataDir, '*.png'))];
folder = fullfile('../../raw/', [target '_0.5']); %save path
mkdir(folder);
scale_factor = 2;

for f_iter = 1:numel(f_lst)
    
    f_info = f_lst(f_iter);
    if f_info.name == '.'
        continue;
    end
    f_path = fullfile(dataDir,f_info.name);
    
    NAME =strsplit(f_info.name, '.');
    NAME = NAME{1};
    
    disp(f_path);
    img_raw = imread(f_path);
    
    LR = imresize(img_raw, 1/scale_factor, 'bicubic');
   
    imwrite(LR, [folder, '/', NAME, '_0.5b', '.bmp'])
    
    display(sprintf('%d/ %d',f_iter, numel(f_lst)));
end
