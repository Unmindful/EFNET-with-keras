
dataDir = 'Validation Data/DIV2K';
folder=fullfile('Validation Data','Test Images','DIV2K_4scale_rgb');
mkdir(folder);
count = 0;
f_lst = [];
f_lst = [f_lst; dir(fullfile(dataDir, '*.jpg'))];
f_lst = [f_lst; dir(fullfile(dataDir, '*.bmp'))];
f_lst = [f_lst; dir(fullfile(dataDir, '*.png'))];
scale=4;

for f_iter = 1:numel(f_lst)

    f_info = f_lst(f_iter);
    if f_info.name == '.'
        continue;
    end
    f_path = fullfile(dataDir,f_info.name);
    disp(f_path);
    img_raw = imread(f_path);
    
    if size(img_raw,3)~=3
        img_raw = repmat(img_raw, [1 1 3]);
    end

    
    img_raw = im2double(img_raw);
    img_size = size(img_raw);
                       
    % Actual Alignment of image            
    patch_name = sprintf('%s/%d.png',folder,count);            
    patch = imresize(img_raw,1/scale,'bicubic');
    imwrite(patch,patch_name);
    count = count + 1;     
    display(count);
    
    
end
