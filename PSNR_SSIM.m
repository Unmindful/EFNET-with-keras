
dataDir1 = 'Validation Data/Set14';
dataDir2 = 'Predicted_image/Set14_EFNet+';
PSNR = [];SSIM = [];
f_lst1 = [];f_lst1 = [f_lst1; dir(fullfile(dataDir1, '*.jpg'))];f_lst1 = [f_lst1; dir(fullfile(dataDir1, '*.bmp'))];f_lst1 = [f_lst1; dir(fullfile(dataDir1, '*.png'))];
f_lst2 = [];f_lst2 = [f_lst2; dir(fullfile(dataDir2, '*.jpg'))];f_lst2 = [f_lst2; dir(fullfile(dataDir2, '*.bmp'))];f_lst2 = [f_lst2; dir(fullfile(dataDir2, '*.png'))];

for f_iter = 1:numel(f_lst1)

    f_info1 = f_lst1(f_iter);    f_info2 = f_lst2(f_iter);
    if f_info1.name == '.'
        continue;
    end
    f_path1 = fullfile(dataDir1,f_info1.name);    f_path2 = fullfile(dataDir2,f_info2.name);
    disp(f_path1);
    img_gt = imread(f_path1);img_pred = imread(f_path2);
    if size(img_gt,3)~=3
        img_gt = repmat(img_gt, [1 1 3]);
    end
    if size(img_pred,3)~=3
        img_pred = repmat(img_pred, [1 1 3]);
    end
    if size(img_gt,1)~=size(img_pred,1)
        gap = abs(size(img_gt,1)-size(img_pred,1));
        if size(img_gt,1)<size(img_pred,1)
            img_pred = img_pred(1:size(img_pred,1)-gap,:,:);
        else
            img_gt = img_gt(1:size(img_gt,1)-gap,:,:);
        end
    end
    if size(img_gt,2)~=size(img_pred,2)
        gap = abs(size(img_gt,2)-size(img_pred,2));
        if size(img_gt,2)<size(img_pred,2)
            img_pred = img_pred(:,1:size(img_pred,2)-gap,:);
        else
            img_gt = img_gt(:,1:size(img_gt,2)-gap,:);
        end
    end
    img_gt = rgb2ycbcr(img_gt);img_pred = rgb2ycbcr(img_pred);
    img_gt = img_gt(:,:,1);img_pred = img_pred(:,:,1);
    PSNR = [PSNR;psnr(img_pred, img_gt)]; 
    SSIM = [SSIM;ssim(img_pred, img_gt)]; 
end
PSNR_sum = sum(PSNR); SSIM_sum = sum(SSIM);
PSNR_avg = PSNR_sum/f_iter;SSIM_avg = SSIM_sum/f_iter;

