%灰度图直方图匹配matlab(不使用histeq)
clear
 
im      = imread('/data/datasets/SMALL/Kodak24/kodim05.png');
imRef   = imread('/data/datasets/SMALL/Kodak24/kodim21.png');
hist    = imhist(im);                % Compute histograms
histRef = imhist(imRef);
cdf     = cumsum(hist) / numel(im);  % Compute CDFs
cdfRef  = cumsum(histRef) / numel(imRef);
 
% Compute the mapping
M   = zeros(1,256);
for idx = 1 : 256
    [tmp,ind] = min(abs(cdf(idx) - cdfRef));
    M(idx)    = ind-1;
end
 
% Now apply the mapping to get first image to make
% the image look like the distribution of the second image
imMatch = uint8(M(double(im)+1));
 
figure;%显示原图像、匹配图像和匹配后的图像
subplot(1,3,1),imshow(im,[]);title('原图像');
subplot(1,3,2),imshow(imRef,[]);title('匹配图像');
subplot(1,3,3),imshow(imMatch,[]);title('匹配之后图像');
figure;%显示原图像、匹配图像和匹配后图像的直方图
subplot(3,1,1),imhist(im,64);title('原图像直方图');
subplot(3,1,2),imhist(imRef,64);title('匹配图像直方图');
subplot(3,1,3),imhist(uint8(imMatch),64);title('匹配之后图像直方图');