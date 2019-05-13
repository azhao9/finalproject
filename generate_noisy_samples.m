files = dir('subset/*.jpg');

for i=1:length(files)
    fname = files(i).name;
    I = imread(strcat('subset/', fname));
    
    % Generates noisy samples from each file
    imwrite(imnoise(I, 'gaussian'), strcat('gaussian/', fname));
    imwrite(imnoise(I, 'speckle'), strcat('speckle/', fname));
    imwrite(imnoise(I, 'salt & pepper'), strcat('saltpepper/', fname));
end