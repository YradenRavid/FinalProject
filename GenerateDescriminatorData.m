function  GenerateDescriminatorData(InputPath,OutputPath,brains,Slices,RangeOfDegrees)
%%%%%%%%%%%%%%%%%%%%%%%%%% instructions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function do:
% Unzip and load image from input path
% Rotate as written
% Save images in output path
% to load the images to matlab use: R5 = matfile('5Degrees.mat'); -> R5.rotate_image

% InputPath - where the data is
% OutputPath - where you want to save the data
% brains - vector: what brains do you want to use?
% Slices - vector: what slices do we take?
% RangeOfDegrees - vector that indicates the range orf degrees. i.e
% -5:0.5:5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('C:\Users\ירדן רביד\Desktop\MRI'); %where nii library is
addpath(InputPath);


%Rotation Matrix
for i=1:numel(RangeOfDegrees)
    teta = RangeOfDegrees(i);
    teta = deg2rad(teta);
    Rotate = [cos(teta) sin(teta) 0 ; -sin(teta) cos(teta) 0 ; 0 0 1]; 
    tform(i) = affine2d(Rotate);
end

fid = fopen('PathFile.txt','w');

for i=brains
    F=dir(fullfile([InputPath '\' num2str(i)],'*.nii.gz'));
    addpath([InputPath '\' num2str(i)]);
    nii_I = load_untouch_nii(F.name);
    I = double(nii_I.img);
    clear nii_I
    I=ReSizeBrain(I);
    mkdir([OutputPath  '\BrainNum' num2str(i)])
    Path = [OutputPath '\BrainNum'  num2str(i)];
    for j=1:numel(Slices)
        mkdir([Path  '\Slice' num2str(Slices(j))])
        currentPath=[Path  '\Slice' num2str(Slices(j))];
        fprintf(fid,'%s\n',currentPath);
        fix_image=I(:,:,Slices(j));
        save([currentPath '\fix_image'],'fix_image'); 
        mkdir([currentPath '\RotateImages'])
        currentPath = [currentPath '\RotateImages'];
        for h=1:numel(RangeOfDegrees)
            Tform = tform(h);
            rotate_image = imwarp(fix_image, Tform , 'OutputView', imref2d(size(fix_image)));
            save([currentPath '\' num2str(RangeOfDegrees(h)) 'Degrees'], 'rotate_image');
        end
    end
end

fclose(fid);

end

